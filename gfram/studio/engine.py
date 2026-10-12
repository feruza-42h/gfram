"""
Recognition engine of the studio.

A thin layer over gfram.SimpleRecognizer that exposes every intermediate result
(landmarks, aligned crop, pose, both embeddings, fusion features, per-stage timings)
so the interface can show how the hybrid method reaches its decision. Matching,
template protection and storage are the library's own code.
"""

import contextlib
import copy
import json
import logging
import pickle
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np

from . import STUDIO_HOME

logger = logging.getLogger(__name__)

STRICT_THRESHOLD_KEY = 'far_1e-3'   # ~0.1% false accepts on LFW


@contextlib.contextmanager
def _native_stderr_silenced():
    """Redirect the process-level stderr (C++ libraries) to /dev/null; Python exceptions still propagate."""
    import os
    import sys
    try:
        sys.stderr.flush()
        saved = os.dup(2)
        devnull = os.open(os.devnull, os.O_WRONLY)
    except OSError:
        yield
        return
    os.dup2(devnull, 2)
    try:
        yield
    finally:
        os.dup2(saved, 2)
        os.close(saved)
        os.close(devnull)


def _now() -> str:
    return datetime.now().strftime('%Y-%m-%d %H:%M:%S')


def _ms(t0: float) -> float:
    return (time.perf_counter() - t0) * 1000.0


class Engine:
    def __init__(self):
        import torch
        torch.set_num_threads(2)
        from gfram.api.simple_recognizer import SimpleRecognizer
        self.lock = threading.RLock()
        self.home = Path(STUDIO_HOME)
        self.rec = SimpleRecognizer(db_path=str(self.home / 'database'), contribute=False)
        self.meta_file = self.home / 'people.json'      # card data of people (no biometrics)
        self.events_file = self.home / 'events.jsonl'   # recognition journal (no images)
        self.meta: Dict[str, Dict] = self._read_json(self.meta_file, {})
        self.events: List[Dict] = self._read_events()
        self._sync_meta()
        self.threshold = None          # None: library's calibrated threshold
        self._gallery = None
        self._gallery_version = -1
        self._version = 0
        self.stolen_copy: Optional[Dict] = None

    # ------------------------------------------------------------------ setup

    def warmup(self, progress=lambda msg: None):
        """Load every model once so the first real request is fast."""
        with _native_stderr_silenced():      # MediaPipe / TFLite print C++ start-up notes to stderr
            progress('MediaPipe FaceMesh (478)…')
            _ = self.rec.detector
            progress('GeometricTransformer…')
            _ = self.rec.embedder
            progress('MobileFaceNet + adaptive fusion…')
            _ = self.rec.hybrid
            img = np.full((240, 240, 3), 128, np.uint8)
            self.rec.detector.detect(img)
            self.hybrid.appearance.embed(np.zeros((1, 112, 112, 3), np.uint8))
            self.rec.embedder.embed(np.random.rand(478, 3).astype(np.float32))

    @property
    def hybrid(self):
        return self.rec.hybrid

    @property
    def fusion(self):
        return self.rec.hybrid.fusion

    @property
    def embedder(self):
        return self.rec.embedder

    def calibrated_thresholds(self) -> Dict[str, float]:
        return dict(self.fusion.thresholds)

    def current_threshold(self) -> float:
        return float(self.threshold) if self.threshold is not None else self.rec.threshold

    def model_info(self) -> Dict:
        geo_params = sum(p.numel() for p in self.embedder.model.parameters())
        from gfram.models.hybrid import ensure_appearance_model
        import os
        app_path = ensure_appearance_model(self.fusion.appearance_model)
        return {
            'geometry_version': self.embedder.version,
            'geometry_params': int(geo_params),
            'geometry_dim': int(self.embedder.dim),
            'appearance_model': self.fusion.appearance_model,
            'appearance_mb': os.path.getsize(app_path) / 1e6,
            'fusion_features': list(self.fusion.features),
            'thresholds': self.calibrated_thresholds(),
        }

    # ------------------------------------------------------------------ analysis

    def analyze(self, image: np.ndarray, max_faces: int = 5) -> Dict:
        """
        All faces in a BGR image with every feature the hybrid method uses.
        Returns {'faces': [...], 'timing': {...}} sorted by face size (largest first).
        """
        from gfram.models.hybrid import align_crop, head_pose, sharpness
        with self.lock:
            t0 = time.perf_counter()
            found = [f for f in self.rec.detector.detect(image) if f.get('num_landmarks') == 478]
            t_det = _ms(t0)
            found.sort(key=lambda f: -(f['bbox'][2] * f['bbox'][3]) if f.get('bbox') else 0)
            faces = []
            t_geo = t_app = 0.0
            for f in found[:max_faces]:
                lm = f['landmarks'].astype(np.float32)
                t = time.perf_counter()
                geo = self.embedder.embed(lm)
                pose = head_pose(lm, self.embedder.reference)
                t_geo += _ms(t)
                t = time.perf_counter()
                crop = align_crop(image, lm)
                app, norm = self.hybrid.appearance.embed(crop[None])
                sharp = sharpness(crop)
                t_app += _ms(t)
                faces.append({
                    'landmarks': lm, 'bbox': f.get('bbox'), 'crop': crop,
                    'geo': geo, 'pose': pose, 'app': app[0].astype(np.float32),
                    'app_norm': float(norm[0]), 'sharp': float(sharp),
                })
            return {'faces': faces,
                    'timing': {'detect': t_det, 'geometry': t_geo, 'appearance': t_app}}

    def analyze_largest(self, image: np.ndarray) -> Optional[Dict]:
        r = self.analyze(image, max_faces=1)
        return r['faces'][0] if r['faces'] else None

    # ------------------------------------------------------------------ fusion

    def pair_breakdown(self, a: Dict, b: Dict) -> Dict:
        """Fusion of two faces, feature by feature (values, standardised values, logit contributions)."""
        fz = self.fusion
        g = {k: np.asarray(b[k])[None] for k in ('geo', 'pose', 'app', 'app_norm', 'sharp')}
        cols = fz.pair_features(a, g)
        x = np.array([float(cols[f][0]) for f in fz.features])
        z = (x - fz.mean) / fz.scale
        contrib = z * fz.coef
        logit = float(contrib.sum() + fz.intercept)
        prob = 1.0 / (1.0 + np.exp(-logit))
        return {'features': list(fz.features), 'values': x, 'z': z, 'contrib': contrib,
                'intercept': fz.intercept, 'logit': logit, 'probability': float(prob),
                's_app': float(cols['s_app'][0]), 's_geo': float(cols['s_geo'][0]),
                'yaw_diff': float(cols['yaw_diff'][0])}

    # ------------------------------------------------------------------ database

    def _changed(self):
        self._version += 1

    def gallery(self):
        with self.lock:
            if not self.rec._sample_names:
                return None
            if self._gallery_version != self._version:
                self._gallery = self.rec._gallery()
                self._gallery_version = self._version
            return self._gallery

    def _protect_query(self, face: Dict, protector=None) -> Dict:
        p = protector or self.rec.protector
        q = dict(face)
        q['geo'] = p.protect(face['geo'], 'geo')
        q['app'] = p.protect(face['app'], 'app')
        return q

    def identify(self, face: Dict, top_k: int = 5) -> Dict:
        """1:N search over the protected database: best candidates per person."""
        with self.lock:
            t = time.perf_counter()
            g = self.gallery()
            if g is None:
                return {'candidates': [], 'time_ms': _ms(t), 'faces': 0}
            probs = self.fusion.probability(self._protect_query(face), g)
            best: Dict[str, float] = {}
            for name, p in zip(self.rec._sample_names, probs):
                if p > best.get(name, -1.0):
                    best[name] = float(p)
            ranked = sorted(best.items(), key=lambda kv: -kv[1])[:top_k]
            return {'candidates': ranked, 'time_ms': _ms(t), 'faces': len(probs)}

    def enroll(self, name: str, face: Dict, save: bool = True, source: str = 'camera') -> int:
        """Add an analysed face. Only protected forms are stored (same as gfram.add)."""
        rec = self.rec
        with self.lock:
            rec._ensure_geometry()
            if name not in rec._persons:
                rec._persons[name] = rec._next_id
                rec._next_id += 1
            p = rec.protector
            rec._sample_names.append(name)
            rec._sample_landmarks_enc.append(p.encrypt_landmarks(face['landmarks']))
            rec._sample_app.append(p.protect(face['app'], 'app'))
            rec._sample_quality.append({'app_norm': face['app_norm'], 'sharp': face['sharp']})
            rec._appearance_model = self.fusion.appearance_model
            geo = p.protect(face['geo'], 'geo')
            rec._geo = geo[None] if rec._geo is None else np.vstack([rec._geo, geo])
            rec._pose = face['pose'][None] if rec._pose is None else np.vstack([rec._pose, face['pose']])
            rec._geo_version = self.embedder.version
            yaw, pitch, _ = (float(x) for x in face['pose'])
            self._person_meta(name)['templates'].append({
                'added': _now(), 'source': source, 'yaw': round(yaw, 1), 'pitch': round(pitch, 1),
                'sharp': round(face['sharp'], 2), 'quality': round(face['app_norm'], 1)})
            if save:
                rec._save_database()
                self._write_meta()
            self._changed()
            return rec._sample_names.count(name)

    def save(self):
        with self.lock:
            self.rec._save_database()
            self._write_meta()

    def persons(self) -> List[Dict]:
        with self.lock:
            counts: Dict[str, int] = {}
            for n in self.rec._sample_names:
                counts[n] = counts.get(n, 0) + 1
            return [{'name': n, 'faces': counts.get(n, 0), 'id': i} for n, i in self.rec._persons.items()]

    def face_indices(self, name: str) -> List[int]:
        return [i for i, n in enumerate(self.rec._sample_names) if n == name]

    def stored_record(self, index: int) -> Dict:
        """Exactly what the database keeps for one enrolled face."""
        rec = self.rec
        with self.lock:
            return {
                'name': rec._sample_names[index],
                'landmarks_enc': rec._sample_landmarks_enc[index],
                'app_protected': rec._sample_app[index],
                'geo_protected': rec._geo[index] if rec._geo is not None else None,
                'pose': rec._pose[index] if rec._pose is not None else None,
                'quality': rec._sample_quality[index],
            }

    def remove(self, name: str) -> bool:
        with self.lock:
            ok = self.rec.remove(name)
            self.meta.pop(name, None)
            self._write_meta()
            self._changed()
            return ok

    def clear(self):
        with self.lock:
            self.rec.clear()
            self.meta = {}
            self._write_meta()
            self.stolen_copy = None
            self._changed()

    # ------------------------------------------------------------------ people management

    @staticmethod
    def _read_json(path: Path, default):
        try:
            return json.loads(path.read_text(encoding='utf-8'))
        except Exception:
            return default

    def _write_meta(self):
        self.home.mkdir(parents=True, exist_ok=True)
        tmp = self.meta_file.with_suffix('.tmp')
        tmp.write_text(json.dumps(self.meta, ensure_ascii=False, indent=1), encoding='utf-8')
        tmp.replace(self.meta_file)

    def _person_meta(self, name: str) -> Dict:
        m = self.meta.setdefault(name, {})
        for k, v in (('position', ''), ('department', ''), ('code', ''), ('note', ''), ('created', _now()),
                     ('templates', []), ('last_seen', None), ('seen', 0)):
            m.setdefault(k, v)
        return m

    def _sync_meta(self):
        """Cards for every enrolled person, one template entry per stored template."""
        counts: Dict[str, int] = {}
        for n in self.rec._sample_names:
            counts[n] = counts.get(n, 0) + 1
        for n in list(self.meta):
            if n not in counts:
                del self.meta[n]
        for n, c in counts.items():
            t = self._person_meta(n)['templates']
            while len(t) < c:
                t.append({'added': None, 'source': '?'})
            del t[c:]

    def person(self, name: str) -> Dict:
        with self.lock:
            return copy.deepcopy(self._person_meta(name))

    def update_person(self, name: str, **fields) -> None:
        with self.lock:
            m = self._person_meta(name)
            for k in ('position', 'department', 'code', 'note'):
                if k in fields:
                    m[k] = str(fields[k]).strip()
            self._write_meta()

    def rename_person(self, old: str, new: str) -> bool:
        new = new.strip()
        with self.lock:
            rec = self.rec
            if not new or old not in rec._persons or new in rec._persons:
                return False
            rec._persons[new] = rec._persons.pop(old)
            rec._sample_names = [new if n == old else n for n in rec._sample_names]
            self.meta[new] = self.meta.pop(old, {})
            rec._save_database()
            self._write_meta()
            self._changed()
            return True

    def delete_template(self, name: str, k: int) -> None:
        """Remove the k-th stored template of a person (the person goes when none are left)."""
        with self.lock:
            rec = self.rec
            idx = self.face_indices(name)
            if not 0 <= k < len(idx):
                return
            if len(idx) == 1:
                self.remove(name)
                return
            i = idx[k]
            for lst in (rec._sample_names, rec._sample_landmarks_enc, rec._sample_app, rec._sample_quality):
                del lst[i]
            keep = np.arange(len(rec._geo)) != i
            rec._geo, rec._pose = rec._geo[keep], rec._pose[keep]
            del self._person_meta(name)['templates'][k]
            rec._save_database()
            self._write_meta()
            self._changed()

    # ------------------------------------------------------------------ journal

    def _read_events(self) -> List[Dict]:
        out = []
        try:
            with open(self.events_file, encoding='utf-8') as f:
                for line in f:
                    out.append(json.loads(line))
        except Exception:
            pass
        return out[-5000:]

    def log_event(self, name: Optional[str], probability: float, recognized: bool, source: str = 'camera'):
        e = {'time': _now(), 'name': name if recognized else None, 'candidate': name,
             'probability': round(float(probability), 4), 'source': source}
        with self.lock:
            self.events.append(e)
            self.events = self.events[-5000:]
            if recognized and name in self.rec._persons:
                m = self._person_meta(name)
                m['last_seen'] = e['time']
                m['seen'] = m.get('seen', 0) + 1
            self.home.mkdir(parents=True, exist_ok=True)
            with open(self.events_file, 'a', encoding='utf-8') as f:
                f.write(json.dumps(e, ensure_ascii=False) + '\n')
        return e

    def clear_events(self):
        with self.lock:
            self.events = []
            self.events_file.unlink(missing_ok=True)

    # ------------------------------------------------------------------ storage audit

    IMAGE_SIGNATURES = {
        'JPEG (JFIF)': b'JFIF\x00', 'JPEG (Exif)': b'Exif\x00\x00', 'PNG': b'\x89PNG\r\n\x1a\n',
        'GIF': b'GIF89a', 'HEIC': b'ftypheic', 'WebP': b'WEBPVP8', 'TIFF': b'II*\x00\x08\x00',
    }

    def storage_report(self) -> Dict:
        """Every file the studio keeps on disk, and a byte-level search for embedded images."""
        with self.lock:
            files = []
            for p in sorted(self.home.rglob('*')):
                if not p.is_file():
                    continue
                raw = p.read_bytes()
                hits = [k for k, sig in self.IMAGE_SIGNATURES.items() if sig in raw]
                files.append({'path': str(p.relative_to(self.home)), 'abs': str(p), 'bytes': len(raw),
                              'images': hits, 'image_ext': p.suffix.lower() in ('.jpg', '.jpeg', '.png', '.bmp', '.heic', '.webp')})
            structure = []
            db = self.rec._db_file
            if db.exists():
                data = pickle.loads(db.read_bytes())
                for k, v in data.items():
                    structure.append({'key': k, 'type': type(v).__name__, 'items': len(v) if hasattr(v, '__len__') else None,
                                      'bytes': len(pickle.dumps(v))})
            return {'root': str(self.home), 'files': files, 'structure': structure,
                    'image_files': sum(1 for f in files if f['image_ext']),
                    'embedded_images': sum(len(f['images']) for f in files)}

    def reload_from_disk(self) -> float:
        """Throw away everything in memory and open the database file again (models are reused)."""
        from gfram.api.simple_recognizer import SimpleRecognizer
        with self.lock:
            t = time.perf_counter()
            old = self.rec
            new = SimpleRecognizer(db_path=str(old.db_path), contribute=False)
            new._detector, new._embedder, new._hybrid = old._detector, old._embedder, old._hybrid
            new._threshold = old._threshold
            self.rec = new
            self.meta = self._read_json(self.meta_file, {})
            self._sync_meta()
            self._changed()
            return _ms(t)

    def stats(self) -> Dict:
        with self.lock:
            db_file = self.rec._db_file
            return {
                'persons': len(self.rec._persons),
                'faces': len(self.rec._sample_names),
                'fingerprint': self.rec.protector.fingerprint,
                'db_path': str(self.rec.db_path),
                'db_bytes': db_file.stat().st_size if db_file.exists() else 0,
            }

    # ------------------------------------------------------------------ template protection demo

    def steal_copy(self):
        """Simulate an attacker copying the database file (protected templates only)."""
        with self.lock:
            g = self.gallery()
            if g is None:
                return False
            self.stolen_copy = {'names': list(self.rec._sample_names), 'gallery': copy.deepcopy(g),
                                'fingerprint': self.rec.protector.fingerprint}
            return True

    def rotate_key(self) -> str:
        with self.lock:
            fp = self.rec.rotate_template_key()
            self._changed()
            return fp

    def match_against_stolen(self, face: Dict) -> Optional[Dict]:
        """Best match of a live query (protected with the current key) in the stolen copy."""
        with self.lock:
            if not self.stolen_copy:
                return None
            probs = self.fusion.probability(self._protect_query(face), self.stolen_copy['gallery'])
            i = int(np.argmax(probs))
            return {'name': self.stolen_copy['names'][i], 'probability': float(probs[i]),
                    'fingerprint': self.stolen_copy['fingerprint']}
