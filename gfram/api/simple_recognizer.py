"""
GFRAM Simple Recognizer
=======================

Simplified face recognizer for end users.

Recognition is hybrid: the geometric module (478 3D landmarks, GeometricTransformer,
head pose) and a deep appearance model are combined by a quality-aware adaptive
fusion that outputs the probability that two faces show the same person.

The local database stores, for every enrolled face, its landmarks and appearance
features. When a newer geometry model arrives from the server the geometric part
is recomputed automatically, so nobody has to be re-enrolled after an update.

Author: Ortiqova F.S.
"""

import logging
import pickle
import shutil
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import torch

logger = logging.getLogger(__name__)

DB_FORMAT = 4  # 4: protected templates (keyed rotation + encrypted landmarks)
DEFAULT_DB_PATH = Path.home() / '.gfram' / 'database'
DEFAULT_THRESHOLD = 0.5  # used only if the model package carries no calibrated threshold


class SimpleRecognizer:
    """
    Simple Face Recognizer with persistent storage.
    """

    def __init__(self, db_path: Optional[str] = None, threshold: Optional[float] = None,
                 contribute: Optional[bool] = None, template_key: Optional[bytes] = None):
        """
        Args:
            db_path: Database directory (default ~/.gfram/database).
            threshold: Match probability needed to accept a match. Defaults to the
                threshold calibrated during training and stored in the model package.
            contribute: Share enrolled face data with the GFRAM server. None (default)
                follows the consent stored with gfram.set_contribution_consent();
                True / False decide for this recognizer only.
            template_key: 32-byte secret protecting the stored templates. Default: the
                installation key in ~/.gfram/keys/template.key (created on first use).
        """
        self.device = torch.device('cpu')
        self.db_path = Path(db_path) if db_path else DEFAULT_DB_PATH
        self.db_path.mkdir(parents=True, exist_ok=True)
        self._threshold = threshold
        self.contribute = contribute

        self._detector = None
        self._embedder = None
        self._hybrid = None
        self._extractor = None

        # Database: one entry per enrolled face
        self._persons: Dict[str, int] = {}
        self._next_id = 0
        self._sample_names: List[str] = []
        self._template_key = template_key
        self._protector = None
        self._sample_landmarks_enc: List[bytes] = []  # AES-GCM encrypted landmarks per face
        self._sample_app: List[np.ndarray] = []      # protected appearance template per face
        self._sample_quality: List[Dict] = []        # app_norm, sharp per face
        self._appearance_model: Optional[str] = None
        self._geo: Optional[np.ndarray] = None       # protected geometric templates, per model version
        self._pose: Optional[np.ndarray] = None
        self._geo_version: Optional[str] = None
        self._total_recognitions = 0

        self._load_database()
        logger.info("GFRAM ready!")

    # ------------------------------------------------------------------
    # Lazy components
    # ------------------------------------------------------------------

    @property
    def detector(self):
        if self._detector is None:
            from ..detectors import FaceDetector
            self._detector = FaceDetector(max_num_faces=5, refine_landmarks=True)
        return self._detector

    @property
    def embedder(self):
        """Geometric module (GeometricTransformer + Procrustes alignment)."""
        if self._embedder is None:
            from ..cloud.model_loader import ensure_model_available
            from ..models.package import FaceEmbedder

            model_path = ensure_model_available()
            if model_path is None:
                raise RuntimeError(
                    'No GFRAM model available: the server is unreachable and no bundled model was found. '
                    'Reinstall gfram or check your connection.')
            self._embedder = FaceEmbedder.from_file(model_path, self.device)
            logger.info(f'Model {self._embedder.version} loaded from {model_path}')
        return self._embedder

    @property
    def hybrid(self):
        """Appearance model + adaptive fusion, or None for a geometry-only model package."""
        if self._hybrid is None and self.embedder.fusion_spec is not None:
            from ..models.hybrid import AdaptiveFusion, HybridAnalyzer
            self._hybrid = HybridAnalyzer(self.embedder, AdaptiveFusion(self.embedder.fusion_spec))
        return self._hybrid

    @property
    def threshold(self) -> float:
        if self._threshold is not None:
            return self._threshold
        thresholds = self.hybrid.fusion.thresholds if self.hybrid else self.embedder.thresholds
        return float(thresholds.get('best_accuracy', DEFAULT_THRESHOLD))

    @property
    def protector(self):
        """Key-bound template protection (see gfram/security.py)."""
        if self._protector is None:
            self._protector = self._make_protector(None)
        return self._protector

    def _make_protector(self, fingerprint: Optional[str]):
        """
        Protector for the configured key. With a stored database fingerprint, also accept
        the key kept aside by an interrupted rotation (template.key.new / .old) and restore it.
        """
        from ..security import TemplateProtector, default_key_path, load_or_create_key, write_key
        if self._template_key is not None:
            protector = TemplateProtector(self._template_key)
            if fingerprint and protector.fingerprint != fingerprint:
                raise RuntimeError('template_key does not match the key this database was protected with')
            return protector

        path = default_key_path()
        protector = TemplateProtector(load_or_create_key(path))
        if not fingerprint or protector.fingerprint == fingerprint:
            return protector
        for alt in (path.with_suffix('.key.new'), path.with_suffix('.key.old')):
            if alt.exists():
                candidate = TemplateProtector(load_or_create_key(alt))
                if candidate.fingerprint == fingerprint:
                    write_key(candidate._key, path)
                    logger.warning(f'Restored the template key from {alt.name} after an interrupted rotation')
                    return candidate
        raise RuntimeError(
            f'The database in {self.db_path} is protected with a different template key than '
            f'{path}. Restore the original key file; without it the stored faces cannot be used.')

    # ------------------------------------------------------------------
    # Persistence
    # ------------------------------------------------------------------

    @property
    def _db_file(self) -> Path:
        return self.db_path / 'persons.pkl'

    def _load_database(self):
        if not self._db_file.exists():
            return
        try:
            with open(self._db_file, 'rb') as f:
                data = pickle.load(f)
        except Exception as e:
            logger.warning(f"Could not load database: {e}")
            return

        if data.get('format') == 3:
            self._migrate_unprotected(data)
            return
        if data.get('format') != DB_FORMAT:
            self._archive_legacy_database(data)
            return

        self._protector = self._make_protector(data['key_fingerprint'])
        self._persons = data['persons']
        self._next_id = data['next_id']
        self._sample_names = data['sample_names']
        self._sample_landmarks_enc = list(data['sample_landmarks_enc'])
        self._sample_app = list(data['sample_app'])
        self._sample_quality = list(data['sample_quality'])
        self._appearance_model = data.get('appearance_model')
        self._geo = data.get('geo')
        self._pose = data.get('pose')
        self._geo_version = data.get('geo_version')
        self._total_recognitions = data.get('total_recognitions', 0)
        logger.info(f"✅ Database loaded: {len(self._persons)} persons, {len(self._sample_names)} faces")

    def _migrate_unprotected(self, data: Dict):
        """gfram 3.2.0 stored plain templates and landmarks: protect them in place, nobody is lost."""
        p = self.protector
        self._persons = data['persons']
        self._next_id = data['next_id']
        self._sample_names = data['sample_names']
        self._sample_landmarks_enc = [p.encrypt_landmarks(lm) for lm in data['sample_landmarks']]
        self._sample_app = [p.protect(a, 'app') for a in data['sample_app']]
        self._sample_quality = list(data['sample_quality'])
        self._appearance_model = data.get('appearance_model')
        self._geo = p.protect(data['geo'], 'geo') if data.get('geo') is not None else None
        self._pose = data.get('pose')
        self._geo_version = data.get('geo_version')
        self._total_recognitions = data.get('total_recognitions', 0)
        self._save_database()
        logger.info(f'Database upgraded to protected templates: {len(self._sample_names)} faces')

    def _archive_legacy_database(self, data: Dict):
        """
        Databases from older versions lack the appearance features needed by hybrid
        recognition and cannot be converted without the original photos. Keep them
        aside instead of deleting.
        """
        version = {2: '3.1'}.get(data.get('format'), '3.0')
        legacy = self.db_path / f'legacy_v{version}'
        legacy.mkdir(exist_ok=True)
        for item in ('persons.pkl', 'geo_index', 'deep_index'):
            src = self.db_path / item
            if src.exists():
                shutil.move(str(src), str(legacy / item))
        names = sorted(data.get('persons', {}).keys())
        logger.warning(
            f'Database from gfram {version} moved to {legacy}. It cannot be upgraded to hybrid '
            f'recognition; please add these persons again: {", ".join(names) or "(none)"}')

    def _save_database(self):
        data = {
            'format': DB_FORMAT,
            'persons': self._persons,
            'next_id': self._next_id,
            'sample_names': self._sample_names,
            'key_fingerprint': self.protector.fingerprint,
            'sample_landmarks_enc': self._sample_landmarks_enc,
            'sample_app': self._sample_app,
            'sample_quality': self._sample_quality,
            'appearance_model': self._appearance_model,
            'geo': self._geo,
            'pose': self._pose,
            'geo_version': self._geo_version,
            'total_recognitions': self._total_recognitions,
        }
        tmp = self._db_file.with_suffix('.tmp')
        with open(tmp, 'wb') as f:
            pickle.dump(data, f)
        tmp.replace(self._db_file)

    def _ensure_geometry(self):
        """Recompute geometric embeddings and poses if built with another geometry model version."""
        n = len(self._sample_landmarks_enc)
        if n == 0:
            self._geo, self._pose = None, None
            return
        if self._geo is not None and self._geo_version == self.embedder.version and len(self._geo) == n:
            return
        from ..models.hybrid import head_pose
        logger.info(f'Re-computing geometry of {n} faces with model {self.embedder.version}')
        landmarks = np.stack([self.protector.decrypt_landmarks(b) for b in self._sample_landmarks_enc])
        self._geo = self.protector.protect(self.embedder.embed(landmarks), 'geo')
        self._pose = np.stack([head_pose(lm, self.embedder.reference) for lm in landmarks])
        self._geo_version = self.embedder.version
        self._save_database()

    def _gallery(self) -> Dict[str, np.ndarray]:
        self._ensure_geometry()
        return {
            'geo': self._geo,
            'pose': self._pose,
            'app': np.stack(self._sample_app),
            'app_norm': np.array([q['app_norm'] for q in self._sample_quality]),
            'sharp': np.array([q['sharp'] for q in self._sample_quality]),
        }

    # ------------------------------------------------------------------
    # Face processing
    # ------------------------------------------------------------------

    def _load_image(self, image: Union[str, np.ndarray]) -> np.ndarray:
        if isinstance(image, (str, Path)):
            import cv2
            img = cv2.imread(str(image))
            if img is None:
                raise ValueError(f"Could not load image: {image}")
            return img
        return image

    def _detect(self, image: np.ndarray) -> Optional[Dict]:
        """Largest face with a full 478-point mesh, or None."""
        faces = [f for f in self.detector.detect(image) if f.get('num_landmarks') == 478]
        if not faces:
            return None
        if len(faces) > 1:
            logger.warning(f"Multiple faces detected ({len(faces)}), using the largest")
        return max(faces, key=lambda f: f['bbox'][2] * f['bbox'][3] if f.get('bbox') else 0)

    def _analyze(self, image: np.ndarray) -> Optional[Dict]:
        """Detect the main face and compute all features used for matching."""
        face = self._detect(image)
        if face is None:
            return None
        landmarks = face['landmarks'].astype(np.float32)
        if self.hybrid is not None:
            features = self.hybrid.analyze(image, landmarks)
        else:
            from ..models.hybrid import head_pose
            features = {'geo': self.embedder.embed(landmarks), 'pose': head_pose(landmarks, self.embedder.reference)}
        features['landmarks'] = landmarks
        features['bbox'] = face.get('bbox')
        return features

    def _scores(self, query: Dict, gallery: Dict) -> np.ndarray:
        """Match probability (hybrid) or geometric cosine similarity (geometry-only package)."""
        if self.hybrid is not None:
            return self.hybrid.fusion.probability(query, gallery)
        return gallery['geo'] @ query['geo']

    def _send_to_server(self, name: str, f: Dict):
        """Share one enrolled face (never the photo) with the GFRAM server."""
        try:
            from ..cloud.consent import installation_id
            from ..cloud.server_client import contribute
            from ..detectors import LandmarkNormalizer
            from ..geometry.features import GeometricFeatureExtractor

            if self._extractor is None:
                self._extractor = (LandmarkNormalizer(), GeometricFeatureExtractor())
            normalizer, extractor = self._extractor
            result = contribute(
                person_id=name,
                landmarks=f['landmarks'],
                geometric_features=extractor.extract(normalizer.normalize(f['landmarks'])),
                embedding=f['geo'],
                extra={
                    'installation_id': installation_id(),
                    'appearance_embedding': f['app'],
                    'appearance_norm': f['app_norm'],
                    'sharpness': f['sharp'],
                    'pose': f['pose'],
                    'model_version': self.embedder.version,
                    'appearance_model': self.hybrid.fusion.appearance_model,
                    'consent': True,
                },
            )
            logger.debug(f"Server contribution: {result.get('status', 'unknown')}")
        except Exception as e:
            logger.debug(f"Could not send to server: {e}")

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def add(self, name: str, image: Union[str, np.ndarray]) -> Dict:
        """Add a face of a person (call several times with different photos for better accuracy)."""
        if self.hybrid is None:
            raise RuntimeError('The installed model package has no appearance fusion; update gfram')
        f = self._analyze(self._load_image(image))
        if f is None:
            return {'success': False, 'error': 'No face detected in image'}

        if self._appearance_model not in (None, self.hybrid.fusion.appearance_model):
            raise RuntimeError(
                f'Database was built with appearance model {self._appearance_model}; '
                f'clear() it and enrol again for {self.hybrid.fusion.appearance_model}')
        self._ensure_geometry()

        if name not in self._persons:
            self._persons[name] = self._next_id
            self._next_id += 1
        self._sample_names.append(name)
        # Only protected forms are stored: encrypted landmarks, key-rotated templates
        self._sample_landmarks_enc.append(self.protector.encrypt_landmarks(f['landmarks']))
        self._sample_app.append(self.protector.protect(f['app'], 'app'))
        self._sample_quality.append({'app_norm': f['app_norm'], 'sharp': f['sharp']})
        self._appearance_model = self.hybrid.fusion.appearance_model
        geo = self.protector.protect(f['geo'], 'geo')
        self._geo = geo[None] if self._geo is None else np.vstack([self._geo, geo])
        self._pose = f['pose'][None] if self._pose is None else np.vstack([self._pose, f['pose']])
        self._geo_version = self.embedder.version
        self._save_database()

        from ..cloud.consent import should_contribute
        if should_contribute(self.contribute):
            self._send_to_server(name, f)

        person_id = self._persons[name]
        logger.info(f"✅ Added: {name} (ID: {person_id})")
        return {'success': True, 'name': name, 'person_id': person_id,
                'faces': self._sample_names.count(name)}

    def recognize(self, image: Union[str, np.ndarray]) -> Dict:
        """Recognize a person in image."""
        f = self._analyze(self._load_image(image))
        if f is None:
            return {'name': 'Unknown', 'confidence': 0.0, 'recognized': False, 'error': 'No face detected'}

        self._total_recognitions += 1
        if not self._sample_names:
            return {'name': 'Unknown', 'confidence': 0.0, 'recognized': False,
                    'error': 'No persons in database', 'bbox': f['bbox']}

        # Compare in the protected domain: rotations preserve every similarity exactly
        query = dict(f, geo=self.protector.protect(f['geo'], 'geo'))
        if 'app' in f:
            query['app'] = self.protector.protect(f['app'], 'app')
        scores = self._scores(query, self._gallery())

        # Best score per person (closest enrolled face)
        best = {}
        for name, s in zip(self._sample_names, scores):
            if s > best.get(name, -np.inf):
                best[name] = float(s)
        name, score = max(best.items(), key=lambda kv: kv[1])

        recognized = score >= self.threshold
        return {
            'name': name if recognized else 'Unknown',
            'confidence': score,
            'recognized': recognized,
            'person_id': self._persons[name] if recognized else None,
            'best_candidate': name,
            'threshold': self.threshold,
            'bbox': f['bbox'],
        }

    def verify(self, image1: Union[str, np.ndarray], image2: Union[str, np.ndarray]) -> Dict:
        """Check whether two photos show the same person (does not use the database)."""
        faces = [self._analyze(self._load_image(img)) for img in (image1, image2)]
        missing = [i + 1 for i, f in enumerate(faces) if f is None]
        if missing:
            return {'same_person': False, 'confidence': 0.0, 'threshold': self.threshold,
                    'error': f'No face detected in image {" and ".join(map(str, missing))}'}

        a, b = faces
        gallery = {k: np.asarray(v)[None] for k, v in b.items() if k in ('geo', 'pose', 'app', 'app_norm', 'sharp')}
        confidence = float(self._scores(a, gallery)[0])
        return {'same_person': confidence >= self.threshold, 'confidence': confidence,
                'threshold': self.threshold}

    def rotate_template_key(self) -> str:
        """
        Revoke the current templates: re-protect the whole database under a new random
        key (no photos needed). Templates copied before the rotation no longer match.
        Returns the new key fingerprint.
        """
        import secrets
        from ..security import TemplateProtector, default_key_path, write_key
        if self._template_key is not None:
            raise RuntimeError('This recognizer uses a key you manage (template_key=...): rotate it in your '
                               'key store and re-open the database with the new key')
        old = self.protector
        new = TemplateProtector(secrets.token_bytes(32))

        landmarks = [old.decrypt_landmarks(b) for b in self._sample_landmarks_enc]
        app = [new.protect(old.unprotect(a, 'app'), 'app') for a in self._sample_app]
        geo = new.protect(old.unprotect(self._geo, 'geo'), 'geo') if self._geo is not None else None

        path = default_key_path()
        # Keep both keys on disk until the database is saved, so an interruption
        # at any point leaves a key that matches the database (see _make_protector)
        write_key(old._key, path.with_suffix('.key.old'))
        write_key(new._key, path.with_suffix('.key.new'))

        self._protector = new
        self._sample_landmarks_enc = [new.encrypt_landmarks(lm) for lm in landmarks]
        self._sample_app, self._geo = app, geo
        self._save_database()

        write_key(new._key, path)
        path.with_suffix('.key.new').unlink()
        path.with_suffix('.key.old').unlink()
        logger.info(f'Template key rotated: {old.fingerprint} -> {new.fingerprint}')
        return new.fingerprint

    def list_persons(self) -> List[str]:
        return list(self._persons.keys())

    def remove(self, name: str) -> bool:
        if name not in self._persons:
            return False
        keep = [i for i, n in enumerate(self._sample_names) if n != name]
        self._sample_names = [self._sample_names[i] for i in keep]
        self._sample_landmarks_enc = [self._sample_landmarks_enc[i] for i in keep]
        self._sample_app = [self._sample_app[i] for i in keep]
        self._sample_quality = [self._sample_quality[i] for i in keep]
        if self._geo is not None:
            self._geo = self._geo[keep] if keep else None
            self._pose = self._pose[keep] if keep else None
        del self._persons[name]
        self._save_database()
        logger.info(f"Removed: {name}")
        return True

    def clear(self):
        """Clear all persons"""
        self._persons.clear()
        self._next_id = 0
        self._sample_names, self._sample_landmarks_enc = [], []
        self._sample_app, self._sample_quality = [], []
        self._appearance_model = None
        self._geo, self._pose = None, None
        if self._db_file.exists():
            self._db_file.unlink()
        logger.info("Cleared all persons")

    def stats(self) -> Dict:
        return {
            'persons': len(self._persons),
            'faces': len(self._sample_names),
            'total_recognitions': self._total_recognitions,
            'device': str(self.device),
            'db_path': str(self.db_path),
            'model_version': self._embedder.version if self._embedder else self._geo_version,
            'mode': ('hybrid' if self._hybrid else 'geometry') if self._embedder else None,
            'template_protection': 'keyed rotation + AES-256-GCM landmarks',
            'key_fingerprint': self.protector.fingerprint,
        }
