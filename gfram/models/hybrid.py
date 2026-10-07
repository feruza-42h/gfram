"""
GFRAM hybrid recognition: geometric module + deep appearance model + adaptive fusion.

1. Geometric module: MediaPipe 478 landmarks -> 3D Procrustes alignment ->
   GeometricTransformer identity embedding, head pose and the face crop alignment.
2. Appearance model: MobileFaceNet (InsightFace w600k_mbf) on the 112x112 crop
   aligned from the landmarks (no separate face detector needed).
3. Quality-aware adaptive fusion: a calibrated model that combines both
   similarities with quality and pose cues into the probability that two faces
   belong to the same person.

The appearance weights are downloaded from the official InsightFace release on
first use and verified by SHA-256 (they are not redistributed with gfram).
"""

import io
import hashlib
import logging
import urllib.request
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import cv2
import numpy as np

from ..geometry.alignment import procrustes_align

logger = logging.getLogger(__name__)

APPEARANCE_MODELS = {
    'w600k_mbf': {
        'url': 'https://github.com/deepinsight/insightface/releases/download/v0.7/buffalo_s.zip',
        'member': 'w600k_mbf.onnx',
        'sha256': '9cc6e4a75f0e2bf0b1aed94578f144d15175f357bdc05e815e5c4a02b319eb4f',
    },
}

# Standard ArcFace 112x112 alignment template: left eye, right eye, nose, left/right mouth corner
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)
IRIS_A, IRIS_B = slice(468, 473), slice(473, 478)
NOSE_TIP, MOUTH_CORNERS = 1, (61, 291)


# ---------------------------------------------------------------- geometry helpers

def five_points(landmarks: np.ndarray) -> np.ndarray:
    """ArcFace anchor points (image coordinates) from 478 MediaPipe landmarks."""
    lm = landmarks[:, :2]
    eyes = sorted([lm[IRIS_A].mean(0), lm[IRIS_B].mean(0)], key=lambda p: p[0])
    mouth = sorted([lm[MOUTH_CORNERS[0]], lm[MOUTH_CORNERS[1]]], key=lambda p: p[0])
    return np.array([eyes[0], eyes[1], lm[NOSE_TIP], mouth[0], mouth[1]], dtype=np.float32)


def similarity_transform(src: np.ndarray, dst: np.ndarray) -> np.ndarray:
    """Umeyama least-squares similarity transform (2x3) mapping src points onto dst."""
    mu_s, mu_d = src.mean(0), dst.mean(0)
    s, d = src - mu_s, dst - mu_d
    u, sig, vt = np.linalg.svd(d.T @ s / len(src))
    fix = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ fix @ vt
    scale = np.trace(np.diag(sig) @ fix) / s.var(0).sum()
    m = np.zeros((2, 3), np.float32)
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def align_crop(image: np.ndarray, landmarks: np.ndarray) -> np.ndarray:
    """112x112 BGR face crop aligned to the ArcFace template using the landmarks."""
    return cv2.warpAffine(image, similarity_transform(five_points(landmarks), ARCFACE_TEMPLATE),
                          (112, 112), borderValue=0)


def head_pose(landmarks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """Yaw, pitch, roll in degrees of the 3D rotation from the reference face to the landmarks."""
    x = landmarks - landmarks.mean(0)
    x = x / np.linalg.norm(x)
    u, _, vt = np.linalg.svd(reference.T @ x)
    r = u @ np.diag([1, 1, np.sign(np.linalg.det(u @ vt))]) @ vt
    yaw = np.degrees(np.arcsin(np.clip(-r[2, 0], -1, 1)))
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return np.array([yaw, pitch, roll], np.float32)


def sharpness(crop: np.ndarray) -> float:
    gray = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    return float(np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var()))


# ---------------------------------------------------------------- appearance model

def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def ensure_appearance_model(name: str = 'w600k_mbf') -> Path:
    """Path to the verified appearance model, downloading it from InsightFace on first use."""
    from ..cloud.server_client import get_cache_dir, _ssl_context

    spec = APPEARANCE_MODELS[name]
    path = get_cache_dir() / spec['member']
    if path.exists() and _sha256(path) == spec['sha256']:
        return path

    print(f'📥 Downloading appearance model {name} from InsightFace (first run only)…')
    request = urllib.request.Request(spec['url'], headers={'User-Agent': 'GFRAM-Client/3.2'})
    with urllib.request.urlopen(request, timeout=600, context=_ssl_context()) as response:
        archive = zipfile.ZipFile(io.BytesIO(response.read()))
    member = next(m for m in archive.namelist() if m.endswith(spec['member']))
    tmp = path.with_suffix('.download')
    tmp.write_bytes(archive.read(member))
    if _sha256(tmp) != spec['sha256']:
        tmp.unlink()
        raise RuntimeError(f'Checksum mismatch for {name}; refusing to use the downloaded file')
    tmp.replace(path)
    print(f'   ✅ Saved {path}')
    return path


class AppearanceEmbedder:
    """512-d appearance embeddings (and their raw norm, a face-quality cue) from aligned crops."""

    def __init__(self, name: str = 'w600k_mbf', threads: int = 2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        opts.log_severity_level = 3  # the ONNX file declares batch 1; batched runs are fine
        self.name = name
        self.session = ort.InferenceSession(str(ensure_appearance_model(name)), opts,
                                            providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name

    def embed(self, crops: np.ndarray):
        """crops (N, 112, 112, 3) BGR uint8 -> (unit embeddings (N, 512), raw norms (N,))."""
        x = crops[..., ::-1].astype(np.float32)
        x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
        raw = self.session.run(None, {self.input: np.ascontiguousarray(x)})[0]
        norms = np.linalg.norm(raw, axis=1)
        return raw / norms[:, None], norms


# ---------------------------------------------------------------- fusion

class AdaptiveFusion:
    """Quality-aware fusion: logistic model over similarities, quality and pose cues."""

    def __init__(self, spec: Dict):
        self.features: List[str] = list(spec['features'])
        self.mean = np.asarray(spec['mean'], np.float64)
        self.scale = np.asarray(spec['scale'], np.float64)
        self.coef = np.asarray(spec['coef'], np.float64)
        self.intercept = float(spec['intercept'])
        self.thresholds = dict(spec['thresholds'])
        self.appearance_model = spec['appearance_model']

    @staticmethod
    def pair_features(q: Dict, g: Dict) -> Dict[str, np.ndarray]:
        """q: one face; g: N gallery faces (arrays with a leading dimension)."""
        s_app = g['app'] @ q['app']
        s_geo = g['geo'] @ q['geo']
        yaw_diff = np.abs(g['pose'][:, 0] - q['pose'][0])
        return {
            's_app': s_app, 's_geo': s_geo, 's_app*s_geo': s_app * s_geo,
            'norm_min': np.minimum(g['app_norm'], q['app_norm']),
            'norm_max': np.maximum(g['app_norm'], q['app_norm']),
            'sharp_min': np.minimum(g['sharp'], q['sharp']),
            'yaw_diff': yaw_diff, 'yaw_max': np.maximum(np.abs(g['pose'][:, 0]), abs(q['pose'][0])),
            's_app*yaw_diff': s_app * yaw_diff, 's_geo*yaw_diff': s_geo * yaw_diff,
        }

    def probability(self, q: Dict, g: Dict) -> np.ndarray:
        """Probability that the query face and each gallery face show the same person."""
        cols = self.pair_features(q, g)
        x = np.column_stack([cols[f] for f in self.features])
        z = ((x - self.mean) / self.scale) @ self.coef + self.intercept
        return 1.0 / (1.0 + np.exp(-z))


class HybridAnalyzer:
    """Turns one detected face into everything the fusion needs."""

    def __init__(self, geometry_embedder, fusion: AdaptiveFusion):
        self.geometry = geometry_embedder
        self.fusion = fusion
        self.appearance = AppearanceEmbedder(fusion.appearance_model)

    def analyze(self, image: np.ndarray, landmarks: np.ndarray) -> Dict:
        crop = align_crop(image, landmarks)
        app, norm = self.appearance.embed(crop[None])
        return {
            'geo': self.geometry.embed(landmarks),
            'app': app[0].astype(np.float32),
            'app_norm': float(norm[0]),
            'sharp': sharpness(crop),
            'pose': head_pose(landmarks, self.geometry.reference),
        }
