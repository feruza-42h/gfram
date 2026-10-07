"""
Appearance branch: ArcFace embeddings from face crops aligned with GFRAM landmarks.

The 478 MediaPipe landmarks already locate the face, so no extra detector is
needed: five anchor points (iris centres, nose tip, mouth corners) are mapped
onto the standard ArcFace 112x112 template with a similarity transform.

Models: InsightFace w600k_r50 (ArcFace ResNet-50) and w600k_mbf (MobileFaceNet),
both trained on WebFace600K.
"""

import os
from pathlib import Path

import cv2
import numpy as np

MODELS = Path(__file__).resolve().parents[2] / 'data' / 'models'

# Standard ArcFace alignment template (112x112): left eye, right eye, nose, left mouth, right mouth
ARCFACE_TEMPLATE = np.array([
    [38.2946, 51.6963], [73.5318, 51.5014], [56.0252, 71.7366],
    [41.5493, 92.3655], [70.7299, 92.2041]], dtype=np.float32)

IRIS_A, IRIS_B = slice(468, 473), slice(473, 478)
NOSE_TIP = 1
MOUTH_CORNERS = (61, 291)


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
    cov = d.T @ s / len(src)
    u, sig, vt = np.linalg.svd(cov)
    fix = np.diag([1.0, np.sign(np.linalg.det(u @ vt))])
    rot = u @ fix @ vt
    scale = np.trace(np.diag(sig) @ fix) / s.var(0).sum()
    m = np.zeros((2, 3), np.float32)
    m[:, :2] = scale * rot
    m[:, 2] = mu_d - scale * rot @ mu_s
    return m


def align_crop(image: np.ndarray, landmarks: np.ndarray, size: int = 112) -> np.ndarray:
    """112x112 aligned BGR face crop."""
    m = similarity_transform(five_points(landmarks), ARCFACE_TEMPLATE * (size / 112))
    return cv2.warpAffine(image, m, (size, size), borderValue=0)


class ArcFaceEmbedder:
    """L2-normalised 512-d appearance embeddings from aligned crops."""

    def __init__(self, name: str = 'w600k_r50', threads: int = 2):
        import onnxruntime as ort
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        opts.inter_op_num_threads = 1
        # The ONNX files declare a fixed output batch of 1 although batched inference
        # works; silence the resulting per-batch shape warning
        opts.log_severity_level = 3
        self.session = ort.InferenceSession(str(MODELS / f'{name}.onnx'), opts,
                                            providers=['CPUExecutionProvider'])
        self.input = self.session.get_inputs()[0].name
        self.name = name

    def embed_crops(self, crops: np.ndarray, batch_size: int = 16, normalize: bool = True) -> np.ndarray:
        """
        crops: (N, 112, 112, 3) BGR uint8 -> (N, 512) float32.

        With normalize=False the raw embedding is returned; its norm is a known
        proxy for face image quality (MagFace/AdaFace), used by adaptive fusion.
        """
        out = []
        for i in range(0, len(crops), batch_size):
            x = crops[i:i + batch_size][..., ::-1].astype(np.float32)  # BGR -> RGB
            x = ((x - 127.5) / 127.5).transpose(0, 3, 1, 2)
            out.append(self.session.run(None, {self.input: np.ascontiguousarray(x)})[0])
        emb = np.concatenate(out)
        return emb / np.linalg.norm(emb, axis=1, keepdims=True) if normalize else emb
