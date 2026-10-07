"""
Procrustes alignment of facial landmarks.

Removes translation, scale and head rotation so that only face shape remains.
Must match the preprocessing used when the model was trained
(scripts/lfw/common.py).
"""

import numpy as np


def _center_scale(x: np.ndarray) -> np.ndarray:
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.linalg.norm(x, axis=(1, 2), keepdims=True)


def _rotate_to(x: np.ndarray, ref: np.ndarray) -> np.ndarray:
    """Batched Kabsch: rotate every shape in x (N, P, 3) onto ref (P, 3)."""
    h = np.einsum('npi,pj->nij', x, ref)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum('nij,njk->nik', u, vt)))
    fix = np.ones((len(x), 3), dtype=x.dtype)
    fix[:, 2] = d
    rot = np.einsum('nij,nj,njk->nik', u, fix, vt)
    return np.einsum('npi,nij->npj', x, rot)


def procrustes_align(landmarks: np.ndarray, reference: np.ndarray) -> np.ndarray:
    """
    Align landmarks to a reference face shape.

    Args:
        landmarks: (P, 3) or (N, P, 3) raw landmarks from FaceDetector.
        reference: (P, 3) mean face shape stored in the model package.

    Returns:
        Aligned landmarks with the same shape as the input (float32).
    """
    single = landmarks.ndim == 2
    x = landmarks[None] if single else landmarks
    x = _rotate_to(_center_scale(x.astype(np.float64)), reference.astype(np.float64))
    x = x.astype(np.float32)
    return x[0] if single else x
