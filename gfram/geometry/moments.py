"""
Geometric moments (Hu moments) for facial landmarks.
"""

import numpy as np
import cv2
from typing import Tuple


def compute_hu_moments(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute Hu moments (7 translation/scale/rotation invariant features).

    Args:
        landmarks: Landmark coordinates (N, 2) or (N, 3).

    Returns:
        Array of 7 Hu moments.
    """
    points_2d = landmarks[:, :2].astype(np.float32)

    # Compute moments
    moments = cv2.moments(points_2d)

    # Compute Hu moments
    hu = cv2.HuMoments(moments).flatten()

    # Log scale
    hu = -np.sign(hu) * np.log10(np.abs(hu) + 1e-10)

    return hu.astype(np.float32)


def compute_central_moments(
        landmarks: np.ndarray,
        order: int = 3
) -> np.ndarray:
    """
    Compute central moments up to given order.

    Args:
        landmarks: Landmark coordinates.
        order: Maximum order of moments.

    Returns:
        Flattened array of central moments.
    """
    points_2d = landmarks[:, :2]

    # Compute centroid
    cx = np.mean(points_2d[:, 0])
    cy = np.mean(points_2d[:, 1])

    # Center points
    x = points_2d[:, 0] - cx
    y = points_2d[:, 1] - cy

    moments = []

    for p in range(order + 1):
        for q in range(order + 1 - p):
            if p + q <= order:
                moment = np.sum((x ** p) * (y ** q))
                moments.append(moment)

    return np.array(moments, dtype=np.float32)


def compute_normalized_moments(landmarks: np.ndarray) -> np.ndarray:
    """
    Compute normalized central moments (scale-invariant).

    Args:
        landmarks: Landmark coordinates.

    Returns:
        Array of normalized moments.
    """
    points_2d = landmarks[:, :2]

    # Compute moments
    m = cv2.moments(points_2d.astype(np.float32))

    # Compute normalized central moments
    mu20 = m['mu20'] / (m['m00'] ** 2)
    mu02 = m['mu02'] / (m['m00'] ** 2)
    mu11 = m['mu11'] / (m['m00'] ** 2)
    mu30 = m['mu30'] / (m['m00'] ** 2.5)
    mu03 = m['mu03'] / (m['m00'] ** 2.5)
    mu21 = m['mu21'] / (m['m00'] ** 2.5)
    mu12 = m['mu12'] / (m['m00'] ** 2.5)

    return np.array([mu20, mu02, mu11, mu30, mu03, mu21, mu12], dtype=np.float32)