"""
Facial symmetry analysis.
"""

import numpy as np
from typing import List, Tuple


def compute_bilateral_symmetry(landmarks: np.ndarray) -> float:
    """
    Compute bilateral symmetry score.

    Args:
        landmarks: Landmark coordinates (468, 2 or 3).

    Returns:
        Symmetry score (0 = perfect symmetry, higher = more asymmetric).
    """
    points_2d = landmarks[:, :2]

    # Compute vertical axis (average x coordinate)
    axis_x = np.mean(points_2d[:, 0])

    # Compute distances from axis
    distances = np.abs(points_2d[:, 0] - axis_x)

    # Symmetry score: normalized standard deviation
    symmetry_score = np.std(distances) / (np.mean(distances) + 1e-8)

    return float(symmetry_score)


def extract_symmetry_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract comprehensive symmetry features (15 features).

    Args:
        landmarks: Landmark coordinates.

    Returns:
        Symmetry feature vector.
    """
    from .landmarks import get_landmark_regions

    features = []
    regions = get_landmark_regions()

    # Paired regions for symmetry analysis
    paired_regions = [
        ('left_eye', 'right_eye'),
        ('left_eyebrow', 'right_eyebrow'),
    ]

    for left_region, right_region in paired_regions:
        left_indices = regions[left_region]
        right_indices = regions[right_region]

        left_points = landmarks[left_indices, :2]
        right_points = landmarks[right_indices, :2]

        # Width difference
        left_width = np.ptp(left_points[:, 0])
        right_width = np.ptp(right_points[:, 0])
        width_diff = abs(left_width - right_width) / (left_width + right_width + 1e-8)
        features.append(width_diff)

        # Height difference
        left_height = np.ptp(left_points[:, 1])
        right_height = np.ptp(right_points[:, 1])
        height_diff = abs(left_height - right_height) / (left_height + right_height + 1e-8)
        features.append(height_diff)

        # Area difference
        left_area = left_width * left_height
        right_area = right_width * right_height
        area_diff = abs(left_area - right_area) / (left_area + right_area + 1e-8)
        features.append(area_diff)

        # Position difference (Y-coordinate)
        left_y = np.mean(left_points[:, 1])
        right_y = np.mean(right_points[:, 1])
        pos_diff = abs(left_y - right_y) / (np.mean(landmarks[:, 1]) + 1e-8)
        features.append(pos_diff)

        # Shape difference (variance)
        left_var = np.mean(np.var(left_points, axis=0))
        right_var = np.mean(np.var(right_points, axis=0))
        shape_diff = abs(left_var - right_var) / (left_var + right_var + 1e-8)
        features.append(shape_diff)

    # Overall symmetry
    overall_symmetry = compute_bilateral_symmetry(landmarks)
    features.extend([overall_symmetry] * 5)  # Pad to 15 features

    return np.array(features[:15], dtype=np.float32)