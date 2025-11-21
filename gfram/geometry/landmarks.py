"""
Landmark processing and manipulation utilities.

Provides functions for processing, transforming, and analyzing facial landmarks.
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import procrustes


class LandmarkProcessor:
    """
    Process and manipulate facial landmarks.
    """

    def __init__(self, num_landmarks: int = 468):
        """
        Initialize landmark processor.

        Args:
            num_landmarks: Expected number of landmarks.
        """
        self.num_landmarks = num_landmarks

    def validate(self, landmarks: np.ndarray) -> bool:
        """
        Validate landmark array.

        Args:
            landmarks: Landmark array (N, 2) or (N, 3).

        Returns:
            True if valid, False otherwise.
        """
        if landmarks is None:
            return False

        if not isinstance(landmarks, np.ndarray):
            return False

        if landmarks.ndim != 2:
            return False

        if landmarks.shape[0] != self.num_landmarks:
            return False

        if landmarks.shape[1] not in [2, 3]:
            return False

        # Check for NaN or Inf
        if not np.isfinite(landmarks).all():
            return False

        return True

    def center(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Center landmarks at origin (subtract centroid).

        Args:
            landmarks: Input landmarks.

        Returns:
            Centered landmarks.
        """
        centroid = np.mean(landmarks, axis=0)
        return landmarks - centroid

    def scale(self, landmarks: np.ndarray, target_scale: float = 1.0) -> np.ndarray:
        """
        Scale landmarks to target scale.

        Args:
            landmarks: Input landmarks.
            target_scale: Target scale.

        Returns:
            Scaled landmarks.
        """
        # Compute current scale (RMS distance from origin)
        current_scale = np.sqrt(np.mean(np.sum(landmarks ** 2, axis=1)))

        if current_scale < 1e-8:
            return landmarks

        scale_factor = target_scale / current_scale
        return landmarks * scale_factor

    def normalize(
            self,
            landmarks: np.ndarray,
            center: bool = True,
            scale: bool = True,
            target_scale: float = 1.0
    ) -> np.ndarray:
        """
        Normalize landmarks (center + scale).

        Args:
            landmarks: Input landmarks.
            center: Whether to center at origin.
            scale: Whether to scale.
            target_scale: Target scale.

        Returns:
            Normalized landmarks.
        """
        result = landmarks.copy()

        if center:
            result = self.center(result)

        if scale:
            result = self.scale(result, target_scale)

        return result


def get_landmark_regions() -> Dict[str, List[int]]:
    """
    Get predefined landmark regions for MediaPipe 468-point mesh.

    Returns:
        Dictionary mapping region names to landmark indices.
    """
    return {
        # Eyes
        'left_eye': [33, 160, 158, 133, 153, 144, 145, 163],
        'right_eye': [362, 385, 387, 263, 373, 380, 374, 390],

        # Eyebrows
        'left_eyebrow': [70, 63, 105, 66, 107, 55, 65],
        'right_eyebrow': [336, 296, 334, 293, 300, 285, 295],

        # Nose
        'nose_bridge': [168, 6, 197, 195, 5],
        'nose_tip': [4, 1, 2],
        'nose_base': [98, 97, 2, 326, 327],

        # Mouth
        'outer_lips': [61, 185, 40, 39, 37, 0, 267, 269, 270, 409, 291, 375, 321, 405, 314, 17, 84, 181, 91, 146],
        'inner_lips': [78, 191, 80, 81, 82, 13, 312, 311, 310, 415, 308, 324, 318, 402, 317, 14, 87, 178, 88, 95],

        # Face outline
        'face_oval': [
            10, 338, 297, 332, 284, 251, 389, 356, 454, 323, 361, 288,
            397, 365, 379, 378, 400, 377, 152, 148, 176, 149, 150, 136,
            172, 58, 132, 93, 234, 127, 162, 21, 54, 103, 67, 109
        ],

        # Jaw
        'jaw': [152, 377, 400, 378, 379, 365, 397, 288, 361, 323, 454, 356, 389, 251, 284, 332, 297, 338],
    }


def compute_interocular_distance(landmarks: np.ndarray) -> float:
    """
    Compute interocular distance (distance between eye centers).

    Args:
        landmarks: Landmark array (468 points).

    Returns:
        Interocular distance.
    """
    regions = get_landmark_regions()
    left_eye = landmarks[regions['left_eye']]
    right_eye = landmarks[regions['right_eye']]

    left_center = np.mean(left_eye, axis=0)
    right_center = np.mean(right_eye, axis=0)

    return np.linalg.norm(left_center - right_center)