"""
Landmark processing and manipulation utilities.

Provides functions for processing, transforming, and analyzing facial landmarks.

3-step normalization algorithm:
1. Centering (Markazlashtirish) - subtract centroid
2. Scaling (Masshtablash) - normalize by interocular distance
3. Alignment (Orientatsiyani moslashtirish) - rotate to horizontal eye line
"""

import numpy as np
from typing import Dict, List, Tuple, Optional
from scipy.spatial import procrustes


def rotation_matrix_2d(theta: float) -> np.ndarray:
    """
    Create 2D rotation matrix.

    Args:
        theta: Rotation angle in radians.

    Returns:
        2x2 rotation matrix.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t],
        [sin_t, cos_t]
    ])


def rotation_matrix_3d(theta: float) -> np.ndarray:
    """
    Create 3D rotation matrix (rotation around Z-axis).

    Formula (3.22):
    R = [cos θ  -sin θ  0]
        [sin θ   cos θ  0]
        [0       0      1]

    Args:
        theta: Rotation angle in radians.

    Returns:
        3x3 rotation matrix.
    """
    cos_t = np.cos(theta)
    sin_t = np.sin(theta)
    return np.array([
        [cos_t, -sin_t, 0],
        [sin_t, cos_t, 0],
        [0, 0, 1]
    ])


class LandmarkProcessor:
    """
    Process and manipulate facial landmarks.

    Implements 3-step normalization from dissertation:
    - Step 1: Centering (Formula 3.17-3.18)
    - Step 2: Scaling (Formula 3.19-3.20)
    - Step 3: Alignment (Formula 3.21-3.23)
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

        Formula (3.17-3.18):
        μ = (1/n) · Σᵢ lᵢ
        l'ᵢ = lᵢ - μ

        Args:
            landmarks: Input landmarks.

        Returns:
            Centered landmarks.
        """
        centroid = np.mean(landmarks, axis=0)
        return landmarks - centroid

    def scale(self, landmarks: np.ndarray, target_scale: float = 1.0) -> np.ndarray:
        """
        Scale landmarks using interocular distance as reference.

        Formula (3.19-3.20):
        d_ref = ||l_left_eye - l_right_eye||₂
        l''ᵢ = l'ᵢ / d_ref

        Args:
            landmarks: Input landmarks (should be centered first).
            target_scale: Target scale (default 1.0).

        Returns:
            Scaled landmarks.
        """
        # Get eye centers for interocular distance
        regions = get_landmark_regions()
        left_eye = landmarks[regions['left_eye']]
        right_eye = landmarks[regions['right_eye']]

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)

        # Interocular distance as reference
        d_ref = np.linalg.norm(left_center - right_center)

        if d_ref < 1e-8:
            return landmarks

        return landmarks * (target_scale / d_ref)

    def align(self, landmarks: np.ndarray) -> np.ndarray:
        """
        Align landmarks by rotating to make eye line horizontal.

        Formula (3.21-3.23):
        θ = arctan((y_right - y_left) / (x_right - x_left))
        R = rotation_matrix(-θ)
        l_norm = R · l''

        Args:
            landmarks: Input landmarks (should be centered and scaled first).

        Returns:
            Aligned landmarks.
        """
        regions = get_landmark_regions()

        # Get eye centers
        left_eye = landmarks[regions['left_eye']]
        right_eye = landmarks[regions['right_eye']]

        left_center = np.mean(left_eye, axis=0)
        right_center = np.mean(right_eye, axis=0)

        # Calculate angle (Formula 3.21)
        dy = right_center[1] - left_center[1]
        dx = right_center[0] - left_center[0]
        theta = np.arctan2(dy, dx)

        # Create rotation matrix (Formula 3.22)
        if landmarks.shape[1] == 3:
            R = rotation_matrix_3d(-theta)
        else:
            R = rotation_matrix_2d(-theta)

        # Apply rotation (Formula 3.23)
        return np.dot(landmarks, R.T)

    def normalize(
            self,
            landmarks: np.ndarray,
            center: bool = True,
            scale: bool = True,
            align: bool = True,
            target_scale: float = 1.0
    ) -> np.ndarray:
        """
        Normalize landmarks using 3-step algorithm.
        Args:
            landmarks: Input landmarks.
            center: Whether to center at origin.
            scale: Whether to scale by interocular distance.
            align: Whether to align (rotate to horizontal eye line).
            target_scale: Target scale for scaling step.

        Returns:
            Normalized landmarks.
        """
        result = landmarks.copy()

        # 1-bosqich: Markazlashtirish
        if center:
            result = self.center(result)

        # 2-bosqich: Masshtablash
        if scale:
            result = self.scale(result, target_scale)

        # 3-bosqich: Orientatsiyani moslashtirish
        if align:
            result = self.align(result)

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