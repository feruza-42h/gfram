"""
Data augmentation for facial landmarks.
"""

import torch
import numpy as np
from typing import Tuple


class LandmarkAugmentation:
    """Augmentation transforms for landmarks."""

    def __init__(
            self,
            rotation_range: float = 0.1,
            scale_range: Tuple[float, float] = (0.9, 1.1),
            translation_range: float = 0.05,
            noise_std: float = 0.01
    ):
        self.rotation_range = rotation_range
        self.scale_range = scale_range
        self.translation_range = translation_range
        self.noise_std = noise_std

    def __call__(self, landmarks: torch.Tensor) -> torch.Tensor:
        """Apply random augmentation."""
        result = landmarks.clone()

        # Random rotation
        if self.rotation_range > 0:
            angle = np.random.uniform(-self.rotation_range, self.rotation_range)
            result = self._rotate(result, angle)

        # Random scale
        if self.scale_range[0] < self.scale_range[1]:
            scale = np.random.uniform(*self.scale_range)
            result = result * scale

        # Random translation
        if self.translation_range > 0:
            tx = np.random.uniform(-self.translation_range, self.translation_range)
            ty = np.random.uniform(-self.translation_range, self.translation_range)
            result[:, 0] += tx
            result[:, 1] += ty

        # Add noise
        if self.noise_std > 0:
            noise = torch.randn_like(result) * self.noise_std
            result = result + noise

        return result

    def _rotate(self, landmarks: torch.Tensor, angle: float) -> torch.Tensor:
        """Rotate landmarks around center."""
        cos_a = np.cos(angle)
        sin_a = np.sin(angle)

        center = landmarks.mean(dim=0)
        centered = landmarks - center

        rotation_matrix = torch.tensor([
            [cos_a, -sin_a, 0],
            [sin_a, cos_a, 0],
            [0, 0, 1]
        ], dtype=landmarks.dtype, device=landmarks.device)

        rotated = torch.matmul(centered, rotation_matrix.T)
        return rotated + center