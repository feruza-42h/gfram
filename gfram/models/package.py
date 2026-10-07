"""
GFRAM model package (formats gfram-model/2 and gfram-model/3).

A single .pth file that carries the model config, weights and the exact
preprocessing used during training, so every client embeds faces the same way
as the model was evaluated.

Format gfram-model/3 additionally carries the adaptive fusion that combines the
geometric embedding with the appearance model (see models/hybrid.py).
"""

import logging
from pathlib import Path
from typing import Dict, Optional, Union

import numpy as np
import torch
import torch.nn.functional as F

from .geometric_transformer import create_geometric_transformer
from ..geometry.alignment import procrustes_align

logger = logging.getLogger(__name__)

PACKAGE_FORMAT = 'gfram-model/3'
SUPPORTED_FORMATS = ('gfram-model/2', 'gfram-model/3')


class ModelPackageError(Exception):
    """Raised when a model file is missing, corrupt or in an unsupported format."""


def read_package(path: Union[str, Path]) -> Dict:
    """Load and validate a model package without building the model."""
    try:
        pkg = torch.load(path, map_location='cpu', weights_only=False)
    except Exception as e:
        raise ModelPackageError(f'Cannot read model file {path}: {e}') from e
    if not isinstance(pkg, dict) or pkg.get('format') not in SUPPORTED_FORMATS:
        raise ModelPackageError(
            f'{path} is not a GFRAM model package ({" / ".join(SUPPORTED_FORMATS)}) '
            f'(found format={pkg.get("format") if isinstance(pkg, dict) else type(pkg).__name__})')
    for key in ('config_name', 'model_state_dict', 'preprocessing', 'model_version'):
        if key not in pkg:
            raise ModelPackageError(f'{path} is missing "{key}"')
    return pkg


class FaceEmbedder:
    """Turns raw MediaPipe landmarks into L2-normalised identity embeddings."""

    def __init__(self, package: Dict, device: Optional[torch.device] = None):
        self.device = device or torch.device('cpu')
        self.version = package['model_version']
        self.thresholds = package.get('thresholds', {})
        self.metrics = package.get('metrics', {})
        self.fusion_spec = package.get('fusion')  # gfram-model/3: hybrid geometry + appearance

        pre = package['preprocessing']
        self.reference = np.asarray(pre['reference_shape'], np.float32)
        self.mean = np.asarray(pre['input_mean'], np.float32)
        self.std = np.asarray(pre['input_std'], np.float32)

        self.model = create_geometric_transformer(config_name=package['config_name'], num_classes=None)
        # strict=True: a mismatched file must fail loudly, never fall back to random weights
        self.model.load_state_dict(package['model_state_dict'], strict=True)
        self.model.to(self.device).eval()
        self.dim = self.model.output_dim

    @classmethod
    def from_file(cls, path: Union[str, Path], device: Optional[torch.device] = None) -> 'FaceEmbedder':
        return cls(read_package(path), device)

    def preprocess(self, landmarks: np.ndarray) -> np.ndarray:
        """(P, 3) or (N, P, 3) raw landmarks -> standardised aligned shapes."""
        return ((procrustes_align(landmarks, self.reference) - self.mean) / self.std).astype(np.float32)

    @torch.no_grad()
    def embed(self, landmarks: np.ndarray, batch_size: int = 64) -> np.ndarray:
        """Embed one face (P, 3) -> (D,) or many faces (N, P, 3) -> (N, D)."""
        single = landmarks.ndim == 2
        x = self.preprocess(landmarks[None] if single else landmarks)
        out = []
        for i in range(0, len(x), batch_size):
            batch = torch.from_numpy(x[i:i + batch_size]).to(self.device)
            out.append(F.normalize(self.model(batch), dim=1).cpu().numpy())
        emb = np.concatenate(out) if out else np.zeros((0, self.dim), np.float32)
        return emb[0] if single else emb
