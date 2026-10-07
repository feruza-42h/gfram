"""
GFRAM Models Module

Contains all AI models for geometric face recognition:
- GeometricTransformer: Transformer-based model for geometric features
- Loss functions for metric learning
"""

__all__ = []

# Try to import models (requires PyTorch)
try:
    from .geometric_transformer import (
        GeometricTransformer,
        create_geometric_transformer,
        PositionalEncoding,
        GeometricEmbedding,
        MultiHeadGeometricAttention,
        GeometricTransformerBlock
    )

    __all__.extend([
        'GeometricTransformer',
        'create_geometric_transformer',
        'PositionalEncoding',
        'GeometricEmbedding',
        'MultiHeadGeometricAttention',
        'GeometricTransformerBlock',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import Transformer models (PyTorch required): {e}")


try:
    from .losses import (
        TripletLoss,
        ArcFaceLoss,
        CosFaceLoss,
        ContrastiveLoss,
        CenterLoss,
        CombinedLoss
    )

    __all__.extend([
        'TripletLoss',
        'ArcFaceLoss',
        'CosFaceLoss',
        'ContrastiveLoss',
        'CenterLoss',
        'CombinedLoss',
    ])
except ImportError as e:
    import warnings
    warnings.warn(f"Could not import loss functions (PyTorch required): {e}")