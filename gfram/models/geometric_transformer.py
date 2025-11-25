"""
Geometric Transformer Model for Face Recognition - UPDATED

UPDATED: Full support for both 468 and 478 landmarks
- Flexible architecture for different landmark counts
- Automatic adjustment of model capacity
- Support for iris landmarks (478 mode)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class PositionalEncoding(nn.Module):
    """
    Positional encoding for geometric coordinates.
    Encodes 2D/3D spatial positions using sinusoidal functions.
    """

    def __init__(self, d_model: int, max_len: int = 500):
        super().__init__()

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.pe[:, :x.size(1), :]


class GeometricEmbedding(nn.Module):
    """
    Embed geometric coordinates (x, y, z) into high-dimensional space.

    UPDATED: Flexible for any number of landmarks
    """

    def __init__(
        self,
        input_dim: int = 3,
        embed_dim: int = 256,
        use_batch_norm: bool = True
    ):
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        self.embedding = nn.Sequential(
            nn.Linear(input_dim, embed_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(embed_dim // 2, embed_dim)
        )

        if use_batch_norm:
            self.norm = nn.BatchNorm1d(embed_dim)
        else:
            self.norm = nn.LayerNorm(embed_dim)

        self.use_batch_norm = use_batch_norm

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_landmarks, input_dim)
        Returns:
            (batch_size, num_landmarks, embed_dim)
        """
        batch_size, num_landmarks, _ = x.shape

        # Embed
        x = self.embedding(x)  # (batch, num_landmarks, embed_dim)

        # Normalize
        if self.use_batch_norm:
            x = x.transpose(1, 2)  # (batch, embed_dim, num_landmarks)
            x = self.norm(x)
            x = x.transpose(1, 2)  # (batch, num_landmarks, embed_dim)
        else:
            x = self.norm(x)

        return x


class GeometricTransformerBlock(nn.Module):
    """
    Single transformer block for geometric data.
    """

    def __init__(
        self,
        embed_dim: int,
        num_heads: int = 8,
        mlp_ratio: int = 4,
        dropout: float = 0.1
    ):
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = nn.MultiheadAttention(
            embed_dim,
            num_heads,
            dropout=dropout,
            batch_first=True
        )

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: (batch_size, num_landmarks, embed_dim)
        """
        # Self-attention with residual
        x = x + self.attn(self.norm1(x), self.norm1(x), self.norm1(x))[0]

        # MLP with residual
        x = x + self.mlp(self.norm2(x))

        return x


class GeometricTransformer(nn.Module):
    """
    Geometric Transformer for face recognition.

    UPDATED: Supports both 468 and 478 landmarks

    Architecture:
    1. Geometric embedding of landmarks
    2. Positional encoding
    3. Multiple transformer blocks
    4. Global pooling
    5. Classification/embedding head
    """

    def __init__(
        self,
        num_landmarks: int = 478,  # UPDATED: Default to 478
        input_dim: int = 3,
        embed_dim: int = 256,
        num_heads: int = 8,
        num_layers: int = 6,
        mlp_ratio: int = 4,
        dropout: float = 0.1,
        num_classes: Optional[int] = None,
        output_dim: int = 256
    ):
        """
        Initialize Geometric Transformer.

        Args:
            num_landmarks: Number of facial landmarks (468 or 478).
            input_dim: Dimension of each landmark (3 for x,y,z).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer blocks.
            mlp_ratio: MLP expansion ratio.
            dropout: Dropout probability.
            num_classes: Number of classes for classification (None for embedding).
            output_dim: Output embedding dimension.
        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.output_dim = output_dim

        # Geometric embedding
        self.embedding = GeometricEmbedding(
            input_dim=input_dim,
            embed_dim=embed_dim
        )

        # Positional encoding
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_landmarks)

        # Transformer blocks
        self.transformer_blocks = nn.ModuleList([
            GeometricTransformerBlock(
                embed_dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(num_layers)
        ])

        # Global pooling
        self.norm = nn.LayerNorm(embed_dim)

        # Output head
        if num_classes is not None:
            # Classification head
            self.head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout),
                nn.Linear(embed_dim, num_classes)
            )
        else:
            # Embedding head
            self.head = nn.Sequential(
                nn.Linear(embed_dim, output_dim),
                nn.LayerNorm(output_dim)
            )

    def forward(
        self,
        landmarks: torch.Tensor,
        return_features: bool = False
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            landmarks: Landmark coordinates (batch_size, num_landmarks, 3).
            return_features: Whether to return intermediate features.

        Returns:
            Output tensor (batch_size, output_dim) or (batch_size, num_classes).
        """
        batch_size, num_lm, _ = landmarks.shape

        # Validate input
        if num_lm != self.num_landmarks:
            # Handle flexible landmark count
            if num_lm == 478 and self.num_landmarks == 468:
                # Take first 468
                landmarks = landmarks[:, :468, :]
            elif num_lm == 468 and self.num_landmarks == 478:
                # Pad with zeros
                padding = torch.zeros(batch_size, 10, 3, device=landmarks.device)
                landmarks = torch.cat([landmarks, padding], dim=1)

        # Embed landmarks
        x = self.embedding(landmarks)  # (batch, num_landmarks, embed_dim)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Transformer blocks
        for block in self.transformer_blocks:
            x = block(x)

        # Normalize
        x = self.norm(x)

        # Global pooling (mean + max)
        x_mean = torch.mean(x, dim=1)  # (batch, embed_dim)
        x_max = torch.max(x, dim=1)[0]  # (batch, embed_dim)
        x = x_mean + x_max  # Combine

        # Output head
        output = self.head(x)

        if return_features:
            return output, x
        else:
            return output

    def extract_embedding(self, landmarks: torch.Tensor) -> torch.Tensor:
        """
        Extract embedding without classification.

        Args:
            landmarks: Landmark coordinates.

        Returns:
            Embedding vector.
        """
        with torch.no_grad():
            return self.forward(landmarks, return_features=False)


def create_geometric_transformer(
    config_name: str = "base",
    num_landmarks: int = 478,  # UPDATED: Default to 478
    num_classes: Optional[int] = None,
    **kwargs
) -> GeometricTransformer:
    """
    Create Geometric Transformer with predefined configuration.

    Args:
        config_name: Configuration name ('tiny', 'small', 'base', 'large').
        num_landmarks: Number of landmarks (468 or 478).
        num_classes: Number of classes (None for embedding mode).
        **kwargs: Override configuration parameters.

    Returns:
        GeometricTransformer instance.
    """
    configs = {
        "tiny": {
            "embed_dim": 128,
            "num_heads": 4,
            "num_layers": 3,
            "mlp_ratio": 2,
            "dropout": 0.1,
            "output_dim": 128
        },
        "small": {
            "embed_dim": 192,
            "num_heads": 6,
            "num_layers": 4,
            "mlp_ratio": 3,
            "dropout": 0.1,
            "output_dim": 192
        },
        "base": {
            "embed_dim": 256,
            "num_heads": 8,
            "num_layers": 6,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "output_dim": 256
        },
        "large": {
            "embed_dim": 384,
            "num_heads": 12,
            "num_layers": 8,
            "mlp_ratio": 4,
            "dropout": 0.1,
            "output_dim": 384
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")

    config = configs[config_name]
    config.update(kwargs)  # Override with custom params

    return GeometricTransformer(
        num_landmarks=num_landmarks,
        num_classes=num_classes,
        **config
    )


# Convenience functions
def create_transformer_468(**kwargs) -> GeometricTransformer:
    """Create transformer for 468 landmarks."""
    return create_geometric_transformer(num_landmarks=468, **kwargs)


def create_transformer_478(**kwargs) -> GeometricTransformer:
    """Create transformer for 478 landmarks."""
    return create_geometric_transformer(num_landmarks=478, **kwargs)


# Backward compatibility alias
MultiHeadGeometricAttention = GeometricTransformerBlock


# Model info
def get_model_info(model: GeometricTransformer) -> dict:
    """
    Get model information.

    Args:
        model: GeometricTransformer instance.

    Returns:
        Dictionary with model info.
    """
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)

    return {
        'num_landmarks': model.num_landmarks,
        'embed_dim': model.embed_dim,
        'num_classes': model.num_classes,
        'output_dim': model.output_dim,
        'total_params': total_params,
        'trainable_params': trainable_params,
        'model_size_mb': total_params * 4 / 1024 / 1024  # Assuming float32
    }