"""
Geometric Transformer Model for Face Recognition.

A custom transformer architecture designed specifically for geometric facial features.
Uses self-attention to capture spatial relationships between facial landmarks.
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
        """
        Initialize positional encoding.

        Args:
            d_model: Embedding dimension.
            max_len: Maximum sequence length.
        """
        super().__init__()

        # Create positional encoding matrix
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        self.register_buffer('pe', pe.unsqueeze(0))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Add positional encoding to input.

        Args:
            x: Input tensor (batch_size, seq_len, d_model).

        Returns:
            Tensor with positional encoding added.
        """
        return x + self.pe[:, :x.size(1), :]


class GeometricEmbedding(nn.Module):
    """
    Embed geometric coordinates (x, y, z) into high-dimensional space.
    """

    def __init__(
            self,
            input_dim: int = 3,  # x, y, z coordinates
            embed_dim: int = 256,
            use_batch_norm: bool = True
    ):
        """
        Initialize geometric embedding.

        Args:
            input_dim: Input dimension (2 for 2D, 3 for 3D landmarks).
            embed_dim: Embedding dimension.
            use_batch_norm: Whether to use batch normalization.
        """
        super().__init__()

        self.input_dim = input_dim
        self.embed_dim = embed_dim

        # Embedding layers
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
        Embed geometric coordinates.

        Args:
            x: Landmark coordinates (batch_size, num_landmarks, input_dim).

        Returns:
            Embedded landmarks (batch_size, num_landmarks, embed_dim).
        """
        batch_size, num_landmarks, _ = x.shape

        # Flatten for embedding
        x_flat = x.view(-1, self.input_dim)
        embedded = self.embedding(x_flat)

        # Normalize
        if self.use_batch_norm:
            embedded = embedded.view(batch_size, num_landmarks, self.embed_dim)
            embedded = embedded.transpose(1, 2)  # (batch, embed_dim, num_landmarks)
            embedded = self.norm(embedded)
            embedded = embedded.transpose(1, 2)  # (batch, num_landmarks, embed_dim)
        else:
            embedded = embedded.view(batch_size, num_landmarks, self.embed_dim)
            embedded = self.norm(embedded)

        return embedded


class MultiHeadGeometricAttention(nn.Module):
    """
    Multi-head self-attention for geometric data.
    """

    def __init__(
            self,
            embed_dim: int,
            num_heads: int = 8,
            dropout: float = 0.1
    ):
        """
        Initialize multi-head attention.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            dropout: Dropout probability.
        """
        super().__init__()

        assert embed_dim % num_heads == 0, "embed_dim must be divisible by num_heads"

        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.scale = self.head_dim ** -0.5

        # Query, Key, Value projections
        self.qkv = nn.Linear(embed_dim, embed_dim * 3)
        self.proj = nn.Linear(embed_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Apply multi-head attention.

        Args:
            x: Input tensor (batch_size, seq_len, embed_dim).
            mask: Attention mask (optional).

        Returns:
            Output tensor (batch_size, seq_len, embed_dim).
        """
        batch_size, seq_len, _ = x.shape

        # Compute Q, K, V
        qkv = self.qkv(x).reshape(batch_size, seq_len, 3, self.num_heads, self.head_dim)
        qkv = qkv.permute(2, 0, 3, 1, 4)  # (3, batch, heads, seq_len, head_dim)
        q, k, v = qkv[0], qkv[1], qkv[2]

        # Attention scores
        attn = (q @ k.transpose(-2, -1)) * self.scale

        if mask is not None:
            attn = attn.masked_fill(mask == 0, float('-inf'))

        attn = F.softmax(attn, dim=-1)
        attn = self.dropout(attn)

        # Apply attention to values
        out = attn @ v
        out = out.transpose(1, 2).reshape(batch_size, seq_len, self.embed_dim)

        # Output projection
        out = self.proj(out)
        out = self.dropout(out)

        return out


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
        """
        Initialize transformer block.

        Args:
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            mlp_ratio: Ratio of MLP hidden dim to embedding dim.
            dropout: Dropout probability.
        """
        super().__init__()

        self.norm1 = nn.LayerNorm(embed_dim)
        self.attn = MultiHeadGeometricAttention(embed_dim, num_heads, dropout)

        self.norm2 = nn.LayerNorm(embed_dim)
        mlp_hidden_dim = int(embed_dim * mlp_ratio)
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, mlp_hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_hidden_dim, embed_dim),
            nn.Dropout(dropout)
        )

    def forward(
            self,
            x: torch.Tensor,
            mask: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Input tensor.
            mask: Attention mask (optional).

        Returns:
            Output tensor.
        """
        # Self-attention with residual connection
        x = x + self.attn(self.norm1(x), mask)

        # MLP with residual connection
        x = x + self.mlp(self.norm2(x))

        return x


class GeometricTransformer(nn.Module):
    """
    Geometric Transformer for face recognition.

    Architecture:
    1. Geometric embedding of landmarks
    2. Positional encoding
    3. Multiple transformer blocks
    4. Global pooling
    5. Classification/embedding head
    """

    def __init__(
            self,
            num_landmarks: int = 468,
            input_dim: int = 3,
            embed_dim: int = 256,
            num_heads: int = 8,
            num_layers: int = 6,
            mlp_ratio: int = 4,
            dropout: float = 0.1,
            num_classes: Optional[int] = None,
            output_embedding: bool = True
    ):
        """
        Initialize Geometric Transformer.

        Args:
            num_landmarks: Number of facial landmarks.
            input_dim: Input dimension (2 for 2D, 3 for 3D).
            embed_dim: Embedding dimension.
            num_heads: Number of attention heads.
            num_layers: Number of transformer layers.
            mlp_ratio: MLP hidden dimension ratio.
            dropout: Dropout probability.
            num_classes: Number of identity classes (None for embedding only).
            output_embedding: Whether to output embedding vector.
        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.embed_dim = embed_dim
        self.num_classes = num_classes
        self.output_embedding = output_embedding

        # Embedding layers
        self.geometric_embed = GeometricEmbedding(input_dim, embed_dim)
        self.pos_encoding = PositionalEncoding(embed_dim, max_len=num_landmarks)

        # Transformer blocks
        self.blocks = nn.ModuleList([
            GeometricTransformerBlock(embed_dim, num_heads, mlp_ratio, dropout)
            for _ in range(num_layers)
        ])

        self.norm = nn.LayerNorm(embed_dim)

        # Output heads
        if output_embedding:
            self.embedding_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)

    def forward(
            self,
            x: torch.Tensor,
            return_embedding: bool = True,
            mask: Optional[torch.Tensor] = None
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input landmarks (batch_size, num_landmarks, input_dim).
            return_embedding: Whether to return embedding vector.
            mask: Attention mask (optional).

        Returns:
            Tuple of (logits, embedding) depending on configuration.
            - If num_classes is None: (None, embedding)
            - If num_classes is set and return_embedding: (logits, embedding)
            - If num_classes is set and not return_embedding: (logits, None)
        """
        # Embed geometric coordinates
        x = self.geometric_embed(x)

        # Add positional encoding
        x = self.pos_encoding(x)

        # Apply transformer blocks
        for block in self.blocks:
            x = block(x, mask)

        x = self.norm(x)

        # Global average pooling
        x = x.mean(dim=1)  # (batch_size, embed_dim)

        # Generate embedding
        embedding = None
        if self.output_embedding and return_embedding:
            embedding = self.embedding_head(x)

        # Classification
        logits = None
        if self.num_classes is not None:
            if self.output_embedding and return_embedding:
                logits = self.classifier(embedding)
            else:
                logits = self.classifier(x)

        return logits, embedding

    def get_embedding(self, x: torch.Tensor) -> torch.Tensor:
        """
        Get embedding vector for input landmarks.

        Args:
            x: Input landmarks.

        Returns:
            Embedding vector.
        """
        with torch.no_grad():
            _, embedding = self.forward(x, return_embedding=True)
        return embedding


def create_geometric_transformer(
        config_name: str = "base",
        num_classes: Optional[int] = None,
        **kwargs
) -> GeometricTransformer:
    """
    Create a Geometric Transformer model with predefined configurations.

    Args:
        config_name: Configuration name ('tiny', 'small', 'base', 'large').
        num_classes: Number of classes for classification head.
        **kwargs: Additional arguments to override defaults.

    Returns:
        GeometricTransformer model.
    """
    configs = {
        'tiny': {
            'embed_dim': 128,
            'num_heads': 4,
            'num_layers': 4,
            'mlp_ratio': 4,
        },
        'small': {
            'embed_dim': 192,
            'num_heads': 6,
            'num_layers': 6,
            'mlp_ratio': 4,
        },
        'base': {
            'embed_dim': 256,
            'num_heads': 8,
            'num_layers': 8,
            'mlp_ratio': 4,
        },
        'large': {
            'embed_dim': 384,
            'num_heads': 12,
            'num_layers': 12,
            'mlp_ratio': 4,
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}. Choose from {list(configs.keys())}")

    config = configs[config_name]
    config.update(kwargs)
    config['num_classes'] = num_classes

    return GeometricTransformer(**config)