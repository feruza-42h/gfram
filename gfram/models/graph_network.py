"""
Geometric Graph Neural Network for Face Recognition.

Uses graph structure of facial landmarks to learn spatial relationships.
Implements message passing and graph convolutions.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple
import numpy as np


class GraphConvolution(nn.Module):
    """
    Simple graph convolution layer.

    GCN layer: H' = σ(D^(-1/2) A D^(-1/2) H W)
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            bias: bool = True
    ):
        """
        Initialize graph convolution layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            bias: Whether to use bias.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features

        self.weight = nn.Parameter(torch.FloatTensor(in_features, out_features))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.weight)
        if self.bias is not None:
            nn.init.zeros_(self.bias)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (batch_size, num_nodes, in_features).
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Output features (batch_size, num_nodes, out_features).
        """
        # Linear transformation
        support = torch.matmul(x, self.weight)

        # Graph convolution
        output = torch.bmm(adj, support)

        if self.bias is not None:
            output = output + self.bias

        return output


class GraphAttentionLayer(nn.Module):
    """
    Graph Attention Layer (GAT).

    Implements attention mechanism for graph convolutions.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            dropout: float = 0.1,
            alpha: float = 0.2,
            concat: bool = True
    ):
        """
        Initialize GAT layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            dropout: Dropout probability.
            alpha: LeakyReLU negative slope.
            concat: Whether to concatenate or average multi-head outputs.
        """
        super().__init__()

        self.in_features = in_features
        self.out_features = out_features
        self.dropout = dropout
        self.alpha = alpha
        self.concat = concat

        # Linear transformation
        self.W = nn.Parameter(torch.FloatTensor(in_features, out_features))

        # Attention parameters
        self.a = nn.Parameter(torch.FloatTensor(2 * out_features, 1))

        self.leakyrelu = nn.LeakyReLU(self.alpha)
        self.dropout_layer = nn.Dropout(dropout)

        self.reset_parameters()

    def reset_parameters(self):
        """Initialize parameters."""
        nn.init.xavier_uniform_(self.W)
        nn.init.xavier_uniform_(self.a)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features (batch_size, num_nodes, in_features).
            adj: Adjacency matrix (batch_size, num_nodes, num_nodes).

        Returns:
            Output features (batch_size, num_nodes, out_features).
        """
        batch_size, num_nodes, _ = x.shape

        # Linear transformation
        Wh = torch.matmul(x, self.W)  # (batch, num_nodes, out_features)

        # Attention mechanism
        # Compute attention coefficients
        a_input = self._prepare_attention_input(Wh)
        e = self.leakyrelu(torch.matmul(a_input, self.a).squeeze(-1))

        # Mask attention coefficients based on adjacency
        zero_vec = -9e15 * torch.ones_like(e)
        attention = torch.where(adj > 0, e, zero_vec)
        attention = F.softmax(attention, dim=-1)
        attention = self.dropout_layer(attention)

        # Apply attention to features
        h_prime = torch.bmm(attention, Wh)

        if self.concat:
            return F.elu(h_prime)
        else:
            return h_prime

    def _prepare_attention_input(self, Wh: torch.Tensor) -> torch.Tensor:
        """
        Prepare input for attention mechanism.

        Creates matrix of concatenated node pairs for attention computation.
        """
        batch_size, num_nodes, out_features = Wh.shape

        # Repeat for all pairs
        Wh_repeat_interleave = Wh.repeat_interleave(num_nodes, dim=1)  # (batch, N*N, out)
        Wh_repeat = Wh.repeat(1, num_nodes, 1)  # (batch, N*N, out)

        # Concatenate
        all_combinations = torch.cat([Wh_repeat_interleave, Wh_repeat], dim=-1)
        all_combinations = all_combinations.view(batch_size, num_nodes, num_nodes, 2 * out_features)

        return all_combinations


class GeometricGNNLayer(nn.Module):
    """
    Custom GNN layer for geometric data.
    Combines graph convolution with geometric features.
    """

    def __init__(
            self,
            in_features: int,
            out_features: int,
            use_attention: bool = True,
            dropout: float = 0.1
    ):
        """
        Initialize geometric GNN layer.

        Args:
            in_features: Input feature dimension.
            out_features: Output feature dimension.
            use_attention: Whether to use attention mechanism.
            dropout: Dropout probability.
        """
        super().__init__()

        self.use_attention = use_attention

        if use_attention:
            self.conv = GraphAttentionLayer(in_features, out_features, dropout)
        else:
            self.conv = GraphConvolution(in_features, out_features)

        self.norm = nn.LayerNorm(out_features)
        self.dropout = nn.Dropout(dropout)

        # Residual connection (if dimensions match)
        self.use_residual = (in_features == out_features)
        if not self.use_residual:
            self.residual_proj = nn.Linear(in_features, out_features)

    def forward(
            self,
            x: torch.Tensor,
            adj: torch.Tensor
    ) -> torch.Tensor:
        """
        Forward pass.

        Args:
            x: Node features.
            adj: Adjacency matrix.

        Returns:
            Output features.
        """
        # Save input for residual
        identity = x

        # Graph convolution
        out = self.conv(x, adj)
        out = self.norm(out)
        out = F.relu(out)
        out = self.dropout(out)

        # Residual connection
        if self.use_residual:
            out = out + identity
        elif hasattr(self, 'residual_proj'):
            out = out + self.residual_proj(identity)

        return out


class GraphPooling(nn.Module):
    """
    Graph pooling layer to aggregate node features.
    """

    def __init__(self, pooling_type: str = "mean"):
        """
        Initialize graph pooling.

        Args:
            pooling_type: Type of pooling ('mean', 'max', 'sum', 'attention').
        """
        super().__init__()
        self.pooling_type = pooling_type

    def forward(
            self,
            x: torch.Tensor,
            adj: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """
        Pool node features to graph-level representation.

        Args:
            x: Node features (batch_size, num_nodes, features).
            adj: Adjacency matrix (optional, for attention pooling).

        Returns:
            Graph-level features (batch_size, features).
        """
        if self.pooling_type == "mean":
            return torch.mean(x, dim=1)
        elif self.pooling_type == "max":
            return torch.max(x, dim=1)[0]
        elif self.pooling_type == "sum":
            return torch.sum(x, dim=1)
        else:
            raise ValueError(f"Unknown pooling type: {self.pooling_type}")


class GeometricGNN(nn.Module):
    """
    Geometric Graph Neural Network for face recognition.

    Uses graph structure of facial landmarks to learn representations.
    """

    def __init__(
            self,
            num_landmarks: int = 468,
            input_dim: int = 3,
            hidden_dims: list = [128, 256, 256],
            use_attention: bool = True,
            dropout: float = 0.1,
            pooling: str = "mean",
            num_classes: Optional[int] = None,
            output_embedding: bool = True
    ):
        """
        Initialize Geometric GNN.

        Args:
            num_landmarks: Number of facial landmarks.
            input_dim: Input feature dimension per node.
            hidden_dims: List of hidden layer dimensions.
            use_attention: Whether to use attention in graph layers.
            dropout: Dropout probability.
            pooling: Pooling method ('mean', 'max', 'sum').
            num_classes: Number of classes for classification.
            output_embedding: Whether to output embedding vector.
        """
        super().__init__()

        self.num_landmarks = num_landmarks
        self.num_classes = num_classes
        self.output_embedding = output_embedding

        # Initial feature projection
        self.input_proj = nn.Sequential(
            nn.Linear(input_dim, hidden_dims[0]),
            nn.LayerNorm(hidden_dims[0]),
            nn.ReLU(),
            nn.Dropout(dropout)
        )

        # Graph layers
        self.graph_layers = nn.ModuleList()
        for i in range(len(hidden_dims) - 1):
            self.graph_layers.append(
                GeometricGNNLayer(
                    hidden_dims[i],
                    hidden_dims[i + 1],
                    use_attention=use_attention,
                    dropout=dropout
                )
            )

        # Pooling
        self.pooling = GraphPooling(pooling)

        # Output heads
        embed_dim = hidden_dims[-1]

        if output_embedding:
            self.embedding_head = nn.Sequential(
                nn.Linear(embed_dim, embed_dim),
                nn.BatchNorm1d(embed_dim),
                nn.ReLU(),
                nn.Dropout(dropout)
            )

        if num_classes is not None:
            self.classifier = nn.Linear(embed_dim, num_classes)

        # Adjacency matrix (learned or fixed)
        self.register_buffer('adj_matrix', self._create_adjacency_matrix(num_landmarks))

    def _create_adjacency_matrix(self, num_landmarks: int) -> torch.Tensor:
        """
        Create adjacency matrix for facial landmarks.

        Uses k-nearest neighbors or predefined facial structure.
        """
        # For now, use simple k-NN connectivity
        # In practice, you would use actual facial structure
        k = 10  # Number of nearest neighbors

        # Create random adjacency for initialization
        # In real implementation, use actual facial landmark connectivity
        adj = torch.eye(num_landmarks)

        # Add k-NN connections (simplified)
        for i in range(num_landmarks):
            for j in range(max(0, i - k // 2), min(num_landmarks, i + k // 2 + 1)):
                if i != j:
                    adj[i, j] = 1.0
                    adj[j, i] = 1.0

        # Normalize adjacency matrix
        deg = adj.sum(dim=1)
        deg_inv_sqrt = torch.pow(deg, -0.5)
        deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.0

        adj_normalized = deg_inv_sqrt.unsqueeze(1) * adj * deg_inv_sqrt.unsqueeze(0)

        return adj_normalized

    def forward(
            self,
            x: torch.Tensor,
            adj: Optional[torch.Tensor] = None,
            return_embedding: bool = True
    ) -> Tuple[Optional[torch.Tensor], Optional[torch.Tensor]]:
        """
        Forward pass.

        Args:
            x: Input landmarks (batch_size, num_landmarks, input_dim).
            adj: Custom adjacency matrix (optional, uses default if None).
            return_embedding: Whether to return embedding vector.

        Returns:
            Tuple of (logits, embedding).
        """
        batch_size = x.size(0)

        # Use default adjacency if not provided
        if adj is None:
            adj = self.adj_matrix.unsqueeze(0).repeat(batch_size, 1, 1)

        # Initial projection
        x = self.input_proj(x)

        # Apply graph layers
        for layer in self.graph_layers:
            x = layer(x, adj)

        # Pool to graph-level representation
        x = self.pooling(x, adj)

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


def create_geometric_gnn(
        config_name: str = "base",
        num_classes: Optional[int] = None,
        **kwargs
) -> GeometricGNN:
    """
    Create a Geometric GNN model with predefined configurations.

    Args:
        config_name: Configuration name ('small', 'base', 'large').
        num_classes: Number of classes for classification head.
        **kwargs: Additional arguments to override defaults.

    Returns:
        GeometricGNN model.
    """
    configs = {
        'small': {
            'hidden_dims': [64, 128, 128],
            'use_attention': False,
        },
        'base': {
            'hidden_dims': [128, 256, 256],
            'use_attention': True,
        },
        'large': {
            'hidden_dims': [256, 512, 512],
            'use_attention': True,
        }
    }

    if config_name not in configs:
        raise ValueError(f"Unknown config: {config_name}")

    config = configs[config_name]
    config.update(kwargs)
    config['num_classes'] = num_classes

    return GeometricGNN(**config)