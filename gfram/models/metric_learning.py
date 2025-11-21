"""
Metric learning utilities.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Tuple


class MetricLearningHead(nn.Module):
    """
    Metric learning head for embedding learning.
    """

    def __init__(
            self,
            embedding_dim: int,
            normalize: bool = True
    ):
        super().__init__()
        self.embedding_dim = embedding_dim
        self.normalize = normalize

    def forward(self, embeddings: torch.Tensor) -> torch.Tensor:
        if self.normalize:
            embeddings = F.normalize(embeddings, p=2, dim=1)
        return embeddings


def compute_pairwise_distances(
        embeddings: torch.Tensor,
        squared: bool = False
) -> torch.Tensor:
    """
    Compute pairwise distances between embeddings.

    Args:
        embeddings: Embedding vectors (batch_size, embedding_dim).
        squared: Whether to return squared distances.

    Returns:
        Distance matrix (batch_size, batch_size).
    """
    dot_product = torch.matmul(embeddings, embeddings.t())
    square_norm = torch.diag(dot_product)

    distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
    distances = torch.clamp(distances, min=0.0)

    if not squared:
        mask = (distances == 0.0).float()
        distances = distances + mask * 1e-16
        distances = torch.sqrt(distances)
        distances = distances * (1.0 - mask)

    return distances


def compute_similarity_matrix(
        embeddings: torch.Tensor,
        temperature: float = 1.0
) -> torch.Tensor:
    """
    Compute cosine similarity matrix.

    Args:
        embeddings: Normalized embeddings.
        temperature: Temperature scaling.

    Returns:
        Similarity matrix.
    """
    embeddings = F.normalize(embeddings, p=2, dim=1)
    similarity = torch.matmul(embeddings, embeddings.t()) / temperature
    return similarity


class OnlineTripletMiner:
    """Mine triplets online during training."""

    def __init__(self, margin: float = 0.3):
        self.margin = margin

    def mine_hard_triplets(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Mine hard triplets (hardest positive and negative)."""
        distances = compute_pairwise_distances(embeddings)

        batch_size = embeddings.size(0)
        anchors = []
        positives = []
        negatives = []

        for i in range(batch_size):
            # Positive mask
            pos_mask = (labels == labels[i]) & (torch.arange(batch_size, device=embeddings.device) != i)

            if not pos_mask.any():
                continue

            # Negative mask
            neg_mask = labels != labels[i]

            if not neg_mask.any():
                continue

            # Hardest positive
            pos_distances = distances[i][pos_mask]
            hardest_pos_idx = pos_mask.nonzero()[pos_distances.argmax()]

            # Hardest negative
            neg_distances = distances[i][neg_mask]
            hardest_neg_idx = neg_mask.nonzero()[neg_distances.argmin()]

            anchors.append(i)
            positives.append(hardest_pos_idx.item())
            negatives.append(hardest_neg_idx.item())

        if len(anchors) == 0:
            return None, None, None

        return (
            torch.tensor(anchors, device=embeddings.device),
            torch.tensor(positives, device=embeddings.device),
            torch.tensor(negatives, device=embeddings.device)
        )