"""
Loss functions for metric learning in face recognition.

Implements:
1. Triplet Loss with various mining strategies
2. ArcFace Loss (Additive Angular Margin)
3. CosFace Loss (Large Margin Cosine Loss)
4. Contrastive Loss
5. Center Loss
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import math
from typing import Optional, Tuple


class TripletLoss(nn.Module):
    """
    Triplet loss with various mining strategies.

    Loss = max(d(a,p) - d(a,n) + margin, 0)
    where d is distance, a is anchor, p is positive, n is negative.
    """

    def __init__(
            self,
            margin: float = 0.3,
            mining: str = "hard",  # "hard", "semi-hard", "all"
            distance: str = "euclidean",  # "euclidean", "cosine"
    ):
        """
        Initialize triplet loss.

        Args:
            margin: Margin for triplet loss.
            mining: Triplet mining strategy ('hard', 'semi-hard', 'all').
            distance: Distance metric ('euclidean', 'cosine').
        """
        super().__init__()
        self.margin = margin
        self.mining = mining
        self.distance = distance

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute triplet loss.

        Args:
            embeddings: Embedding vectors (batch_size, embedding_dim).
            labels: Labels (batch_size,).

        Returns:
            Loss value.
        """
        # Normalize embeddings if using cosine distance
        if self.distance == "cosine":
            embeddings = F.normalize(embeddings, p=2, dim=1)

        # Compute pairwise distances
        dist_mat = self._pairwise_distance(embeddings)

        # Mine triplets based on strategy
        if self.mining == "hard":
            loss = self._hard_mining(dist_mat, labels)
        elif self.mining == "semi-hard":
            loss = self._semi_hard_mining(dist_mat, labels)
        else:
            loss = self._all_mining(dist_mat, labels)

        return loss

    def _pairwise_distance(self, embeddings: torch.Tensor) -> torch.Tensor:
        """Compute pairwise distances."""
        if self.distance == "euclidean":
            dot_product = torch.matmul(embeddings, embeddings.t())
            square_norm = torch.diag(dot_product)
            distances = square_norm.unsqueeze(0) - 2.0 * dot_product + square_norm.unsqueeze(1)
            distances = torch.clamp(distances, min=0.0).sqrt()
        else:  # cosine
            distances = 1.0 - torch.matmul(embeddings, embeddings.t())

        return distances

    def _hard_mining(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Hard negative mining."""
        batch_size = dist_mat.size(0)

        # Get positive and negative masks
        labels = labels.unsqueeze(0)
        pos_mask = labels == labels.t()
        neg_mask = ~pos_mask

        # For each anchor, find hardest positive and negative
        pos_dist = dist_mat * pos_mask.float()
        pos_dist = pos_dist.masked_fill(~pos_mask, float('-inf'))
        hardest_positive = pos_dist.max(dim=1)[0]

        neg_dist = dist_mat.clone()
        neg_dist = neg_dist.masked_fill(pos_mask, float('inf'))
        hardest_negative = neg_dist.min(dim=1)[0]

        # Compute triplet loss
        loss = F.relu(hardest_positive - hardest_negative + self.margin)

        return loss.mean()

    def _semi_hard_mining(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Semi-hard negative mining."""
        batch_size = dist_mat.size(0)

        # Get masks
        labels = labels.unsqueeze(0)
        pos_mask = labels == labels.t()
        neg_mask = ~pos_mask

        # For each anchor-positive pair, find semi-hard negatives
        losses = []
        for i in range(batch_size):
            # Get positive samples for anchor i
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            if len(pos_indices) <= 1:  # Skip if no positive pairs
                continue

            for pos_idx in pos_indices:
                if pos_idx == i:
                    continue

                pos_dist = dist_mat[i, pos_idx]

                # Find semi-hard negatives: d(a,n) > d(a,p) but d(a,n) < d(a,p) + margin
                neg_dists = dist_mat[i] * neg_mask[i].float()
                semi_hard_mask = (neg_dists > pos_dist) & (neg_dists < pos_dist + self.margin)

                if semi_hard_mask.any():
                    neg_dist = neg_dists[semi_hard_mask].min()
                    losses.append(F.relu(pos_dist - neg_dist + self.margin))

        if len(losses) == 0:
            return torch.tensor(0.0, device=dist_mat.device)

        return torch.stack(losses).mean()

    def _all_mining(self, dist_mat: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Use all valid triplets."""
        batch_size = dist_mat.size(0)

        # Get masks
        labels = labels.unsqueeze(0)
        pos_mask = labels == labels.t()
        neg_mask = ~pos_mask

        # Compute all triplet losses
        losses = []
        for i in range(batch_size):
            pos_indices = pos_mask[i].nonzero(as_tuple=True)[0]
            neg_indices = neg_mask[i].nonzero(as_tuple=True)[0]

            if len(pos_indices) <= 1 or len(neg_indices) == 0:
                continue

            for pos_idx in pos_indices:
                if pos_idx == i:
                    continue

                pos_dist = dist_mat[i, pos_idx]

                for neg_idx in neg_indices:
                    neg_dist = dist_mat[i, neg_idx]
                    loss = F.relu(pos_dist - neg_dist + self.margin)
                    if loss > 0:
                        losses.append(loss)

        if len(losses) == 0:
            return torch.tensor(0.0, device=dist_mat.device)

        return torch.stack(losses).mean()


class ArcFaceLoss(nn.Module):
    """
    ArcFace: Additive Angular Margin Loss for Deep Face Recognition.

    Reference: https://arxiv.org/abs/1801.07698
    """

    def __init__(
            self,
            embedding_dim: int,
            num_classes: int,
            margin: float = 0.5,
            scale: float = 64.0,
    ):
        """
        Initialize ArcFace loss.

        Args:
            embedding_dim: Dimension of embeddings.
            num_classes: Number of identity classes.
            margin: Angular margin (m in paper).
            scale: Feature scale (s in paper).
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        # Weight matrix
        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

        self.cos_m = math.cos(margin)
        self.sin_m = math.sin(margin)
        self.th = math.cos(math.pi - margin)
        self.mm = math.sin(math.pi - margin) * margin

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute ArcFace loss.

        Args:
            embeddings: Embedding vectors (batch_size, embedding_dim).
            labels: Ground truth labels (batch_size,).

        Returns:
            Loss value.
        """
        # Normalize features and weights
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)
        sine = torch.sqrt(1.0 - torch.pow(cosine, 2))

        # cos(theta + m)
        phi = cosine * self.cos_m - sine * self.sin_m

        # Handle numerical stability
        phi = torch.where(cosine > self.th, phi, cosine - self.mm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Combine original and margin-adjusted logits
        output = (one_hot * phi) + ((1.0 - one_hot) * cosine)
        output *= self.scale

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class CosFaceLoss(nn.Module):
    """
    CosFace: Large Margin Cosine Loss.

    Reference: https://arxiv.org/abs/1801.09414
    """

    def __init__(
            self,
            embedding_dim: int,
            num_classes: int,
            margin: float = 0.35,
            scale: float = 64.0,
    ):
        """
        Initialize CosFace loss.

        Args:
            embedding_dim: Dimension of embeddings.
            num_classes: Number of identity classes.
            margin: Cosine margin.
            scale: Feature scale.
        """
        super().__init__()
        self.embedding_dim = embedding_dim
        self.num_classes = num_classes
        self.margin = margin
        self.scale = scale

        self.weight = nn.Parameter(torch.FloatTensor(num_classes, embedding_dim))
        nn.init.xavier_uniform_(self.weight)

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute CosFace loss.

        Args:
            embeddings: Embedding vectors (batch_size, embedding_dim).
            labels: Ground truth labels (batch_size,).

        Returns:
            Loss value.
        """
        # Normalize
        embeddings = F.normalize(embeddings, p=2, dim=1)
        weight_norm = F.normalize(self.weight, p=2, dim=1)

        # Cosine similarity
        cosine = F.linear(embeddings, weight_norm)

        # One-hot encoding
        one_hot = torch.zeros(cosine.size(), device=embeddings.device)
        one_hot.scatter_(1, labels.view(-1, 1).long(), 1)

        # Add margin
        output = self.scale * (cosine - one_hot * self.margin)

        # Cross entropy loss
        loss = F.cross_entropy(output, labels)

        return loss


class ContrastiveLoss(nn.Module):
    """
    Contrastive loss for pairs of samples.
    """

    def __init__(self, margin: float = 1.0):
        """
        Initialize contrastive loss.

        Args:
            margin: Margin for dissimilar pairs.
        """
        super().__init__()
        self.margin = margin

    def forward(
            self,
            embeddings1: torch.Tensor,
            embeddings2: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute contrastive loss.

        Args:
            embeddings1: First set of embeddings.
            embeddings2: Second set of embeddings.
            labels: 1 for similar pairs, 0 for dissimilar pairs.

        Returns:
            Loss value.
        """
        # Euclidean distance
        distances = F.pairwise_distance(embeddings1, embeddings2)

        # Contrastive loss
        loss_similar = labels * torch.pow(distances, 2)
        loss_dissimilar = (1 - labels) * torch.pow(
            torch.clamp(self.margin - distances, min=0.0), 2
        )

        loss = (loss_similar + loss_dissimilar).mean()

        return loss


class CenterLoss(nn.Module):
    """
    Center loss to minimize intra-class variations.

    Reference: https://ydwen.github.io/papers/WenECCV16.pdf
    """

    def __init__(
            self,
            num_classes: int,
            embedding_dim: int,
            lambda_c: float = 0.003
    ):
        """
        Initialize center loss.

        Args:
            num_classes: Number of classes.
            embedding_dim: Dimension of embeddings.
            lambda_c: Weight for center loss.
        """
        super().__init__()
        self.num_classes = num_classes
        self.embedding_dim = embedding_dim
        self.lambda_c = lambda_c

        # Centers
        self.centers = nn.Parameter(torch.randn(num_classes, embedding_dim))

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """
        Compute center loss.

        Args:
            embeddings: Embedding vectors.
            labels: Ground truth labels.

        Returns:
            Loss value.
        """
        batch_size = embeddings.size(0)

        # Compute distances to centers
        distmat = torch.pow(embeddings, 2).sum(dim=1, keepdim=True).expand(batch_size, self.num_classes) + \
                  torch.pow(self.centers, 2).sum(dim=1, keepdim=True).expand(self.num_classes, batch_size).t()
        distmat.addmm_(embeddings, self.centers.t(), beta=1, alpha=-2)

        # Select distances for ground truth classes
        classes = torch.arange(self.num_classes, device=embeddings.device).long()
        labels = labels.unsqueeze(1).expand(batch_size, self.num_classes)
        mask = labels.eq(classes.expand(batch_size, self.num_classes))

        dist = distmat * mask.float()
        loss = dist.clamp(min=1e-12, max=1e+12).sum() / batch_size

        return self.lambda_c * loss


class CombinedLoss(nn.Module):
    """
    Combines multiple loss functions for comprehensive training.
    """

    def __init__(
            self,
            embedding_dim: int,
            num_classes: int,
            use_arcface: bool = True,
            use_triplet: bool = True,
            use_center: bool = True,
            arcface_margin: float = 0.5,
            arcface_scale: float = 64.0,
            triplet_margin: float = 0.3,
            center_lambda: float = 0.003,
    ):
        """
        Initialize combined loss.

        Args:
            embedding_dim: Dimension of embeddings.
            num_classes: Number of classes.
            use_arcface: Whether to use ArcFace loss.
            use_triplet: Whether to use Triplet loss.
            use_center: Whether to use Center loss.
            arcface_margin: Margin for ArcFace.
            arcface_scale: Scale for ArcFace.
            triplet_margin: Margin for Triplet loss.
            center_lambda: Weight for Center loss.
        """
        super().__init__()

        self.use_arcface = use_arcface
        self.use_triplet = use_triplet
        self.use_center = use_center

        if use_arcface:
            self.arcface = ArcFaceLoss(
                embedding_dim=embedding_dim,
                num_classes=num_classes,
                margin=arcface_margin,
                scale=arcface_scale
            )

        if use_triplet:
            self.triplet = TripletLoss(margin=triplet_margin, mining="hard")

        if use_center:
            self.center = CenterLoss(
                num_classes=num_classes,
                embedding_dim=embedding_dim,
                lambda_c=center_lambda
            )

    def forward(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> Tuple[torch.Tensor, dict]:
        """
        Compute combined loss.

        Args:
            embeddings: Embedding vectors.
            labels: Ground truth labels.

        Returns:
            Total loss and dictionary of individual losses.
        """
        losses = {}
        total_loss = 0.0

        if self.use_arcface:
            arcface_loss = self.arcface(embeddings, labels)
            losses['arcface'] = arcface_loss.item()
            total_loss += arcface_loss

        if self.use_triplet:
            triplet_loss = self.triplet(embeddings, labels)
            losses['triplet'] = triplet_loss.item()
            total_loss += triplet_loss

        if self.use_center:
            center_loss = self.center(embeddings, labels)
            losses['center'] = center_loss.item()
            total_loss += center_loss

        losses['total'] = total_loss.item()

        return total_loss, losses