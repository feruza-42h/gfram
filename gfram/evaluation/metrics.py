"""
Face Recognition Metrics
========================

Standard metrics for face verification and identification.

PhD Thesis: Ortiqova F.S.
"""

import numpy as np
from typing import Tuple, List, Optional
from dataclasses import dataclass
from sklearn.metrics import roc_curve, auc, precision_recall_curve


def compute_roc(
        similarities: np.ndarray,
        labels: np.ndarray
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Compute ROC curve.

    Args:
        similarities: Pairwise similarity scores
        labels: Ground truth labels (1 for same person, 0 for different)

    Returns:
        (fpr, tpr, thresholds) arrays
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    return fpr, tpr, thresholds


def compute_tar_at_far(
        similarities: np.ndarray,
        labels: np.ndarray,
        target_far: float = 0.001
) -> float:
    """
    Compute True Accept Rate at given False Accept Rate.

    TAR@FAR is a key metric for face verification.

    Args:
        similarities: Pairwise similarity scores
        labels: Ground truth labels
        target_far: Target FAR (e.g., 0.001 for 0.1%)

    Returns:
        TAR at the given FAR
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)

    # Find TAR at target FAR
    # Interpolate if needed
    idx = np.searchsorted(fpr, target_far)

    if idx == 0:
        return tpr[0]
    if idx >= len(fpr):
        return tpr[-1]

    # Linear interpolation
    far_low, far_high = fpr[idx - 1], fpr[idx]
    tar_low, tar_high = tpr[idx - 1], tpr[idx]

    if far_high == far_low:
        return tar_low

    tar = tar_low + (tar_high - tar_low) * (target_far - far_low) / (far_high - far_low)

    return float(tar)


def compute_eer(
        similarities: np.ndarray,
        labels: np.ndarray
) -> Tuple[float, float]:
    """
    Compute Equal Error Rate (EER).

    EER is the point where FAR = FRR (False Reject Rate).

    Args:
        similarities: Pairwise similarity scores
        labels: Ground truth labels

    Returns:
        (eer, threshold) - EER value and corresponding threshold
    """
    fpr, tpr, thresholds = roc_curve(labels, similarities)
    fnr = 1 - tpr  # False Negative Rate = 1 - True Positive Rate

    # Find where FAR = FNR
    # EER is where the ROC curve crosses the diagonal
    eer_idx = np.nanargmin(np.abs(fpr - fnr))
    eer = (fpr[eer_idx] + fnr[eer_idx]) / 2

    threshold = thresholds[eer_idx] if eer_idx < len(thresholds) else 0.5

    return float(eer), float(threshold)


def compute_accuracy(
        similarities: np.ndarray,
        labels: np.ndarray,
        threshold: Optional[float] = None
) -> Tuple[float, float]:
    """
    Compute verification accuracy.

    Args:
        similarities: Pairwise similarity scores
        labels: Ground truth labels
        threshold: Decision threshold (finds best if None)

    Returns:
        (accuracy, best_threshold)
    """
    if threshold is None:
        # Find best threshold
        best_acc = 0.0
        best_thresh = 0.5

        for thresh in np.linspace(0, 1, 1000):
            predictions = (similarities >= thresh).astype(int)
            acc = np.mean(predictions == labels)

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        return float(best_acc), float(best_thresh)
    else:
        predictions = (similarities >= threshold).astype(int)
        acc = np.mean(predictions == labels)
        return float(acc), threshold


@dataclass
class FaceVerificationMetrics:
    """
    Comprehensive face verification metrics.

    Usage:
        metrics = FaceVerificationMetrics(similarities, labels)
        print(f"Accuracy: {metrics.accuracy}")
        print(f"AUC: {metrics.auc}")
        print(f"EER: {metrics.eer}")
        print(f"TAR@FAR=0.1%: {metrics.tar_at_far(0.001)}")
    """

    def __init__(
            self,
            similarities: np.ndarray,
            labels: np.ndarray
    ):
        """
        Compute all metrics.

        Args:
            similarities: Pairwise similarity scores
            labels: Ground truth labels
        """
        self.similarities = similarities
        self.labels = labels

        # ROC curve
        self.fpr, self.tpr, self.thresholds = compute_roc(similarities, labels)

        # AUC
        self.auc = float(auc(self.fpr, self.tpr))

        # EER
        self.eer, self.eer_threshold = compute_eer(similarities, labels)

        # Best accuracy
        self.accuracy, self.best_threshold = compute_accuracy(similarities, labels)

        # Confusion matrix at best threshold
        predictions = (similarities >= self.best_threshold).astype(int)
        self.tp = np.sum((predictions == 1) & (labels == 1))
        self.tn = np.sum((predictions == 0) & (labels == 0))
        self.fp = np.sum((predictions == 1) & (labels == 0))
        self.fn = np.sum((predictions == 0) & (labels == 1))

    def tar_at_far(self, target_far: float) -> float:
        """Get TAR at specific FAR"""
        return compute_tar_at_far(self.similarities, self.labels, target_far)

    @property
    def precision(self) -> float:
        """Precision = TP / (TP + FP)"""
        return self.tp / (self.tp + self.fp + 1e-8)

    @property
    def recall(self) -> float:
        """Recall = TP / (TP + FN)"""
        return self.tp / (self.tp + self.fn + 1e-8)

    @property
    def f1(self) -> float:
        """F1 Score = 2 * Precision * Recall / (Precision + Recall)"""
        p, r = self.precision, self.recall
        return 2 * p * r / (p + r + 1e-8)

    @property
    def far(self) -> float:
        """False Accept Rate at best threshold"""
        return self.fp / (self.fp + self.tn + 1e-8)

    @property
    def frr(self) -> float:
        """False Reject Rate at best threshold"""
        return self.fn / (self.fn + self.tp + 1e-8)

    def summary(self) -> dict:
        """Get summary of all metrics"""
        return {
            'accuracy': self.accuracy,
            'auc': self.auc,
            'eer': self.eer,
            'precision': self.precision,
            'recall': self.recall,
            'f1': self.f1,
            'far': self.far,
            'frr': self.frr,
            'best_threshold': self.best_threshold,
            'eer_threshold': self.eer_threshold,
            'tar_at_far_1e-1': self.tar_at_far(0.1),
            'tar_at_far_1e-2': self.tar_at_far(0.01),
            'tar_at_far_1e-3': self.tar_at_far(0.001),
            'tar_at_far_1e-4': self.tar_at_far(0.0001),
            'confusion_matrix': {
                'tp': int(self.tp),
                'tn': int(self.tn),
                'fp': int(self.fp),
                'fn': int(self.fn)
            }
        }

    def __repr__(self) -> str:
        return (
            f"FaceVerificationMetrics(\n"
            f"  accuracy={self.accuracy:.4f},\n"
            f"  auc={self.auc:.4f},\n"
            f"  eer={self.eer:.4f},\n"
            f"  tar@far=0.1%={self.tar_at_far(0.001):.4f}\n"
            f")"
        )


class IdentificationMetrics:
    """
    Face identification (1:N) metrics.

    Unlike verification (1:1), identification finds a person
    from a gallery of N candidates.
    """

    def __init__(
            self,
            query_embeddings: np.ndarray,
            query_labels: np.ndarray,
            gallery_embeddings: np.ndarray,
            gallery_labels: np.ndarray
    ):
        """
        Compute identification metrics.

        Args:
            query_embeddings: Embeddings of query faces
            query_labels: Labels of query faces
            gallery_embeddings: Embeddings in gallery
            gallery_labels: Labels in gallery
        """
        self.query_embeddings = query_embeddings
        self.query_labels = query_labels
        self.gallery_embeddings = gallery_embeddings
        self.gallery_labels = gallery_labels

        # Compute similarity matrix
        query_norm = query_embeddings / (np.linalg.norm(query_embeddings, axis=1, keepdims=True) + 1e-8)
        gallery_norm = gallery_embeddings / (np.linalg.norm(gallery_embeddings, axis=1, keepdims=True) + 1e-8)

        self.similarity_matrix = np.dot(query_norm, gallery_norm.T)

    def rank_k_accuracy(self, k: int = 1) -> float:
        """
        Rank-k identification accuracy.

        Correct if true identity is in top-k matches.

        Args:
            k: Rank (1 for top-1 accuracy)

        Returns:
            Rank-k accuracy
        """
        correct = 0

        for i, query_label in enumerate(self.query_labels):
            # Get top-k indices
            top_k_indices = np.argsort(self.similarity_matrix[i])[::-1][:k]
            top_k_labels = self.gallery_labels[top_k_indices]

            if query_label in top_k_labels:
                correct += 1

        return correct / len(self.query_labels)

    def cmc_curve(self, max_rank: int = 20) -> np.ndarray:
        """
        Compute Cumulative Match Characteristic (CMC) curve.

        Args:
            max_rank: Maximum rank to compute

        Returns:
            CMC curve values
        """
        cmc = np.zeros(max_rank)

        for k in range(1, max_rank + 1):
            cmc[k - 1] = self.rank_k_accuracy(k)

        return cmc

    def summary(self) -> dict:
        """Get summary of identification metrics"""
        cmc = self.cmc_curve(20)

        return {
            'rank_1': float(cmc[0]),
            'rank_5': float(cmc[4]) if len(cmc) > 4 else None,
            'rank_10': float(cmc[9]) if len(cmc) > 9 else None,
            'rank_20': float(cmc[19]) if len(cmc) > 19 else None,
            'num_queries': len(self.query_labels),
            'gallery_size': len(self.gallery_labels),
            'num_identities': len(np.unique(self.gallery_labels))
        }


# Utility functions
def compute_distance_matrix(
        embeddings1: np.ndarray,
        embeddings2: np.ndarray,
        metric: str = 'cosine'
) -> np.ndarray:
    """
    Compute pairwise distance matrix.

    Args:
        embeddings1: First set of embeddings (N, D)
        embeddings2: Second set of embeddings (M, D)
        metric: Distance metric ('cosine', 'euclidean')

    Returns:
        Distance matrix (N, M)
    """
    if metric == 'cosine':
        # Normalize
        e1_norm = embeddings1 / (np.linalg.norm(embeddings1, axis=1, keepdims=True) + 1e-8)
        e2_norm = embeddings2 / (np.linalg.norm(embeddings2, axis=1, keepdims=True) + 1e-8)

        # Cosine distance = 1 - cosine similarity
        similarity = np.dot(e1_norm, e2_norm.T)
        return 1 - similarity

    elif metric == 'euclidean':
        # Efficient euclidean distance computation
        e1_sq = np.sum(embeddings1 ** 2, axis=1, keepdims=True)
        e2_sq = np.sum(embeddings2 ** 2, axis=1, keepdims=True)
        cross = np.dot(embeddings1, embeddings2.T)

        distances = np.sqrt(e1_sq + e2_sq.T - 2 * cross + 1e-8)
        return distances

    else:
        raise ValueError(f"Unknown metric: {metric}")


def print_metrics_table(metrics: FaceVerificationMetrics):
    """Print metrics in a nice table format"""
    print("\n" + "=" * 50)
    print("FACE VERIFICATION METRICS")
    print("=" * 50)
    print(f"{'Metric':<25} {'Value':>15}")
    print("-" * 50)
    print(f"{'Accuracy':<25} {metrics.accuracy * 100:>14.2f}%")
    print(f"{'AUC':<25} {metrics.auc:>15.4f}")
    print(f"{'EER':<25} {metrics.eer * 100:>14.2f}%")
    print(f"{'Precision':<25} {metrics.precision * 100:>14.2f}%")
    print(f"{'Recall':<25} {metrics.recall * 100:>14.2f}%")
    print(f"{'F1 Score':<25} {metrics.f1:>15.4f}")
    print("-" * 50)
    print(f"{'TAR@FAR=10%':<25} {metrics.tar_at_far(0.1) * 100:>14.2f}%")
    print(f"{'TAR@FAR=1%':<25} {metrics.tar_at_far(0.01) * 100:>14.2f}%")
    print(f"{'TAR@FAR=0.1%':<25} {metrics.tar_at_far(0.001) * 100:>14.2f}%")
    print(f"{'TAR@FAR=0.01%':<25} {metrics.tar_at_far(0.0001) * 100:>14.2f}%")
    print("-" * 50)
    print(f"{'Best Threshold':<25} {metrics.best_threshold:>15.4f}")
    print(f"{'EER Threshold':<25} {metrics.eer_threshold:>15.4f}")
    print("=" * 50 + "\n")
