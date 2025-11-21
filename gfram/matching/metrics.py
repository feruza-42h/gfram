"""
Distance metrics for face matching.
"""

import numpy as np
from typing import Callable


def euclidean_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Euclidean (L2) distance."""
    return float(np.linalg.norm(a - b))


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Cosine distance (1 - cosine similarity)."""
    sim = np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-8)
    return float(1.0 - sim)


def manhattan_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Manhattan (L1) distance."""
    return float(np.sum(np.abs(a - b)))


def chebyshev_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Chebyshev (L-infinity) distance."""
    return float(np.max(np.abs(a - b)))


def mahalanobis_distance(
        a: np.ndarray,
        b: np.ndarray,
        cov_inv: np.ndarray
) -> float:
    """Mahalanobis distance."""
    diff = a - b
    return float(np.sqrt(diff @ cov_inv @ diff.T))


class DistanceMetric:
    """Distance metric wrapper."""

    def __init__(self, metric: str = 'euclidean'):
        self.metric = metric
        self._metric_fn = self._get_metric_fn(metric)

    def _get_metric_fn(self, metric: str) -> Callable:
        metrics = {
            'euclidean': euclidean_distance,
            'cosine': cosine_distance,
            'manhattan': manhattan_distance,
            'chebyshev': chebyshev_distance,
        }

        if metric not in metrics:
            raise ValueError(f"Unknown metric: {metric}")

        return metrics[metric]

    def __call__(self, a: np.ndarray, b: np.ndarray) -> float:
        return self._metric_fn(a, b)

    def batch(self, queries: np.ndarray, database: np.ndarray) -> np.ndarray:
        """Compute pairwise distances."""
        from scipy.spatial.distance import cdist

        if self.metric == 'cosine':
            return cdist(queries, database, metric='cosine')
        else:
            return cdist(queries, database, metric=self.metric)