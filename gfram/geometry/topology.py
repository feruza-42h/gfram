"""
Topological features (persistent homology) for facial landmarks.
"""

import numpy as np
from typing import List, Tuple
from scipy.spatial.distance import pdist, squareform


def compute_persistence_diagram(
        landmarks: np.ndarray,
        max_dimension: int = 1
) -> List[Tuple[float, float]]:
    """
    Compute persistence diagram (simplified version).

    Args:
        landmarks: Landmark coordinates.
        max_dimension: Maximum homology dimension.

    Returns:
        List of (birth, death) pairs.
    """
    # Simplified persistence computation using distance matrix
    points_2d = landmarks[:, :2]

    # Compute pairwise distances
    distances = squareform(pdist(points_2d))

    # Extract persistence pairs (simplified)
    persistence_pairs = []

    # Use distance thresholds as birth/death times
    thresholds = np.linspace(0, np.max(distances), 20)

    for i in range(len(thresholds) - 1):
        birth = thresholds[i]
        death = thresholds[i + 1]
        persistence_pairs.append((birth, death))

    return persistence_pairs


def compute_betti_numbers(landmarks: np.ndarray) -> Tuple[int, int]:
    """
    Compute Betti numbers (simplified).

    Args:
        landmarks: Landmark coordinates.

    Returns:
        Tuple of (β0, β1) - number of connected components and cycles.
    """
    # Simplified Betti number computation
    from scipy.sparse.csgraph import connected_components

    points_2d = landmarks[:, :2]
    distances = squareform(pdist(points_2d))

    # Create adjacency matrix (threshold = median distance)
    threshold = np.median(distances)
    adjacency = (distances < threshold).astype(int)

    # β0: number of connected components
    n_components, _ = connected_components(adjacency, directed=False)

    # β1: simplified estimate (number of edges - vertices + components)
    n_vertices = len(landmarks)
    n_edges = np.sum(adjacency) // 2
    beta1 = max(0, n_edges - n_vertices + n_components)

    return int(n_components), int(beta1)


def extract_topological_features(landmarks: np.ndarray) -> np.ndarray:
    """
    Extract topological features (20 features).

    Args:
        landmarks: Landmark coordinates.

    Returns:
        Topological feature vector.
    """
    features = []

    # Betti numbers
    beta0, beta1 = compute_betti_numbers(landmarks)
    features.extend([beta0, beta1])

    # Persistence diagram statistics
    persistence_pairs = compute_persistence_diagram(landmarks)

    if len(persistence_pairs) > 0:
        births = [p[0] for p in persistence_pairs]
        deaths = [p[1] for p in persistence_pairs]
        lifetimes = [d - b for b, d in persistence_pairs]

        # Statistics of births
        features.extend([
            np.mean(births),
            np.std(births),
            np.max(births),
            np.min(births),
        ])

        # Statistics of deaths
        features.extend([
            np.mean(deaths),
            np.std(deaths),
            np.max(deaths),
            np.min(deaths),
        ])

        # Statistics of lifetimes
        features.extend([
            np.mean(lifetimes),
            np.std(lifetimes),
            np.max(lifetimes),
            np.min(lifetimes),
        ])

        # Additional features
        features.extend([
            len(persistence_pairs),
            np.sum(lifetimes),
            np.median(lifetimes),
            np.percentile(lifetimes, 75),
            np.percentile(lifetimes, 25),
        ])
    else:
        # Fill with zeros if no persistence pairs
        features.extend([0.0] * 18)

    return np.array(features[:20], dtype=np.float32)