"""
Delaunay triangulation for facial landmarks.
"""

import numpy as np
from scipy.spatial import Delaunay
from typing import Optional
import logging

logger = logging.getLogger(__name__)


def compute_delaunay(landmarks: np.ndarray) -> Optional[Delaunay]:
    """Compute Delaunay triangulation of 2D landmarks."""
    try:
        points_2d = landmarks[:, :2]
        tri = Delaunay(points_2d)
        return tri
    except Exception as e:
        logger.error(f"Delaunay triangulation failed: {e}")
        return None


def extract_triangulation_features(landmarks: np.ndarray) -> np.ndarray:
    """Extract statistical features from triangulation (15 features)."""
    tri = compute_delaunay(landmarks)
    if tri is None:
        return np.zeros(15)

    features = []
    points_2d = landmarks[:, :2]
    num_points = len(landmarks)

    # Triangle count
    num_triangles = len(tri.simplices)
    features.append(num_triangles / num_points)

    # Triangle areas
    areas = []
    for simplex in tri.simplices:
        pts = points_2d[simplex]
        x1, y1 = pts[0]
        x2, y2 = pts[1]
        x3, y3 = pts[2]
        area = 0.5 * abs((x2 - x1) * (y3 - y1) - (x3 - x1) * (y2 - y1))
        areas.append(area)

    areas = np.array(areas)
    features.extend([np.mean(areas), np.std(areas), np.max(areas), np.min(areas)])

    # Triangle regularity
    regularities = []
    for simplex in tri.simplices:
        pts = points_2d[simplex]
        sides = [
            np.linalg.norm(pts[1] - pts[0]),
            np.linalg.norm(pts[2] - pts[1]),
            np.linalg.norm(pts[0] - pts[2])
        ]
        regularity = np.std(sides) / (np.mean(sides) + 1e-8)
        regularities.append(regularity)

    features.extend([np.mean(regularities), np.std(regularities)])

    # Edge statistics
    edges = set()
    for simplex in tri.simplices:
        edges.add(tuple(sorted([simplex[0], simplex[1]])))
        edges.add(tuple(sorted([simplex[1], simplex[2]])))
        edges.add(tuple(sorted([simplex[2], simplex[0]])))

    num_edges = len(edges)
    features.append(num_edges / num_points)

    # Edge lengths
    edge_lengths = []
    for i, j in edges:
        length = np.linalg.norm(points_2d[i] - points_2d[j])
        edge_lengths.append(length)

    edge_lengths = np.array(edge_lengths)
    features.extend([np.mean(edge_lengths), np.std(edge_lengths),
                     np.max(edge_lengths), np.min(edge_lengths)])

    # Connectivity
    adjacency = np.zeros((num_points, num_points))
    for simplex in tri.simplices:
        for i in range(3):
            for j in range(i + 1, 3):
                adjacency[simplex[i], simplex[j]] = 1
                adjacency[simplex[j], simplex[i]] = 1

    degrees = np.sum(adjacency, axis=1)
    features.extend([np.mean(degrees), np.std(degrees)])

    return np.array(features, dtype=np.float32)