"""
Visualization utilities for GFRAM.
"""

import cv2
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Optional, Tuple
import logging

logger = logging.getLogger(__name__)


def draw_landmarks(
        image: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        radius: int = 2
) -> np.ndarray:
    """
    Draw landmarks on image.

    Args:
        image: Input image.
        landmarks: Landmark coordinates (N, 2) or (N, 3).
        color: BGR color tuple.
        radius: Point radius.

    Returns:
        Image with landmarks drawn.
    """
    result = image.copy()
    points_2d = landmarks[:, :2]

    for point in points_2d:
        x, y = int(point[0]), int(point[1])
        cv2.circle(result, (x, y), radius, color, -1)

    return result


def draw_face_mesh(
        image: np.ndarray,
        landmarks: np.ndarray,
        color: Tuple[int, int, int] = (0, 255, 0),
        thickness: int = 1
) -> np.ndarray:
    """
    Draw face mesh connections on image.

    Args:
        image: Input image.
        landmarks: Landmark coordinates.
        color: Line color.
        thickness: Line thickness.

    Returns:
        Image with mesh drawn.
    """
    result = image.copy()
    points_2d = landmarks[:, :2].astype(int)

    # Simplified connections (just draw some representative lines)
    # In production, use MediaPipe's FACEMESH_TESSELATION
    connections = [
        (70, 63), (63, 105), (105, 66), (66, 107),  # Left eyebrow
        (336, 296), (296, 334), (334, 293), (293, 300),  # Right eyebrow
        (168, 6), (6, 197), (197, 195), (195, 5),  # Nose bridge
    ]

    for start, end in connections:
        if start < len(points_2d) and end < len(points_2d):
            pt1 = tuple(points_2d[start])
            pt2 = tuple(points_2d[end])
            cv2.line(result, pt1, pt2, color, thickness)

    return result


def draw_bounding_box(
        image: np.ndarray,
        bbox: Tuple[int, int, int, int],
        color: Tuple[int, int, int] = (255, 0, 0),
        thickness: int = 2
) -> np.ndarray:
    """
    Draw bounding box on image.

    Args:
        image: Input image.
        bbox: Bounding box (x, y, w, h).
        color: Box color.
        thickness: Line thickness.

    Returns:
        Image with box drawn.
    """
    result = image.copy()
    x, y, w, h = bbox
    cv2.rectangle(result, (x, y), (x + w, y + h), color, thickness)
    return result


def plot_feature_distribution(
        features: np.ndarray,
        feature_names: Optional[List[str]] = None,
        title: str = "Feature Distribution"
):
    """
    Plot distribution of features.

    Args:
        features: Feature array (N, D).
        feature_names: Names of features.
        title: Plot title.
    """
    plt.figure(figsize=(15, 5))

    if feature_names is None:
        feature_names = [f"F{i}" for i in range(features.shape[1])]

    # Box plot
    plt.subplot(1, 2, 1)
    plt.boxplot(features, labels=feature_names[:min(20, len(feature_names))])
    plt.xticks(rotation=90)
    plt.title(f"{title} - Box Plot")
    plt.tight_layout()

    # Histogram
    plt.subplot(1, 2, 2)
    for i in range(min(5, features.shape[1])):
        plt.hist(features[:, i], alpha=0.5, label=feature_names[i])
    plt.legend()
    plt.title(f"{title} - Histogram")
    plt.tight_layout()

    plt.show()


def visualize_recognition_result(
        image: np.ndarray,
        face_data: dict,
        show_landmarks: bool = True,
        show_bbox: bool = True
) -> np.ndarray:
    """
    Visualize face recognition result.

    Args:
        image: Input image.
        face_data: Face data with 'landmarks', 'bbox', 'name', 'confidence'.
        show_landmarks: Whether to show landmarks.
        show_bbox: Whether to show bounding box.

    Returns:
        Annotated image.
    """
    result = image.copy()

    # Draw bounding box
    if show_bbox and 'bbox' in face_data:
        result = draw_bounding_box(result, face_data['bbox'])

    # Draw landmarks
    if show_landmarks and 'landmarks' in face_data:
        result = draw_landmarks(result, face_data['landmarks'])

    # Draw name and confidence
    if 'name' in face_data and 'bbox' in face_data:
        x, y, w, h = face_data['bbox']
        name = face_data['name']
        confidence = face_data.get('confidence', 0.0)

        text = f"{name}: {confidence:.2%}"
        cv2.putText(result, text, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    return result