"""
Input/Output utilities for GFRAM.
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Optional, List
import logging

logger = logging.getLogger(__name__)


def load_image(path: str) -> Optional[np.ndarray]:
    """
    Load image from file.

    Args:
        path: Path to image file.

    Returns:
        Image as numpy array (BGR) or None if failed.
    """
    try:
        image = cv2.imread(str(path))
        if image is None:
            logger.error(f"Failed to load image: {path}")
            return None
        return image
    except Exception as e:
        logger.error(f"Error loading image {path}: {e}")
        return None


def save_image(image: np.ndarray, path: str) -> bool:
    """
    Save image to file.

    Args:
        image: Image array.
        path: Output path.

    Returns:
        True if successful, False otherwise.
    """
    try:
        Path(path).parent.mkdir(parents=True, exist_ok=True)
        success = cv2.imwrite(str(path), image)
        if not success:
            logger.error(f"Failed to save image: {path}")
        return success
    except Exception as e:
        logger.error(f"Error saving image {path}: {e}")
        return False


def load_images_from_directory(
        directory: str,
        extensions: List[str] = ['.jpg', '.jpeg', '.png']
) -> List[tuple]:
    """
    Load all images from directory.

    Args:
        directory: Directory path.
        extensions: List of valid extensions.

    Returns:
        List of (path, image) tuples.
    """
    directory = Path(directory)
    images = []

    for ext in extensions:
        for path in directory.glob(f'*{ext}'):
            image = load_image(str(path))
            if image is not None:
                images.append((str(path), image))

    return images


def save_landmarks(landmarks: np.ndarray, path: str):
    """Save landmarks to numpy file."""
    np.save(path, landmarks)


def load_landmarks(path: str) -> Optional[np.ndarray]:
    """Load landmarks from numpy file."""
    try:
        return np.load(path)
    except Exception as e:
        logger.error(f"Error loading landmarks {path}: {e}")
        return None