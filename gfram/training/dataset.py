"""
Dataset classes for training GFRAM models.
"""

import torch
from torch.utils.data import Dataset
import numpy as np
from pathlib import Path
from typing import List, Tuple, Optional, Callable
import logging

logger = logging.getLogger(__name__)


class FaceLandmarkDataset(Dataset):
    """
    Dataset for face landmarks and labels.
    """

    def __init__(
            self,
            landmarks: np.ndarray,
            labels: np.ndarray,
            transform: Optional[Callable] = None
    ):
        """
        Initialize dataset.

        Args:
            landmarks: Array of landmarks (N, num_landmarks, 3).
            labels: Array of labels (N,).
            transform: Optional transform function.
        """
        self.landmarks = torch.from_numpy(landmarks).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor]:
        landmarks = self.landmarks[idx]
        label = self.labels[idx]

        if self.transform:
            landmarks = self.transform(landmarks)

        return landmarks, label


class TripletFaceDataset(Dataset):
    """
    Dataset for triplet learning (anchor, positive, negative).
    """

    def __init__(
            self,
            landmarks: np.ndarray,
            labels: np.ndarray,
            transform: Optional[Callable] = None
    ):
        """
        Initialize triplet dataset.

        Args:
            landmarks: Array of landmarks.
            labels: Array of labels.
            transform: Optional transform function.
        """
        self.landmarks = torch.from_numpy(landmarks).float()
        self.labels = torch.from_numpy(labels).long()
        self.transform = transform

        # Create label to indices mapping
        self.label_to_indices = {}
        for idx, label in enumerate(self.labels):
            label_item = label.item()
            if label_item not in self.label_to_indices:
                self.label_to_indices[label_item] = []
            self.label_to_indices[label_item].append(idx)

    def __len__(self) -> int:
        return len(self.landmarks)

    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        # Anchor
        anchor = self.landmarks[idx]
        anchor_label = self.labels[idx].item()

        # Positive (same label, different sample)
        positive_indices = self.label_to_indices[anchor_label]
        positive_idx = np.random.choice([i for i in positive_indices if i != idx])
        positive = self.landmarks[positive_idx]

        # Negative (different label)
        negative_label = np.random.choice([l for l in self.label_to_indices.keys() if l != anchor_label])
        negative_idx = np.random.choice(self.label_to_indices[negative_label])
        negative = self.landmarks[negative_idx]

        if self.transform:
            anchor = self.transform(anchor)
            positive = self.transform(positive)
            negative = self.transform(negative)

        return anchor, positive, negative


def load_dataset_from_directory(
        data_dir: str,
        detector,
        feature_extractor=None
) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load dataset from directory structure: data_dir/person_name/*.jpg

    Args:
        data_dir: Root directory.
        detector: Face detector instance.
        feature_extractor: Feature extractor (if None, use raw landmarks).

    Returns:
        Tuple of (features, labels).
    """
    data_dir = Path(data_dir)
    all_features = []
    all_labels = []

    label_map = {}
    current_label = 0

    for person_dir in sorted(data_dir.iterdir()):
        if not person_dir.is_dir():
            continue

        person_name = person_dir.name
        if person_name not in label_map:
            label_map[person_name] = current_label
            current_label += 1

        label = label_map[person_name]

        for img_path in person_dir.glob('*.jpg'):
            try:
                import cv2
                image = cv2.imread(str(img_path))
                if image is None:
                    continue

                faces = detector.detect(image)
                if not faces:
                    continue

                landmarks = faces[0]['landmarks']

                if feature_extractor:
                    features = feature_extractor.extract(landmarks)
                else:
                    features = landmarks.flatten()

                all_features.append(features)
                all_labels.append(label)

            except Exception as e:
                logger.error(f"Error processing {img_path}: {e}")
                continue

    if not all_features:
        raise ValueError("No features extracted from dataset")

    features = np.array(all_features)
    labels = np.array(all_labels)

    return features, labels, label_map