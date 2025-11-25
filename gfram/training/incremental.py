"""
Online Incremental Learning Module
PhD Thesis Innovation: Learning without Catastrophic Forgetting
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from typing import Dict, List, Optional
from collections import deque
import logging

logger = logging.getLogger(__name__)


class MemoryBank:
    """
    Memory Bank for storing representative samples.
    Prevents catastrophic forgetting during incremental updates.
    """

    def __init__(
            self,
            max_samples_per_class: int = 10,
            max_total_samples: int = 1000
    ):
        self.max_samples_per_class = max_samples_per_class
        self.max_total_samples = max_total_samples
        self.memory = {}  # {class_id: deque of samples}
        self.class_counts = {}

    def add(
            self,
            embedding: np.ndarray,
            label: int,
            landmarks: Optional[np.ndarray] = None,
            geo_features: Optional[np.ndarray] = None
    ):
        """Add sample to memory bank"""
        if label not in self.memory:
            self.memory[label] = deque(maxlen=self.max_samples_per_class)
            self.class_counts[label] = 0

        sample = {
            'embedding': embedding,
            'landmarks': landmarks,
            'geo_features': geo_features
        }

        self.memory[label].append(sample)
        self.class_counts[label] += 1

        # Check total size
        total = sum(len(samples) for samples in self.memory.values())
        if total > self.max_total_samples:
            self._evict_oldest()

    def _evict_oldest(self):
        """Evict oldest sample from largest class"""
        largest_class = max(self.class_counts, key=self.class_counts.get)
        if self.memory[largest_class]:
            self.memory[largest_class].popleft()
            self.class_counts[largest_class] -= 1

    def sample(self, batch_size: int = 32) -> List[Dict]:
        """Sample batch from memory for replay"""
        all_samples = []
        for label, samples in self.memory.items():
            for sample in samples:
                all_samples.append({
                    'label': label,
                    **sample
                })

        if not all_samples:
            return []

        indices = np.random.choice(
            len(all_samples),
            size=min(batch_size, len(all_samples)),
            replace=False
        )

        return [all_samples[i] for i in indices]


class IncrementalTrainer:
    """
    Incremental Trainer for Online Learning.

    PhD Innovation:
    - Updates model as new persons are added
    - Uses memory replay to prevent forgetting
    - Fast: < 100ms per update
    """

    def __init__(
            self,
            model: nn.Module,
            memory_bank: MemoryBank,
            device: torch.device,
            learning_rate: float = 0.0001,
            num_update_steps: int = 5
    ):
        self.model = model
        self.memory_bank = memory_bank
        self.device = device
        self.num_update_steps = num_update_steps

        # Optimizer with small LR for stability
        self.optimizer = optim.Adam(
            self.model.parameters(),
            lr=learning_rate,
            weight_decay=1e-4
        )

        # Loss functions
        self.triplet_loss = nn.TripletMarginLoss(margin=1.0)
        self.mse_loss = nn.MSELoss()

        # Keep old model for distillation
        self.old_model = None

    def incremental_update(
            self,
            embedding: np.ndarray,
            label: int,
            landmarks: Optional[np.ndarray] = None,
            geo_features: Optional[np.ndarray] = None
    ) -> Dict:
        """
        Perform incremental model update.

        THIS IS THE PhD INNOVATION!
        """
        logger.info(f"Incremental update for person {label}")

        # 1. Add to memory bank
        self.memory_bank.add(embedding, label, landmarks, geo_features)

        # 2. Training mode
        self.model.train()

        # 3. Copy model for distillation
        if self.old_model is None:
            self.old_model = self._copy_model()

        # 4. Gradient updates
        total_loss = 0.0
        losses = []

        for step in range(self.num_update_steps):
            # Sample replay batch
            replay_samples = self.memory_bank.sample(batch_size=16)

            if len(replay_samples) < 2:
                logger.warning("Not enough samples for training")
                break

            # Prepare batch
            batch = self._prepare_batch(replay_samples)

            # Compute loss
            loss = self._compute_loss(batch, label, embedding)

            # Backward
            self.optimizer.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), max_norm=1.0)
            self.optimizer.step()

            total_loss += loss.item()
            losses.append(loss.item())

        # 5. Update old model
        self.old_model = self._copy_model()

        # 6. Eval mode
        self.model.eval()

        avg_loss = total_loss / max(1, self.num_update_steps)

        logger.info(f"Update complete. Avg loss: {avg_loss:.4f}")

        return {
            'loss': avg_loss,
            'num_steps': self.num_update_steps,
            'losses': losses,
            'memory_size': sum(len(s) for s in self.memory_bank.memory.values())
        }

    def _prepare_batch(self, samples: List[Dict]) -> Dict:
        """Prepare batch tensors"""
        batch = {
            'embeddings': [],
            'labels': [],
            'landmarks': [],
            'geo_features': []
        }

        for sample in samples:
            batch['embeddings'].append(sample['embedding'])
            batch['labels'].append(sample['label'])

            if sample.get('landmarks') is not None:
                batch['landmarks'].append(sample['landmarks'])

            if sample.get('geo_features') is not None:
                batch['geo_features'].append(sample['geo_features'])

        # To tensors
        batch['embeddings'] = torch.FloatTensor(np.array(batch['embeddings'])).to(self.device)
        batch['labels'] = torch.LongTensor(batch['labels']).to(self.device)

        if batch['landmarks']:
            batch['landmarks'] = torch.FloatTensor(np.array(batch['landmarks'])).to(self.device)
        else:
            batch['landmarks'] = None

        if batch['geo_features']:
            batch['geo_features'] = torch.FloatTensor(np.array(batch['geo_features'])).to(self.device)
        else:
            batch['geo_features'] = None

        return batch

    def _compute_loss(
            self,
            batch: Dict,
            new_label: int,
            new_embedding: np.ndarray
    ) -> torch.Tensor:
        """Compute combined loss"""
        total_loss = 0.0

        # Forward pass on replay samples
        if batch['landmarks'] is not None and batch['geo_features'] is not None:
            embeddings = self.model(batch['landmarks'], batch['geo_features'])
        else:
            embeddings = batch['embeddings']

        # 1. Contrastive Loss
        labels = batch['labels']
        unique_labels = torch.unique(labels)

        if len(unique_labels) > 1:
            contrastive_loss = self._contrastive_loss(embeddings, labels)
            total_loss += contrastive_loss

        # 2. Distillation Loss (prevent forgetting)
        if self.old_model is not None and batch['landmarks'] is not None:
            with torch.no_grad():
                old_embeddings = self.old_model(batch['landmarks'], batch['geo_features'])

            distillation_loss = self.mse_loss(embeddings, old_embeddings)
            total_loss += 0.5 * distillation_loss

        # 3. L2 regularization
        l2_reg = 0.0
        for param in self.model.parameters():
            l2_reg += torch.norm(param)

        total_loss += 1e-5 * l2_reg

        return total_loss

    def _contrastive_loss(
            self,
            embeddings: torch.Tensor,
            labels: torch.Tensor
    ) -> torch.Tensor:
        """Contrastive loss: same person close, different far"""
        # Normalize
        embeddings = nn.functional.normalize(embeddings, p=2, dim=1)

        # Pairwise similarities
        similarities = torch.matmul(embeddings, embeddings.t())

        # Mask for positive pairs
        labels = labels.unsqueeze(1)
        mask = (labels == labels.t()).float()

        # Positive loss
        pos_loss = (1 - similarities) * mask
        pos_loss = pos_loss.sum() / (mask.sum() + 1e-8)

        # Negative loss
        neg_mask = 1 - mask - torch.eye(len(labels), device=self.device)
        neg_loss = torch.relu(similarities - 0.5) * neg_mask
        neg_loss = neg_loss.sum() / (neg_mask.sum() + 1e-8)

        return pos_loss + neg_loss

    def _copy_model(self) -> nn.Module:
        """Create copy of model"""
        import copy
        model_copy = copy.deepcopy(self.model)
        model_copy.eval()
        for param in model_copy.parameters():
            param.requires_grad = False
        return model_copy