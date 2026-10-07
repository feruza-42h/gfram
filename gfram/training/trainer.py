"""
Training utilities for GFRAM models.
"""

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Optional, Dict, Callable
from tqdm import tqdm
import logging

logger = logging.getLogger(__name__)


class Trainer:
    """Trainer for GFRAM models."""

    def __init__(
            self,
            model: nn.Module,
            optimizer: torch.optim.Optimizer,
            criterion: nn.Module,
            device: str = 'cpu',
            scheduler: Optional[torch.optim.lr_scheduler._LRScheduler] = None
    ):
        self.model = model.to(device)
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device
        self.scheduler = scheduler

        self.train_losses = []
        self.val_losses = []

    def train_epoch(self, train_loader: DataLoader) -> float:
        """Train for one epoch."""
        self.model.train()
        total_loss = 0.0

        pbar = tqdm(train_loader, desc="Training")
        for batch in pbar:
            landmarks, labels = batch
            landmarks = landmarks.to(self.device)
            labels = labels.to(self.device)

            self.optimizer.zero_grad()

            logits, embeddings = self.model(landmarks)

            if isinstance(self.criterion, nn.CrossEntropyLoss):
                loss = self.criterion(logits, labels)
            else:
                loss, _ = self.criterion(embeddings, labels)

            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
            pbar.set_postfix({'loss': loss.item()})

        avg_loss = total_loss / len(train_loader)
        return avg_loss

    def validate(self, val_loader: DataLoader) -> float:
        """Validate model."""
        self.model.eval()
        total_loss = 0.0

        with torch.no_grad():
            for batch in val_loader:
                landmarks, labels = batch
                landmarks = landmarks.to(self.device)
                labels = labels.to(self.device)

                logits, embeddings = self.model(landmarks)

                if isinstance(self.criterion, nn.CrossEntropyLoss):
                    loss = self.criterion(logits, labels)
                else:
                    loss, _ = self.criterion(embeddings, labels)

                total_loss += loss.item()

        avg_loss = total_loss / len(val_loader)
        return avg_loss

    def train(
            self,
            train_loader: DataLoader,
            val_loader: Optional[DataLoader] = None,
            epochs: int = 100,
            save_path: Optional[str] = None
    ):
        """Train model for multiple epochs."""
        best_val_loss = float('inf')

        for epoch in range(epochs):
            train_loss = self.train_epoch(train_loader)
            self.train_losses.append(train_loss)

            logger.info(f"Epoch {epoch + 1}/{epochs} - Train Loss: {train_loss:.4f}")

            if val_loader:
                val_loss = self.validate(val_loader)
                self.val_losses.append(val_loss)
                logger.info(f"  Val Loss: {val_loss:.4f}")

                if val_loss < best_val_loss and save_path:
                    best_val_loss = val_loss
                    torch.save(self.model.state_dict(), save_path)
                    logger.info(f"  Saved best model to {save_path}")

            if self.scheduler:
                self.scheduler.step()

        logger.info("Training completed!")