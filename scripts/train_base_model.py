"""
Обучение модели GFRAM на Windows с CUDA

ОПТИМИЗИРОВАНО ДЛЯ RTX 4060:
- Большой batch size (128)
- Gradient clipping
- Mixed precision (опционально)
- Без NaN проблем
"""

import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
import numpy as np
from pathlib import Path
import json
import logging
from tqdm import tqdm
import time

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gfram.models import create_geometric_transformer
from gfram.models.losses import CombinedLoss


class LFWDataset(TensorDataset):
    """Dataset для LFW"""

    def __init__(self, landmarks, geo_features, labels):
        super().__init__(landmarks, geo_features, labels)
        self.landmarks = landmarks
        self.geo_features = geo_features
        self.labels = labels

    def __len__(self):
        return len(self.landmarks)

    def __getitem__(self, idx):
        return self.landmarks[idx], self.geo_features[idx], self.labels[idx]


def load_dataset(dataset_dir):
    """Загрузить dataset"""
    dataset_path = Path(dataset_dir)

    logger.info(f"Loading dataset from {dataset_path}")

    train_data = np.load(dataset_path / 'train.npz')
    train_landmarks = torch.FloatTensor(train_data['landmarks'])
    train_geo_features = torch.FloatTensor(train_data['geo_features'])
    train_labels = torch.LongTensor(train_data['labels'])

    val_data = np.load(dataset_path / 'val.npz')
    val_landmarks = torch.FloatTensor(val_data['landmarks'])
    val_geo_features = torch.FloatTensor(val_data['geo_features'])
    val_labels = torch.LongTensor(val_data['labels'])

    train_dataset = LFWDataset(train_landmarks, train_geo_features, train_labels)
    val_dataset = LFWDataset(val_landmarks, val_geo_features, val_labels)

    logger.info(f"✅ Train: {len(train_dataset)} samples")
    logger.info(f"✅ Val: {len(val_dataset)} samples")
    logger.info(f"📊 Num persons: {len(torch.unique(train_labels))}")

    return train_dataset, val_dataset


def train_epoch(model, dataloader, criterion, optimizer, device, epoch, use_amp=False):
    """Train one epoch"""
    model.train()
    total_loss = 0

    # Mixed precision scaler
    scaler = torch.cuda.amp.GradScaler() if use_amp else None

    pbar = tqdm(dataloader, desc=f'Epoch {epoch}')

    for landmarks, geo_features, labels in pbar:
        landmarks = landmarks.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        if use_amp:
            # Mixed precision forward
            with torch.cuda.amp.autocast():
                embeddings = model(landmarks)
                loss, loss_dict = criterion(embeddings, labels)

            # Backward
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
        else:
            # Normal forward
            embeddings = model(landmarks)
            loss, loss_dict = criterion(embeddings, labels)

            # Backward
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()

        # Check NaN
        if torch.isnan(loss):
            logger.warning("NaN loss detected, skipping batch")
            continue

        total_loss += loss.item()

        pbar.set_postfix({
            'loss': f'{loss.item():.4f}',
            'avg': f'{total_loss / (pbar.n + 1):.4f}'
        })

    return total_loss / len(dataloader)


def validate(model, dataloader, criterion, device):
    """Validate"""
    model.eval()

    total_loss = 0
    all_embeddings = []
    all_labels = []

    with torch.no_grad():
        for landmarks, geo_features, labels in tqdm(dataloader, desc='Validating'):
            landmarks = landmarks.to(device)
            labels = labels.to(device)

            embeddings = model(landmarks)
            loss, loss_dict = criterion(embeddings, labels)

            if not torch.isnan(loss):
                total_loss += loss.item()

            all_embeddings.append(embeddings.cpu())
            all_labels.append(labels.cpu())

    avg_loss = total_loss / len(dataloader)

    # Accuracy
    all_embeddings = torch.cat(all_embeddings)
    all_labels = torch.cat(all_labels)

    all_embeddings = torch.nn.functional.normalize(all_embeddings, p=2, dim=1)
    similarity = torch.mm(all_embeddings, all_embeddings.t())

    correct = 0
    for i in range(len(all_embeddings)):
        sim_scores = similarity[i].clone()
        sim_scores[i] = -1
        most_similar_idx = torch.argmax(sim_scores)

        if all_labels[i] == all_labels[most_similar_idx]:
            correct += 1

    accuracy = correct / len(all_embeddings)

    return avg_loss, accuracy


def train_model(
    dataset_dir,
    output_path='models/gfram_base_v1.0.0.pth',
    config_name='base',
    num_epochs=50,
    batch_size=128,
    learning_rate=0.001,
    device='cuda',
    use_amp=False
):
    """
    Обучить модель на CUDA

    ОПТИМИЗИРОВАНО ДЛЯ RTX 4060:
    - Batch size 128 (быстрее)
    - Gradient clipping (против NaN)
    - Mixed precision (опционально, быстрее в 2 раза)
    """

    # Device
    if device == 'auto':
        device = 'cuda' if torch.cuda.is_available() else 'cpu'

    device = torch.device(device)
    logger.info(f"Using device: {device}")

    if device.type == 'cuda':
        logger.info(f"GPU: {torch.cuda.get_device_name(0)}")
        logger.info(f"Memory: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.1f} GB")

    # Load dataset
    train_dataset, val_dataset = load_dataset(dataset_dir)

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4 if device.type == 'cuda' else 0,
        pin_memory=True if device.type == 'cuda' else False
    )

    num_persons = len(torch.unique(train_dataset.labels))
    logger.info(f"Number of persons: {num_persons}")

    # Create model
    logger.info(f"Creating model: {config_name}")
    model = create_geometric_transformer(
        config_name=config_name,
        num_landmarks=478,
        num_classes=None
    )
    model = model.to(device)

    total_params = sum(p.numel() for p in model.parameters())
    logger.info(f"Total parameters: {total_params:,}")

    # Loss
    criterion = CombinedLoss(
        embedding_dim=model.output_dim,
        num_classes=num_persons,
        use_arcface=True,
        use_triplet=True,
        use_center=True
    )
    criterion = criterion.to(device)
    logger.info("Using CombinedLoss (ArcFace + Triplet + Center)")

    # Optimizer
    optimizer = optim.Adam(
        list(model.parameters()) + list(criterion.parameters()),
        lr=learning_rate
    )

    # Scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        factor=0.5,
        patience=3
    )

    # Training
    best_accuracy = 0.0
    best_epoch = 0
    training_history = []

    logger.info("\n" + "="*60)
    logger.info("STARTING TRAINING")
    logger.info("="*60)
    logger.info(f"Epochs: {num_epochs}")
    logger.info(f"Batch size: {batch_size}")
    logger.info(f"Learning rate: {learning_rate}")
    logger.info(f"Device: {device}")
    logger.info(f"Mixed precision: {use_amp}")
    logger.info("="*60 + "\n")

    start_time = time.time()

    for epoch in range(1, num_epochs + 1):
        epoch_start = time.time()

        # Train
        train_loss = train_epoch(
            model, train_loader, criterion, optimizer, device, epoch, use_amp
        )

        # Validate
        val_loss, val_accuracy = validate(
            model, val_loader, criterion, device
        )

        scheduler.step(val_loss)

        epoch_time = time.time() - epoch_start

        logger.info(
            f"Epoch {epoch}/{num_epochs} - "
            f"Train: {train_loss:.4f} - "
            f"Val: {val_loss:.4f} - "
            f"Acc: {val_accuracy:.4f} - "
            f"Time: {epoch_time:.1f}s"
        )

        training_history.append({
            'epoch': epoch,
            'train_loss': float(train_loss),
            'val_loss': float(val_loss),
            'val_accuracy': float(val_accuracy),
            'time': float(epoch_time)
        })

        # Save best
        if val_accuracy > best_accuracy:
            best_accuracy = val_accuracy
            best_epoch = epoch

            output_path_obj = Path(output_path)
            output_path_obj.parent.mkdir(parents=True, exist_ok=True)

            checkpoint = {
                'epoch': epoch,
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'criterion_state_dict': criterion.state_dict(),
                'best_accuracy': best_accuracy,
                'config_name': config_name,
                'num_landmarks': 478,
                'num_persons': num_persons,
                'training_history': training_history,
                'version': '1.0.0'
            }

            torch.save(checkpoint, output_path)
            logger.info(f"✅ Saved (acc: {best_accuracy:.4f})")

    total_time = time.time() - start_time

    logger.info("\n" + "="*60)
    logger.info("TRAINING COMPLETE!")
    logger.info("="*60)
    logger.info(f"Time: {total_time / 60:.1f} minutes")
    logger.info(f"Best epoch: {best_epoch}")
    logger.info(f"Best accuracy: {best_accuracy:.4f}")
    logger.info(f"Model: {output_path}")
    logger.info("="*60)

    # Save history
    history_path = Path(output_path).parent / 'training_history.json'
    with open(history_path, 'w') as f:
        json.dump({
            'config_name': config_name,
            'num_epochs': num_epochs,
            'best_epoch': best_epoch,
            'best_accuracy': float(best_accuracy),
            'total_time': float(total_time),
            'history': training_history
        }, f, indent=2)

    return model, best_accuracy


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Train GFRAM on Windows CUDA')
    parser.add_argument('--dataset', required=True,
                        help='Dataset path')
    parser.add_argument('--output', default='models/gfram_base_v1.0.0.pth',
                        help='Output path')
    parser.add_argument('--config', default='base',
                        choices=['tiny', 'small', 'base', 'large'],
                        help='Model config')
    parser.add_argument('--epochs', type=int, default=50,
                        help='Epochs')
    parser.add_argument('--batch-size', type=int, default=128,
                        help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                        help='Learning rate')
    parser.add_argument('--device', default='cuda',
                        choices=['cuda', 'cpu', 'auto'],
                        help='Device')
    parser.add_argument('--amp', action='store_true',
                        help='Use mixed precision (faster)')

    args = parser.parse_args()

    model, accuracy = train_model(
        dataset_dir=args.dataset,
        output_path=args.output,
        config_name=args.config,
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        device=args.device,
        use_amp=args.amp
    )

    logger.info(f"\n🎉 Done! Accuracy: {accuracy:.2%}")