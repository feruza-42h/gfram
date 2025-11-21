"""Тест training модуля"""
import torch
import numpy as np
from gfram.training import (
    FaceLandmarkDataset,
    LandmarkAugmentation,
    Trainer
)
from gfram.models import create_geometric_transformer, TripletLoss

print("🔍 Тест: Training модуль")

# Создать случайные данные
landmarks = np.random.rand(10, 468, 3).astype(np.float32)
labels = np.array([0, 0, 1, 1, 2, 2, 3, 3, 4, 4])
print("✅ Данные созданы")

# Тест 1: Dataset
dataset = FaceLandmarkDataset(landmarks, labels)
assert len(dataset) == 10, "Датасет должен содержать 10 образцов"
print(f"✅ Dataset: {len(dataset)} образцов")

# Тест 2: Аугментация
augment = LandmarkAugmentation()
augmented = augment(torch.from_numpy(landmarks[0]))
assert augmented.shape == landmarks[0].shape
print(f"✅ Аугментация: {augmented.shape}")

# Тест 3: DataLoader
from torch.utils.data import DataLoader
loader = DataLoader(dataset, batch_size=2)
batch_lm, batch_labels = next(iter(loader))
assert batch_lm.shape[0] == 2, "Batch size должен быть 2"
print(f"✅ DataLoader: batch {batch_lm.shape}")

# Тест 4: Trainer (базовая проверка)
model = create_geometric_transformer('tiny', num_classes=5)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
criterion = TripletLoss()
trainer = Trainer(model, optimizer, criterion)
print("✅ Trainer создан")

print("\n✅ ВСЕ TRAINING ТЕСТЫ ПРОЙДЕНЫ!")