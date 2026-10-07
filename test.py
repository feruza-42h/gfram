import ssl
ssl._create_default_https_context = ssl._create_unverified_context
import os
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import torch
from pathlib import Path

# Загрузить checkpoint
model_path = Path.home() / '.gfram' / 'cache' / 'gfram_model.pth'
checkpoint = torch.load(model_path, map_location='cpu')

# Создать модель
from gfram.models import create_geometric_transformer
model = create_geometric_transformer(config_name='base', num_classes=None)

# Попробовать загрузить с strict=True чтобы увидеть ошибки
try:
    model.load_state_dict(checkpoint['model_state_dict'], strict=True)
    print("✅ All weights loaded correctly!")
except Exception as e:
    print(f"❌ Error: {e}")

# Посмотреть какие ключи не совпадают
model_keys = set(model.state_dict().keys())
checkpoint_keys = set(checkpoint['model_state_dict'].keys())

missing = model_keys - checkpoint_keys
unexpected = checkpoint_keys - model_keys

print(f"\nMissing in checkpoint: {len(missing)}")
for k in list(missing)[:5]:
    print(f"  - {k}")

print(f"\nUnexpected in checkpoint: {len(unexpected)}")
for k in list(unexpected)[:5]:
    print(f"  - {k}")