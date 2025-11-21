"""Тест utils модуля"""
import numpy as np
import tempfile
import os
from gfram.utils import Config, save_landmarks, load_landmarks

print("🔍 Тест: Utils модуль")

# Тест 1: Config
config = Config()
print("✅ Config создан")

# Проверка настроек
batch_size = config.get('training.batch_size')
assert batch_size == 32, "Дефолтный batch_size должен быть 32"
print(f"✅ Config.get(): batch_size = {batch_size}")

# Изменение настроек
config.set('training.batch_size', 64)
assert config.get('training.batch_size') == 64
print("✅ Config.set() работает")

# Тест 2: Сохранение/загрузка landmarks
landmarks = np.random.rand(468, 3).astype(np.float32)

with tempfile.TemporaryDirectory() as tmpdir:
    path = os.path.join(tmpdir, 'test_landmarks.npy')

    # Сохранение
    save_landmarks(landmarks, path)
    assert os.path.exists(path), "Файл landmarks должен существовать"
    print("✅ save_landmarks() работает")

    # Загрузка
    loaded = load_landmarks(path)
    assert loaded is not None, "Landmarks должны загрузиться"
    assert np.allclose(landmarks, loaded), "Landmarks должны совпадать"
    print("✅ load_landmarks() работает")

print("\n✅ ВСЕ UTILS ТЕСТЫ ПРОЙДЕНЫ!")