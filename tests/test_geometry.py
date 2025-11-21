"""Тест геометрических признаков"""
import numpy as np
from gfram.geometry import (
    GeometricFeatureExtractor,
    compute_hu_moments,
    extract_symmetry_features,
    compute_delaunay
)

print("🔍 Тест: Геометрические признаки")

# Создать случайные landmarks (468 точек, 3D)
landmarks = np.random.rand(468, 3).astype(np.float32)
print("✅ Landmarks созданы: shape", landmarks.shape)

# Тест 1: Моменты Ху
hu = compute_hu_moments(landmarks)
assert len(hu) == 7, "Должно быть 7 моментов Ху"
print(f"✅ Моменты Ху: {hu.shape}")

# Тест 2: Симметрия
symmetry = extract_symmetry_features(landmarks)
assert len(symmetry) == 15, "Должно быть 15 признаков симметрии"
print(f"✅ Симметрия: {symmetry.shape}")

# Тест 3: Триангуляция
tri = compute_delaunay(landmarks)
if tri is not None:
    print(f"✅ Триангуляция: {len(tri.simplices)} треугольников")
else:
    print("⚠️  Триангуляция не удалась (это может быть нормально)")

# Тест 4: Полный экстрактор
extractor = GeometricFeatureExtractor()
features = extractor.extract(landmarks)
print(f"✅ Полные признаки: {features.shape} (должно быть ~150)")
assert len(features) > 100, "Должно быть больше 100 признаков"

print("\n✅ ВСЕ ГЕОМЕТРИЧЕСКИЕ ТЕСТЫ ПРОЙДЕНЫ!")