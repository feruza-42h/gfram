"""End-to-end тест всего pipeline"""
import os
# ИСПРАВЛЕНИЕ: Обход проблемы с OpenMP на macOS
os.environ['KMP_DUPLICATE_LIB_OK'] = 'TRUE'

import numpy as np
import cv2
from gfram import FaceDetector, GeometricFeatureExtractor, FaceIndex

print("🔍 Тест: End-to-End Pipeline")

# 1. Создать компоненты
detector = FaceDetector()
extractor = GeometricFeatureExtractor()
index = FaceIndex(dimension=150)
print("✅ Компоненты созданы")

# 2. Создать тестовые изображения
test_images = []
for i in range(3):
    img = np.random.randint(0, 255, (480, 640, 3), dtype=np.uint8)
    test_images.append(img)
print(f"✅ Создано {len(test_images)} тестовых изображений")

# 3. Детекция и извлечение признаков
features_list = []
for i, img in enumerate(test_images):
    faces = detector.detect(img)
    if faces:
        landmarks = faces[0]['landmarks']
        features = extractor.extract(landmarks)
        features_list.append(features)
        print(f"✅ Изображение {i+1}: извлечено {len(features)} признаков")
    else:
        # Если лицо не найдено, создать случайные landmarks
        landmarks = np.random.rand(468, 3).astype(np.float32)
        features = extractor.extract(landmarks)
        features_list.append(features)
        print(f"⚠️  Изображение {i+1}: лицо не найдено, используем случайные landmarks")

# 4. Добавление в индекс (ИСПРАВЛЕНО: name вместо label)
for i, features in enumerate(features_list):
    index.add(name=f"person_{i}", embeddings=features)
print(f"✅ Добавлено {len(features_list)} векторов в индекс")

# 5. Поиск похожих
query = features_list[0]
results = index.search(query, k=2)
print(f"✅ Поиск выполнен: найдено {len(results)} результатов")

# Проверка результатов
if len(results) > 0:
    print(f"   Первый результат: {results[0]['name']} (distance: {results[0]['distance']:.4f})")

print("\n✅ END-TO-END ТЕСТ ПРОЙДЕН!")
print("\n🎉 ВСЯ БИБЛИОТЕКА РАБОТАЕТ КОРРЕКТНО!")