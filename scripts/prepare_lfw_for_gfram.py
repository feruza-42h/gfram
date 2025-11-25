"""
Обработка LFW dataset для GFRAM
Извлекает landmarks, features И создаёт training splits
"""

from pathlib import Path
import numpy as np
import cv2
from tqdm import tqdm
import json
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from gfram import FaceDetector, GeometricFeatureExtractor, LandmarkNormalizer


def process_lfw(
    lfw_dir='/Users/nnnaimov/datasets/lfw',
    output_dir='/Users/nnnaimov/datasets/lfw_processed'
):
    """Обработать LFW dataset"""
    lfw_path = Path(lfw_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector()
    extractor = GeometricFeatureExtractor()
    normalizer = LandmarkNormalizer()

    processed = 0
    failed = 0

    person_dirs = sorted([d for d in lfw_path.iterdir() if d.is_dir()])

    print(f"Processing {len(person_dirs)} persons...")

    for person_dir in tqdm(person_dirs, desc="Processing persons"):
        person_name = person_dir.name
        person_output = output_path / person_name
        person_output.mkdir(exist_ok=True)

        for image_path in person_dir.glob('*.jpg'):
            try:
                # Load
                image = cv2.imread(str(image_path))
                if image is None:
                    failed += 1
                    continue

                # Detect
                faces = detector.detect(image)
                if not faces:
                    failed += 1
                    continue

                # Extract
                landmarks = faces[0]['landmarks']

                # Normalize
                try:
                    landmarks_normalized = normalizer.normalize(landmarks)
                except:
                    landmarks_normalized = landmarks

                # Extract features
                geo_features = extractor.extract(landmarks_normalized)

                # Save
                output_file = person_output / f'{image_path.stem}.npz'
                np.savez(
                    output_file,
                    landmarks=landmarks_normalized,
                    geo_features=geo_features,
                    original_landmarks=landmarks,
                    num_landmarks=len(landmarks),
                    image_path=str(image_path)
                )

                processed += 1

            except Exception as e:
                logger.error(f"Error: {image_path}: {e}")
                failed += 1

    # Metadata
    metadata = {
        'dataset': 'LFW',
        'processed': processed,
        'failed': failed,
        'persons': len(person_dirs)
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info(f"\n✅ Processed: {processed}")
    logger.info(f"❌ Failed: {failed}")
    logger.info(f"📊 Output: {output_path}")

    return processed, failed


def create_training_splits(
    processed_dir='/Users/nnnaimov/datasets/lfw_processed',
    output_dir='/Users/nnnaimov/datasets/lfw_training',
    train_ratio=0.8
):
    """
    Создать train/val splits из обработанного dataset

    Args:
        processed_dir: Директория с обработанными данными
        output_dir: Куда сохранить training splits
        train_ratio: Процент для training (0.8 = 80%)
    """
    processed_path = Path(processed_dir)
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    logger.info("Creating training splits...")

    # Собрать все данные
    all_landmarks = []
    all_geo_features = []
    all_labels = []

    label_map = {}
    current_label = 0

    person_dirs = sorted([d for d in processed_path.iterdir() if d.is_dir()])

    logger.info(f"Loading data from {len(person_dirs)} persons...")

    for person_dir in tqdm(person_dirs, desc="Loading data"):
        person_name = person_dir.name

        if person_name not in label_map:
            label_map[person_name] = current_label
            current_label += 1

        label = label_map[person_name]

        for npz_file in person_dir.glob('*.npz'):
            try:
                data = np.load(npz_file)
                all_landmarks.append(data['landmarks'])
                all_geo_features.append(data['geo_features'])
                all_labels.append(label)
            except Exception as e:
                logger.error(f"Error loading {npz_file}: {e}")
                continue

    if not all_landmarks:
        logger.error("No data loaded!")
        return

    logger.info(f"Loaded {len(all_landmarks)} samples from {len(label_map)} persons")

    # Конвертировать в numpy arrays
    landmarks = np.array(all_landmarks, dtype=np.float32)
    geo_features = np.array(all_geo_features, dtype=np.float32)
    labels = np.array(all_labels, dtype=np.int64)

    logger.info(f"Landmarks shape: {landmarks.shape}")
    logger.info(f"Geo features shape: {geo_features.shape}")
    logger.info(f"Labels shape: {labels.shape}")

    # Shuffle
    indices = np.random.permutation(len(landmarks))
    landmarks = landmarks[indices]
    geo_features = geo_features[indices]
    labels = labels[indices]

    # Split
    split_idx = int(len(landmarks) * train_ratio)

    train_landmarks = landmarks[:split_idx]
    train_geo_features = geo_features[:split_idx]
    train_labels = labels[:split_idx]

    val_landmarks = landmarks[split_idx:]
    val_geo_features = geo_features[split_idx:]
    val_labels = labels[split_idx:]

    # Сохранить
    logger.info("Saving training split...")
    np.savez(
        output_path / 'train.npz',
        landmarks=train_landmarks,
        geo_features=train_geo_features,
        labels=train_labels
    )

    logger.info("Saving validation split...")
    np.savez(
        output_path / 'val.npz',
        landmarks=val_landmarks,
        geo_features=val_geo_features,
        labels=val_labels
    )

    # Сохранить label map
    with open(output_path / 'label_map.json', 'w') as f:
        json.dump(label_map, f, indent=2)

    # Metadata
    metadata = {
        'total_samples': len(landmarks),
        'train_samples': len(train_landmarks),
        'val_samples': len(val_landmarks),
        'num_persons': len(label_map),
        'landmarks_shape': list(landmarks.shape),
        'geo_features_shape': list(geo_features.shape),
        'train_ratio': train_ratio
    }

    with open(output_path / 'metadata.json', 'w') as f:
        json.dump(metadata, f, indent=2)

    logger.info("\n" + "="*60)
    logger.info("TRAINING SPLITS CREATED!")
    logger.info("="*60)
    logger.info(f"✅ Train samples: {len(train_landmarks)}")
    logger.info(f"✅ Val samples: {len(val_landmarks)}")
    logger.info(f"📊 Persons: {len(label_map)}")
    logger.info(f"📁 Output: {output_path}")

    return metadata


if __name__ == '__main__':
    import argparse

    parser = argparse.ArgumentParser(description='Process LFW dataset for GFRAM')
    parser.add_argument('--lfw-dir', default='/Users/nnnaimov/datasets/lfw',
                        help='LFW dataset directory')
    parser.add_argument('--processed-dir', default='/Users/nnnaimov/datasets/lfw_processed',
                        help='Output directory for processed data')
    parser.add_argument('--training-dir', default='/Users/nnnaimov/datasets/lfw_training',
                        help='Output directory for training splits')
    parser.add_argument('--skip-processing', action='store_true',
                        help='Skip processing, only create splits')
    parser.add_argument('--train-ratio', type=float, default=0.8,
                        help='Training split ratio (default: 0.8)')

    args = parser.parse_args()

    # Шаг 1: Обработка LFW (если уже не обработан)
    if not args.skip_processing:
        logger.info("="*60)
        logger.info("STEP 1: Processing LFW dataset")
        logger.info("="*60)
        processed, failed = process_lfw(args.lfw_dir, args.processed_dir)

        if processed == 0:
            logger.error("No samples processed! Exiting.")
            exit(1)
    else:
        logger.info("Skipping processing (--skip-processing flag)")

    # Шаг 2: Создание training splits
    logger.info("\n" + "="*60)
    logger.info("STEP 2: Creating training splits")
    logger.info("="*60)

    try:
        metadata = create_training_splits(
            args.processed_dir,
            args.training_dir,
            args.train_ratio
        )

        logger.info("\n" + "="*60)
        logger.info("✅ ALL DONE!")
        logger.info("="*60)
        logger.info(f"📁 Processed data: {args.processed_dir}")
        logger.info(f"📁 Training data: {args.training_dir}")
        logger.info("\n🚀 Next steps:")
        logger.info("   1. python scripts/train_base_model.py")
        logger.info("   2. Test model")
        logger.info("   3. Deploy server")
        logger.info("   4. Publish library!")

    except Exception as e:
        logger.error(f"Error creating training splits: {e}")
        logger.error("Make sure processed data exists!")
        exit(1)