#!/usr/bin/env python3
"""
GFRAM Geometric Features Example
================================

Shows the two representations GFRAM computes from a face:

1. hand-crafted geometric features (153 values: distances, curvatures,
   topology, statistics, symmetry, Delaunay graph, iris) - useful for analysis
   and research;
2. the learned geometric identity embedding (128 values) used by the hybrid
   recogniser together with the appearance model.

Usage:
    python geometric_features.py FACE.jpg     # real face
    python geometric_features.py              # synthetic landmarks (no image needed)

Requirements:
    pip install gfram
"""

import sys

import numpy as np


def load_landmarks():
    """Raw 478-point landmarks from an image, or a synthetic set without one."""
    if len(sys.argv) > 1:
        import cv2
        from gfram.detectors import FaceDetector

        image = cv2.imread(sys.argv[1])
        if image is None:
            sys.exit(f"❌ Could not load {sys.argv[1]}")
        faces = FaceDetector(refine_landmarks=True).detect(image)
        if not faces:
            sys.exit(f"❌ No face detected in {sys.argv[1]}")
        print(f"\n📷 {sys.argv[1]}: {faces[0]['num_landmarks']} landmarks detected")
        return faces[0]['landmarks']

    print("\n🎲 No image given: using synthetic landmarks (values are meaningless)")
    rng = np.random.default_rng(42)
    return (rng.normal(size=(478, 3)) * 20 + 200).astype(np.float32)


def main():
    print("=" * 60)
    print("🔷 GFRAM Geometric Features Example")
    print("=" * 60)

    from gfram.detectors import LandmarkNormalizer
    from gfram.geometry.features import GeometricFeatureExtractor
    from gfram.cloud.model_loader import ensure_model_available
    from gfram.models.package import FaceEmbedder

    landmarks = load_landmarks()
    print(f"   Shape: {landmarks.shape}")

    # 1. Hand-crafted geometric features
    extractor = GeometricFeatureExtractor(num_landmarks=478)
    features = extractor.extract(LandmarkNormalizer().normalize(landmarks))
    names = extractor.get_feature_names()

    print(f"\n📐 Hand-crafted features: {extractor.get_feature_count()}")
    print(f"   Range [{features.min():.4f}, {features.max():.4f}], mean {features.mean():.4f}")
    print(f"\n📋 First 10:")
    for i, name in enumerate(names[:10]):
        print(f"   {i + 1:2d}. {name}: {features[i]:.4f}")
    print(f"\n👁️ Iris features:")
    for name, value in zip(names[-3:], features[-3:]):
        print(f"   {name}: {value:.4f}")

    # 2. Learned identity embedding (what recognition compares)
    embedder = FaceEmbedder.from_file(ensure_model_available())
    embedding = embedder.embed(landmarks)
    print(f"\n🤖 Geometric identity embedding (model {embedder.version}): {embedding.shape[0]} values, "
          f"L2 norm {np.linalg.norm(embedding):.3f}")
    print(f"   First 8: {np.array2string(embedding[:8], precision=3)}")

    # The embedding ignores head pose: rotate and rescale the face, compare
    angle = np.deg2rad(20)
    rot = np.array([[np.cos(angle), -np.sin(angle), 0], [np.sin(angle), np.cos(angle), 0], [0, 0, 1]])
    moved = (landmarks - landmarks.mean(0)) @ rot.T * 1.5 + landmarks.mean(0)
    print(f"   Similarity to the same face rotated 20° and scaled 1.5×: "
          f"{float(embedding @ embedder.embed(moved.astype(np.float32))):.4f}")

    print("\n" + "=" * 60)
    print("✅ Done")
    print("=" * 60)


if __name__ == '__main__':
    main()
