#!/usr/bin/env python3
"""
Step 3: untrained baselines on all 10 LFW folds.

  current_pipeline  - LandmarkNormalizer + 153 hand-crafted features + cosine (what gfram ships today)
  features_zscore   - same features, standardised per dimension before cosine
  procrustes_raw    - Procrustes-aligned landmarks flattened + cosine

Usage:
    nice -n 19 python scripts/lfw/baseline.py
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import sys
import logging
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import ROOT, RUNS, load_landmarks, load_pairs, fit_reference, align, verification_report  # noqa: E402

sys.path.insert(0, str(ROOT))
from gfram.detectors import LandmarkNormalizer  # noqa: E402
from gfram.geometry.features import GeometricFeatureExtractor  # noqa: E402

logging.disable(logging.WARNING)


def cosine_scores(vectors, path_index, pairs):
    v = vectors / (np.linalg.norm(vectors, axis=1, keepdims=True) + 1e-12)
    return np.array([v[path_index[a]] @ v[path_index[b]] for a, b in pairs])


def main():
    landmarks, paths, _ = load_landmarks()
    path_index = {p: i for i, p in enumerate(paths)}
    pairs = [(a, b, s) for _, a, b, s in load_pairs() if a in path_index and b in path_index]
    ab = [(a, b) for a, b, _ in pairs]
    labels = np.array([s for _, _, s in pairs])
    print(f'{len(pairs)}/6000 pairs have landmarks for both images')

    normalizer, extractor = LandmarkNormalizer(), GeometricFeatureExtractor()
    feats = np.stack([extractor.extract(normalizer.normalize(lm)) for lm in landmarks]).astype(np.float64)
    feats = np.nan_to_num(feats)

    z = (feats - feats.mean(0)) / (feats.std(0) + 1e-8)
    aligned = align(landmarks, fit_reference(landmarks))
    flat = aligned.reshape(len(aligned), -1)
    flat = flat - flat.mean(0)

    results = {
        'current_pipeline': verification_report(cosine_scores(feats, path_index, ab), labels),
        'features_zscore': verification_report(cosine_scores(z, path_index, ab), labels),
        'procrustes_raw': verification_report(cosine_scores(flat, path_index, ab), labels),
    }
    for k, v in results.items():
        print(f'{k:18s} ' + '  '.join(f'{m}={x:.4f}' if isinstance(x, float) else f'{m}={x}' for m, x in v.items()))

    RUNS.mkdir(exist_ok=True)
    (RUNS / 'baselines.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
