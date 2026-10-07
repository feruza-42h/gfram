#!/usr/bin/env python3
"""
Is the geometry contribution on CPLFW statistically significant?

Compares QAF without geometry (appearance + quality cues) against full QAF
(+ landmark pose + geometric similarity) on identical pairs:
  - McNemar exact test on per-pair correctness,
  - 95% bootstrap confidence intervals (resampling pairs) for the difference in
    accuracy and in TAR@FAR=1%.

Usage:
    python scripts/hybrid/significance.py --crop centre
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'lfw'))
import cplfw_benchmark as cb  # noqa: E402
from robustness import threshold_acc  # noqa: E402
from common import RUNS, tar_at_far  # noqa: E402

QUALITY = ['norm_min', 'norm_max', 'sharp_min']
POSE = ['yaw_diff', 'yaw_max', 's_app*yaw_diff']
GEOMETRY = ['s_geo', 's_app*s_geo', 's_geo*yaw_diff', 'geo_ok']


def cv_predictions(X, labels, folds, cols):
    """Out-of-fold scores and per-pair correctness (threshold from training folds)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    M = np.column_stack([X[c] for c in cols])
    scores, correct = np.zeros(len(labels)), np.zeros(len(labels), bool)
    for k in range(10):
        tr, te = folds != k, folds == k
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(M[tr], labels[tr])
        s_tr, s_te = model.decision_function(M[tr]), model.decision_function(M[te])
        correct[te] = threshold_acc(s_tr, labels[tr], s_te, labels[te])
        scores[te] = s_te
    return scores, correct


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', default='centre', choices=['landmarks', 'centre', 'insightface'])
    ap.add_argument('--boot', type=int, default=2000)
    args = ap.parse_args()

    paths, ia, ib, labels, folds = cb.load_pairs()
    F = cb.features(paths, ['w600k_mbf', 'w600k_r50'], args.crop)
    rng = np.random.default_rng(0)
    out = {}
    for m in [m for m in ('w600k_mbf', 'w600k_r50') if m in F]:
        X = cb.pair_matrix(F, m, ia, ib)
        s0, c0 = cv_predictions(X, labels, folds, ['s_app'] + QUALITY)
        s1, c1 = cv_predictions(X, labels, folds, ['s_app'] + QUALITY + POSE + GEOMETRY)

        only_full, only_base = int((c1 & ~c0).sum()), int((c0 & ~c1).sum())
        p = binomtest(only_full, only_full + only_base, 0.5).pvalue if only_full + only_base else 1.0

        d_acc, d_tar = [], []
        n = len(labels)
        for _ in range(args.boot):
            idx = rng.integers(0, n, n)
            d_acc.append(c1[idx].mean() - c0[idx].mean())
            d_tar.append(tar_at_far(s1[idx], labels[idx], 1e-2) - tar_at_far(s0[idx], labels[idx], 1e-2))
        r = {
            'acc_without_geometry': float(c0.mean()), 'acc_with_geometry': float(c1.mean()),
            'acc_gain_ci95': [float(np.percentile(d_acc, 2.5)), float(np.percentile(d_acc, 97.5))],
            'tar1_without_geometry': float(tar_at_far(s0, labels, 1e-2)),
            'tar1_with_geometry': float(tar_at_far(s1, labels, 1e-2)),
            'tar1_gain_ci95': [float(np.percentile(d_tar, 2.5)), float(np.percentile(d_tar, 97.5))],
            'mcnemar': {'fixed_by_geometry': only_full, 'broken_by_geometry': only_base, 'p_value': float(p)},
        }
        out[m] = r
        print(f'{m} [{args.crop} crops]')
        print(f'  accuracy {r["acc_without_geometry"]:.4f} -> {r["acc_with_geometry"]:.4f}  '
              f'gain 95% CI [{r["acc_gain_ci95"][0]:+.4f}, {r["acc_gain_ci95"][1]:+.4f}]')
        print(f'  TAR@1%   {r["tar1_without_geometry"]:.4f} -> {r["tar1_with_geometry"]:.4f}  '
              f'gain 95% CI [{r["tar1_gain_ci95"][0]:+.4f}, {r["tar1_gain_ci95"][1]:+.4f}]')
        print(f'  McNemar: geometry fixed {only_full} pairs, broke {only_base}, p = {p:.2g}', flush=True)
    (RUNS / f'cplfw_significance_{args.crop}.json').write_text(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
