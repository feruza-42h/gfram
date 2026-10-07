#!/usr/bin/env python3
"""
Train the quality-aware adaptive fusion (QAF) shipped with gfram 3.2.

Pipeline matches the library exactly: MediaPipe landmarks -> geometry embedding,
landmark-aligned crop -> MobileFaceNet embedding, plus quality/pose cues.

Data (cached by robustness.py and cplfw_benchmark.py):
  - LFW, 8 capture conditions (clean gallery vs degraded probe), 10 folds
  - CPLFW cross-pose pairs, 10 folds
Only pairs where both faces have landmarks are used (the library needs them).

Reports cross-dataset generalisation (train on one dataset, test on the other),
then fits the final model on everything and writes data/models/fusion_mbf.json.

Usage:
    python scripts/hybrid/train_fusion.py
"""

import csv
import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lfw'))
from common import ROOT, DATA as LFW_DATA, load_landmarks, load_pairs, roc_auc, tar_at_far  # noqa: E402

CPLFW = ROOT / 'data' / 'cplfw'
OUT = ROOT / 'data' / 'models' / 'fusion_mbf.json'

FEATURES = ['s_app', 's_geo', 's_app*s_geo', 'norm_min', 'norm_max', 'sharp_min',
            'yaw_diff', 'yaw_max', 's_app*yaw_diff', 's_geo*yaw_diff']


def pair_features(A, B):
    """A, B: dicts of per-face arrays aligned pairwise (geo, app, app_norm, sharp, pose)."""
    s_app = (A['app'] * B['app']).sum(1)
    s_geo = (A['geo'] * B['geo']).sum(1)
    yaw_diff = np.abs(A['pose'][:, 0] - B['pose'][:, 0])
    cols = {
        's_app': s_app, 's_geo': s_geo, 's_app*s_geo': s_app * s_geo,
        'norm_min': np.minimum(A['app_norm'], B['app_norm']),
        'norm_max': np.maximum(A['app_norm'], B['app_norm']),
        'sharp_min': np.minimum(A['sharp'], B['sharp']),
        'yaw_diff': yaw_diff, 'yaw_max': np.maximum(np.abs(A['pose'][:, 0]), np.abs(B['pose'][:, 0])),
        's_app*yaw_diff': s_app * yaw_diff, 's_geo*yaw_diff': s_geo * yaw_diff,
    }
    return np.column_stack([cols[f] for f in FEATURES]).astype(np.float64)


def take(F, idx):
    return {k: F[k][idx] for k in ('geo', 'app', 'app_norm', 'sharp', 'pose')}


def lfw_sets():
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    raw = [(f, a, b, s) for f, a, b, s in load_pairs() if a in index and b in index]
    gallery, probes = sorted({a for _, a, _, _ in raw}), sorted({b for _, _, b, _ in raw})
    gpos, ppos = {p: i for i, p in enumerate(gallery)}, {p: i for i, p in enumerate(probes)}
    ia = np.array([gpos[a] for _, a, _, _ in raw])
    ib = np.array([ppos[b] for _, _, b, _ in raw])
    labels = np.array([s for *_, s in raw])
    folds = np.array([f for f, *_ in raw])
    A = dict(np.load(LFW_DATA / 'robustness' / 'gallery_clean.npz'))
    sets = {}
    for f in sorted((LFW_DATA / 'robustness').glob('*.npz')):
        if f.stem == 'gallery_clean':
            continue
        B = dict(np.load(f))
        ok = A['found'][ia] & B['found'][ib]
        sets[f'lfw/{f.stem}'] = (pair_features(take(A, ia[ok]), take(B, ib[ok])), labels[ok], folds[ok])
    return sets


def cplfw_set():
    rows = list(csv.DictReader(open(CPLFW / 'pairs.csv')))
    paths = sorted({p for r in rows for p in (r['image_a_path'], r['image_b_path'])})
    pos = {p: i for i, p in enumerate(paths)}
    ia = np.array([pos[r['image_a_path']] for r in rows])
    ib = np.array([pos[r['image_b_path']] for r in rows])
    labels = np.array([r['is_same'] == '1' for r in rows])
    folds = np.array([int(r['fold_id']) - 1 for r in rows])
    F = dict(np.load(CPLFW / 'features.npz'))
    F['app'], F['app_norm'] = F['w600k_mbf'], F['w600k_mbf_norm']
    ok = F['found'][ia] & F['found'][ib]
    return {'cplfw': (pair_features(take(F, ia[ok]), take(F, ib[ok])), labels[ok], folds[ok])}


# ---------------------------------------------------------------- model

def fit(X, y):
    from sklearn.linear_model import LogisticRegression
    mean, scale = X.mean(0), X.std(0) + 1e-9
    lr = LogisticRegression(C=1.0, max_iter=5000).fit((X - mean) / scale, y)
    return {'mean': mean, 'scale': scale, 'coef': lr.coef_[0], 'intercept': float(lr.intercept_[0])}


def predict(m, X):
    """Probability that the pair shows the same person (pure numpy, as in the library)."""
    z = ((X - m['mean']) / m['scale']) @ m['coef'] + m['intercept']
    return 1 / (1 + np.exp(-z))


def best_threshold(p, y):
    cands = np.unique(np.quantile(p, np.linspace(0, 1, 2000)))
    return float(cands[np.argmax([((p >= t) == y).mean() for t in cands])])


def far_threshold(p, y, far):
    neg = np.sort(p[~y])[::-1]
    return float(neg[max(int(far * len(neg)), 1) - 1])


def acc(p, y, t):
    return float(((p >= t) == y).mean())


# ---------------------------------------------------------------- main

def main():
    lfw, cpl = lfw_sets(), cplfw_set()
    print('pairs used:', {k: len(v[1]) for k, v in {**lfw, **cpl}.items()})

    def stack(sets):
        return (np.vstack([s[0] for s in sets.values()]), np.concatenate([s[1] for s in sets.values()]))

    report = {}
    # 1) Cross-dataset: a model fitted on one dataset, threshold included, applied unchanged to the other
    for train_name, train_sets, test_sets in (('LFW', lfw, cpl), ('CPLFW', cpl, lfw)):
        Xtr, ytr = stack(train_sets)
        m = fit(Xtr, ytr)
        t_fused = best_threshold(predict(m, Xtr), ytr)
        t_app = best_threshold(Xtr[:, 0], ytr)
        print(f'\ntrained on {train_name} -> tested on the other dataset (thresholds from training data)')
        for name, (X, y, _) in test_sets.items():
            r = {'appearance_acc': acc(X[:, 0], y, t_app), 'qaf_acc': acc(predict(m, X), y, t_fused),
                 'appearance_auc': float(roc_auc(X[:, 0], y)), 'qaf_auc': float(roc_auc(predict(m, X), y))}
            report[f'{train_name}->{name}'] = r
            print(f'  {name:18s} appearance {r["appearance_acc"]:.4f} (auc {r["appearance_auc"]:.4f})   '
                  f'QAF {r["qaf_acc"]:.4f} (auc {r["qaf_auc"]:.4f})')

    # 2) Final model on everything
    X, y = stack({**lfw, **cpl})
    m = fit(X, y)
    p = predict(m, X)
    Xc, yc, _ = lfw['lfw/clean']
    thresholds = {'best_accuracy': best_threshold(p, y),
                  'far_1e-2': far_threshold(predict(m, Xc), yc, 1e-2),
                  'far_1e-3': far_threshold(predict(m, Xc), yc, 1e-3)}
    print('\nfinal thresholds:', {k: round(v, 4) for k, v in thresholds.items()})
    print('final weights:', {f: round(float(c), 3) for f, c in zip(FEATURES, m['coef'])})

    OUT.write_text(json.dumps({
        'type': 'logistic', 'appearance_model': 'w600k_mbf', 'features': FEATURES,
        'mean': m['mean'].tolist(), 'scale': m['scale'].tolist(), 'coef': m['coef'].tolist(),
        'intercept': m['intercept'], 'thresholds': thresholds,
        'trained_on': {k: int(len(v[1])) for k, v in {**lfw, **cpl}.items()},
        'cross_dataset': report,
    }, indent=2))
    print(f'-> {OUT}')


if __name__ == '__main__':
    main()
