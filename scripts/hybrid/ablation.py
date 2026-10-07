#!/usr/bin/env python3
"""
Which cues make quality-aware fusion work? Logistic QAF retrained with groups of
pair features removed (same leave-one-fold-out protocol, cached features only).

Usage:
    python scripts/hybrid/ablation.py
"""

import json
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'lfw'))
import robustness as rb  # noqa: E402
from common import RUNS, load_landmarks, load_pairs  # noqa: E402

# Columns produced by robustness.pair_inputs()
GROUPS = {
    'geometry identity (s_geo)': [1, 2],
    'embedding norms': [3, 4, 5],
    'image sharpness/brightness': [6, 7],
    'head-pose difference (from landmarks)': [8, 9, 10],
}


def load_data():
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    raw = [(f, a, b, s) for f, a, b, s in load_pairs() if a in index and b in index]
    gallery, probes = sorted({a for _, a, _, _ in raw}), sorted({b for _, _, b, _ in raw})
    gpos, ppos = {p: i for i, p in enumerate(gallery)}, {p: i for i, p in enumerate(probes)}
    ia = np.array([gpos[a] for _, a, _, _ in raw])
    ib = np.array([ppos[b] for _, _, b, _ in raw])
    A = dict(np.load(rb.CACHE / 'gallery_clean.npz'))
    conds = [f.stem for f in sorted(rb.CACHE.glob('*.npz')) if f.stem != 'gallery_clean']
    data = {}
    for c in conds:
        s_app, s_geo, q = rb.pair_inputs(A, dict(np.load(rb.CACHE / f'{c}.npz')), ia, ib)
        data[c] = q
    return data, np.array([f for f, *_ in raw]), np.array([s for *_, s in raw]), conds


def qaf_accuracy(data, folds, labels, conds, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    correct = {c: [] for c in conds}
    for k in range(10):
        tr, te = folds != k, folds == k
        Xtr = np.vstack([data[c][tr][:, cols] for c in conds])
        ytr = np.concatenate([labels[tr]] * len(conds))
        model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000)).fit(Xtr, ytr)
        s_tr = model.decision_function(Xtr)
        for c in conds:
            correct[c].append(rb.threshold_acc(s_tr, ytr, model.decision_function(data[c][te][:, cols]), labels[te]))
    acc = {c: float(np.concatenate(v).mean()) for c, v in correct.items()}
    return acc['clean'], float(np.mean([v for c, v in acc.items() if c != 'clean']))


def main():
    data, folds, labels, conds = load_data()
    all_cols = list(range(data[conds[0]].shape[1]))
    rows = {'full QAF': qaf_accuracy(data, folds, labels, conds, all_cols),
            'appearance score only': qaf_accuracy(data, folds, labels, conds, [0])}
    for name, cols in GROUPS.items():
        rows[f'without {name}'] = qaf_accuracy(data, folds, labels, conds, [i for i in all_cols if i not in cols])
    for name, cols in GROUPS.items():
        rows[f'appearance + {name} only'] = qaf_accuracy(data, folds, labels, conds, [0, 11] + cols)

    print(f'{"variant":48s} {"clean":>8s} {"degraded":>9s}')
    for name, (clean, degr) in rows.items():
        print(f'{name:48s} {clean:8.4f} {degr:9.4f}')
    (RUNS / 'qaf_ablation.json').write_text(json.dumps(rows, indent=2))


if __name__ == '__main__':
    main()
