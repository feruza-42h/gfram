#!/usr/bin/env python3
"""
LFW benchmark: geometry vs appearance vs score fusion.

Appearance embeddings are cached in data/lfw/appearance_<model>.npz, so the
heavy part runs once. Fusion weights are chosen on 9 folds and tested on the
10th (standard LFW protocol), never on the pairs being scored.

Usage:
    nice -n 19 python scripts/hybrid/lfw_benchmark.py
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'lfw'))
from appearance import ArcFaceEmbedder, align_crop  # noqa: E402
from common import ROOT, DATA, RUNS, load_landmarks, load_pairs, roc_auc, tar_at_far  # noqa: E402

sys.path.insert(0, str(ROOT))
from gfram.models.package import FaceEmbedder  # noqa: E402


def appearance_embeddings(model_name, paths, landmarks):
    cache = DATA / f'appearance_{model_name}.npz'
    if cache.exists():
        d = np.load(cache)
        if list(d['paths']) == list(paths):
            return d['emb']
    crops = np.stack([align_crop(cv2.imread(str(DATA / 'lfw_funneled' / p)), lm)
                      for p, lm in zip(paths, landmarks)])
    t = time.time()
    emb = ArcFaceEmbedder(model_name).embed_crops(crops)
    print(f'  {model_name}: {len(crops)} faces in {time.time() - t:.0f}s '
          f'({1000 * (time.time() - t) / len(crops):.1f} ms/face)', flush=True)
    np.savez_compressed(cache, paths=np.array(paths), emb=emb)
    return emb


def fold_accuracy(scores, labels, folds):
    """Standard LFW: threshold picked on 9 folds, accuracy measured on the 10th."""
    correct = 0
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        cands = np.unique(scores[train])
        t = cands[np.argmax([((scores[train] >= c) == labels[train]).mean() for c in cands])]
        correct += ((scores[test] >= t) == labels[test]).sum()
    return correct / len(scores)


def fused_fold_accuracy(a, g, labels, folds, weights=np.linspace(0, 1, 21)):
    """Score fusion a + w*g with w and threshold both chosen on the 9 training folds."""
    correct, chosen = 0, []
    for k in np.unique(folds):
        train, test = folds != k, folds == k
        best = (-1, None, None)
        for w in weights:
            s = a[train] + w * g[train]
            cands = np.quantile(s, np.linspace(0, 1, 400))
            accs = [((s >= c) == labels[train]).mean() for c in cands]
            i = int(np.argmax(accs))
            if accs[i] > best[0]:
                best = (accs[i], w, cands[i])
        _, w, t = best
        chosen.append(w)
        correct += ((a[test] + w * g[test] >= t) == labels[test]).sum()
    return correct / len(labels), float(np.mean(chosen))


def report(name, scores, labels, folds):
    r = {'accuracy': float(fold_accuracy(scores, labels, folds)), 'auc': float(roc_auc(scores, labels)),
         'tar@far=1%': float(tar_at_far(scores, labels, 1e-2)),
         'tar@far=0.1%': float(tar_at_far(scores, labels, 1e-3))}
    print(f'{name:28s} acc={r["accuracy"]:.4f}  auc={r["auc"]:.4f}  '
          f'tar@1%={r["tar@far=1%"]:.4f}  tar@0.1%={r["tar@far=0.1%"]:.4f}', flush=True)
    return r


def main():
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    raw = [(f, a, b, s) for f, a, b, s in load_pairs() if a in index and b in index]
    needed = sorted({index[p] for _, a, b, _ in raw for p in (a, b)})
    pos = {k: j for j, k in enumerate(needed)}
    ia = np.array([pos[index[a]] for _, a, _, _ in raw])
    ib = np.array([pos[index[b]] for _, _, b, _ in raw])
    labels = np.array([s for *_, s in raw])
    folds = np.array([f for f, *_ in raw])
    sub_paths, sub_lm = [str(paths[i]) for i in needed], lm[needed]
    print(f'{len(raw)} LFW pairs, {len(needed)} faces', flush=True)

    geo = FaceEmbedder.from_file(ROOT / 'gfram' / 'pretrained' / 'gfram_model.pth').embed(sub_lm)
    emb = {'geometry': geo}
    for m in ('w600k_mbf', 'w600k_r50'):
        emb[m] = appearance_embeddings(m, sub_paths, sub_lm)

    score = {k: (e[ia] * e[ib]).sum(1) for k, e in emb.items()}
    results = {k: report(k, s, labels, folds) for k, s in score.items()}
    for m in ('w600k_mbf', 'w600k_r50'):
        acc, w = fused_fold_accuracy(score[m], score['geometry'], labels, folds)
        results[f'{m}+geometry'] = {'accuracy': float(acc), 'geometry_weight': w}
        print(f'{m + " + geometry":28s} acc={acc:.4f}  (mean geometry weight {w:.2f})', flush=True)

    RUNS.mkdir(exist_ok=True)
    (RUNS / 'hybrid_lfw.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
