#!/usr/bin/env python3
"""
CPLFW (Cross-Pose LFW) benchmark: does 3D geometry help when head pose differs?

CPLFW pairs deliberately mix frontal and profile views, the setting where
appearance models lose the most accuracy. 3D landmarks are Procrustes-aligned in
3D, so geometry is, in principle, pose-invariant.

If MediaPipe finds no face (extreme profiles), the appearance branch falls back
to the pre-aligned centre crop and the geometry branch is marked unavailable.

Usage:
    nice -n 19 python scripts/hybrid/cplfw_benchmark.py
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import csv
import json
import logging
import sys
import time
from pathlib import Path

import cv2
import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'lfw'))
from appearance import ArcFaceEmbedder, align_crop  # noqa: E402
from robustness import head_pose, central_face, threshold_acc  # noqa: E402
from common import ROOT, RUNS, roc_auc, tar_at_far  # noqa: E402

sys.path.insert(0, str(ROOT))
from gfram.detectors import FaceDetector  # noqa: E402
from gfram.models.package import FaceEmbedder  # noqa: E402

logging.disable(logging.WARNING)
DATA = ROOT / 'data' / 'cplfw'


def load_pairs():
    rows = list(csv.DictReader(open(DATA / 'pairs.csv')))
    paths = sorted({p for r in rows for p in (r['image_a_path'], r['image_b_path'])})
    pos = {p: i for i, p in enumerate(paths)}
    ia = np.array([pos[r['image_a_path']] for r in rows])
    ib = np.array([pos[r['image_b_path']] for r in rows])
    labels = np.array([r['is_same'] == '1' for r in rows])
    folds = np.array([int(r['fold_id']) - 1 for r in rows])
    return paths, ia, ib, labels, folds


def features(paths, app_models, crop_mode='landmarks'):
    """
    crop_mode='landmarks': appearance crops aligned with MediaPipe landmarks (fallback: centre crop).
    crop_mode='centre': appearance always uses the dataset's own pre-aligned centre crop, i.e. the
    strongest standard baseline; landmarks are still used for geometry and pose.
    """
    cache = DATA / ('features.npz' if crop_mode == 'landmarks' else f'features_{crop_mode}.npz')
    if cache.exists():
        return dict(np.load(cache))

    detector = FaceDetector(max_num_faces=3, refine_landmarks=True)
    geo = FaceEmbedder.from_file(ROOT / 'gfram' / 'pretrained' / 'gfram_model.pth')
    n = len(paths)
    found = np.zeros(n, bool)
    lms = np.zeros((n, 478, 3), np.float32)
    crops = np.zeros((n, 112, 112, 3), np.uint8)
    t = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(str(DATA / p))
        face = central_face(detector.detect(img), img.shape[1], img.shape[0])
        if face is not None:
            found[i] = True
            lms[i] = face['landmarks']
        if face is not None and crop_mode == 'landmarks':
            crops[i] = align_crop(img, face['landmarks'])
        else:
            crops[i] = cv2.resize(img, (112, 112), interpolation=cv2.INTER_AREA)
    print(f'landmarks found on {found.mean():.1%} of {n} images ({time.time() - t:.0f}s)', flush=True)

    out = {'found': found, 'geo': np.zeros((n, geo.dim), np.float32), 'pose': np.zeros((n, 3), np.float32)}
    out['geo'][found] = geo.embed(lms[found])
    out['pose'][found] = np.stack([head_pose(lm, geo.reference) for lm in lms[found]])
    gray = [cv2.cvtColor(c, cv2.COLOR_BGR2GRAY) for c in crops]
    out['sharp'] = np.array([np.log1p(cv2.Laplacian(g, cv2.CV_32F).var()) for g in gray], np.float32)
    for m in app_models:
        t = time.time()
        raw = ArcFaceEmbedder(m).embed_crops(crops, normalize=False)
        out[f'{m}_norm'] = np.linalg.norm(raw, axis=1)
        out[m] = raw / out[f'{m}_norm'][:, None]
        print(f'{m}: {time.time() - t:.0f}s', flush=True)
    np.savez_compressed(cache, **out)
    return out


def pair_matrix(F, m, ia, ib):
    ok = F['found'][ia] & F['found'][ib]
    s_app = (F[m][ia] * F[m][ib]).sum(1)
    s_geo = np.where(ok, (F['geo'][ia] * F['geo'][ib]).sum(1), 0.0)
    na, nb = F[f'{m}_norm'][ia], F[f'{m}_norm'][ib]
    yaw_a, yaw_b = np.abs(F['pose'][ia, 0]), np.abs(F['pose'][ib, 0])
    cols = {
        's_app': s_app, 's_geo': s_geo, 's_app*s_geo': s_app * s_geo,
        'norm_min': np.minimum(na, nb), 'norm_max': np.maximum(na, nb),
        'sharp_min': np.minimum(F['sharp'][ia], F['sharp'][ib]),
        'yaw_diff': np.abs(F['pose'][ia, 0] - F['pose'][ib, 0]), 'yaw_max': np.maximum(yaw_a, yaw_b),
        's_geo*yaw_diff': s_geo * np.abs(F['pose'][ia, 0] - F['pose'][ib, 0]),
        's_app*yaw_diff': s_app * np.abs(F['pose'][ia, 0] - F['pose'][ib, 0]),
        'geo_ok': ok.astype(float),
    }
    return {k: v.astype(np.float32) for k, v in cols.items()}


def cv_eval(X, labels, folds, cols):
    from sklearn.linear_model import LogisticRegression
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    M = np.column_stack([X[c] for c in cols])
    correct, scores = [], np.zeros(len(labels))
    for k in range(10):
        tr, te = folds != k, folds == k
        if cols == ['s_app'] or cols == ['s_geo']:
            s_tr, s_te = M[tr, 0], M[te, 0]
        else:
            model = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000)).fit(M[tr], labels[tr])
            s_tr, s_te = model.decision_function(M[tr]), model.decision_function(M[te])
        correct.append(threshold_acc(s_tr, labels[tr], s_te, labels[te]))
        scores[te] = s_te
    return {'accuracy': float(np.concatenate(correct).mean()), 'auc': float(roc_auc(scores, labels)),
            'tar@far=1%': float(tar_at_far(scores, labels, 1e-2))}


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--crop', default='landmarks', choices=['landmarks', 'centre', 'insightface'])
    args = ap.parse_args()

    paths, ia, ib, labels, folds = load_pairs()
    models = ['w600k_mbf', 'w600k_r50']
    F = features(paths, models, args.crop)
    print(f'{len(labels)} pairs, geometry available for '
          f'{(F["found"][ia] & F["found"][ib]).mean():.1%} of pairs', flush=True)

    quality = ['norm_min', 'norm_max', 'sharp_min']
    pose = ['yaw_diff', 'yaw_max', 's_app*yaw_diff']
    geometry = ['s_geo', 's_app*s_geo', 's_geo*yaw_diff', 'geo_ok']
    variants = {
        'geometry only': ['s_geo'],
        'appearance only': ['s_app'],
        '+ quality (QAF w/o geometry)': ['s_app'] + quality,
        '+ quality + pose': ['s_app'] + quality + pose,
        '+ quality + geometry': ['s_app'] + quality + geometry,
        '+ quality + pose + geometry (full)': ['s_app'] + quality + pose + geometry,
    }
    results = {}
    models = [m for m in models if m in F]  # the insightface baseline only provides R50
    for m in models:
        X = pair_matrix(F, m, ia, ib)
        results[m] = {}
        print(f'\n{m}')
        for name, cols in variants.items():
            r = cv_eval(X, labels, folds, cols)
            results[m][name] = r
            print(f'  {name:38s} acc={r["accuracy"]:.4f}  auc={r["auc"]:.4f}  tar@1%={r["tar@far=1%"]:.4f}', flush=True)

    (RUNS / f'cplfw_{args.crop}.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
