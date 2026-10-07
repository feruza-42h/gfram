#!/usr/bin/env python3
"""
Robustness study + quality-aware adaptive fusion (QAF) on LFW.

Protocol: in every LFW pair the first photo stays clean (enrolment) and the
second is degraded (probe camera). Landmarks are re-detected on the degraded
photo, so both branches see exactly what a deployed system would see.

Phase 1 (cached per condition): detection, geometry embedding, MobileFaceNet
embedding and quality cues for every probe.
Phase 2: verification accuracy per condition for
  - appearance only, geometry only,
  - fixed fusion (one weight for all conditions),
  - QAF: a small network that reads quality cues of the pair (embedding norms,
    sharpness, brightness, head-pose difference) and decides how much to trust
    each branch for that particular comparison.
All learned parts are fitted on 9 identity-disjoint folds and tested on the 10th.

Usage:
    nice -n 19 python scripts/hybrid/robustness.py
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

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
from degrade import conditions  # noqa: E402
from common import ROOT, DATA, RUNS, load_landmarks, load_pairs, roc_auc, tar_at_far  # noqa: E402

sys.path.insert(0, str(ROOT))
from gfram.detectors import FaceDetector  # noqa: E402
from gfram.models.package import FaceEmbedder  # noqa: E402

logging.disable(logging.WARNING)
CACHE = DATA / 'robustness'
GEOMETRY_COLUMNS = (1, 2)  # s_geo and s_app*s_geo in pair_inputs()
APP_MODEL = 'w600k_mbf'


# ------------------------------------------------------------------ phase 1

def head_pose(lm, ref):
    """Yaw, pitch, roll (degrees) of the rotation that maps the reference face onto lm."""
    x = lm - lm.mean(0)
    x /= np.linalg.norm(x)
    u, _, vt = np.linalg.svd(ref.T @ x)
    d = np.sign(np.linalg.det(u @ vt))
    r = u @ np.diag([1, 1, d]) @ vt
    yaw = np.degrees(np.arcsin(np.clip(-r[2, 0], -1, 1)))
    pitch = np.degrees(np.arctan2(r[2, 1], r[2, 2]))
    roll = np.degrees(np.arctan2(r[1, 0], r[0, 0]))
    return np.array([yaw, pitch, roll], np.float32)


def central_face(faces, w, h):
    faces = [f for f in faces if f.get('num_landmarks') == 478]
    if not faces:
        return None
    return min(faces, key=lambda f: (f['landmarks'][:, 0].mean() - w / 2) ** 2 + (f['landmarks'][:, 1].mean() - h / 2) ** 2)


def probe_features(name, degrade, paths, clean_lm, detector, geo, app):
    """Detect + embed degraded probes; cached in data/lfw/robustness/<name>.npz."""
    out = CACHE / f'{name}.npz'
    if out.exists():
        return dict(np.load(out))

    t = time.time()
    n = len(paths)
    found = np.zeros(n, bool)
    lms = np.zeros((n, 478, 3), np.float32)
    crops = np.zeros((n, 112, 112, 3), np.uint8)
    sharp, bright = np.zeros(n, np.float32), np.zeros(n, np.float32)
    for i, (p, lm0) in enumerate(zip(paths, clean_lm)):
        img = degrade(cv2.imread(str(DATA / 'lfw_funneled' / p)), lm0)
        face = central_face(detector.detect(img), img.shape[1], img.shape[0])
        if face is None:
            continue
        found[i] = True
        lms[i] = face['landmarks']
        crops[i] = align_crop(img, face['landmarks'])
        gray = cv2.cvtColor(crops[i], cv2.COLOR_BGR2GRAY)
        sharp[i] = np.log1p(cv2.Laplacian(gray, cv2.CV_32F).var())
        bright[i] = gray.mean() / 255

    geo_emb = np.zeros((n, geo.dim), np.float32)
    app_emb = np.zeros((n, 512), np.float32)
    app_norm = np.zeros(n, np.float32)
    pose = np.zeros((n, 3), np.float32)
    if found.any():
        geo_emb[found] = geo.embed(lms[found])
        raw = app.embed_crops(crops[found], normalize=False)
        app_norm[found] = np.linalg.norm(raw, axis=1)
        app_emb[found] = raw / app_norm[found, None]
        pose[found] = np.stack([head_pose(lm, geo.reference) for lm in lms[found]])

    feats = dict(paths=np.array(paths), found=found, geo=geo_emb, app=app_emb, app_norm=app_norm,
                 sharp=sharp, bright=bright, pose=pose)
    np.savez_compressed(out, **feats)
    print(f'  {name:12s} detected {found.mean():6.1%} of {n} probes ({time.time() - t:.0f}s)', flush=True)
    return feats


# ------------------------------------------------------------------ phase 2

def pair_inputs(A, B, ia, ib):
    """Per-pair similarity scores and quality cues (A = clean gallery, B = probe)."""
    ok = B['found'][ib] & A['found'][ia]
    s_app = np.where(ok, (A['app'][ia] * B['app'][ib]).sum(1), -1.0)
    s_geo = np.where(ok, (A['geo'][ia] * B['geo'][ib]).sum(1), -1.0)
    q = np.column_stack([
        s_app, s_geo, s_app * s_geo,
        A['app_norm'][ia], B['app_norm'][ib], np.minimum(A['app_norm'][ia], B['app_norm'][ib]),
        B['sharp'][ib], B['bright'][ib],
        np.abs(A['pose'][ia] - B['pose'][ib]),
        ok.astype(np.float32),
    ]).astype(np.float32)
    return s_app, s_geo, q


def threshold_acc(train_s, train_y, test_s, test_y):
    cands = np.quantile(train_s, np.linspace(0, 1, 500))
    t = cands[np.argmax([((train_s >= c) == train_y).mean() for c in cands])]
    return ((test_s >= t) == test_y)


def evaluate(data, folds, labels, conds):
    """Leave-one-fold-out over identity-disjoint folds; learned parts see 9 folds of all conditions."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.neural_network import MLPClassifier
    from sklearn.preprocessing import StandardScaler
    from sklearn.pipeline import make_pipeline

    methods = ['appearance', 'appearance_cond_thr', 'geometry', 'fixed_fusion',
               'qaf_no_geometry', 'qaf_logistic', 'qaf_mlp']
    no_geo = [i for i in range(data[conds[0]]['q'].shape[1]) if i not in GEOMETRY_COLUMNS]
    correct = {m: {c: [] for c in conds} for m in methods}
    scores = {m: {c: np.zeros(len(labels)) for c in conds} for m in methods}

    for k in range(10):
        tr, te = folds != k, folds == k
        Xtr = np.vstack([data[c]['q'][tr] for c in conds])
        ytr = np.concatenate([labels[tr]] * len(conds))
        sa_tr = np.concatenate([data[c]['s_app'][tr] for c in conds])
        sg_tr = np.concatenate([data[c]['s_geo'][tr] for c in conds])

        # one global weight for all conditions, chosen on training folds
        best_w = max(np.linspace(0, 1, 11),
                     key=lambda w: threshold_acc(sa_tr + w * sg_tr, ytr, sa_tr + w * sg_tr, ytr).mean())
        logit = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr, ytr)
        logit_ng = make_pipeline(StandardScaler(), LogisticRegression(C=1.0, max_iter=2000)).fit(Xtr[:, no_geo], ytr)
        mlp = make_pipeline(StandardScaler(), MLPClassifier(hidden_layer_sizes=(32, 16), alpha=1e-3,
                                                            max_iter=400, early_stopping=True,
                                                            random_state=0)).fit(Xtr, ytr)
        train_scores = {
            'appearance': sa_tr, 'geometry': sg_tr, 'fixed_fusion': sa_tr + best_w * sg_tr,
            'qaf_logistic': logit.decision_function(Xtr), 'qaf_mlp': mlp.predict_proba(Xtr)[:, 1],
            'qaf_no_geometry': logit_ng.decision_function(Xtr[:, no_geo]),
        }
        for c in conds:
            d = data[c]
            test_scores = {
                'appearance': d['s_app'][te], 'geometry': d['s_geo'][te],
                'fixed_fusion': d['s_app'][te] + best_w * d['s_geo'][te],
                'qaf_logistic': logit.decision_function(d['q'][te]), 'qaf_mlp': mlp.predict_proba(d['q'][te])[:, 1],
                'qaf_no_geometry': logit_ng.decision_function(d['q'][te][:, no_geo]),
            }
            for m in methods:
                if m == 'appearance_cond_thr':
                    # oracle-style baseline: threshold tuned on training folds of this very condition
                    correct[m][c].append(threshold_acc(d['s_app'][tr], labels[tr], d['s_app'][te], labels[te]))
                    scores[m][c][te] = d['s_app'][te]
                    continue
                # threshold for every other method is chosen on the mixed-condition training folds
                correct[m][c].append(threshold_acc(train_scores[m], ytr, test_scores[m], labels[te]))
                scores[m][c][te] = test_scores[m]

    results = {}
    for m in methods:
        results[m] = {}
        for c in conds:
            acc = float(np.concatenate(correct[m][c]).mean())
            results[m][c] = {'accuracy': acc, 'auc': float(roc_auc(scores[m][c], labels)),
                             'tar@far=1%': float(tar_at_far(scores[m][c], labels, 1e-2))}
    return results


def main():
    CACHE.mkdir(parents=True, exist_ok=True)
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    raw = [(f, a, b, s) for f, a, b, s in load_pairs() if a in index and b in index]
    folds = np.array([f for f, *_ in raw])
    labels = np.array([s for *_, s in raw])

    gallery = sorted({a for _, a, _, _ in raw})
    probes = sorted({b for _, _, b, _ in raw})
    gpos, ppos = {p: i for i, p in enumerate(gallery)}, {p: i for i, p in enumerate(probes)}
    ia = np.array([gpos[a] for _, a, _, _ in raw])
    ib = np.array([ppos[b] for _, _, b, _ in raw])
    print(f'{len(raw)} pairs | {len(gallery)} gallery photos (clean) | {len(probes)} probe photos (degraded)', flush=True)

    detector = FaceDetector(max_num_faces=3, refine_landmarks=True)
    geo = FaceEmbedder.from_file(ROOT / 'gfram' / 'pretrained' / 'gfram_model.pth')
    app = ArcFaceEmbedder(APP_MODEL)
    rng = np.random.default_rng(0)
    conds = conditions(rng)

    A = probe_features('gallery_clean', conds['clean'], gallery, lm[[index[p] for p in gallery]], detector, geo, app)
    data = {}
    for name, fn in conds.items():
        B = probe_features(name, fn, probes, lm[[index[p] for p in probes]], detector, geo, app)
        s_app, s_geo, q = pair_inputs(A, B, ia, ib)
        data[name] = {'s_app': s_app, 's_geo': s_geo, 'q': q, 'detected': float(B['found'].mean())}

    results = evaluate(data, folds, labels, list(conds))
    print(f'\n{"condition":12s} {"detect":>7s} ' + ' '.join(f'{m[:13]:>13s}' for m in results), flush=True)
    for c in conds:
        print(f'{c:12s} {data[c]["detected"]:7.1%} ' +
              ' '.join(f'{results[m][c]["accuracy"]:13.4f}' for m in results), flush=True)
    mean = {m: np.mean([results[m][c]['accuracy'] for c in conds if c != 'clean']) for m in results}
    print(f'{"mean(degr.)":12s} {"":7s} ' + ' '.join(f'{mean[m]:13.4f}' for m in results), flush=True)

    RUNS.mkdir(exist_ok=True)
    (RUNS / 'robustness.json').write_text(json.dumps(
        {'results': results, 'detected': {c: data[c]['detected'] for c in conds},
         'appearance_model': APP_MODEL}, indent=2))


if __name__ == '__main__':
    main()
