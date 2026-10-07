#!/usr/bin/env python3
"""
Strongest appearance baseline for CPLFW: the stock InsightFace pipeline
(SCRFD detector + 5-point alignment + ArcFace R50, buffalo_l), as used in
published CPLFW results. Geometry and pose cues still come from GFRAM's
MediaPipe landmarks (data/cplfw/features.npz).

Writes data/cplfw/features_insightface.npz in the schema used by
cplfw_benchmark.py / significance.py (--crop insightface).

Usage:
    nice -n 19 python scripts/hybrid/cplfw_insightface.py
"""

import os

os.environ.setdefault('OMP_NUM_THREADS', '2')

import sys
import time
from pathlib import Path

import certifi
import cv2
import numpy as np

# insightface downloads its model pack with urllib; python.org builds need certifi
os.environ.setdefault('SSL_CERT_FILE', certifi.where())

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import cplfw_benchmark as cb  # noqa: E402

MODEL_ROOT = cb.ROOT / 'data' / 'models' / 'insightface'


def main():
    from insightface.app import FaceAnalysis

    paths, *_ = cb.load_pairs()
    base = dict(np.load(cb.DATA / 'features.npz'))  # MediaPipe geometry/pose from the main run

    app = FaceAnalysis(name='buffalo_l', root=str(MODEL_ROOT), allowed_modules=['detection', 'recognition'],
                       providers=['CPUExecutionProvider'])
    app.prepare(ctx_id=-1, det_size=(320, 320))
    rec = app.models['recognition']

    n = len(paths)
    emb = np.zeros((n, 512), np.float32)
    norm = np.zeros(n, np.float32)
    det = np.zeros(n, np.float32)
    t = time.time()
    for i, p in enumerate(paths):
        img = cv2.imread(str(cb.DATA / p))
        faces = app.get(img)
        if faces:
            h, w = img.shape[:2]
            f = min(faces, key=lambda f: ((f.bbox[0] + f.bbox[2]) / 2 - w / 2) ** 2 + ((f.bbox[1] + f.bbox[3]) / 2 - h / 2) ** 2)
            raw, det[i] = f.embedding, f.det_score
        else:
            # no detection: recognise the dataset's centre crop directly
            raw = rec.get_feat(cv2.resize(img, (112, 112)))[0]
        norm[i] = np.linalg.norm(raw)
        emb[i] = raw / norm[i]
        if (i + 1) % 1000 == 0:
            print(f'  {i + 1}/{n} ({time.time() - t:.0f}s)', flush=True)
    print(f'InsightFace detected {np.mean(det > 0):.1%} of {n} images ({time.time() - t:.0f}s)', flush=True)

    out = {k: base[k] for k in ('found', 'geo', 'pose', 'sharp')}
    out['w600k_r50'], out['w600k_r50_norm'] = emb, norm
    np.savez_compressed(cb.DATA / 'features_insightface.npz', **out)


if __name__ == '__main__':
    main()
