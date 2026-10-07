#!/usr/bin/env python3
"""
Step 2: extract MediaPipe 478-point landmarks from LFW images.

Light on resources: processes images in chunks, writes each chunk to disk,
and resumes from the last finished chunk if restarted.

Usage:
    nice -n 19 python scripts/lfw/extract_landmarks.py [--chunk 500] [--pause 2]
"""

import os

# Keep CPU usage low: limit native thread pools before heavy imports
for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import sys
import time
import logging
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gfram.detectors import FaceDetector  # noqa: E402

DATA = ROOT / 'data' / 'lfw'
IMAGES = DATA / 'lfw_funneled'
OUT = DATA / 'landmarks'

logging.basicConfig(level=logging.WARNING)


def pick_central_face(faces, width, height):
    """LFW-funneled images are centred on the target face; pick the face closest to centre."""
    cx, cy = width / 2, height / 2

    def dist(face):
        lm = face['landmarks']
        return (lm[:, 0].mean() - cx) ** 2 + (lm[:, 1].mean() - cy) ** 2

    return min(faces, key=dist)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--chunk', type=int, default=500, help='images per saved chunk')
    parser.add_argument('--pause', type=float, default=2.0, help='seconds to rest between chunks')
    args = parser.parse_args()

    paths = sorted(IMAGES.glob('*/*.jpg'))
    if not paths:
        sys.exit(f'No images in {IMAGES}. Run the download step first.')
    OUT.mkdir(parents=True, exist_ok=True)

    detector = FaceDetector(max_num_faces=3, refine_landmarks=True)
    num_chunks = (len(paths) + args.chunk - 1) // args.chunk
    print(f'{len(paths)} images -> {num_chunks} chunks of {args.chunk}')

    for c in range(num_chunks):
        out_file = OUT / f'chunk_{c:04d}.npz'
        if out_file.exists():
            continue

        start = time.time()
        batch = paths[c * args.chunk:(c + 1) * args.chunk]
        landmarks, rel_paths, failed = [], [], []

        for p in batch:
            rel = str(p.relative_to(IMAGES))
            img = cv2.imread(str(p))
            faces = detector.detect(img) if img is not None else []
            faces = [f for f in faces if f.get('num_landmarks') == 478]
            if not faces:
                failed.append(rel)
                continue
            face = pick_central_face(faces, img.shape[1], img.shape[0])
            landmarks.append(face['landmarks'].astype(np.float32))
            rel_paths.append(rel)

        np.savez_compressed(
            out_file,
            landmarks=np.stack(landmarks) if landmarks else np.zeros((0, 478, 3), np.float32),
            paths=np.array(rel_paths),
            failed=np.array(failed),
        )
        print(f'chunk {c + 1}/{num_chunks}: ok={len(rel_paths)} failed={len(failed)} '
              f'({time.time() - start:.0f}s)', flush=True)
        time.sleep(args.pause)

    # Merge all chunks into one file
    parts = [np.load(f) for f in sorted(OUT.glob('chunk_*.npz'))]
    all_lm = np.concatenate([p['landmarks'] for p in parts])
    all_paths = np.concatenate([p['paths'] for p in parts])
    all_failed = np.concatenate([p['failed'] for p in parts])
    names = np.array([p.split('/')[0] for p in all_paths])
    np.savez_compressed(DATA / 'lfw_landmarks.npz', landmarks=all_lm, paths=all_paths, names=names)
    print(f'Done: {len(all_paths)} faces, {len(set(names))} identities, '
          f'{len(all_failed)} images without a face -> {DATA / "lfw_landmarks.npz"}')


if __name__ == '__main__':
    main()
