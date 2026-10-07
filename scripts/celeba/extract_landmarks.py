#!/usr/bin/env python3
"""
CelebA: download shard -> extract MediaPipe 478 landmarks -> delete shard.

Source: Hugging Face mirror flwrlabs/celeba (25 parquet shards, ~500 MB each,
images + celeb_id). Only one shard is on disk at a time; finished shards are
skipped on restart, so the script can be stopped and resumed at any point.

Usage:
    nice -n 19 python scripts/celeba/extract_landmarks.py [--shards train-00000,...] [--pause 5]
"""

import os

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import logging
import sys
import subprocess
import time
from pathlib import Path

import cv2
import numpy as np
import pyarrow.parquet as pq

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from gfram.detectors import FaceDetector  # noqa: E402

DATA = ROOT / 'data' / 'celeba'
OUT = DATA / 'landmarks'
BASE_URL = 'https://huggingface.co/datasets/flwrlabs/celeba/resolve/main/img_align%2Bidentity%2Battr/'
SHARDS = ([f'train-{i:05d}-of-00019' for i in range(19)]
          + [f'valid-{i:05d}-of-00003' for i in range(3)]
          + [f'test-{i:05d}-of-00003' for i in range(3)])

logging.basicConfig(level=logging.WARNING)


def download(shard, dest):
    # curl uses the system certificate store (python.org builds on macOS ship without one)
    # and -C - resumes a partial file after an interrupted run
    tmp = dest.with_suffix('.part')
    cmd = ['curl', '-sSL', '--fail', '--retry', '5', '--retry-delay', '10', '-C', '-',
           '-o', str(tmp), BASE_URL + shard + '.parquet']
    if subprocess.run(cmd).returncode != 0:
        raise RuntimeError(f'could not download {shard}')
    tmp.rename(dest)


def process_shard(shard, detector):
    out_file = OUT / f'{shard}.npz'
    parquet = DATA / f'{shard}.parquet'
    if not parquet.exists():
        t = time.time()
        download(shard, parquet)
        print(f'  downloaded {shard} ({time.time() - t:.0f}s)', flush=True)

    t = time.time()
    landmarks, ids, rows, failed = [], [], [], 0
    row = 0
    pf = pq.ParquetFile(parquet)
    # Small batches keep memory flat even though the shard is ~500 MB
    for batch in pf.iter_batches(batch_size=64, columns=['image', 'celeb_id']):
        images = batch.column('image').to_pylist()
        celeb = batch.column('celeb_id').to_pylist()
        for img_struct, cid in zip(images, celeb):
            img = cv2.imdecode(np.frombuffer(img_struct['bytes'], np.uint8), cv2.IMREAD_COLOR)
            faces = detector.detect(img) if img is not None else []
            faces = [f for f in faces if f.get('num_landmarks') == 478]
            if faces:
                landmarks.append(faces[0]['landmarks'].astype(np.float32))
                ids.append(cid)
                rows.append(row)
            else:
                failed += 1
            row += 1

    np.savez_compressed(out_file, landmarks=np.stack(landmarks), celeb_id=np.array(ids),
                        row=np.array(rows), split=shard.split('-')[0], failed=failed)
    parquet.unlink()
    print(f'{shard}: ok={len(ids)} failed={failed} ({time.time() - t:.0f}s)', flush=True)


def merge():
    parts = [np.load(f) for f in sorted(OUT.glob('*.npz'))]
    lm = np.concatenate([p['landmarks'] for p in parts])
    ids = np.concatenate([p['celeb_id'] for p in parts])
    split = np.concatenate([np.full(len(p['celeb_id']), str(p['split'])) for p in parts])
    np.savez_compressed(DATA / 'celeba_landmarks.npz', landmarks=lm, celeb_id=ids, split=split)
    print(f'merged {len(parts)} shards: {len(ids)} faces, {len(np.unique(ids))} identities '
          f'-> {DATA / "celeba_landmarks.npz"}', flush=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--shards', default=None, help='comma-separated subset, default: all 25')
    p.add_argument('--pause', type=float, default=5.0, help='seconds to rest between shards')
    args = p.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    shards = args.shards.split(',') if args.shards else SHARDS
    detector = FaceDetector(max_num_faces=1, refine_landmarks=True)

    for i, shard in enumerate(shards):
        if (OUT / f'{shard}.npz').exists():
            continue
        print(f'[{i + 1}/{len(shards)}] {shard}', flush=True)
        process_shard(shard, detector)
        time.sleep(args.pause)

    if len(list(OUT.glob('*.npz'))) == len(SHARDS):
        merge()


if __name__ == '__main__':
    main()
