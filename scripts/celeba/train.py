#!/usr/bin/env python3
"""
Train GeometricTransformer on CelebA landmarks, evaluate on all 6000 LFW pairs
(standard 10-fold protocol). LFW is never used for training.

Memory-aware for 8 GB machines: landmarks are aligned in place chunk by chunk
and standardised per batch instead of materialising extra copies.

Usage:
    nice -n 19 python scripts/celeba/train.py --config tiny --steps-per-epoch 2000 --epochs 30
"""

import os

for var in ('OMP_NUM_THREADS', 'MKL_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'VECLIB_MAXIMUM_THREADS'):
    os.environ.setdefault(var, '2')
os.environ.setdefault('KMP_DUPLICATE_LIB_OK', 'TRUE')

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parent / 'lfw'))
from common import ROOT, RUNS, load_landmarks, load_pairs, fit_reference, align, verification_report  # noqa: E402

# scripts/lfw/train.py shares this file's module name, so load it explicitly
import importlib.util  # noqa: E402
_spec = importlib.util.spec_from_file_location('lfw_train', HERE.parent / 'lfw' / 'train.py')
_lfw_train = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_lfw_train)
augment, embed = _lfw_train.augment, _lfw_train.embed

sys.path.insert(0, str(ROOT))
from gfram.models import create_geometric_transformer  # noqa: E402
from gfram.models.losses import ArcFaceLoss  # noqa: E402

torch.set_num_threads(2)
CELEBA = ROOT / 'data' / 'celeba' / 'celeba_landmarks.npz'


def align_in_place(x, ref, chunk=4000):
    for i in range(0, len(x), chunk):
        x[i:i + chunk] = align(x[i:i + chunk], ref)
    return x


def chunked_mean_std(x, chunk=8000):
    """Per-coordinate mean/std without allocating a full-size temporary."""
    total = np.zeros(x.shape[1:], np.float64)
    sq = np.zeros(x.shape[1:], np.float64)
    for i in range(0, len(x), chunk):
        c = x[i:i + chunk].astype(np.float64)
        total += c.sum(0)
        sq += (c ** 2).sum(0)
    mean = total / len(x)
    std = np.sqrt(np.maximum(sq / len(x) - mean ** 2, 0))
    return mean.astype(np.float32), (std + 1e-6).astype(np.float32)


def lfw_eval_set(ref, mu, sd):
    """Standardised LFW inputs for every image used by pairs.txt, plus the pair list."""
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    pairs = [(a, b, s) for _, a, b, s in load_pairs() if a in index and b in index]
    needed = sorted({index[p] for a, b, _ in pairs for p in (a, b)})
    x = ((align(lm[needed], ref) - mu) / sd).astype(np.float32)
    pos = {k: j for j, k in enumerate(needed)}
    pair_idx = np.array([(pos[index[a]], pos[index[b]]) for a, b, _ in pairs])
    labels = np.array([s for _, _, s in pairs])
    return x, pair_idx, labels


def evaluate(model, lfw, device):
    x, pair_idx, labels = lfw
    emb = embed(model, x, device)
    scores = (emb[pair_idx[:, 0]] * emb[pair_idx[:, 1]]).sum(1)
    return verification_report(scores, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--config', default='tiny', choices=['tiny', 'small', 'base'])
    p.add_argument('--epochs', type=int, default=30)
    p.add_argument('--steps-per-epoch', type=int, default=2000)
    p.add_argument('--batch', type=int, default=16)
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--scale', type=float, default=30.0)
    p.add_argument('--min-images', type=int, default=5)
    p.add_argument('--name', default=None)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = p.parse_args()

    name = args.name or f'celeba_{args.config}'
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- data ----
    # Keep a single landmark array in memory: select identities by index, not by copying
    d = np.load(CELEBA)
    x_all, ids_all = d['landmarks'], d['celeb_id']
    del d
    uniq, counts = np.unique(ids_all, return_counts=True)
    train_idx = np.where(np.isin(ids_all, uniq[counts >= args.min_images]))[0]

    rng0 = np.random.default_rng(0)
    ref = fit_reference(x_all[rng0.choice(train_idx, 5000, replace=False)])
    align_in_place(x_all, ref)
    mu, sd = chunked_mean_std(x_all)
    class_names, y_all = np.unique(ids_all[train_idx], return_inverse=True)
    lfw = lfw_eval_set(ref, mu, sd)
    print(f'train: {len(train_idx)} CelebA faces, {len(class_names)} identities | '
          f'test: {len(lfw[2])} LFW pairs', flush=True)

    # ---- model ----
    model = create_geometric_transformer(config_name=args.config, num_classes=None).to(device)
    criterion = ArcFaceLoss(model.output_dim, len(class_names), margin=args.margin, scale=args.scale).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    total = args.epochs * args.steps_per_epoch
    warmup = args.steps_per_epoch
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / warmup) * 0.5 * (1 + math.cos(math.pi * min(s, total) / total)))

    start_epoch, best, history = 0, None, []
    last = run_dir / 'last.pt'
    if last.exists():
        ck = torch.load(last, map_location=device, weights_only=False)
        model.load_state_dict(ck['model'])
        criterion.load_state_dict(ck['criterion'])
        opt.load_state_dict(ck['opt'])
        sched.load_state_dict(ck['sched'])
        start_epoch, best, history = ck['epoch'] + 1, ck['best'], ck['history']
        print(f'resumed from epoch {start_epoch}', flush=True)

    for epoch in range(start_epoch, args.epochs):
        rng = np.random.default_rng(1000 + epoch)
        model.train()
        t0 = time.time()
        losses = []
        for _ in range(args.steps_per_epoch):
            pick = rng.integers(0, len(train_idx), args.batch)
            xb = ((augment(x_all[train_idx[pick]], sd, rng) - mu) / sd).astype(np.float32)
            loss = criterion(model(torch.from_numpy(xb).to(device)), torch.from_numpy(y_all[pick]).to(device))
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        entry = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'sec': round(time.time() - t0, 1)}
        entry.update(evaluate(model, lfw, device))
        if best is None or entry['auc'] > best['auc']:
            best = entry
            torch.save({'model_state_dict': model.state_dict(), 'reference_shape': ref,
                        'input_mean': mu, 'input_std': sd, 'config': args.config,
                        'metrics': entry, 'args': vars(args)}, run_dir / 'best.pt')
        history.append(entry)
        print(json.dumps(entry), flush=True)
        torch.save({'model': model.state_dict(), 'criterion': criterion.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best': best, 'history': history}, last)

    (run_dir / 'history.json').write_text(json.dumps({'best': best, 'history': history}, indent=2))
    print('BEST', json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
