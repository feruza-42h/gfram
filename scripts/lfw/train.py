#!/usr/bin/env python3
"""
Step 4: train GeometricTransformer on LFW landmarks with ArcFace.

Identity-disjoint protocol: every person who appears in the test fold is
removed from training, so the reported numbers are on unseen people.

Resource-friendly: 2 CPU threads, small batches, checkpoint every epoch,
resumes automatically from runs/<name>/last.pt.

Usage:
    nice -n 19 python scripts/lfw/train.py --fold 0 --config tiny --epochs 60
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

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(__import__('pathlib').Path(__file__).resolve().parent))
from common import (ROOT, RUNS, load_landmarks, load_pairs, fold_identities,  # noqa: E402
                    fit_reference, align, verification_report)

sys.path.insert(0, str(ROOT))
from gfram.models import create_geometric_transformer  # noqa: E402
from gfram.models.losses import ArcFaceLoss  # noqa: E402

torch.set_num_threads(2)


def random_rotation(n, max_deg, rng):
    """Small random 3D rotations (n, 3, 3) to simulate residual pose error."""
    a = np.deg2rad(rng.uniform(-max_deg, max_deg, size=(n, 3)))
    cx, cy, cz = np.cos(a).T
    sx, sy, sz = np.sin(a).T
    one, zero = np.ones(n), np.zeros(n)
    rx = np.stack([one, zero, zero, zero, cx, -sx, zero, sx, cx], 1).reshape(n, 3, 3)
    ry = np.stack([cy, zero, sy, zero, one, zero, -sy, zero, cy], 1).reshape(n, 3, 3)
    rz = np.stack([cz, -sz, zero, sz, cz, zero, zero, zero, one], 1).reshape(n, 3, 3)
    return (rx @ ry @ rz).astype(np.float32)


def augment(x, sd, rng):
    """
    Augmentations sized relative to between-person variation (sd per coordinate,
    ~1e-3 after Procrustes). Stronger ones erase identity: a 6 deg rotation alone
    moves points more than the difference between two people.
    """
    n = len(x)
    x = np.einsum('npi,nij->npj', x, random_rotation(n, 1.0, rng))
    return x + (rng.normal(size=x.shape) * 0.15 * sd).astype(np.float32)


@torch.no_grad()
def embed(model, x, device, bs=32):
    model.eval()
    out = []
    for i in range(0, len(x), bs):
        out.append(F.normalize(model(torch.from_numpy(x[i:i + bs]).to(device)), dim=1).cpu())
    return torch.cat(out).numpy()


def evaluate(model, aligned, path_index, pairs, fold, device):
    test = [(a, b, s) for f, a, b, s in pairs if f == fold and a in path_index and b in path_index]
    needed = sorted({path_index[p] for a, b, _ in test for p in (a, b)})
    emb = dict(zip(needed, embed(model, aligned[needed], device)))
    scores = np.array([emb[path_index[a]] @ emb[path_index[b]] for a, b, _ in test])
    labels = np.array([s for _, _, s in test])
    rep = verification_report(scores, labels)
    rep['coverage'] = len(test) / sum(1 for p in pairs if p[0] == fold)
    return rep


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--fold', type=int, default=0)
    p.add_argument('--config', default='tiny', choices=['tiny', 'small', 'base'])
    p.add_argument('--epochs', type=int, default=40)
    p.add_argument('--batch', type=int, default=16, help='16 fits in 8 GB unified memory')
    p.add_argument('--lr', type=float, default=5e-4)
    p.add_argument('--margin', type=float, default=0.3)
    p.add_argument('--scale', type=float, default=30.0)
    p.add_argument('--min-images', type=int, default=2, help='min images per training identity')
    p.add_argument('--eval-every', type=int, default=2)
    p.add_argument('--name', default=None)
    p.add_argument('--device', default='mps' if torch.backends.mps.is_available() else 'cpu')
    args = p.parse_args()

    name = args.name or f'lfw_fold{args.fold}_{args.config}'
    run_dir = RUNS / name
    run_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    # ---- data ----
    landmarks, paths, names = load_landmarks()
    pairs = load_pairs()
    test_ids = fold_identities(pairs, args.fold)

    ref = fit_reference(landmarks)
    aligned = align(landmarks, ref)
    path_index = {p: i for i, p in enumerate(paths)}

    train_mask = np.array([n not in test_ids for n in names])
    uniq, counts = np.unique(names[train_mask], return_counts=True)
    keep = set(uniq[counts >= args.min_images])
    train_mask &= np.array([n in keep for n in names])
    class_names = sorted(keep)
    class_of = {n: i for i, n in enumerate(class_names)}
    x_train = aligned[train_mask]
    # Identity lives in tiny deviations from the mean face; feed standardised deviations
    mu, sd = x_train.mean(0), x_train.std(0) + 1e-6
    standardise = lambda a: ((a - mu) / sd).astype(np.float32)
    inputs = standardise(aligned)
    y_train = np.array([class_of[n] for n in names[train_mask]])
    print(f'train: {len(x_train)} faces, {len(class_names)} identities | '
          f'test fold {args.fold}: {len(test_ids)} unseen identities', flush=True)

    # ---- model ----
    model = create_geometric_transformer(config_name=args.config, num_classes=None).to(device)
    criterion = ArcFaceLoss(model.output_dim, len(class_names), margin=args.margin, scale=args.scale).to(device)
    params = list(model.parameters()) + list(criterion.parameters())
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=5e-4)
    steps_per_epoch = math.ceil(len(x_train) / args.batch)
    total = args.epochs * steps_per_epoch
    warmup = 2 * steps_per_epoch
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

    rng = np.random.default_rng(start_epoch)
    for epoch in range(start_epoch, args.epochs):
        model.train()
        t0 = time.time()
        perm = rng.permutation(len(x_train))
        losses = []
        for i in range(0, len(perm), args.batch):
            idx = perm[i:i + args.batch]
            if len(idx) < 2:
                continue
            xb = torch.from_numpy(standardise(augment(x_train[idx], sd, rng))).to(device)
            yb = torch.from_numpy(y_train[idx]).to(device)
            loss = criterion(model(xb), yb)
            opt.zero_grad()
            loss.backward()
            torch.nn.utils.clip_grad_norm_(params, 5.0)
            opt.step()
            sched.step()
            losses.append(loss.item())

        entry = {'epoch': epoch + 1, 'loss': float(np.mean(losses)), 'sec': round(time.time() - t0, 1)}
        if (epoch + 1) % args.eval_every == 0 or epoch + 1 == args.epochs:
            entry.update(evaluate(model, inputs, path_index, pairs, args.fold, device))
            if best is None or entry['auc'] > best['auc']:
                best = entry
                torch.save({'model_state_dict': model.state_dict(), 'reference_shape': ref, 'input_mean': mu, 'input_std': sd,
                            'config': args.config, 'metrics': entry, 'args': vars(args)},
                           run_dir / 'best.pt')
        history.append(entry)
        print(json.dumps(entry), flush=True)

        torch.save({'model': model.state_dict(), 'criterion': criterion.state_dict(),
                    'opt': opt.state_dict(), 'sched': sched.state_dict(),
                    'epoch': epoch, 'best': best, 'history': history}, last)

    (run_dir / 'history.json').write_text(json.dumps({'best': best, 'history': history}, indent=2))
    print('BEST', json.dumps(best), flush=True)


if __name__ == '__main__':
    main()
