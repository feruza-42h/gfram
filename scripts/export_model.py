#!/usr/bin/env python3
"""
Package a trained run into the single file served by gfram.uz (/api/model/download).

The file carries everything a client needs to reproduce training-time preprocessing:
model config, weights, Procrustes reference shape and input standardisation.

Usage:
    python scripts/export_model.py runs/celeba_tiny/best.pt dist/gfram_model_v3.1.0.pth --version 3.1.0
"""

import argparse
import hashlib
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / 'scripts' / 'lfw'))


def calibrate(package):
    """
    Score all LFW pairs through the client-side FaceEmbedder (same code path users run)
    and derive decision thresholds.
    """
    from gfram.models.package import FaceEmbedder
    from common import load_landmarks, load_pairs, verification_report

    embedder = FaceEmbedder(package)
    lm, paths, _ = load_landmarks()
    index = {p: i for i, p in enumerate(paths)}
    pairs = [(a, b, s) for _, a, b, s in load_pairs() if a in index and b in index]
    needed = sorted({index[p] for a, b, _ in pairs for p in (a, b)})
    pos = {k: j for j, k in enumerate(needed)}
    emb = embedder.embed(lm[needed])
    scores = np.array([emb[pos[index[a]]] @ emb[pos[index[b]]] for a, b, _ in pairs])
    labels = np.array([s for _, _, s in pairs])

    cands = np.unique(scores)
    accs = np.array([((scores >= t) == labels).mean() for t in cands])
    neg = np.sort(scores[~labels])[::-1]
    thresholds = {
        'best_accuracy': float(cands[accs.argmax()]),
        'far_1e-2': float(neg[int(0.01 * len(neg)) - 1]),
        'far_1e-3': float(neg[max(int(0.001 * len(neg)), 1) - 1]),
    }
    return thresholds, verification_report(scores, labels)


def main():
    p = argparse.ArgumentParser()
    p.add_argument('checkpoint')
    p.add_argument('output')
    p.add_argument('--version', required=True, help='model version, e.g. 3.2.0')
    p.add_argument('--fusion', default=None,
                   help='fusion JSON from scripts/hybrid/train_fusion.py (makes a gfram-model/3 hybrid package)')
    args = p.parse_args()

    ck = torch.load(args.checkpoint, map_location='cpu', weights_only=False)
    package = {
        'format': 'gfram-model/3' if args.fusion else 'gfram-model/2',
        'model_version': args.version,
        'min_client_version': args.version,
        'created_at': datetime.now(timezone.utc).isoformat(),
        'architecture': 'GeometricTransformer',
        'config_name': ck['config'],
        'num_landmarks': 478,
        'model_state_dict': ck['model_state_dict'],
        'preprocessing': {
            'align': 'procrustes',
            'reference_shape': np.asarray(ck['reference_shape'], np.float32),
            'input_mean': np.asarray(ck['input_mean'], np.float32),
            'input_std': np.asarray(ck['input_std'], np.float32),
        },
        'metrics': {'dataset': 'LFW (6000 pairs, 10-fold, trained on CelebA)', **ck['metrics']},
        'train_args': ck['args'],
    }

    if args.fusion:
        spec = json.loads(Path(args.fusion).read_text())
        package['fusion'] = {k: spec[k] for k in ('type', 'appearance_model', 'features', 'mean', 'scale',
                                                  'coef', 'intercept', 'thresholds')}
        package['fusion']['cross_dataset'] = spec.get('cross_dataset')

    package['thresholds'], client_metrics = calibrate(package)
    print('thresholds:', package['thresholds'])
    print('client-side LFW check:', client_metrics)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    torch.save(package, out)
    sha = hashlib.sha256(out.read_bytes()).hexdigest()
    out.with_suffix('.json').write_text(json.dumps({
        'model_version': args.version, 'file': out.name, 'size_bytes': out.stat().st_size,
        'sha256': sha, 'metrics': package['metrics'], 'thresholds': package['thresholds'],
    }, indent=2))
    print(f'{out} ({out.stat().st_size / 2**20:.1f} MB) sha256={sha[:16]}…')


if __name__ == '__main__':
    main()
