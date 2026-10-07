#!/usr/bin/env python3
"""
Protected templates: dissertation experiment.

Uses the cached LFW (8 capture conditions) and CPLFW pair features and the shipped
adaptive fusion (data/models/fusion_mbf.json) to measure:

  1. Score preservation  - fused match probability with vs without protection
                           (max |dp|, decisions changed at the shipped threshold).
  2. Unlinkability       - mated pairs whose two templates were protected under
                           DIFFERENT keys vs non-mated pairs: ROC AUC (0.5 = unlinkable).
  3. Revocability        - stolen templates (old key) against probes after key
                           rotation (new key): genuine accept rate at the threshold.
  4. Overhead            - time to protect a template, bytes stored per face.

Usage:
    python scripts/hybrid/template_protection.py
"""

import json
import secrets
import sys
import time
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parent / 'lfw'))
import train_fusion as tf  # noqa: E402
from common import ROOT, RUNS, roc_auc  # noqa: E402

sys.path.insert(0, str(ROOT))
from gfram.security import TemplateProtector  # noqa: E402

SPEC = json.loads((ROOT / 'data' / 'models' / 'fusion_mbf.json').read_text())
MODEL = {k: np.asarray(SPEC[k], np.float64) for k in ('mean', 'scale', 'coef')}
MODEL['intercept'] = SPEC['intercept']
THRESHOLD = SPEC['thresholds']['best_accuracy']


def pair_sets():
    """name -> (A side, B side, labels) with per-face arrays, as used by the fusion."""
    import csv
    out = {}
    lm_paths = tf.load_landmarks()[1]
    index = {p: i for i, p in enumerate(lm_paths)}
    raw = [(f, a, b, s) for f, a, b, s in tf.load_pairs() if a in index and b in index]
    gallery, probes = sorted({a for _, a, _, _ in raw}), sorted({b for _, _, b, _ in raw})
    gpos, ppos = {p: i for i, p in enumerate(gallery)}, {p: i for i, p in enumerate(probes)}
    ia = np.array([gpos[a] for _, a, _, _ in raw]); ib = np.array([ppos[b] for _, _, b, _ in raw])
    labels = np.array([s for *_, s in raw])
    A = dict(np.load(tf.LFW_DATA / 'robustness' / 'gallery_clean.npz'))
    for f in sorted((tf.LFW_DATA / 'robustness').glob('*.npz')):
        if f.stem == 'gallery_clean':
            continue
        B = dict(np.load(f))
        ok = A['found'][ia] & B['found'][ib]
        out[f'LFW {f.stem}'] = (tf.take(A, ia[ok]), tf.take(B, ib[ok]), labels[ok])

    rows = list(csv.DictReader(open(tf.CPLFW / 'pairs.csv')))
    paths = sorted({p for r in rows for p in (r['image_a_path'], r['image_b_path'])})
    pos = {p: i for i, p in enumerate(paths)}
    ia = np.array([pos[r['image_a_path']] for r in rows]); ib = np.array([pos[r['image_b_path']] for r in rows])
    labels = np.array([r['is_same'] == '1' for r in rows])
    F = dict(np.load(tf.CPLFW / 'features.npz'))
    F['app'], F['app_norm'] = F['w600k_mbf'], F['w600k_mbf_norm']
    ok = F['found'][ia] & F['found'][ib]
    out['CPLFW'] = (tf.take(F, ia[ok]), tf.take(F, ib[ok]), labels[ok])
    return out


def protect(side, protector):
    return dict(side, geo=protector.protect(side['geo'], 'geo'), app=protector.protect(side['app'], 'app'))


def probability(A, B):
    return tf.predict(MODEL, tf.pair_features(A, B))


def main():
    sets = pair_sets()
    k1, k2 = TemplateProtector(secrets.token_bytes(32)), TemplateProtector(secrets.token_bytes(32))
    results = {}
    print(f'{"set":20s} {"pairs":>6s} {"max|dp|":>9s} {"flips":>6s} {"acc plain":>10s} {"acc prot.":>10s} '
          f'{"link AUC":>9s} {"revoked GAR":>12s}')
    for name, (A, B, y) in sets.items():
        p_plain = probability(A, B)
        p_prot = probability(protect(A, k1), protect(B, k1))               # same key: normal operation
        p_cross = probability(protect(A, k1), protect(B, k2))              # different keys / after rotation
        flips = int(((p_plain >= THRESHOLD) != (p_prot >= THRESHOLD)).sum())
        r = {
            'pairs': int(len(y)),
            'max_abs_prob_diff': float(np.abs(p_plain - p_prot).max()),
            'decisions_changed': flips,
            'accuracy_plain': float(((p_plain >= THRESHOLD) == y).mean()),
            'accuracy_protected': float(((p_prot >= THRESHOLD) == y).mean()),
            # 0.5 = mated and non-mated cross-key pairs are indistinguishable
            'unlinkability_auc': float(roc_auc(p_cross, y)),
            'revoked_genuine_accept_rate': float((p_cross[y] >= THRESHOLD).mean()),
            'plain_genuine_accept_rate': float((p_plain[y] >= THRESHOLD).mean()),
        }
        results[name] = r
        print(f'{name:20s} {r["pairs"]:6d} {r["max_abs_prob_diff"]:9.1e} {flips:6d} {r["accuracy_plain"]:10.4f} '
              f'{r["accuracy_protected"]:10.4f} {r["unlinkability_auc"]:9.3f} {r["revoked_genuine_accept_rate"]:12.4f}')

    # Overhead
    e_geo, e_app = np.random.default_rng(0).normal(size=(2, 1, 512))[:, 0]
    t = time.perf_counter()
    for _ in range(1000):
        k1.protect(e_geo[:128], 'geo'); k1.protect(e_app, 'app')
    protect_us = (time.perf_counter() - t) / 1000 * 1e6
    blob = k1.encrypt_landmarks(np.zeros((478, 3), np.float32))
    t = time.perf_counter()
    for _ in range(1000):
        k1.decrypt_landmarks(blob)
    decrypt_us = (time.perf_counter() - t) / 1000 * 1e6
    t = time.perf_counter(); TemplateProtector(secrets.token_bytes(32)).protect(e_app, 'app'); setup_ms = (time.perf_counter() - t) * 1e3
    results['overhead'] = {'protect_geo_and_app_us': protect_us, 'decrypt_landmarks_us': decrypt_us,
                           'key_setup_ms': setup_ms, 'landmarks_bytes_plain': 478 * 3 * 4,
                           'landmarks_bytes_encrypted': len(blob)}
    print(f'\noverhead: protect both templates {protect_us:.1f} us, decrypt landmarks {decrypt_us:.1f} us, '
          f'key setup {setup_ms:.0f} ms, landmarks {478 * 3 * 4} -> {len(blob)} bytes')
    (RUNS / 'template_protection.json').write_text(json.dumps(results, indent=2))


if __name__ == '__main__':
    main()
