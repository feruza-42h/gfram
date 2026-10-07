"""
Shared helpers for LFW experiments: data loading, Procrustes alignment,
identity-disjoint splits and verification metrics.
"""

from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
DATA = ROOT / 'data' / 'lfw'
RUNS = ROOT / 'runs'


# ------------------------------------------------------------------
# Data
# ------------------------------------------------------------------

def load_landmarks():
    """Return landmarks (N, 478, 3), relative paths (N,), identity names (N,)."""
    d = np.load(DATA / 'lfw_landmarks.npz')
    return d['landmarks'].astype(np.float32), d['paths'], d['names']


def load_pairs():
    """
    Parse LFW View 2 pairs.txt.

    Returns list of (fold, path_a, path_b, same) where paths match load_landmarks().
    Folds are identity-disjoint by construction of the LFW protocol.
    """
    lines = (DATA / 'pairs.txt').read_text().strip().split('\n')
    num_folds, per_fold = map(int, lines[0].split())
    rel = lambda name, idx: f'{name}/{name}_{int(idx):04d}.jpg'

    pairs = []
    for i, line in enumerate(lines[1:]):
        fold = i // (2 * per_fold)
        t = line.split()
        if len(t) == 3:
            pairs.append((fold, rel(t[0], t[1]), rel(t[0], t[2]), True))
        else:
            pairs.append((fold, rel(t[0], t[1]), rel(t[2], t[3]), False))
    return pairs


def fold_identities(pairs, fold):
    """All identity names that appear in a given test fold."""
    names = set()
    for f, a, b, _ in pairs:
        if f == fold:
            names.add(a.split('/')[0])
            names.add(b.split('/')[0])
    return names


# ------------------------------------------------------------------
# Alignment
# ------------------------------------------------------------------

def _center_scale(x):
    x = x - x.mean(axis=1, keepdims=True)
    return x / np.linalg.norm(x, axis=(1, 2), keepdims=True)


def _rotate_to(x, ref):
    """Batched Kabsch: rotate every shape in x (N, P, 3) onto ref (P, 3)."""
    h = np.einsum('npi,pj->nij', x, ref)
    u, _, vt = np.linalg.svd(h)
    d = np.sign(np.linalg.det(np.einsum('nij,njk->nik', u, vt)))
    fix = np.ones((len(x), 3), dtype=x.dtype)
    fix[:, 2] = d
    rot = np.einsum('nij,nj,njk->nik', u, fix, vt)
    return np.einsum('npi,nij->npj', x, rot)


def fit_reference(landmarks, iters=3):
    """Generalised Procrustes: mean face shape that all faces are aligned to."""
    x = _center_scale(landmarks.astype(np.float64))
    ref = x[0]
    for _ in range(iters):
        ref = _rotate_to(x, ref).mean(axis=0)
        ref /= np.linalg.norm(ref)
    return ref.astype(np.float32)


def align(landmarks, ref):
    """Remove translation, scale and head rotation; keeps only face shape."""
    x = _center_scale(landmarks.astype(np.float64))
    return _rotate_to(x, ref.astype(np.float64)).astype(np.float32)


# ------------------------------------------------------------------
# Metrics
# ------------------------------------------------------------------

def roc_auc(scores, labels):
    """Threshold-free ROC AUC (probability a genuine pair outscores an impostor pair)."""
    order = np.argsort(scores)
    ranks = np.empty(len(scores))
    ranks[order] = np.arange(1, len(scores) + 1)
    pos = labels.astype(bool)
    n_pos, n_neg = pos.sum(), (~pos).sum()
    return (ranks[pos].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def _best_threshold(scores, labels):
    cands = np.unique(scores)
    accs = [((scores >= t) == labels).mean() for t in cands]
    return cands[int(np.argmax(accs))]


def cv_accuracy(scores, labels, k=10, seed=0):
    """Accuracy where each threshold is chosen on held-out pairs, not on the pairs it is tested on."""
    idx = np.random.RandomState(seed).permutation(len(scores))
    parts = np.array_split(idx, k)
    correct = 0
    for i in range(k):
        test = parts[i]
        train = np.concatenate([parts[j] for j in range(k) if j != i])
        t = _best_threshold(scores[train], labels[train])
        correct += ((scores[test] >= t) == labels[test]).sum()
    return correct / len(scores)


def tar_at_far(scores, labels, far=1e-2):
    """True accept rate at a fixed false accept rate."""
    neg = np.sort(scores[~labels.astype(bool)])[::-1]
    k = max(int(np.floor(far * len(neg))), 1)
    thr = neg[k - 1]
    return (scores[labels.astype(bool)] > thr).mean()


def verification_report(scores, labels):
    labels = np.asarray(labels, dtype=bool)
    return {
        'pairs': int(len(scores)),
        'accuracy': float(cv_accuracy(scores, labels)),
        'auc': float(roc_auc(scores, labels)),
        'tar@far=1%': float(tar_at_far(scores, labels, 1e-2)),
    }
