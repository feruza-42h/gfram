"""Experiment results (runs/*.json) and benchmark datasets (LFW, CPLFW) of the dissertation."""

import csv
import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Tuple

from . import ROOT

RESULTS = Path(__file__).parent / 'results'          # experiment results shipped with the package
RUNS = ROOT / 'runs'                                  # full results when run from the repository
LFW_DIR = ROOT / 'data' / 'lfw'
LFW_IMAGES = LFW_DIR / 'lfw_funneled'
CPLFW_DIR = ROOT / 'data' / 'cplfw'


@lru_cache(None)
def run(name: str) -> Dict:
    for path in (RESULTS / f'{name}.json', RUNS / f'{name}.json'):
        if path.exists():
            return json.loads(path.read_text())
    return {}


def training_history() -> List[Dict]:
    for path in (RESULTS / 'celeba_history.json', RUNS / 'celeba_tiny' / 'history.json'):
        if path.exists():
            return json.loads(path.read_text()).get('history', [])
    return []


# ---------------------------------------------------------------- datasets

def lfw_available() -> bool:
    return (LFW_DIR / 'pairs.txt').exists() and LFW_IMAGES.exists()


def cplfw_available() -> bool:
    return (CPLFW_DIR / 'pairs.csv').exists()


@lru_cache(None)
def lfw_pairs() -> List[Tuple[int, Path, Path, bool]]:
    """LFW View 2: (fold, image_a, image_b, same)."""
    lines = (LFW_DIR / 'pairs.txt').read_text().strip().split('\n')
    _, per_fold = map(int, lines[0].split())
    rel = lambda name, idx: LFW_IMAGES / name / f'{name}_{int(idx):04d}.jpg'
    pairs = []
    for i, line in enumerate(lines[1:]):
        fold = i // (2 * per_fold)
        t = line.split()
        if len(t) == 3:
            pairs.append((fold, rel(t[0], t[1]), rel(t[0], t[2]), True))
        else:
            pairs.append((fold, rel(t[0], t[1]), rel(t[2], t[3]), False))
    return pairs


@lru_cache(None)
def cplfw_pairs() -> List[Tuple[int, Path, Path, bool]]:
    pairs = []
    with open(CPLFW_DIR / 'pairs.csv') as f:
        for row in csv.DictReader(f):
            pairs.append((int(row['fold_id']), CPLFW_DIR / row['image_a_path'],
                          CPLFW_DIR / row['image_b_path'], row['is_same'] == '1'))
    return pairs


@lru_cache(None)
def lfw_people(min_images: int = 1) -> Dict[str, List[Path]]:
    """Identity -> sorted image paths."""
    people = {}
    for d in sorted(LFW_IMAGES.iterdir()):
        if d.is_dir():
            imgs = sorted(d.glob('*.jpg'))
            if len(imgs) >= min_images:
                people[d.name] = imgs
    return people


def pretty_name(lfw_name: str) -> str:
    return lfw_name.replace('_', ' ')
