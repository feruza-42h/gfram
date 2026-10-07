"""
Face Recognition Benchmarks for GFRAM
=====================================

Standard benchmarks for evaluating face recognition performance.

PhD Thesis: Ortiqova F.S.
"""

import numpy as np
from pathlib import Path
from typing import Optional, List, Dict, Tuple
import logging
from tqdm import tqdm
from abc import ABC, abstractmethod

from .metrics import (
    compute_roc,
    compute_tar_at_far,
    compute_eer,
    compute_accuracy,
    FaceVerificationMetrics
)

logger = logging.getLogger(__name__)


class BaseBenchmark(ABC):
    """Base class for face recognition benchmarks"""

    def __init__(self, data_dir: str, name: str = "benchmark"):
        self.data_dir = Path(data_dir)
        self.name = name
        self.pairs: List[Tuple] = []
        self.labels: List[int] = []

    @abstractmethod
    def load_pairs(self):
        """Load verification pairs"""
        pass

    @abstractmethod
    def get_image_path(self, identifier) -> Path:
        """Get path to image"""
        pass

    def evaluate(
            self,
            recognizer,
            num_folds: int = 10
    ) -> Dict:
        """
        Evaluate recognizer on benchmark.

        Args:
            recognizer: GFRAM recognizer instance
            num_folds: Number of folds for cross-validation

        Returns:
            Evaluation results dictionary
        """
        if not self.pairs:
            self.load_pairs()

        logger.info(f"Evaluating on {self.name}: {len(self.pairs)} pairs")

        # Compute similarities
        similarities = []
        valid_labels = []

        for (img1_id, img2_id), label in tqdm(zip(self.pairs, self.labels),
                                              total=len(self.pairs),
                                              desc=f"Evaluating {self.name}"):
            try:
                img1_path = self.get_image_path(img1_id)
                img2_path = self.get_image_path(img2_id)

                # Get embeddings
                emb1 = recognizer.get_embedding(str(img1_path))
                emb2 = recognizer.get_embedding(str(img2_path))

                if emb1 is None or emb2 is None:
                    continue

                # Compute similarity (cosine)
                sim = np.dot(emb1, emb2) / (np.linalg.norm(emb1) * np.linalg.norm(emb2) + 1e-8)
                similarities.append(sim)
                valid_labels.append(label)

            except Exception as e:
                logger.warning(f"Error processing pair: {e}")
                continue

        similarities = np.array(similarities)
        valid_labels = np.array(valid_labels)

        # Compute metrics
        metrics = FaceVerificationMetrics(similarities, valid_labels)

        # Cross-validation for accuracy
        fold_size = len(similarities) // num_folds
        accuracies = []

        for fold in range(num_folds):
            start = fold * fold_size
            end = start + fold_size

            # Test fold
            test_sims = similarities[start:end]
            test_labels = valid_labels[start:end]

            # Train folds (find best threshold)
            train_mask = np.ones(len(similarities), dtype=bool)
            train_mask[start:end] = False
            train_sims = similarities[train_mask]
            train_labels = valid_labels[train_mask]

            # Find best threshold on train
            best_thresh, best_acc = self._find_best_threshold(train_sims, train_labels)

            # Evaluate on test
            predictions = (test_sims > best_thresh).astype(int)
            fold_acc = np.mean(predictions == test_labels)
            accuracies.append(fold_acc)

        results = {
            'benchmark': self.name,
            'num_pairs': len(self.pairs),
            'valid_pairs': len(valid_labels),
            'accuracy': np.mean(accuracies),
            'accuracy_std': np.std(accuracies),
            'auc': metrics.auc,
            'eer': metrics.eer,
            'tar_at_far_1e-3': metrics.tar_at_far(0.001),
            'tar_at_far_1e-4': metrics.tar_at_far(0.0001),
            'best_threshold': float(metrics.best_threshold),
            'fold_accuracies': accuracies
        }

        logger.info(f"{self.name} Results:")
        logger.info(f"  Accuracy: {results['accuracy'] * 100:.2f}% ± {results['accuracy_std'] * 100:.2f}%")
        logger.info(f"  AUC: {results['auc']:.4f}")
        logger.info(f"  EER: {results['eer'] * 100:.2f}%")
        logger.info(f"  TAR@FAR=0.1%: {results['tar_at_far_1e-3'] * 100:.2f}%")

        return results

    def _find_best_threshold(
            self,
            similarities: np.ndarray,
            labels: np.ndarray
    ) -> Tuple[float, float]:
        """Find threshold that maximizes accuracy"""
        best_thresh = 0.5
        best_acc = 0.0

        for thresh in np.linspace(0, 1, 100):
            predictions = (similarities > thresh).astype(int)
            acc = np.mean(predictions == labels)

            if acc > best_acc:
                best_acc = acc
                best_thresh = thresh

        return best_thresh, best_acc


class LFWBenchmark(BaseBenchmark):
    """
    LFW (Labeled Faces in the Wild) Benchmark

    Standard benchmark for unconstrained face verification.
    6,000 face pairs (3,000 positive, 3,000 negative)

    Reference: Huang et al., "Labeled Faces in the Wild"
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir, name="LFW")
        self.pairs_file = self.data_dir / "pairs.txt"

    def load_pairs(self):
        """Load LFW verification pairs"""
        if not self.pairs_file.exists():
            raise FileNotFoundError(f"LFW pairs file not found: {self.pairs_file}")

        self.pairs = []
        self.labels = []

        with open(self.pairs_file, 'r') as f:
            lines = f.readlines()

        # Skip header line
        num_folds = int(lines[0].strip().split()[0])
        pairs_per_fold = int(lines[0].strip().split()[1])

        current_line = 1

        for fold in range(num_folds):
            # Positive pairs
            for i in range(pairs_per_fold):
                parts = lines[current_line].strip().split()
                name = parts[0]
                idx1 = int(parts[1])
                idx2 = int(parts[2])

                self.pairs.append(((name, idx1), (name, idx2)))
                self.labels.append(1)
                current_line += 1

            # Negative pairs
            for i in range(pairs_per_fold):
                parts = lines[current_line].strip().split()
                name1 = parts[0]
                idx1 = int(parts[1])
                name2 = parts[2]
                idx2 = int(parts[3])

                self.pairs.append(((name1, idx1), (name2, idx2)))
                self.labels.append(0)
                current_line += 1

        logger.info(f"Loaded {len(self.pairs)} LFW pairs ({sum(self.labels)} positive)")

    def get_image_path(self, identifier) -> Path:
        """Get path to LFW image"""
        name, idx = identifier
        filename = f"{name}_{idx:04d}.jpg"
        return self.data_dir / name / filename


class CFPBenchmark(BaseBenchmark):
    """
    CFP (Celebrities in Frontal-Profile) Benchmark

    Tests cross-pose face verification (frontal vs profile).

    Protocols:
    - CFP-FF: Frontal-Frontal pairs
    - CFP-FP: Frontal-Profile pairs (harder)
    """

    def __init__(self, data_dir: str, protocol: str = "FP"):
        super().__init__(data_dir, name=f"CFP-{protocol}")
        self.protocol = protocol

    def load_pairs(self):
        """Load CFP verification pairs"""
        pairs_file = self.data_dir / f"Protocol" / self.protocol / "pair_list.txt"

        if not pairs_file.exists():
            logger.warning(f"CFP pairs file not found: {pairs_file}")
            return

        self.pairs = []
        self.labels = []

        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                img1 = parts[0]
                img2 = parts[1]
                label = int(parts[2])

                self.pairs.append((img1, img2))
                self.labels.append(label)

        logger.info(f"Loaded {len(self.pairs)} CFP-{self.protocol} pairs")

    def get_image_path(self, identifier) -> Path:
        """Get path to CFP image"""
        return self.data_dir / "Data" / identifier


class AgeDBBenchmark(BaseBenchmark):
    """
    AgeDB-30 Benchmark

    Tests face verification across age gaps.
    Minimum age difference of 30 years between pairs.
    """

    def __init__(self, data_dir: str):
        super().__init__(data_dir, name="AgeDB-30")

    def load_pairs(self):
        """Load AgeDB verification pairs"""
        pairs_file = self.data_dir / "agedb_30_pair.txt"

        if not pairs_file.exists():
            logger.warning(f"AgeDB pairs file not found: {pairs_file}")
            return

        self.pairs = []
        self.labels = []

        with open(pairs_file, 'r') as f:
            for line in f:
                parts = line.strip().split()
                if len(parts) < 3:
                    continue

                img1 = parts[0]
                img2 = parts[1]
                label = int(parts[2])

                self.pairs.append((img1, img2))
                self.labels.append(label)

        logger.info(f"Loaded {len(self.pairs)} AgeDB-30 pairs")

    def get_image_path(self, identifier) -> Path:
        """Get path to AgeDB image"""
        return self.data_dir / identifier


def run_all_benchmarks(
        recognizer,
        benchmark_dirs: Dict[str, str],
        output_file: Optional[str] = None
) -> Dict:
    """
    Run all available benchmarks.

    Args:
        recognizer: GFRAM recognizer instance
        benchmark_dirs: Dict mapping benchmark name to directory path
            e.g., {'LFW': '/path/to/lfw', 'CFP': '/path/to/cfp'}
        output_file: Optional path to save results

    Returns:
        Combined results dictionary
    """
    results = {}

    # LFW
    if 'LFW' in benchmark_dirs and Path(benchmark_dirs['LFW']).exists():
        try:
            lfw = LFWBenchmark(benchmark_dirs['LFW'])
            results['LFW'] = lfw.evaluate(recognizer)
        except Exception as e:
            logger.error(f"LFW benchmark failed: {e}")
            results['LFW'] = {'error': str(e)}

    # CFP-FP
    if 'CFP' in benchmark_dirs and Path(benchmark_dirs['CFP']).exists():
        try:
            cfp = CFPBenchmark(benchmark_dirs['CFP'], protocol='FP')
            results['CFP-FP'] = cfp.evaluate(recognizer)
        except Exception as e:
            logger.error(f"CFP-FP benchmark failed: {e}")
            results['CFP-FP'] = {'error': str(e)}

    # AgeDB
    if 'AgeDB' in benchmark_dirs and Path(benchmark_dirs['AgeDB']).exists():
        try:
            agedb = AgeDBBenchmark(benchmark_dirs['AgeDB'])
            results['AgeDB-30'] = agedb.evaluate(recognizer)
        except Exception as e:
            logger.error(f"AgeDB benchmark failed: {e}")
            results['AgeDB-30'] = {'error': str(e)}

    # Summary
    summary = {
        'benchmarks_run': len(results),
        'results': results
    }

    # Print summary
    print("\n" + "=" * 60)
    print("GFRAM BENCHMARK RESULTS")
    print("=" * 60)

    for name, res in results.items():
        if 'error' in res:
            print(f"\n{name}: ERROR - {res['error']}")
        else:
            print(f"\n{name}:")
            print(f"  Accuracy: {res['accuracy'] * 100:.2f}%")
            print(f"  AUC: {res['auc']:.4f}")
            print(f"  TAR@FAR=0.1%: {res.get('tar_at_far_1e-3', 0) * 100:.2f}%")

    print("=" * 60 + "\n")

    # Save if requested
    if output_file:
        import json

        # Convert numpy types for JSON
        def convert(obj):
            if isinstance(obj, np.ndarray):
                return obj.tolist()
            if isinstance(obj, (np.float32, np.float64)):
                return float(obj)
            if isinstance(obj, (np.int32, np.int64)):
                return int(obj)
            return obj

        with open(output_file, 'w') as f:
            json.dump(summary, f, indent=2, default=convert)

        logger.info(f"Results saved to {output_file}")

    return summary
