"""
GFRAM Evaluation Module
=======================

Benchmarks and metrics for face recognition evaluation.

Supported benchmarks:
- LFW (Labeled Faces in the Wild)
- CFP-FP (Celebrities in Frontal-Profile)
- AgeDB-30

Metrics:
- TAR@FAR (True Accept Rate at False Accept Rate)
- EER (Equal Error Rate)
- AUC (Area Under Curve)
- Accuracy

PhD Thesis: Ortiqova F.S.
"""

from .benchmarks import (
    LFWBenchmark,
    CFPBenchmark,
    AgeDBBenchmark,
    run_all_benchmarks
)

from .metrics import (
    compute_roc,
    compute_tar_at_far,
    compute_eer,
    compute_accuracy,
    FaceVerificationMetrics
)

__all__ = [
    'LFWBenchmark',
    'CFPBenchmark',
    'AgeDBBenchmark',
    'run_all_benchmarks',
    'compute_roc',
    'compute_tar_at_far',
    'compute_eer',
    'compute_accuracy',
    'FaceVerificationMetrics'
]
