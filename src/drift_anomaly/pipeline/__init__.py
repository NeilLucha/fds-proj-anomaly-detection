"""drift_anomaly.pipeline
=======================

Public API for the pipeline layer.

Exports
-------
tune_threshold      : Grid-search over tau on validation data.
evaluate_detector   : Run a detector on test series and return EvalMetrics.
compute_metrics     : Compute EvalMetrics from a DetectorResult + TimeSeries.
generate_report     : Write JSON / CSV / plot reports to disk.
"""

from .tune import tune_threshold, temporal_split, walk_forward_validate
from .evaluate import evaluate_detector, evaluate_all_variants, run_full_evaluation
from .metrics import compute_metrics, compute_point_metrics, compute_drift_diagnostics
from .reporting import generate_report, save_metrics_json, save_metrics_csv

__all__ = [
    # tuning
    "tune_threshold",
    "temporal_split",
    "walk_forward_validate",
    # evaluation
    "evaluate_detector",
    "evaluate_all_variants",
    "run_full_evaluation",
    # metrics
    "compute_metrics",
    "compute_point_metrics",
    "compute_drift_diagnostics",
    # reporting
    "generate_report",
    "save_metrics_json",
    "save_metrics_csv",
]
