"""
tests/test_pipeline.py
======================

Unit tests for pipeline components:
- Temporal train/val/test splitting (strictly causal, non-shuffled)
- Threshold tuning with zero test-set leakage
- Point metrics & drift diagnostics
- evaluate_detector smoke execution
"""
from __future__ import annotations

import numpy as np
import pytest

from drift_anomaly.data.synthetic import SyntheticGenerator
from drift_anomaly.detectors.version_a_cautious_rolling import VersionA
from drift_anomaly.pipeline.metrics import compute_point_metrics, compute_drift_diagnostics
from drift_anomaly.pipeline.tune import temporal_split, tune_threshold, _run_detector_from_prefix
from drift_anomaly.pipeline.evaluate import evaluate_detector
from drift_anomaly.types import DetectorConfig, TimeSeriesPoint, TimeSeries


def test_temporal_split_order():
    """Verify temporal split maintains strict chronological order without shuffle."""
    pts = [TimeSeriesPoint(timestamp=float(i), value=float(i)) for i in range(100)]
    ts = TimeSeries(points=pts, name="split_test")
    train, val, test = temporal_split(ts, val_fraction=0.2, test_fraction=0.2)
    assert len(train) == 60
    assert len(val) == 20
    assert len(test) == 20
    # Chronological timestamps
    assert train.timestamps[-1] < val.timestamps[0]
    assert val.timestamps[-1] < test.timestamps[0]


def test_tune_threshold_no_test_leakage():
    """Tuning threshold uses only validation series labels."""
    gen = SyntheticGenerator(seed=42)
    val_series = gen.step_shift(n=300, shift_at=150, n_anomalies=10)
    cfg = DetectorConfig(window_size=40, candidate_size=15, tau=3.0)
    best_tau, results = tune_threshold(
        DetectorClass=VersionA,
        config=cfg,
        val_series=val_series,
        tau_grid=[2.0, 3.0, 4.0],
    )
    assert best_tau in [2.0, 3.0, 4.0]
    assert len(results) == 3


def test_metrics_all_correct():
    """Perfect predictions yield precision=1, recall=1, F1=1."""
    y_true = np.array([0, 1, 0, 1, 0, 1])
    y_pred = np.array([0, 1, 0, 1, 0, 1])
    scores = np.array([0.1, 4.0, 0.2, 5.0, 0.3, 6.0])
    res = compute_point_metrics(y_true, y_pred, scores)
    assert res["precision"] == 1.0
    assert res["recall"] == 1.0
    assert res["f1"] == 1.0
    assert res["roc_auc"] == 1.0


def test_metrics_all_wrong():
    """Zero recall and precision when no positive matches occur."""
    y_true = np.array([0, 1, 0, 1])
    y_pred = np.array([0, 0, 0, 0])
    scores = np.array([0.1, 0.2, 0.1, 0.2])
    res = compute_point_metrics(y_true, y_pred, scores)
    assert res["recall"] == 0.0
    assert res["precision"] == 0.0
    assert res["f1"] == 0.0


def test_evaluate_smoke():
    """End-to-end evaluation runs without error on small synthetic series."""
    gen = SyntheticGenerator(seed=42)
    test_ts = gen.stationary(n=100)
    cfg = DetectorConfig(window_size=30, tau=3.0)
    metrics_list = evaluate_detector(
        DetectorClass=VersionA,
        config=cfg,
        test_series=[test_ts],
    )
    assert len(metrics_list) == 1
    m = metrics_list[0]
    assert m.runtime_s >= 0.0
    assert 0.0 <= m.roc_auc <= 1.0


def test_prefix_warmup_scores_only_target_slice():
    """Causal prefix initializes the detector without leaking outputs into metrics."""
    from drift_anomaly.detectors.version_b_rolling_adwin import VersionB

    points = [TimeSeriesPoint(timestamp=float(i), value=float(i % 7)) for i in range(50)]
    series = TimeSeries(points=points, name="prefix")
    prefix = TimeSeries(points=points[:40], name="prefix")
    scored = TimeSeries(points=points[40:], name="prefix")
    result = _run_detector_from_prefix(VersionB, DetectorConfig(window_size=10), prefix, scored)
    assert len(result.scores) == len(scored)
    assert result.scores[0] > 0.0
