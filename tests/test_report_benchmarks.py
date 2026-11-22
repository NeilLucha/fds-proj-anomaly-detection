from __future__ import annotations

from types import SimpleNamespace

import numpy as np

from scripts.run_report_benchmarks import (
    EWMABaseline,
    LagIForestBaseline,
    RiverHSTBaseline,
    score_baseline_on_split,
    tune_score_threshold,
)
from drift_anomaly.types import TimeSeries, TimeSeriesPoint


def _series(values, labels=None, name="test"):
    labels = labels or [False] * len(values)
    return TimeSeries(
        points=[TimeSeriesPoint(float(i), float(x), bool(labels[i])) for i, x in enumerate(values)],
        name=name,
    )


def test_ewma_scores_current_value_before_state_update():
    model = EWMABaseline(alpha=0.1, variance_floor=0.1)
    model.fit([0.0] * 20)
    scores, _ = model.score_stream([5.0, 5.0])
    assert scores[0] > scores[1]


def test_river_hst_scores_before_learning_every_observation():
    trace = []

    class FakeHST:
        def __init__(self, **kwargs):
            self.count = 0

        def score_one(self, x):
            trace.append("score")
            return float(self.count)

        def learn_one(self, x):
            trace.append("learn")
            self.count += 1

    baseline = RiverHSTBaseline(SimpleNamespace(HalfSpaceTrees=FakeHST))
    baseline.fit([0.0, 1.0, 2.0])
    trace.clear()
    scores, _ = baseline.score_stream([3.0, 4.0])
    assert trace == ["score", "learn", "score", "learn"]
    assert scores.tolist() == [3.0, 4.0]


def test_lag_iforest_uses_only_prior_values_and_runs_validation_test():
    baseline = LagIForestBaseline(lookback=3, n_estimators=10)
    assert baseline._window([1.0, 2.0, 3.0]).tolist() == [3.0, 2.0, 1.0]
    train = _series(np.sin(np.arange(40) / 3))
    validation = _series(np.sin(np.arange(40, 50) / 3), [False] * 9 + [True])
    test = _series(np.sin(np.arange(50, 60) / 3), [False] * 10)
    result = score_baseline_on_split(lambda: LagIForestBaseline(lookback=3, n_estimators=10), train, validation, test)
    assert len(result["scores"]) == len(test)
    assert np.isfinite(result["scores"]).all()
    assert np.isfinite(result["threshold"])


def test_score_threshold_tuning_uses_validation_labels_and_scores():
    y = np.array([0, 0, 1, 1], dtype=bool)
    scores = np.array([0.1, 0.2, 0.8, 0.9])
    threshold, details = tune_score_threshold(y, scores)
    assert threshold <= 0.8
    assert details["validation_f1"] == 1.0
    assert details["validation_negative_fpr"] <= details["fpr_cap"]


def test_report_runner_fails_clearly_if_river_is_missing(monkeypatch):
    import sys
    import pytest
    from scripts.run_report_benchmarks import require_river

    monkeypatch.setitem(sys.modules, "river", None)
    with pytest.raises(RuntimeError, match="pip install river"):
        require_river()
