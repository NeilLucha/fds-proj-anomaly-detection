"""drift_anomaly.pipeline.tune
==============================

Threshold and hyperparameter tuning on validation data.

Design principles
-----------------
* **Strict temporal splits** — time series are NEVER shuffled.
* **Test-data isolation** — test labels are NEVER used during tuning.
* Walk-forward (expanding-window) cross-validation respects causality.
"""
from __future__ import annotations

import copy
import dataclasses
import logging
import time
from typing import Dict, List, Optional, Tuple, Type

import numpy as np

from ..types import DetectorConfig, DetectorResult, EvalMetrics, TimeSeries, TimeSeriesPoint
from ..detectors.base import BaseDetector
from .metrics import compute_metrics

logger = logging.getLogger(__name__)

# Default tau grid for threshold search.
_DEFAULT_TAU_GRID: List[float] = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


# ---------------------------------------------------------------------------
# Temporal splitting helpers
# ---------------------------------------------------------------------------

def _slice_series(series: TimeSeries, start: int, stop: int) -> TimeSeries:
    """Return a new TimeSeries containing points[start:stop]."""
    return TimeSeries(
        points=series.points[start:stop],
        name=series.name,
        metadata=dict(series.metadata),
    )


def _run_detector_from_prefix(
    DetectorClass: Type[BaseDetector],
    config: DetectorConfig,
    prefix: TimeSeries,
    scored: TimeSeries,
):
    """Warm a fresh detector on a causal prefix, returning outputs on *scored*."""
    detector = DetectorClass(config)
    detector.reset()
    start = time.perf_counter()
    for index, point in enumerate(prefix.points):
        detector.step(point.value, index)

    scores, alerts, drift_events, adaptations = [], [], [], []
    offset = len(prefix)
    for local_index, point in enumerate(scored.points):
        # Use a contiguous stream index. Detector implementations use t only
        # for outputs/telemetry; the fresh instance already owns prior state.
        out = detector.step(point.value, offset + local_index)
        scores.append(out.score)
        alerts.append(out.alert)
        if out.drift_detected:
            drift_events.append(local_index)
        if out.baseline_updated:
            adaptations.append(local_index)
    return DetectorResult(
        series_name=scored.name,
        variant=detector.name,
        scores=scores,
        alerts=alerts,
        drift_events=drift_events,
        adaptation_events=adaptations,
        runtime_s=time.perf_counter() - start,
        config=config,
    )


def temporal_split(
    series: TimeSeries,
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Tuple[TimeSeries, TimeSeries, TimeSeries]:
    """Split *series* into train / val / test temporally (no shuffling).

    The split is performed by index order:

    ``[  train  |  val  |  test  ]``

    Parameters
    ----------
    series : TimeSeries
        The full labelled series.
    val_fraction : float
        Fraction of points reserved for validation (default 0.2).
    test_fraction : float
        Fraction of points reserved for test (default 0.2).

    Returns
    -------
    train_series, val_series, test_series : tuple[TimeSeries, TimeSeries, TimeSeries]

    Raises
    ------
    ValueError
        If fractions are out of range or the series is empty.
    """
    if not (0.0 < val_fraction < 1.0):
        raise ValueError(f"val_fraction must be in (0, 1); got {val_fraction}.")
    if not (0.0 < test_fraction < 1.0):
        raise ValueError(f"test_fraction must be in (0, 1); got {test_fraction}.")
    if val_fraction + test_fraction >= 1.0:
        raise ValueError(
            "val_fraction + test_fraction must be < 1; "
            f"got {val_fraction + test_fraction}."
        )
    n = len(series)
    if n == 0:
        empty = TimeSeries(points=[], name=series.name, metadata=dict(series.metadata))
        return empty, empty, empty

    test_start = int(n * (1.0 - test_fraction))
    val_start = int(n * (1.0 - test_fraction - val_fraction))
    val_start = max(0, val_start)

    train = _slice_series(series, 0, val_start)
    val = _slice_series(series, val_start, test_start)
    test = _slice_series(series, test_start, n)

    logger.debug(
        "temporal_split: n=%d → train=%d, val=%d, test=%d",
        n, len(train), len(val), len(test),
    )
    return train, val, test


# ---------------------------------------------------------------------------
# Threshold grid search
# ---------------------------------------------------------------------------

def tune_threshold(
    DetectorClass: Type[BaseDetector],
    config: DetectorConfig,
    val_series: TimeSeries,
    tau_grid: Optional[List[float]] = None,
    metric: str = "f1",
    warmup_series: Optional[TimeSeries] = None,
) -> Tuple[float, dict]:
    """Grid-search over ``tau`` values on the validation series.

    Parameters
    ----------
    DetectorClass : Type[BaseDetector]
        The detector class to instantiate (e.g. ``VariantA``).
    config : DetectorConfig
        Base configuration; ``tau`` is overwritten for each grid point.
    val_series : TimeSeries
        Validation series **with ground-truth labels**.
        Test labels are NEVER used here.
    tau_grid : list[float] or None
        Values of ``tau`` to try.  Defaults to
        ``[1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]``.
    metric : str
        Optimisation target — one of ``"f1"``, ``"precision"``,
        ``"recall"``, ``"roc_auc"``, ``"pr_auc"`` (default ``"f1"``).

    Returns
    -------
    best_tau : float
        The ``tau`` value that maximises *metric* on the validation data.
    results : dict
        Mapping ``{tau: EvalMetrics}`` for every grid point evaluated.

    Notes
    -----
    If ``val_series`` is empty, the original ``config.tau`` is returned
    unchanged and ``results`` is empty.
    """
    if tau_grid is None:
        tau_grid = _DEFAULT_TAU_GRID

    if len(val_series) == 0:
        logger.warning("tune_threshold: val_series is empty; returning default tau.")
        return config.tau, {}

    valid_metrics = {"f1", "precision", "recall", "roc_auc", "pr_auc"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}; got '{metric}'.")

    results: Dict[float, EvalMetrics] = {}
    best_tau: float = tau_grid[0]
    best_score: float = -1.0

    for tau in tau_grid:
        # Create a fresh config copy with this tau.
        trial_config = dataclasses.replace(config, tau=tau)
        try:
            det_result = (
                _run_detector_from_prefix(DetectorClass, trial_config, warmup_series, val_series)
                if warmup_series is not None
                else DetectorClass(trial_config).run(val_series)
            )
            metrics = compute_metrics(det_result, val_series)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "tune_threshold: tau=%.2f raised %s; skipping.", tau, exc
            )
            continue

        results[tau] = metrics
        score = getattr(metrics, metric)


        logger.debug(
            "tune_threshold: tau=%.2f → %s=%.4f", tau, metric, score
        )

        if score > best_score:
            best_score = score
            best_tau = tau

    if not np.any(val_series.labels):
        logger.warning(
            "tune_threshold: validation series '%s' has no positive labels; "
            "F1 cannot identify a threshold (selected %.2f by grid tie order).",
            val_series.name,
            best_tau,
        )

    logger.info(
        "tune_threshold: best tau=%.2f (%s=%.4f) over %d grid points.",
        best_tau, metric, best_score, len(results),
    )
    return best_tau, results


# ---------------------------------------------------------------------------
# Walk-forward cross-validation
# ---------------------------------------------------------------------------

def walk_forward_validate(
    DetectorClass: Type[BaseDetector],
    config: DetectorConfig,
    series: TimeSeries,
    n_splits: int = 5,
    metric: str = "f1",
) -> dict:
    """Walk-forward cross-validation on *series*.

    For each fold ``k`` in ``[1 … n_splits]``:

    * **Train** on ``points[ : train_end_k ]``
      (the detector is re-initialised; it only *sees* the train prefix
      during run-time, so it naturally ignores future data).
    * **Validate** on ``points[ train_end_k : val_end_k ]``.

    No data ever flows backwards in time, and test/future labels are
    never touched.

    Parameters
    ----------
    DetectorClass : Type[BaseDetector]
        Detector class to evaluate.
    config : DetectorConfig
        Configuration (tau fixed for this call; tune separately if needed).
    series : TimeSeries
        The full labelled series (excluding the held-out test set).
    n_splits : int
        Number of walk-forward folds (default 5).
    metric : str
        Metric to report per fold.

    Returns
    -------
    dict with keys:
        ``"fold_scores"`` : list of per-fold metric values,
        ``"mean_{metric}"`` : float,
        ``"std_{metric}"``  : float,
        ``"fold_metrics"``  : list of :class:`EvalMetrics` per fold.
    """
    n = len(series)
    if n == 0:
        logger.warning("walk_forward_validate: series is empty.")
        return {"fold_scores": [], f"mean_{metric}": 0.0, f"std_{metric}": 0.0, "fold_metrics": []}

    if n_splits < 2:
        raise ValueError(f"n_splits must be ≥ 2; got {n_splits}.")

    valid_metrics = {"f1", "precision", "recall", "roc_auc", "pr_auc"}
    if metric not in valid_metrics:
        raise ValueError(f"metric must be one of {valid_metrics}; got '{metric}'.")

    # We need at least 2 points per fold.
    min_fold_size = max(2, n // (n_splits + 1))
    fold_scores: List[float] = []
    fold_metrics: List[EvalMetrics] = []

    for k in range(1, n_splits + 1):
        # Expanding training window; each validation window slides forward.
        train_end = int(n * k / (n_splits + 1))
        val_end = int(n * (k + 1) / (n_splits + 1))

        if train_end < 1 or val_end <= train_end:
            logger.debug("walk_forward_validate: fold %d is degenerate; skipping.", k)
            continue

        val_slice = _slice_series(series, train_end, val_end)
        if len(val_slice) == 0:
            logger.debug(
                "walk_forward_validate: fold %d validation slice is empty; skipping.", k
            )
            continue

        try:
            detector = DetectorClass(config)
            # Warm-up: feed the training portion without collecting output.
            train_slice = _slice_series(series, 0, train_end)
            for point in train_slice.points:
                detector.step(point.value)

            # Evaluate on the validation slice.
            det_result = detector.run(val_slice)
            metrics = compute_metrics(det_result, val_slice)
        except Exception as exc:  # noqa: BLE001
            logger.warning(
                "walk_forward_validate: fold %d raised %s; skipping.", k, exc
            )
            continue

        score = getattr(metrics, metric)
        fold_scores.append(score)
        fold_metrics.append(metrics)
        logger.debug(
            "walk_forward_validate: fold %d/%d → %s=%.4f", k, n_splits, metric, score
        )

    mean_score = float(np.mean(fold_scores)) if fold_scores else 0.0
    std_score = float(np.std(fold_scores)) if fold_scores else 0.0

    logger.info(
        "walk_forward_validate: %d folds, mean %s=%.4f ± %.4f",
        len(fold_scores), metric, mean_score, std_score,
    )

    return {
        "fold_scores": fold_scores,
        f"mean_{metric}": mean_score,
        f"std_{metric}": std_score,
        "fold_metrics": fold_metrics,
    }
