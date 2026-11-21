"""drift_anomaly.pipeline.metrics
================================

Evaluation metrics for anomaly detection.

Point-level metrics
-------------------
Precision, recall, F1 computed directly on binary labels.
ROC-AUC and PR-AUC computed on continuous anomaly scores.

Drift diagnostics
-----------------
drift trigger counts, false drift rate on stationary segments.
"""
from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np
from sklearn.metrics import (
    auc,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
)

from ..types import DetectorResult, EvalMetrics, TimeSeries

logger = logging.getLogger(__name__)

# Tolerance window (steps) within which a drift trigger is considered
# "near" a true change-point.
_CHANGE_POINT_TOLERANCE: int = 50


# ---------------------------------------------------------------------------
# Point-level classification metrics
# ---------------------------------------------------------------------------

def compute_point_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    scores: np.ndarray,
    zero_division: float = 0.0,
) -> dict:
    """Compute precision, recall, F1, ROC-AUC, and PR-AUC.

    Parameters
    ----------
    y_true : np.ndarray of shape (n,), dtype bool or int
        Ground-truth binary anomaly labels.
    y_pred : np.ndarray of shape (n,), dtype bool or int
        Predicted binary anomaly labels (threshold-based alerts).
    scores : np.ndarray of shape (n,), dtype float
        Continuous anomaly scores (larger ⟹ more anomalous).
    zero_division : float
        Value returned when a metric is undefined due to no positives.

    Returns
    -------
    dict with keys: precision, recall, f1, roc_auc, pr_auc
    """
    y_true = np.asarray(y_true, dtype=int)
    y_pred = np.asarray(y_pred, dtype=int)
    scores = np.asarray(scores, dtype=float)

    result: dict = {
        "precision": zero_division,
        "recall": zero_division,
        "f1": zero_division,
        "roc_auc": 0.5,
        "pr_auc": zero_division,
    }

    if len(y_true) == 0:
        logger.warning("compute_point_metrics: empty arrays supplied.")
        return result

    # Replace NaN scores with 0 so sklearn doesn't crash.
    nan_mask = ~np.isfinite(scores)
    if nan_mask.any():
        logger.warning(
            "compute_point_metrics: %d non-finite scores replaced with 0.",
            nan_mask.sum(),
        )
        scores = scores.copy()
        scores[nan_mask] = 0.0

    n_pos = int(y_true.sum())
    n_neg = int((1 - y_true).sum())
    all_same_true = n_pos == 0 or n_neg == 0
    all_same_pred = y_pred.sum() == 0 or y_pred.sum() == len(y_pred)

    # ----- Precision / Recall / F1 ----------------------------------------
    if not all_same_pred or n_pos > 0:
        result["precision"] = float(
            precision_score(y_true, y_pred, zero_division=zero_division)
        )
        result["recall"] = float(
            recall_score(y_true, y_pred, zero_division=zero_division)
        )
        result["f1"] = float(
            f1_score(y_true, y_pred, zero_division=zero_division)
        )
    else:
        # All predictions are the same class and there are no positives.
        result["precision"] = zero_division
        result["recall"] = zero_division
        result["f1"] = zero_division

    # ----- ROC-AUC --------------------------------------------------------
    if all_same_true:
        # Only one class present in y_true: AUC is undefined.
        logger.warning(
            "compute_point_metrics: only one class in y_true; "
            "ROC-AUC set to 0.5."
        )
        result["roc_auc"] = 0.5
    else:
        try:
            result["roc_auc"] = float(roc_auc_score(y_true, scores))
        except ValueError as exc:
            logger.warning("roc_auc_score failed (%s); defaulting to 0.5.", exc)
            result["roc_auc"] = 0.5

    # ----- PR-AUC ---------------------------------------------------------
    if n_pos == 0:
        logger.warning(
            "compute_point_metrics: no positive labels; PR-AUC set to 0."
        )
        result["pr_auc"] = 0.0
    else:
        try:
            precision_vals, recall_vals, _ = precision_recall_curve(y_true, scores)
            result["pr_auc"] = float(auc(recall_vals, precision_vals))
        except ValueError as exc:
            logger.warning("precision_recall_curve failed (%s); PR-AUC set to 0.", exc)
            result["pr_auc"] = 0.0

    # A no-positive slice has undefined ROC/PR ranking quality. Keep the
    # legacy neutral ROC placeholder for API compatibility, while making the
    # condition explicit in logs and split-count fields in benchmark reports.

    return result


# ---------------------------------------------------------------------------
# Drift diagnostics
# ---------------------------------------------------------------------------

def compute_drift_diagnostics(
    drift_events: List[int],
    is_anomaly: np.ndarray,
    is_change_point: np.ndarray,
    adaptation_accepted: int,
    adaptation_rejected: int,
    stationary_mask: Optional[np.ndarray] = None,
) -> dict:
    """Compute drift diagnostics.

    Parameters
    ----------
    drift_events : list[int]
        Time indices at which the detector fired a drift signal.
    is_anomaly : np.ndarray of shape (n,), dtype bool
        Ground-truth anomaly labels.
    is_change_point : np.ndarray of shape (n,), dtype bool
        Ground-truth change-point labels.
    adaptation_accepted : int
        Count of candidate windows accepted as new baseline.
    adaptation_rejected : int
        Count of candidate windows rejected by the contamination gate.
    stationary_mask : np.ndarray of shape (n,) or None
        Boolean mask where ``True`` indicates a stationary (non-change)
        segment.  Used to compute *false drift rate on stationary segments*.
        If ``None``, it is derived as ``~is_change_point`` after padding.

    Returns
    -------
    dict with keys:
        n_drift_triggers, false_drift_rate,
        adaptation_accepted, adaptation_rejected
    """
    is_change_point = np.asarray(is_change_point, dtype=bool)
    n = len(is_change_point)

    n_drift_triggers = len(drift_events)

    if n_drift_triggers == 0:
        return {
            "n_drift_triggers": 0,
            "false_drift_rate": 0.0,
            "adaptation_accepted": adaptation_accepted,
            "adaptation_rejected": adaptation_rejected,
        }

    # Build a boolean array: True at positions within tolerance of a change-point.
    near_change_point = np.zeros(n, dtype=bool)
    cp_indices = np.where(is_change_point)[0]
    for cp in cp_indices:
        lo = max(0, cp - _CHANGE_POINT_TOLERANCE)
        hi = min(n, cp + _CHANGE_POINT_TOLERANCE + 1)
        near_change_point[lo:hi] = True

    # Count drift triggers that are NOT near any change-point.
    false_triggers = 0
    for t in drift_events:
        if 0 <= t < n:
            if not near_change_point[t]:
                false_triggers += 1
        else:
            # Out of range ⟹ definitely false.
            false_triggers += 1

    false_drift_rate = false_triggers / n_drift_triggers

    return {
        "n_drift_triggers": n_drift_triggers,
        "false_drift_rate": false_drift_rate,
        "adaptation_accepted": adaptation_accepted,
        "adaptation_rejected": adaptation_rejected,
    }


# ---------------------------------------------------------------------------
# Combined metric computation
# ---------------------------------------------------------------------------

def compute_metrics(result: DetectorResult, series: TimeSeries) -> EvalMetrics:
    """Combine point metrics + drift diagnostics into an :class:`EvalMetrics`.

    Parameters
    ----------
    result : DetectorResult
        Output from running a detector on *series*.
    series : TimeSeries
        The labelled time series that was evaluated.

    Returns
    -------
    EvalMetrics
    """
    # Extract arrays whether labels are properties or methods
    labels_val = series.labels() if callable(getattr(series, "labels", None)) else series.labels
    cp_val = series.change_points() if callable(getattr(series, "change_points", None)) else series.change_points
    y_true = np.array(labels_val, dtype=int)
    y_pred = np.array(result.alerts, dtype=int)
    scores = np.array(result.scores, dtype=float)
    is_change_point = np.array(cp_val, dtype=bool)

    # Align lengths (detector might not emit outputs for warm-up steps).
    n = min(len(y_true), len(y_pred), len(scores))
    if n == 0:
        logger.warning(
            "compute_metrics: zero-length overlap for series '%s'.",
            series.name,
        )
        return EvalMetrics(
            precision=0.0,
            recall=0.0,
            f1=0.0,
            roc_auc=0.5,
            pr_auc=0.0,
            n_drift_triggers=len(result.drift_events),
            false_drift_rate=0.0,
            adaptation_accepted=len(result.adaptation_events),
            adaptation_rejected=0,
            runtime_s=result.runtime_s,
        )

    y_true = y_true[:n]
    y_pred = y_pred[:n]
    scores = scores[:n]
    is_change_point = is_change_point[:n]

    point = compute_point_metrics(y_true, y_pred, scores)

    # Count accepted vs rejected from adaptation_events vs drift_events.
    adaptation_accepted = len(result.adaptation_events)
    adaptation_rejected = max(0, len(result.drift_events) - adaptation_accepted)

    drift_diag = compute_drift_diagnostics(
        drift_events=result.drift_events,
        is_anomaly=y_true.astype(bool),
        is_change_point=is_change_point,
        adaptation_accepted=adaptation_accepted,
        adaptation_rejected=adaptation_rejected,
    )

    return EvalMetrics(
        precision=point["precision"],
        recall=point["recall"],
        f1=point["f1"],
        roc_auc=point["roc_auc"],
        pr_auc=point["pr_auc"],
        n_drift_triggers=drift_diag["n_drift_triggers"],
        false_drift_rate=drift_diag["false_drift_rate"],
        adaptation_accepted=drift_diag["adaptation_accepted"],
        adaptation_rejected=drift_diag["adaptation_rejected"],
        runtime_s=result.runtime_s,
    )


def event_window_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    alert_gap: int = 5,
) -> dict:
    """Compute event/window precision, recall, F1 and false alert clusters.

    A labeled window is a contiguous positive run. An event is detected when
    any alert lands in that run. Predicted positives outside all true windows
    are grouped when separated by at most ``alert_gap`` points.
    """
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    if truth.shape != pred.shape:
        raise ValueError("y_true and y_pred must have the same shape")
    true_events = _boolean_runs(truth)
    alert_points = np.flatnonzero(pred)
    false_alert_points = [int(i) for i in alert_points if not truth[i]]
    false_clusters = 0
    previous = None
    for index in false_alert_points:
        if previous is None or index - previous > alert_gap + 1:
            false_clusters += 1
        previous = index
    tp = sum(bool(pred[start:end].any()) for start, end in true_events)
    fn = len(true_events) - tp
    fp = false_clusters
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0
    # For a zero-event series, summarize the raw false alert rate and alert
    # clusters; event F1 has no positive-event denominator and is not pooled.
    return {
        "event_tp": tp,
        "event_fp": fp,
        "event_fn": fn,
        "event_precision": precision,
        "event_recall": recall,
        "event_f1": f1,
        "false_alert_clusters": fp,
        "zero_event_fpr": len(false_alert_points) / len(truth) if not true_events and len(truth) else 0.0,
        "false_alert_count": len(false_alert_points),
    }


def _boolean_runs(mask: np.ndarray) -> list[tuple[int, int]]:
    padded = np.pad(np.asarray(mask, dtype=bool).astype(np.int8), (1, 1))
    starts = np.flatnonzero(np.diff(padded) == 1)
    ends = np.flatnonzero(np.diff(padded) == -1)
    return list(zip(starts.tolist(), ends.tolist()))


def pooled_point_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict:
    """Pooled pointwise precision/recall/F1 and zero-event false alarm rate."""
    truth = np.asarray(y_true, dtype=bool)
    pred = np.asarray(y_pred, dtype=bool)
    tp = int(np.sum(truth & pred))
    fp = int(np.sum(~truth & pred))
    fn = int(np.sum(truth & ~pred))
    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0
    return {
        "precision": precision,
        "recall": recall,
        "f1": 2 * precision * recall / (precision + recall) if precision + recall else 0.0,
        "false_alert_count": fp,
        "zero_event_fpr": fp / max(1, int(np.sum(~truth))),
    }


def evaluate_pointwise_macro(metrics_by_series: dict[str, tuple[np.ndarray, np.ndarray]]) -> dict:
    """Pooled micro scores and positive-series macro F1 from raw predictions."""
    true_all, pred_all, f1_positive, zero_fprs, zero_false = [], [], [], [], 0
    for truth_raw, pred_raw in metrics_by_series.values():
        truth = np.asarray(truth_raw, dtype=bool)
        pred = np.asarray(pred_raw, dtype=bool)
        true_all.extend(truth.tolist())
        pred_all.extend(pred.tolist())
        if truth.any():
            tp, fp, fn = int(np.sum(truth & pred)), int(np.sum(~truth & pred)), int(np.sum(truth & ~pred))
            p = tp / (tp + fp) if tp + fp else 0.0
            r = tp / (tp + fn) if tp + fn else 0.0
            f1_positive.append(2 * p * r / (p + r) if p + r else 0.0)
        else:
            zero_fprs.append(float(np.mean(pred)) if len(pred) else 0.0)
            zero_false += int(np.sum(pred))
    return {
        "micro": pooled_point_metrics(np.asarray(true_all), np.asarray(pred_all)),
        "macro_f1_positive_series": float(np.mean(f1_positive)) if f1_positive else 0.0,
        "zero_event_series_mean_fpr": float(np.mean(zero_fprs)) if zero_fprs else 0.0,
        "zero_event_series_false_alerts": zero_false,
    }
