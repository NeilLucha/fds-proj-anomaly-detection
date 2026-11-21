"""drift_anomaly.pipeline.evaluate
==================================

Evaluation pipeline: run detectors on test sets and compute metrics.

The entry-points are:

* :func:`evaluate_detector`     — single detector, list of test series.
* :func:`evaluate_all_variants` — all three variants + baselines.
* :func:`run_full_evaluation`   — full pipeline: tune → evaluate → save.
"""
from __future__ import annotations

import dataclasses
import logging
import time
from pathlib import Path
from typing import Dict, List, Optional, Type

import numpy as np

from ..detectors.base import BaseDetector
from ..types import DetectorConfig, DetectorResult, EvalMetrics, TimeSeries
from .metrics import aggregate_literature_metrics, compute_metrics, literature_series_metrics
from .tune import temporal_split, tune_threshold

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Single detector evaluation
# ---------------------------------------------------------------------------

def evaluate_detector(
    DetectorClass: Type[BaseDetector],
    config: DetectorConfig,
    test_series: List[TimeSeries],
    series_names: Optional[List[str]] = None,
) -> List[EvalMetrics]:
    """Run *DetectorClass* on each series in *test_series* and return metrics.

    Parameters
    ----------
    DetectorClass : Type[BaseDetector]
        Detector class to instantiate and run.
    config : DetectorConfig
        Configuration for the detector.
    test_series : list[TimeSeries]
        Test series to evaluate on.  Labels are used **only for metric
        computation**, never fed back to the detector.
    series_names : list[str] or None
        Human-readable names for each series (used in log messages).
        Defaults to the ``name`` attribute of each series.

    Returns
    -------
    list[EvalMetrics]
        One :class:`EvalMetrics` per series, in the same order as
        *test_series*.  Failed series yield zero-valued metrics.
    """
    if series_names is None:
        series_names = [s.name for s in test_series]
    elif len(series_names) != len(test_series):
        raise ValueError(
            "series_names and test_series must have the same length; "
            f"got {len(series_names)} vs {len(test_series)}."
        )

    all_metrics: List[EvalMetrics] = []

    for name, series in zip(series_names, test_series):
        if len(series) == 0:
            logger.warning(
                "evaluate_detector: series '%s' is empty; returning zero metrics.", name
            )
            all_metrics.append(
                EvalMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    roc_auc=0.5,
                    pr_auc=0.0,
                    n_drift_triggers=0,
                    false_drift_rate=0.0,
                    adaptation_accepted=0,
                    adaptation_rejected=0,
                    runtime_s=0.0,
                )
            )
            continue

        try:
            t0 = time.perf_counter()
            detector = DetectorClass(config)
            det_result: DetectorResult = detector.run(series)
            runtime = time.perf_counter() - t0
            # Patch runtime in case the detector's own timer differs.
            det_result = dataclasses.replace(det_result, runtime_s=runtime)

            metrics = compute_metrics(det_result, series)
            logger.info(
                "evaluate_detector: '%s' → F1=%.4f, ROC-AUC=%.4f, "
                "drift_triggers=%d, runtime=%.3fs",
                name,
                metrics.f1,
                metrics.roc_auc,
                metrics.n_drift_triggers,
                metrics.runtime_s,
            )
            all_metrics.append(metrics)

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "evaluate_detector: series '%s' failed with %s; "
                "returning zero metrics.",
                name, exc, exc_info=True,
            )
            all_metrics.append(
                EvalMetrics(
                    precision=0.0,
                    recall=0.0,
                    f1=0.0,
                    roc_auc=0.5,
                    pr_auc=0.0,
                    n_drift_triggers=0,
                    false_drift_rate=0.0,
                    adaptation_accepted=0,
                    adaptation_rejected=0,
                    runtime_s=0.0,
                )
            )

    return all_metrics


# ---------------------------------------------------------------------------
# Multi-variant evaluation
# ---------------------------------------------------------------------------

def evaluate_all_variants(
    series_list: List[TimeSeries],
    configs: Dict[str, DetectorConfig],
    variant_classes: Dict[str, Type[BaseDetector]],
) -> Dict[str, List[EvalMetrics]]:
    """Evaluate all variants (and optional baselines) on *series_list*.

    Parameters
    ----------
    series_list : list[TimeSeries]
        Test series.
    configs : dict[str, DetectorConfig]
        Mapping ``{variant_name: config}`` — one entry per variant.
    variant_classes : dict[str, Type[BaseDetector]]
        Mapping ``{variant_name: DetectorClass}`` — must have the same
        keys as *configs*.

    Returns
    -------
    dict[str, list[EvalMetrics]]
        One list per variant, each list containing one
        :class:`EvalMetrics` per series.

    Notes
    -----
    Keys in *configs* and *variant_classes* must match.  Extra keys in
    either mapping are silently ignored.
    """
    common_keys = set(configs.keys()) & set(variant_classes.keys())
    if not common_keys:
        logger.warning("evaluate_all_variants: no common keys between configs and variant_classes.")
        return {}

    results: Dict[str, List[EvalMetrics]] = {}

    for variant_name in sorted(common_keys):
        logger.info("evaluate_all_variants: evaluating variant '%s'.", variant_name)
        cls = variant_classes[variant_name]
        cfg = configs[variant_name]
        metrics_list = evaluate_detector(cls, cfg, series_list)
        results[variant_name] = metrics_list

    return results


def evaluate_literature_results(
    results_by_series: Dict[str, tuple[DetectorResult, TimeSeries]],
) -> dict:
    """Compute optional literature-aligned metrics from already-scored output.

    Detector predictions and continuous scores are consumed as-is; no model
    is rerun and no strict metric or detector behavior is modified.
    """
    per_series = {}
    arrays = {}
    for name, (result, series) in results_by_series.items():
        truth = np.asarray(series.labels, dtype=bool)
        pred = np.asarray(result.alerts, dtype=bool)
        scores = np.asarray(result.scores, dtype=float)
        n = min(len(truth), len(pred), len(scores))
        truth, pred, scores = truth[:n], pred[:n], scores[:n]
        per_series[name] = literature_series_metrics(
            truth, pred, scores
        )
        arrays[name] = (truth, pred, scores)
    return {
        "per_series": per_series,
        "aggregate": aggregate_literature_metrics(arrays),
    }


# ---------------------------------------------------------------------------
# Full pipeline
# ---------------------------------------------------------------------------

def run_full_evaluation(
    series_list: List[TimeSeries],
    config: DetectorConfig,
    output_dir: str = "outputs/",
) -> Dict:
    """Full evaluation pipeline: tune on val, evaluate on test, save results.

    For each series in *series_list*:
    1. Temporally split into train / val / test.
    2. Tune ``tau`` on the validation split.
    3. Evaluate the tuned detector on the test split.
    4. Collect metrics.

    After all series are processed, results are written to *output_dir*
    via :func:`~drift_anomaly.pipeline.reporting.generate_report`.

    Parameters
    ----------
    series_list : list[TimeSeries]
        Full labelled series.
    config : DetectorConfig
        Base configuration; ``tau`` may be overridden per-series.
    output_dir : str
        Directory where JSON / CSV / plots are written.

    Returns
    -------
    dict with keys:
        ``"metrics_by_series"`` : dict[str, EvalMetrics],
        ``"best_taus"``         : dict[str, float],
        ``"output_dir"``        : str.

    Notes
    -----
    Import is deferred to avoid circular imports when ``reporting`` or
    ``detectors`` are still being loaded.
    """
    # Deferred imports to avoid circular dependency at module load time.
    from ..detectors import VariantA  # type: ignore[attr-defined]
    from .reporting import generate_report

    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)

    metrics_by_series: Dict[str, EvalMetrics] = {}
    best_taus: Dict[str, float] = {}

    for series in series_list:
        name = series.name
        if len(series) == 0:
            logger.warning("run_full_evaluation: series '%s' is empty; skipping.", name)
            continue

        logger.info("run_full_evaluation: processing series '%s'.", name)

        try:
            train, val, test = temporal_split(series)

            # Tune on validation data only.
            best_tau, _tune_results = tune_threshold(VariantA, config, val)
            best_taus[name] = best_tau

            # Build tuned config.
            tuned_config = dataclasses.replace(config, tau=best_tau)

            # Evaluate on test.
            test_metrics_list = evaluate_detector(VariantA, tuned_config, [test])
            if test_metrics_list:
                metrics_by_series[name] = test_metrics_list[0]

        except Exception as exc:  # noqa: BLE001
            logger.error(
                "run_full_evaluation: series '%s' failed with %s.", name, exc, exc_info=True
            )
            continue

    # Generate report (best-effort; non-fatal if reporting fails).
    if metrics_by_series:
        try:
            generate_report(
                metrics_by_variant={"VariantA": list(metrics_by_series.values())},
                output_dir=out_path,
                make_plots=True,
            )
        except Exception as exc:  # noqa: BLE001
            logger.error("run_full_evaluation: reporting failed: %s", exc, exc_info=True)

    return {
        "metrics_by_series": metrics_by_series,
        "best_taus": best_taus,
        "output_dir": str(out_path),
    }
