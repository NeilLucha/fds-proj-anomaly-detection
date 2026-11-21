#!/usr/bin/env python3
"""
scripts/run_real_benchmarks.py
==============================

End-to-end training and evaluation script for REAL datasets (NAB and Yahoo S5).

Pipeline stages:
1. Data verification & automatic download (NAB from GitHub; Yahoo S5 check).
2. Temporal train / validation / test splits (strict chronological causality, no shuffle).
3. Leak-free threshold (tau) tuning on validation split (never uses test labels).
4. Out-of-sample evaluation across A, B, C, D, E, E2, and LSTM on held-out test series.
5. Telemetry and drift diagnostics (triggers, adaptations accepted/rejected, runtimes).
6. Generation of structured reports (JSON, CSV) and comparative visual plots.
"""
from __future__ import annotations

import argparse
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
from tqdm.auto import tqdm

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drift_anomaly.data.nab import NABLoader
from drift_anomaly.data.yahoo_s5 import YahooS5Loader
from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2
from drift_anomaly.detectors.lstm import LSTMAnomalyDetector
from drift_anomaly.pipeline.evaluate import evaluate_detector
from drift_anomaly.pipeline.metrics import (
    aggregate_literature_metrics,
    compute_metrics,
    event_window_metrics,
    evaluate_pointwise_macro,
    literature_series_metrics,
)
from drift_anomaly.pipeline.reporting import (
    plot_metrics_comparison,
    plot_series_with_anomalies,
    save_metrics_csv,
    save_metrics_json,
)
from drift_anomaly.pipeline.tune import temporal_split, tune_threshold, _run_detector_from_prefix
from drift_anomaly.types import DetectorConfig, EvalMetrics, TimeSeries

logging.basicConfig(
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    level=logging.WARNING,
)
logger = logging.getLogger("run_real_benchmarks")


VARIANTS = {
    "A": VersionA,
    "B": VersionB,
    "C": VersionC,
    "D": VersionD,
}

LSTM_SETTINGS = {"lookback": 24, "hidden_size": 16, "epochs": 5, "learning_rate": 0.01, "seed": 42}


def tune_pooled_thresholds(series_list, base_config, tau_grid):
    """Choose one tau per variant by pooled F1 on validation series."""
    from sklearn.model_selection import train_test_split

    positive = [s for s in series_list if sum(s.labels) > 0]
    negative = [s for s in series_list if sum(s.labels) == 0]
    if len(positive) >= 2:
        _, val_pos = train_test_split(positive, test_size=0.3, random_state=42)
    else:
        val_pos = positive
    _, val_neg = train_test_split(negative, test_size=0.3, random_state=42) if len(negative) >= 2 else (negative, [])
    validation = val_pos + val_neg
    if not validation:
        validation = series_list

    best_taus = {}
    for name, cls in tqdm(VARIANTS.items(), desc="Pooled threshold tuning", leave=True):
        best_tau, best_f1 = tau_grid[0], -1.0
        for tau in tqdm(tau_grid, desc=f"Tune {name} tau", leave=False):
            truth, pred = [], []
            cfg = dataclasses.replace(base_config, tau=tau)
            for series in validation:
                res = cls(cfg).run(series)
                truth.extend(series.labels)
                pred.extend(res.alerts)
            from drift_anomaly.pipeline.metrics import pooled_point_metrics
            metrics = pooled_point_metrics(np.asarray(truth), np.asarray(pred))
            if metrics["f1"] > best_f1:
                best_tau, best_f1 = tau, metrics["f1"]
        best_taus[name] = best_tau
    calibration_start = time.perf_counter()
    pooled_validation_values = np.concatenate([
        np.asarray(s.values, dtype=float) for s in validation
    ]) if validation else np.asarray([], dtype=float)
    calibration_model = LSTMAnomalyDetector(**LSTM_SETTINGS)
    calibration_model.fit(pooled_validation_values)
    calibration_residuals = []
    for series in validation:
        scored = calibration_model.score(series.values, context=[])
        calibration_residuals.extend(
            float(score) for score, label in zip(scored.scores, series.labels) if not label
        )
    best_taus["LSTM"] = calibration_model.threshold_from(calibration_residuals, target_fpr=0.01)
    lstm_calibration = {
        "training_time_s": time.perf_counter() - calibration_start,
        "normal_residual_count": len(calibration_residuals),
        "model": calibration_model,
    }
    validation_ids = {id(s) for s in validation}
    test_series = [s for s in series_list if id(s) not in validation_ids]
    return best_taus, validation, test_series, lstm_calibration


def validate_e_ablation(series_list, base_config, detector_cls=VersionE, dual_tier=False):
    """Validation-only grid for E or E2; never reads a test suffix label."""
    from drift_anomaly.pipeline.metrics import pooled_point_metrics

    # Bound ablation cost while preserving both error classes.
    positive = [s for s in series_list if sum(s.labels) > 0][:10]
    controls = [s for s in series_list if not sum(s.labels)][:5]
    series_list = positive + controls
    candidates = []
    parameter_combinations = [
        (floor, lag, auto_lag, persistence, low, high)
        for floor in (0.10, 0.50)
        for lag, auto_lag in ((1, False), (24, True))
        for persistence in ((2, 3) if dual_tier else (1, 2))
        for low, high in ((2.0, 5.0), (3.0, 8.0))
    ]
    for floor, lag, auto_lag, persistence, low, high in tqdm(
        parameter_combinations, desc=f"{detector_cls.__name__} validation ablation"
    ):
        params = {
            "e_scale_floor_ratio": floor,
            "e_seasonal_lag": lag,
            "e_auto_lag": auto_lag,
        }
        if dual_tier:
            params.update({
                "e2_persistence_k": persistence,
                "e2_tau_moderate": low,
                "e2_tau_extreme": high,
            })
        else:
            params.update({
                "e_persistence_k": persistence,
                "e_tau_low": low,
                "e_tau_high": high,
            })
        cfg = dataclasses.replace(base_config, **params)
        truth, pred = [], []
        positive_recalls, positive_event_f1, zero_fprs = [], [], []
        for series in series_list:
            train, val, _ = temporal_split(series, 0.2, 0.2)
            val = TimeSeries(points=val.points[: min(600, len(val))], name=series.name)
            context = TimeSeries(points=train.points, name=series.name)
            result = _run_detector_from_prefix(detector_cls, cfg, context, val)
            y, p = np.asarray(val.labels, bool), np.asarray(result.alerts, bool)
            truth.extend(y.tolist())
            pred.extend(p.tolist())
            if y.any():
                positive_recalls.append(float(np.sum(y & p) / np.sum(y)))
                positive_event_f1.append(event_window_metrics(y, p)["event_f1"])
            else:
                zero_fprs.append(float(np.mean(p)) if len(p) else 0.0)
        pooled = pooled_point_metrics(np.asarray(truth), np.asarray(pred))
        # Select using balanced validation objective; explicitly
        # penalize false alarms without test data.
        macro_recall = float(np.mean(positive_recalls)) if positive_recalls else 0.0
        macro_fpr = float(np.mean(zero_fprs)) if zero_fprs else 0.0
        macro_event_f1 = float(np.mean(positive_event_f1)) if positive_event_f1 else 0.0
        if dual_tier:
            objective = 0.5 * pooled["f1"] + 0.5 * macro_event_f1
            fpr_cap = base_config.e2_validation_zero_event_fpr_cap
            eligible = macro_fpr <= fpr_cap if zero_fprs else True
        else:
            objective = pooled["f1"] + 0.15 * macro_recall - 0.20 * macro_fpr
            fpr_cap = None
            eligible = True
        candidates.append({
            "config": params,
            "point_f1": pooled["f1"],
            "positive_macro_event_f1": macro_event_f1,
            "positive_macro_recall": macro_recall,
            "zero_event_macro_fpr": macro_fpr,
            "zero_event_fpr_cap": fpr_cap,
            "fpr_cap_satisfied": eligible,
            "objective": objective,
        })
    eligible_candidates = [item for item in candidates if item["fpr_cap_satisfied"]]
    if dual_tier and not eligible_candidates:
        # Report a feasible fallback rather than selecting an over-triggering
        # config as though it had met the requested FPR constraint.
        candidates.sort(key=lambda x: (x["zero_event_macro_fpr"], -x["objective"]))
        selected = candidates[0]
        selected["fpr_cap_fallback"] = True
    else:
        eligible_candidates.sort(key=lambda x: x["objective"], reverse=True)
        selected = eligible_candidates[0]
    cfg = dataclasses.replace(base_config, **selected["config"])
    error_cases = []
    for series in series_list:
        train, val, _ = temporal_split(series, 0.2, 0.2)
        val = TimeSeries(points=val.points[: min(600, len(val))], name=series.name)
        result = _run_detector_from_prefix(
            detector_cls, cfg, TimeSeries(points=train.points, name=series.name), val
        )
        truth, pred = np.asarray(val.labels, bool), np.asarray(result.alerts, bool)
        ev = event_window_metrics(truth, pred)
        if ev["event_fn"] or (not truth.any() and pred.any()):
            error_cases.append({
                "series": series.name,
                "validation_positive_points": int(truth.sum()),
                "missed_events": ev["event_fn"],
                "false_alert_clusters": ev["event_fp"],
                "false_alert_points": int(np.sum(pred & ~truth)),
                "validation_alerts": int(pred.sum()),
            })
    selected["validation_error_cases"] = error_cases
    return selected, candidates


def inspect_e2_validation_score_distributions(series_list, config):
    """Summarize E2 validation scores for single spikes, wide events and controls."""
    def runs(mask):
        padded = np.pad(np.asarray(mask, dtype=np.int8), (1, 1))
        starts = np.flatnonzero(np.diff(padded) == 1)
        ends = np.flatnonzero(np.diff(padded) == -1)
        return list(zip(starts.tolist(), ends.tolist()))

    val_data = []
    singles, wide, controls = [], [], []
    for series in series_list:
        train, val, _ = temporal_split(series, 0.2, 0.2)
        if not len(val):
            continue
        val = TimeSeries(points=val.points[: min(600, len(val))], name=series.name)
        event_runs = runs(val.labels)
        if not event_runs:
            controls.append(series.name)
        if any(end - start == 1 for start, end in event_runs):
            singles.append(series.name)
        if any(end - start >= 5 for start, end in event_runs):
            wide.append(series.name)
        val_data.append((series, train, val, event_runs))

    selected_names = set(singles[:20] + wide[:20] + controls[:10])
    spike_scores, wide_interior_scores, control_cluster_peaks = [], [], []
    scored_series = 0
    for series, train, val, event_runs in val_data:
        if series.name not in selected_names:
            continue
        result = _run_detector_from_prefix(
            VersionE2, config, train, val
        )
        scores = np.asarray(result.scores, dtype=float)
        alerts = np.asarray(result.alerts, dtype=bool)
        truth = np.asarray(val.labels, dtype=bool)
        scored_series += 1
        if not truth.any():
            control_cluster_peaks.extend(
                float(np.max(scores[start:end]))
                for start, end in runs(alerts)
                if end > start
            )
        for start, end in event_runs:
            if end - start == 1 and series.name in singles[:20]:
                spike_scores.append(float(scores[start]))
            if end - start >= 5 and series.name in wide[:20]:
                wide_interior_scores.extend(scores[start + 1:end - 1].tolist())

    def describe(values):
        vals = np.asarray(values, dtype=float)
        if not vals.size:
            return {"count": 0}
        return {
            "count": int(vals.size),
            "q50": float(np.quantile(vals, 0.50)),
            "q90": float(np.quantile(vals, 0.90)),
            "q95": float(np.quantile(vals, 0.95)),
            "q99": float(np.quantile(vals, 0.99)),
            "max": float(np.max(vals)),
        }

    moderate, extreme = config.e2_tau_moderate, config.e2_tau_extreme
    return {
        "protocol": "temporal train-prefix replay; validation labels used only to select event categories",
        "scored_validation_series": scored_series,
        "tau_moderate": moderate,
        "tau_extreme": extreme,
        "isolated_one_point_spikes": {
            **describe(spike_scores),
            "fraction_above_extreme": float(np.mean(np.asarray(spike_scores) > extreme)) if spike_scores else 0.0,
        },
        "wide_event_interior_points": {
            **describe(wide_interior_scores),
            "fraction_moderate": float(np.mean((np.asarray(wide_interior_scores) > moderate) & (np.asarray(wide_interior_scores) <= extreme))) if wide_interior_scores else 0.0,
            "fraction_above_extreme": float(np.mean(np.asarray(wide_interior_scores) > extreme)) if wide_interior_scores else 0.0,
        },
        "zero_event_false_positive_cluster_peak_scores": describe(control_cluster_peaks),
    }


def load_dataset(
    dataset_name: str,
    max_series: Optional[int] = None,
    use_windows: bool = True,
    filter_anomalous: bool = True,
) -> List[TimeSeries]:
    """Load real benchmark dataset."""
    if dataset_name.lower() == "nab":
        loader = NABLoader()
        if not loader.is_available():
            logger.info("NAB data not found locally. Initiating download...")
            success = loader.download()
            if not success:
                logger.error("Failed to download NAB dataset.")
                sys.exit(1)
        series_list = loader.load_all(use_windows=use_windows)
        
        # Keep normal controls available for zero-event false-alarm reporting.
        # Preserve normal controls for FPR diagnostics; the default argument
        # remains for backward compatibility with direct callers.

        if max_series and len(series_list) > max_series:
            series_list = series_list[:max_series]
        return series_list

    elif dataset_name.lower() in ("yahoo", "yahoo_s5"):
        loader = YahooS5Loader()
        if not loader.is_available():
            logger.warning(
                "Yahoo S5 unavailable locally; place licensed A1Benchmark through "
                "A4Benchmark directories under %s.", loader.data_root
            )
            return []
        series_list = []
        for benchmark in ("A1", "A2", "A3", "A4"):
            try:
                series_list.extend(loader.load_benchmark(benchmark))
            except (FileNotFoundError, RuntimeError):
                logger.warning("Yahoo benchmark %s is unavailable locally.", benchmark)

        if max_series and len(series_list) > max_series:
            series_list = series_list[:max_series]
        return series_list
    elif dataset_name.lower() == "synthetic":
        from drift_anomaly.data.synthetic import SyntheticGenerator

        generator = SyntheticGenerator(seed=42)
        series_list = [
            generator.stationary(),
            generator.step_shift(),
            generator.variance_shift(),
            generator.spike_burst(),
            generator.isolated_spikes(),
            generator.compound(),
        ]
        return series_list[:max_series] if max_series else series_list
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'synthetic', 'nab', or 'yahoo'.")


def run_pipeline_for_series(
    series: TimeSeries,
    base_config: DetectorConfig,
    tau_grid: List[float],
    val_fraction: float = 0.2,
    test_fraction: float = 0.2,
) -> Tuple[Dict[str, EvalMetrics], Dict[str, float], TimeSeries, TimeSeries]:
    """Execute leak-free train/val/test pipeline for a single series across all variants."""
    train_ts, val_ts, test_ts = temporal_split(
        series, val_fraction=val_fraction, test_fraction=test_fraction
    )

    metrics_by_variant: Dict[str, EvalMetrics] = {}
    best_taus: Dict[str, float] = {}

    for var_name, VarCls in VARIANTS.items():
        # Tune validation while replaying the causal training prefix to warm
        # detector state. Labels remain isolated in the scored validation slice.
        best_tau, _ = tune_threshold(
            DetectorClass=VarCls,
            config=base_config,
            val_series=val_ts,
            tau_grid=tau_grid,
            metric="f1",
            warmup_series=train_ts,
        )
        best_taus[var_name] = best_tau

        # Phase 2: Evaluation on held-out test split with tuned tau
        tuned_cfg = dataclasses.replace(base_config, tau=best_tau)
        # Recreate state by replaying train + validation using the frozen tau,
        # then score test. This preserves chronology without consulting labels.
        context = TimeSeries(
            points=train_ts.points + val_ts.points,
            name=series.name,
            metadata=dict(series.metadata),
        )
        result = _run_detector_from_prefix(VarCls, tuned_cfg, context, test_ts)
        metrics_by_variant[var_name] = compute_metrics(result, test_ts)

    return metrics_by_variant, best_taus, val_ts, test_ts


def main():
    parser = argparse.ArgumentParser(
        description="Tune/train and evaluate A-D, E, E2, and LSTM on synthetic, NAB, or Yahoo S5."
    )
    parser.add_argument(
        "--dataset",
        choices=["synthetic", "nab", "yahoo"],
        default="nab",
        help="Target dataset (default: nab)",
    )
    parser.add_argument(
        "--max-series",
        type=int,
        default=None,
        help="Optional max series count (default: full available dataset)",
    )
    parser.add_argument(
        "--window-size",
        type=int,
        default=200,
        help="Baseline reference window size L (default: 200)",
    )
    parser.add_argument(
        "--candidate-size",
        type=int,
        default=50,
        help="Candidate window size M for Version A (default: 50)",
    )
    parser.add_argument(
        "--drift-detector",
        choices=["adwin", "page-hinkley"],
        default="adwin",
        help="Drift detector algorithm (default: adwin)",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=0.2,
        help="Validation split fraction (default: 0.2)",
    )
    parser.add_argument(
        "--test-fraction",
        type=float,
        default=0.3,
        help="Test split fraction (default: 0.3)",
    )
    parser.add_argument(
        "--use-exact-points",
        action="store_true",
        default=False,
        help="Use NAB's exact labeled timestamps instead of anomaly windows (default: windows)",
    )
    parser.add_argument(
        "--include-all",
        action="store_true",
        default=False,
        help="Include non-anomalous baseline series as well",
    )
    parser.add_argument(
        "--series-split",
        action="store_true",
        help="Tune pooled thresholds on a stratified 30%% series holdout instead of temporal per-series validation.",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="outputs/real_benchmarks",
        help="Output directory for reports and figures",
    )
    args = parser.parse_args()

    out_dir = Path(args.output_dir) / args.dataset
    out_dir.mkdir(parents=True, exist_ok=True)
    plots_dir = out_dir / "plots"
    plots_dir.mkdir(parents=True, exist_ok=True)

    logger.info("=================================================================")
    logger.info("Real Benchmark Training & Evaluation: %s", args.dataset.upper())
    logger.info("Output directory: %s", out_dir.resolve())
    logger.info("=================================================================")

    # 1. Load data
    series_list = load_dataset(
        args.dataset,
        max_series=None,
        use_windows=(args.dataset.lower() == "nab"),
        filter_anomalous=not args.include_all,
    )
    if not series_list:
        logger.error("No series loaded for '%s'. Exiting.", args.dataset)
        return

    if args.max_series and len(series_list) > args.max_series:
        positives = [s for s in series_list if sum(s.labels) > 0]
        controls = [s for s in series_list if sum(s.labels) == 0]
        n_pos = min(len(positives), max(1, int(round(args.max_series * 0.8))))
        n_ctrl = min(len(controls), args.max_series - n_pos)
        selected = positives[:n_pos] + controls[:n_ctrl]
        if len(selected) < args.max_series:
            remaining = [s for s in series_list if s not in selected]
            selected.extend(remaining[: args.max_series - len(selected)])
        series_list = selected

    dataset_series = list(series_list)

    logger.info("Loaded %d series from %s", len(series_list), args.dataset.upper())

    n_no_validation_labels = sum(
        sum(temporal_split(s, args.val_fraction, args.test_fraction)[1].labels) == 0
        for s in series_list
    )
    if n_no_validation_labels and not args.series_split:
        logger.warning(
            "%d/%d series have no validation anomalies; F1-based tau tuning "
            "cannot distinguish thresholds on those series.",
            n_no_validation_labels,
            len(series_list),
        )


    base_config = DetectorConfig(
        window_size=args.window_size,
        candidate_size=args.candidate_size,
        tau=3.0,
        drift_detector=args.drift_detector,
        freeze_window=args.window_size,
        hold_steps=args.candidate_size,
    )
    tau_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]

    all_results_by_variant: Dict[str, List[EvalMetrics]] = {v: [] for v in [*VARIANTS, "E", "E2", "LSTM"]}
    detailed_log: List[dict] = []
    pooled_taus = {}
    pooled_validation, pooled_test = [], []
    if args.series_split:
        pooled_taus, pooled_validation, pooled_test, pooled_lstm_calibration = tune_pooled_thresholds(
            series_list, base_config, tau_grid
        )
        # Select E settings only from the validation-side series; the test
        # series must remain unseen during ablation selection.
        best_e, e_validation_report = validate_e_ablation(pooled_validation, base_config)
        for key, value in best_e["config"].items():
            setattr(base_config, key, value)
        best_e2, e2_validation_report = validate_e_ablation(
            pooled_validation, base_config, detector_cls=VersionE2, dual_tier=True
        )
        e2_config = dataclasses.replace(base_config, **best_e2["config"])
        diagnostics_series = pooled_validation
        series_list = pooled_test
        pooled_validation_ids = {id(s) for s in pooled_validation}
        logger.warning(
            "Pooled threshold holdout: %d validation series (%d positive), %d test series (%d positive).",
            len(pooled_validation), sum(sum(s.labels) > 0 for s in pooled_validation),
            len(pooled_test), sum(sum(s.labels) > 0 for s in pooled_test),
        )
        logger.info("Pooled series holdout: %d validation, %d test", len(pooled_validation), len(pooled_test))

    e_validation_report = locals().get("e_validation_report", [])
    if not args.series_split:
        # Keep temporal test suffixes out of hyperparameter selection. For this
        # protocol E uses its declared config defaults; use --series-split to
        # select E settings on a held-out validation set.
        e_validation_report = []
        best_e = {
            "config": {
                key: getattr(base_config, key)
                for key in (
                    "e_scale_floor_ratio", "e_sigma_floor", "e_tau_low",
                    "e_tau_high", "e_persistence_k", "e_seasonal_lag",
                    "e_auto_lag", "e_max_lag", "e_online_calibration",
                    "e_calibration_quantile",
                )
            },
            "selection": "configured_defaults_temporal_mode",
        }
        logger.info("Temporal mode: using configured Version E settings without ablation tuning.")
        best_e2, e2_validation_report = validate_e_ablation(
            dataset_series, base_config, detector_cls=VersionE2, dual_tier=True
        )
        e2_config = dataclasses.replace(base_config, **best_e2["config"])
        diagnostics_series = dataset_series
    e2_validation_report = locals().get("e2_validation_report", [])
    e2_score_diagnostics = inspect_e2_validation_score_distributions(
        diagnostics_series, e2_config
    )

    # 2. Iterate through series
    for idx, s in enumerate(tqdm(series_list, desc="Benchmark test series")):
        logger.info(
            "[%d/%d] Processing series: %s (length=%d, anomalies=%d)",
            idx + 1,
            len(series_list),
            s.name,
            len(s),
            sum(s.labels),
        )

        if args.series_split:
            train_ts = TimeSeries(points=[], name=s.name)
            val_ts = TimeSeries(points=[], name=s.name)
            test_ts = s
            best_taus = pooled_taus
            metrics_map = {}
            for v, Cls in VARIANTS.items():
                cfg = dataclasses.replace(base_config, tau=best_taus[v])
                metrics_map[v] = compute_metrics(Cls(cfg).run(test_ts), test_ts)
            metrics_map["E"] = compute_metrics(VersionE(base_config).run(test_ts), test_ts)
            metrics_map["E2"] = compute_metrics(VersionE2(e2_config).run(test_ts), test_ts)
            # Series holdout starts each test series from its own beginning;
            # score every variant on the same split used for its metrics.
            comparison_prefix = TimeSeries(points=[], name=s.name, metadata=dict(s.metadata))
            lstm_train = TimeSeries(points=[], name=s.name)
            lstm_scored = s
            lstm_context = []
        else:
            train_ts, val_ts, test_ts = temporal_split(
                s, val_fraction=args.val_fraction, test_fraction=args.test_fraction
            )
            metrics_map, best_taus, val_ts, test_ts = run_pipeline_for_series(
                series=s,
                base_config=base_config,
                tau_grid=tau_grid,
                val_fraction=args.val_fraction,
                test_fraction=args.test_fraction,
            )
            e_tau = base_config.tau
            best_taus["E"] = e_tau
            e_context = TimeSeries(points=train_ts.points + val_ts.points, name=s.name,
                                   metadata=dict(s.metadata))
            comparison_prefix = e_context
            e_res = _run_detector_from_prefix(VersionE, dataclasses.replace(base_config, tau=e_tau), e_context, test_ts)
            metrics_map["E"] = compute_metrics(e_res, test_ts)
            e2_res = _run_detector_from_prefix(VersionE2, e2_config, e_context, test_ts)
            metrics_map["E2"] = compute_metrics(e2_res, test_ts)
            lstm_train = train_ts
            lstm_scored = test_ts
            lstm_context = train_ts.values + val_ts.values

        lstm = LSTMAnomalyDetector(**LSTM_SETTINGS)
        if args.series_split:
            lstm_model = pooled_lstm_calibration["model"]
            lstm_training_time = pooled_lstm_calibration["training_time_s"]
            lstm_result = lstm_model.score(
                lstm_scored.values, tau=best_taus.get("LSTM", 3.0), context=[]
            )
        else:
            lstm_training_time = lstm.fit(lstm_train.values)
            # Select the LSTM residual threshold from normal validation points
            # only, then score the test suffix with train+validation context.
            lstm_val = lstm.score(val_ts.values, context=lstm_train.values)
            normal_val_scores = [
                score for score, is_anomaly in zip(lstm_val.scores, val_ts.labels)
                if not is_anomaly
            ]
            lstm_tau = lstm.threshold_from(normal_val_scores, target_fpr=0.01)
            best_taus["LSTM"] = lstm_tau
            lstm_result = lstm.score(
                lstm_scored.values, tau=lstm_tau, context=lstm_context
            )
        lstm_result.training_time_s = lstm_training_time
        lstm_truth = np.asarray(lstm_scored.labels, dtype=int)
        from drift_anomaly.pipeline.metrics import compute_point_metrics
        lstm_point = compute_point_metrics(lstm_truth, lstm_result.alerts.astype(int), lstm_result.scores)
        lstm_metric_map = {
            **lstm_point,
            "n_drift_triggers": 0,
            "false_drift_rate": 0.0,
            "adaptation_accepted": 0,
            "adaptation_rejected": 0,
            "runtime_s": lstm_result.inference_time_s,
            "training_time_s": lstm_result.training_time_s,
            "inference_time_s": lstm_result.inference_time_s,
            "parameter_count": lstm_result.parameter_count,
            "complexity_training": f"O(E*N*L*H^2), E={lstm.epochs}, L={lstm.lookback}, H={lstm.hidden_size}",
            "complexity_inference_per_step": f"O(L*H^2), L={lstm.lookback}, H={lstm.hidden_size}",
        }
        e_metric_map = dataclasses.asdict(metrics_map["E"])
        e2_metric_map = dataclasses.asdict(metrics_map["E2"])
        metrics_map["LSTM"] = EvalMetrics(
            precision=lstm_point["precision"], recall=lstm_point["recall"], f1=lstm_point["f1"],
            roc_auc=lstm_point["roc_auc"], pr_auc=lstm_point["pr_auc"], n_drift_triggers=0,
            false_drift_rate=0.0, adaptation_accepted=0, adaptation_rejected=0,
            runtime_s=lstm_result.inference_time_s,
        )

        series_entry = {
            "series": s.name,
            "length": len(s),
            "test_length": len(test_ts),
            "train_anomalies": int(sum(train_ts.labels)) if not args.series_split else 0,
            "validation_anomalies": int(sum(val_ts.labels)) if not args.series_split else 0,
            "test_anomalies": int(sum(test_ts.labels)),
            "best_taus": best_taus,
            "tuning_protocol": "pooled_series_holdout" if args.series_split else "temporal_within_series",
            "metrics": {v: dataclasses.asdict(metrics_map[v]) for v in [*VARIANTS, "E", "E2"]},
        }
        series_entry["metrics"]["LSTM"] = lstm_metric_map
        series_entry["metrics"]["E"] = e_metric_map
        series_entry["metrics"]["E2"] = e2_metric_map
        series_entry["runtime"] = {
            "lstm_training_time_s": lstm_result.training_time_s,
            "lstm_inference_time_s": lstm_result.inference_time_s,
            "lstm_parameter_count": lstm_result.parameter_count,
            "lstm_complexity_training": lstm_metric_map["complexity_training"],
            "lstm_complexity_inference_per_step": lstm_metric_map["complexity_inference_per_step"],
        }

        prefix = TimeSeries(
            points=(s.points[: len(s) - len(test_ts)] if not args.series_split else []),
            name=s.name,
            metadata=dict(s.metadata),
        )
        point_data = {}
        score_data = {}
        detector_configs = {
            **{v: base_config for v in VARIANTS},
            "E": base_config,
            "E2": e2_config,
        }
        for v, VarCls in {**VARIANTS, "E": VersionE, "E2": VersionE2}.items():
            variant_config = detector_configs[v]
            tuned_cfg = dataclasses.replace(variant_config, tau=best_taus.get(v, variant_config.tau))
            result = _run_detector_from_prefix(VarCls, tuned_cfg, comparison_prefix, test_ts)
            truth = np.asarray(test_ts.labels, dtype=bool)
            pred = np.asarray(result.alerts, dtype=bool)
            series_entry["metrics"][v].update(event_window_metrics(truth, pred))
            series_entry["metrics"][v]["false_alert_count"] = int(np.sum(pred & ~truth))
            series_entry["metrics"][v]["zero_event_fpr"] = (
                float(np.mean(pred)) if not truth.any() and len(pred) else 0.0
            )
            point_data[v] = {"truth": truth.tolist(), "pred": pred.tolist()}
            score_data[v] = np.asarray(result.scores, dtype=float).tolist()
        series_entry["metrics"]["E"].update(event_window_metrics(
            np.asarray(point_data["E"]["truth"], dtype=bool),
            np.asarray(point_data["E"]["pred"], dtype=bool),
        ))
        series_entry["metrics"]["E"]["false_alert_count"] = int(np.sum(
            np.asarray(point_data["E"]["pred"], dtype=bool) &
            ~np.asarray(point_data["E"]["truth"], dtype=bool)
        ))
        series_entry["metrics"]["E"]["zero_event_fpr"] = (
            float(np.mean(point_data["E"]["pred"])) if not any(point_data["E"]["truth"]) else 0.0
        )
        series_entry["metrics"]["E2"].update(event_window_metrics(
            np.asarray(point_data["E2"]["truth"], dtype=bool),
            np.asarray(point_data["E2"]["pred"], dtype=bool),
        ))
        series_entry["metrics"]["E2"]["false_alert_count"] = int(np.sum(
            np.asarray(point_data["E2"]["pred"], dtype=bool) &
            ~np.asarray(point_data["E2"]["truth"], dtype=bool)
        ))
        series_entry["metrics"]["E2"]["zero_event_fpr"] = (
            float(np.mean(point_data["E2"]["pred"])) if not any(point_data["E2"]["truth"]) else 0.0
        )
        lstm_truth_bool = lstm_truth.astype(bool)
        series_entry["metrics"]["LSTM"].update(event_window_metrics(lstm_truth_bool, lstm_result.alerts))
        series_entry["metrics"]["LSTM"]["false_alert_count"] = int(np.sum(lstm_result.alerts & ~lstm_truth_bool))
        series_entry["metrics"]["LSTM"]["zero_event_fpr"] = (
            float(np.mean(lstm_result.alerts)) if not lstm_truth_bool.any() and len(lstm_truth_bool) else 0.0
        )
        point_data["LSTM"] = {"truth": lstm_truth_bool.tolist(), "pred": lstm_result.alerts.tolist()}
        score_data["LSTM"] = np.asarray(lstm_result.scores, dtype=float).tolist()
        for variant in [*VARIANTS, "E", "E2", "LSTM"]:
            series_entry["metrics"][variant]["literature_aligned"] = literature_series_metrics(
                np.asarray(point_data[variant]["truth"], dtype=bool),
                np.asarray(point_data[variant]["pred"], dtype=bool),
                np.asarray(score_data[variant], dtype=float),
            )
        series_entry["_point_data"] = point_data
        series_entry["_score_data"] = score_data
        detailed_log.append(series_entry)
        for v in [*VARIANTS, "E", "E2", "LSTM"]:
            all_results_by_variant[v].append(metrics_map[v])

        # Plot individual test series diagnostics for the first 3 series
        if idx < 3:
            for v, VarCls in {**VARIANTS, "E": VersionE, "E2": VersionE2}.items():
                variant_config = detector_configs[v]
                tuned_cfg = dataclasses.replace(variant_config, tau=best_taus.get(v, variant_config.tau))
                best_taus_for_plot = {
                    **best_taus,
                    "E": base_config.e_tau_high,
                    "E2": e2_config.e2_tau_extreme,
                }
                res = _run_detector_from_prefix(VarCls, tuned_cfg, comparison_prefix, test_ts)
                clean_name = s.name.replace("/", "_").replace("\\", "_").replace(".csv", "")
                plot_file = plots_dir / f"{clean_name}_variant_{v}.png"
                plot_series_with_anomalies(
                    values=np.array(test_ts.values),
                    true_labels=np.array(test_ts.labels, dtype=int),
                    pred_labels=np.array(res.alerts, dtype=int),
                    scores=np.array(res.scores, dtype=float),
                    drift_events=res.drift_events,
                    title=f"{s.name} (Test Split) — Variant {v} (τ={best_taus_for_plot[v]:.1f})",
                    output_path=plot_file,
                )
            plot_series_with_anomalies(
                values=np.asarray(test_ts.values),
                true_labels=lstm_truth,
                pred_labels=lstm_result.alerts.astype(int),
                scores=lstm_result.scores,
                drift_events=[],
                title=f"{s.name} (Test Split) — LSTM",
                output_path=plots_dir / f"{clean_name}_variant_LSTM.png",
            )

    # 3. Save aggregated reports
    summary_json_path = out_dir / "evaluation_summary.json"
    with summary_json_path.open("w") as fh:
        json.dump(detailed_log, fh, indent=2)
    logger.info("Saved JSON summary to %s", summary_json_path)

    for v in [*VARIANTS, "E", "E2", "LSTM"]:
        csv_path = out_dir / f"metrics_variant_{v}.csv"
        save_metrics_csv(all_results_by_variant[v], csv_path)

    # 4. Generate comparison bar chart across variants
    comparison_plot_path = out_dir / "variant_comparison.png"
    plot_metrics_comparison(all_results_by_variant, output_path=comparison_plot_path)
    logger.info("Saved comparative bar chart to %s", comparison_plot_path)

    pooled_summary = {}
    for v in [*VARIANTS, "E", "E2", "LSTM"]:
        macro_f1, event_f1 = [], []
        for entry in detailed_log:
            truth = np.asarray(entry["_point_data"][v]["truth"], dtype=bool)
            pred = np.asarray(entry["_point_data"][v]["pred"], dtype=bool)
            if truth.any():
                tp = int(np.sum(truth & pred))
                fp = int(np.sum(~truth & pred))
                fn = int(np.sum(truth & ~pred))
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                macro_f1.append(2 * p * r / (p + r) if p + r else 0.0)
                event_f1.append(entry["metrics"][v]["event_f1"])
        series_point_data = {
            entry["series"]: (
                np.asarray(entry["_point_data"][v]["truth"], dtype=bool),
                np.asarray(entry["_point_data"][v]["pred"], dtype=bool),
            )
            for entry in detailed_log
        }
        point_aggregate = evaluate_pointwise_macro(series_point_data)
        literature_series_data = {
            entry["series"]: (
                np.asarray(entry["_point_data"][v]["truth"], dtype=bool),
                np.asarray(entry["_point_data"][v]["pred"], dtype=bool),
                np.asarray(entry["_score_data"][v], dtype=float),
            )
            for entry in detailed_log
        }
        literature_aggregate = aggregate_literature_metrics(literature_series_data)
        pooled_summary[v] = {
            "pointwise_micro": point_aggregate["micro"],
            "pointwise_macro_positive_series_f1": point_aggregate["macro_f1_positive_series"],
            "zero_event_point_fpr": point_aggregate["micro"]["zero_event_point_fpr"],
            "zero_event_mean_fpr": point_aggregate["zero_event_series_mean_fpr"],
            "zero_event_false_alert_count": point_aggregate["zero_event_series_false_alerts"],
            "event_window_macro_f1": float(np.mean(event_f1)) if event_f1 else 0.0,
            "literature_aligned": literature_aggregate,
        }
    pooled_summary["LSTM"]["training_time_s_total"] = float(sum(
        entry["metrics"]["LSTM"]["training_time_s"] for entry in detailed_log
    )) if not args.series_split else float(pooled_lstm_calibration["training_time_s"])
    pooled_summary["E"]["training_time_s_total"] = 0.0
    pooled_summary["E2"]["training_time_s_total"] = 0.0
    for entry in detailed_log:
        entry.pop("_point_data", None)
        entry.pop("_score_data", None)
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump({"series": detailed_log, "pooled": pooled_summary,
                   "validation_series": [s.name for s in pooled_validation],
                   "shared_best_taus": pooled_taus,
                   "version_e_validation_selection": best_e,
                   "version_e_validation_ablation": e_validation_report,
                   "version_e2_validation_selection": best_e2,
                   "version_e2_validation_ablation": e2_validation_report,
                   "version_e2_validation_score_diagnostics": e2_score_diagnostics}, fh, indent=2)
    (out_dir / "pooled_metrics.json").write_text(json.dumps(pooled_summary, indent=2), encoding="utf-8")
    timing = {}
    for variant, cls in {**VARIANTS, "E": VersionE, "E2": VersionE2}.items():
        timing[variant] = {
            "training_time_s": 0.0,
            "inference_time_s": float(sum(m.runtime_s for m in all_results_by_variant[variant])),
            "time_complexity_per_point": "O(W log W) robust median/MAD update",
            "memory_complexity": "O(W)",
        }
    timing["LSTM"] = {
        "training_time_s": float(sum(e["metrics"]["LSTM"]["training_time_s"] for e in detailed_log)),
        "inference_time_s": float(sum(e["metrics"]["LSTM"]["inference_time_s"] for e in detailed_log)),
        "parameter_count": int(np.mean([e["metrics"]["LSTM"]["parameter_count"] for e in detailed_log])) if detailed_log else 0,
        "time_complexity_training": "O(E*N*L*H^2) per series, minibatched BPTT",
        "time_complexity_inference": "O(L*H^2) per forecasted point",
        "memory_complexity": "O(L*H + P + B*L), P parameters, B minibatch size",
        "settings": LSTM_SETTINGS,
    }
    (out_dir / "complexity_timing.json").write_text(json.dumps(timing, indent=2), encoding="utf-8")

    # 5. Print terminal summary
    print("\n" + "=" * 80)
    print(f"BENCHMARK EVALUATION SUMMARY ({args.dataset.upper()})")
    print("=" * 80)
    print(f"{'Variant':<10} {'Point F1':<12} {'Event F1':<12} {'Zero FPR point':<16} {'Zero FPR mean':<15} {'Macro F1+':<12}")
    print("-" * 84)
    for v in [*VARIANTS, "E", "E2", "LSTM"]:
        v_metrics = all_results_by_variant[v]
        agg = pooled_summary[v]
        print(f"{v:<10} {agg['pointwise_micro']['f1']:<12.4f} {agg['event_window_macro_f1']:<12.4f} "
              f"{agg['zero_event_point_fpr']:<16.4f} {agg['zero_event_mean_fpr']:<15.4f} "
              f"{agg['pointwise_macro_positive_series_f1']:<12.4f}")
    print("=" * 84)
    print("LITERATURE-ALIGNED METRICS (POINT-ADJUSTED & ORACLE-BEST)")
    print(f"{'Variant':<10} {'PA-F1 micro':<14} {'PA-F1 macro+':<15} {'PR-AUC+':<12} {'ROC-AUC+':<12} {'Oracle F1+':<13} {'Oracle PA-F1+':<14}")
    print("-" * 92)
    for v in [*VARIANTS, "E", "E2", "LSTM"]:
        literature = pooled_summary[v]["literature_aligned"]
        print(f"{v:<10} {literature['pa_micro_f1']:<14.4f} "
              f"{literature['pa_macro_positive_series']['pa_f1']:<15.4f} "
              f"{literature['positive_series_mean_pr_auc']:<12.4f} "
              f"{literature['positive_series_mean_roc_auc']:<12.4f} "
              f"{literature['oracle_best_f1_macro_positive_series']:<13.4f} "
              f"{literature['oracle_best_pa_f1_macro_positive_series']:<14.4f}")
    print("=" * 92)
    print(f"Artifacts and diagnostic plots available in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
