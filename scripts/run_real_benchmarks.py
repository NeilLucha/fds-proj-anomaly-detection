#!/usr/bin/env python3
"""
scripts/run_real_benchmarks.py
==============================

End-to-end training and evaluation script for REAL datasets (NAB and Yahoo S5).

Pipeline stages:
1. Data verification & automatic download (NAB from GitHub; Yahoo S5 check).
2. Temporal train / validation / test splits (strict chronological causality, no shuffle).
3. Leak-free threshold (tau) tuning on validation split (never uses test labels).
4. Out-of-sample evaluation across all three variants (A, B, C) on held-out test series.
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

# Ensure project root is in sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT / "src") not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT / "src"))

from drift_anomaly.data.nab import NABLoader
from drift_anomaly.data.yahoo_s5 import YahooS5Loader
from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD
from drift_anomaly.detectors.lstm import LSTMAnomalyDetector
from drift_anomaly.pipeline.evaluate import evaluate_detector
from drift_anomaly.pipeline.metrics import compute_metrics, event_window_metrics, evaluate_pointwise_macro
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
    for name, cls in VARIANTS.items():
        best_tau, best_f1 = tau_grid[0], -1.0
        for tau in tau_grid:
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
    else:
        raise ValueError(f"Unknown dataset '{dataset_name}'. Choose 'nab' or 'yahoo'.")


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
        description="End-to-end training and evaluation on real benchmarks (NAB / Yahoo S5)."
    )
    parser.add_argument(
        "--dataset",
        choices=["nab", "yahoo"],
        default="nab",
        help="Target benchmark dataset (default: nab)",
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

    all_results_by_variant: Dict[str, List[EvalMetrics]] = {v: [] for v in [*VARIANTS, "LSTM"]}
    detailed_log: List[dict] = []
    pooled_taus = {}
    pooled_validation, pooled_test = [], []
    if args.series_split:
        pooled_taus, pooled_validation, pooled_test, pooled_lstm_calibration = tune_pooled_thresholds(
            series_list, base_config, tau_grid
        )
        series_list = pooled_test
        pooled_validation_ids = {id(s) for s in pooled_validation}
        logger.warning(
            "Pooled threshold holdout: %d validation series (%d positive), %d test series (%d positive).",
            len(pooled_validation), sum(sum(s.labels) > 0 for s in pooled_validation),
            len(pooled_test), sum(sum(s.labels) > 0 for s in pooled_test),
        )
        logger.info("Pooled series holdout: %d validation, %d test", len(pooled_validation), len(pooled_test))

    # 2. Iterate through series
    for idx, s in enumerate(series_list):
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
            lstm_result = lstm.score(
                lstm_scored.values, tau=best_taus.get("LSTM", 3.0), context=lstm_context
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
            "metrics": {v: dataclasses.asdict(metrics_map[v]) for v in VARIANTS},
        }
        series_entry["metrics"]["LSTM"] = lstm_metric_map
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
        for v, VarCls in VARIANTS.items():
            tuned_cfg = dataclasses.replace(base_config, tau=best_taus[v])
            result = _run_detector_from_prefix(VarCls, tuned_cfg, prefix, test_ts)
            truth = np.asarray(test_ts.labels, dtype=bool)
            pred = np.asarray(result.alerts, dtype=bool)
            series_entry["metrics"][v].update(event_window_metrics(truth, pred))
            series_entry["metrics"][v]["false_alert_count"] = int(np.sum(pred & ~truth))
            series_entry["metrics"][v]["zero_event_fpr"] = (
                float(np.mean(pred)) if not truth.any() and len(pred) else 0.0
            )
            point_data[v] = {"truth": truth.tolist(), "pred": pred.tolist()}
        lstm_truth_bool = lstm_truth.astype(bool)
        series_entry["metrics"]["LSTM"].update(event_window_metrics(lstm_truth_bool, lstm_result.alerts))
        series_entry["metrics"]["LSTM"]["false_alert_count"] = int(np.sum(lstm_result.alerts & ~lstm_truth_bool))
        series_entry["metrics"]["LSTM"]["zero_event_fpr"] = (
            float(np.mean(lstm_result.alerts)) if not lstm_truth_bool.any() and len(lstm_truth_bool) else 0.0
        )
        point_data["LSTM"] = {"truth": lstm_truth_bool.tolist(), "pred": lstm_result.alerts.tolist()}
        series_entry["_point_data"] = point_data
        detailed_log.append(series_entry)
        for v in [*VARIANTS, "LSTM"]:
            all_results_by_variant[v].append(metrics_map[v])

        # Plot individual test series diagnostics for the first 3 series
        if idx < 3:
            for v, VarCls in VARIANTS.items():
                tuned_cfg = dataclasses.replace(base_config, tau=best_taus[v])
                context = TimeSeries(
                    points=s.points[: len(s) - len(test_ts)] if not args.series_split else [],
                    name=s.name,
                    metadata=dict(s.metadata),
                )
                res = _run_detector_from_prefix(VarCls, tuned_cfg, context, test_ts)
                clean_name = s.name.replace("/", "_").replace("\\", "_").replace(".csv", "")
                plot_file = plots_dir / f"{clean_name}_variant_{v}.png"
                plot_series_with_anomalies(
                    values=np.array(test_ts.values),
                    true_labels=np.array(test_ts.labels, dtype=int),
                    pred_labels=np.array(res.alerts, dtype=int),
                    scores=np.array(res.scores, dtype=float),
                    drift_events=res.drift_events,
                    title=f"{s.name} (Test Split) — Variant {v} (τ={best_taus[v]:.1f})",
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

    for v in [*VARIANTS, "LSTM"]:
        csv_path = out_dir / f"metrics_variant_{v}.csv"
        save_metrics_csv(all_results_by_variant[v], csv_path)

    # 4. Generate comparison bar chart across variants
    comparison_plot_path = out_dir / "variant_comparison.png"
    plot_metrics_comparison(all_results_by_variant, output_path=comparison_plot_path)
    logger.info("Saved comparative bar chart to %s", comparison_plot_path)

    pooled_summary = {}
    for v in [*VARIANTS, "LSTM"]:
        truth_parts, pred_parts, macro_f1, event_f1 = [], [], [], []
        zero_fprs, zero_false_alerts = [], 0
        zero_fprs, zero_false_alerts = [], 0
        for entry in detailed_log:
            truth = np.asarray(entry["_point_data"][v]["truth"], dtype=bool)
            pred = np.asarray(entry["_point_data"][v]["pred"], dtype=bool)
            truth_parts.extend(truth.tolist())
            pred_parts.extend(pred.tolist())
            if truth.any():
                tp = int(np.sum(truth & pred))
                fp = int(np.sum(~truth & pred))
                fn = int(np.sum(truth & ~pred))
                p = tp / (tp + fp) if tp + fp else 0.0
                r = tp / (tp + fn) if tp + fn else 0.0
                macro_f1.append(2 * p * r / (p + r) if p + r else 0.0)
                event_f1.append(entry["metrics"][v]["event_f1"])
            else:
                zero_fprs.append(float(np.mean(pred)) if len(pred) else 0.0)
                zero_false_alerts += int(np.sum(pred))
        series_point_data = {
            entry["series"]: (
                np.asarray(entry["_point_data"][v]["truth"], dtype=bool),
                np.asarray(entry["_point_data"][v]["pred"], dtype=bool),
            )
            for entry in detailed_log
        }
        point_aggregate = evaluate_pointwise_macro(series_point_data)
        pooled_summary[v] = {
            "pointwise_micro": point_aggregate["micro"],
            "pointwise_macro_positive_series_f1": point_aggregate["macro_f1_positive_series"],
            "zero_event_mean_fpr": point_aggregate["zero_event_series_mean_fpr"],
            "zero_event_false_alert_count": point_aggregate["zero_event_series_false_alerts"],
            "event_window_macro_f1": float(np.mean(event_f1)) if event_f1 else 0.0,
        }
    pooled_summary["LSTM"]["training_time_s_total"] = float(sum(
        entry["metrics"]["LSTM"]["training_time_s"] for entry in detailed_log
    )) if not args.series_split else float(pooled_lstm_calibration["training_time_s"])
    for entry in detailed_log:
        entry.pop("_point_data", None)
    with summary_json_path.open("w", encoding="utf-8") as fh:
        json.dump({"series": detailed_log, "pooled": pooled_summary,
                   "validation_series": [s.name for s in pooled_validation],
                   "shared_best_taus": pooled_taus}, fh, indent=2)
    (out_dir / "pooled_metrics.json").write_text(json.dumps(pooled_summary, indent=2), encoding="utf-8")
    timing = {}
    for variant, cls in VARIANTS.items():
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
    print(f"{'Variant':<10} {'Point F1':<12} {'Event F1':<12} {'Zero-event FPR':<15} {'Macro F1+':<12}")
    print("-" * 68)
    for v in [*VARIANTS, "LSTM"]:
        v_metrics = all_results_by_variant[v]
        agg = pooled_summary[v]
        print(f"{v:<10} {agg['pointwise_micro']['f1']:<12.4f} {agg['event_window_macro_f1']:<12.4f} "
              f"{agg['zero_event_mean_fpr']:<15.4f} {agg['pointwise_macro_positive_series_f1']:<12.4f}")
    print("=" * 68)
    print(f"Artifacts and diagnostic plots available in: {out_dir.resolve()}\n")


if __name__ == "__main__":
    main()
