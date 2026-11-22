#!/usr/bin/env python3
"""Report-oriented strict-causal comparison with reference baselines.

This is a standalone runner. It does not modify the existing benchmark runner
or any detector implementation. It compares A-D, E, E2, LSTM, EWMA, River HST,
and causal lag-window Isolation Forest on NAB or Yahoo S5.
"""
from __future__ import annotations

import argparse
import copy
import dataclasses
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
from sklearn.ensemble import IsolationForest
from tqdm.auto import tqdm

ROOT = Path(__file__).resolve().parent.parent
if str(ROOT / "src") not in sys.path:
    sys.path.insert(0, str(ROOT / "src"))
if str(ROOT / "scripts") not in sys.path:
    sys.path.insert(0, str(ROOT / "scripts"))

import run_real_benchmarks as existing
from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2
from drift_anomaly.detectors.lstm import LSTMAnomalyDetector
from drift_anomaly.pipeline.metrics import (
    aggregate_literature_metrics,
    compute_point_metrics,
    event_window_metrics,
    evaluate_pointwise_macro,
    literature_series_metrics,
)
from drift_anomaly.pipeline.tune import _run_detector_from_prefix, temporal_split, tune_threshold
from drift_anomaly.types import DetectorConfig, TimeSeries

logging.basicConfig(level=logging.WARNING, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("run_report_benchmarks")

STATISTICAL = {**existing.VARIANTS, "E": VersionE, "E2": VersionE2}
VARIANT_ORDER = ["A", "B", "C", "D", "E", "E2", "LSTM", "EWMA", "River_HST", "Lag_IForest"]


def require_river():
    """Import River eagerly for actual runs and explain how to install it."""
    try:
        from river import anomaly
    except ImportError as exc:
        raise RuntimeError(
            "River HST baseline requires River. Install it with `pip install river` "
            "and rerun this benchmark."
        ) from exc
    return anomaly


class EWMABaseline:
    """Causal standardized residual; each point is scored before updating."""

    def __init__(self, alpha: float = 0.05, variance_floor: float = 1e-8):
        self.alpha = float(alpha)
        self.variance_floor = float(variance_floor)
        self.mean = 0.0
        self.variance = self.variance_floor
        self.initialized = False

    def fit(self, values):
        start = time.perf_counter()
        for raw in values:
            x = float(raw)
            # Observe the same score-before-update ordering during the
            # unlabeled training prefix; discard scores, retain state.
            _ = abs(x - self.mean) / np.sqrt(max(self.variance, self.variance_floor)) if self.initialized else 0.0
            self.update(x)
        return time.perf_counter() - start

    def update(self, x: float):
        if not self.initialized:
            self.mean = x
            self.variance = self.variance_floor
            self.initialized = True
            return
        residual = x - self.mean
        a = self.alpha
        self.mean += a * residual
        self.variance = max((1.0 - a) * (self.variance + a * residual * residual), self.variance_floor)

    def score_stream(self, values):
        scores = []
        start = time.perf_counter()
        for raw in values:
            x = float(raw)
            score = abs(x - self.mean) / np.sqrt(max(self.variance, self.variance_floor)) if self.initialized else 0.0
            scores.append(float(score))
            self.update(x)  # update only after scoring x_t
        return np.asarray(scores), time.perf_counter() - start


class RiverHSTBaseline:
    """River Half-Space Trees with a fixed train-derived robust scaler."""

    def __init__(self, anomaly_module, *, n_trees=25, height=8, window_size=250, seed=42):
        self.anomaly_module = anomaly_module
        self.n_trees, self.height, self.window_size, self.seed = n_trees, height, window_size, seed
        self.median = 0.0
        self.scale = 1.0
        self.model = None

    def _transform(self, x):
        return {"value": float(np.clip((float(x) - self.median) / self.scale, -8.0, 8.0))}

    def fit(self, values):
        start = time.perf_counter()
        train = np.asarray(values, dtype=float)
        self.median = float(np.median(train)) if train.size else 0.0
        q25, q75 = np.quantile(train, [0.25, 0.75]) if train.size else (0.0, 1.0)
        self.scale = max(float(q75 - q25) / 1.349, float(np.std(train)) * 1e-3, 1e-6)
        self.model = self.anomaly_module.HalfSpaceTrees(
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            limits={"value": (-8.0, 8.0)},
            seed=self.seed,
        )
        # Training observations are learned sequentially; never scored against
        # a model that has already seen their future observations.
        for x in train:
            features = self._transform(x)
            self.model.score_one(features)
            self.model.learn_one(features)
        return time.perf_counter() - start

    def score_stream(self, values):
        if self.model is None:
            raise RuntimeError("River HST must be fit on the training split first")
        scores = []
        start = time.perf_counter()
        for x in values:
            features = self._transform(x)
            score = self.model.score_one(features)  # score under F_(t-1)
            scores.append(float(score if score is not None else 0.0))
            self.model.learn_one(features)  # learn only after scoring x_t
        return np.asarray(scores), time.perf_counter() - start

    def clone_fitted(self):
        """Copy fitted River state without trying to pickle its module handle."""
        duplicate = RiverHSTBaseline(
            self.anomaly_module,
            n_trees=self.n_trees,
            height=self.height,
            window_size=self.window_size,
            seed=self.seed,
        )
        duplicate.median = self.median
        duplicate.scale = self.scale
        duplicate.model = copy.deepcopy(self.model)
        return duplicate


class LagIForestBaseline:
    """Fixed train-only Isolation Forest over causal lag vectors."""

    def __init__(self, lookback=10, n_estimators=100, seed=42):
        self.lookback = int(lookback)
        self.n_estimators = int(n_estimators)
        self.seed = int(seed)
        self.model = None

    def _window(self, history):
        hist = np.asarray(history, dtype=float)
        if hist.size == 0:
            return np.zeros(self.lookback, dtype=float)
        tail = hist[-self.lookback:]
        if tail.size < self.lookback:
            tail = np.pad(tail, (self.lookback - tail.size, 0), mode="edge")
        return tail[::-1].copy()  # [x_(t-1), x_(t-2), ...]

    def fit(self, values):
        start = time.perf_counter()
        train = np.asarray(values, dtype=float)
        features = [self._window(train[:t]) for t in range(1, len(train))]
        if len(features) < 2:
            raise ValueError("Lag_IForest needs at least three training points")
        self.model = IsolationForest(
            n_estimators=self.n_estimators,
            contamination="auto",
            random_state=self.seed,
            n_jobs=1,
        ).fit(np.asarray(features))
        return time.perf_counter() - start

    def score_stream(self, context, values):
        if self.model is None:
            raise RuntimeError("Lag_IForest must be fit on the training split first")
        history = list(map(float, context))
        scores = []
        start = time.perf_counter()
        for x in values:
            features = self._window(history).reshape(1, -1)
            # sklearn's decision_function is larger for normal points.
            scores.append(float(-self.model.decision_function(features)[0]))
            history.append(float(x))
        return np.asarray(scores), time.perf_counter() - start


def tune_score_threshold(y_true, scores, fpr_cap=0.10):
    """Tune a score threshold on validation Point/Event F1 under an FPR cap."""
    truth = np.asarray(y_true, dtype=bool)
    score = np.nan_to_num(np.asarray(scores, dtype=float), nan=0.0)
    if truth.shape != score.shape:
        raise ValueError("validation labels and scores must have the same shape")
    if not score.size:
        return 0.0, {"validation_f1": 0.0, "validation_event_f1": 0.0,
                     "validation_negative_fpr": 0.0, "fpr_cap": fpr_cap}
    if not truth.any():
        threshold = float(np.nextafter(np.max(score), np.inf))
        return threshold, {"validation_f1": 0.0, "validation_event_f1": 0.0,
                           "validation_negative_fpr": 0.0, "fpr_cap": fpr_cap,
                           "fpr_cap_fallback": False}
    candidates = np.unique(np.quantile(score, np.linspace(0.0, 1.0, min(65, max(2, score.size)))))
    candidates = np.r_[np.nextafter(np.max(score), np.inf), candidates]
    eligible, fallback = [], []
    for threshold in candidates:
        pred = score >= threshold
        point_f1 = compute_point_metrics(truth, pred, score)["f1"]
        event_f1 = event_window_metrics(truth, pred)["event_f1"]
        negatives = ~truth
        negative_fpr = float(np.mean(pred[negatives])) if negatives.any() else 0.0
        item = (0.5 * point_f1 + 0.5 * event_f1, -negative_fpr, float(threshold),
                point_f1, event_f1, negative_fpr)
        fallback.append(item)
        if negative_fpr <= fpr_cap:
            eligible.append(item)
    selected = max(eligible) if eligible else max(fallback, key=lambda x: (x[1], x[0]))
    return selected[2], {
        "validation_objective": selected[0],
        "validation_f1": selected[3],
        "validation_event_f1": selected[4],
        "validation_negative_fpr": selected[5],
        "fpr_cap": fpr_cap,
        "fpr_cap_fallback": not bool(eligible),
        "rule": "maximize 0.5*Point-F1 + 0.5*Event-F1 subject to validation negative-point FPR cap; score >= threshold",
    }


def score_baseline_on_split(factory, train, validation, test, *, river_anomaly=None):
    """Fit training-only state, tune on validation, then replay through test."""
    model = factory(river_anomaly) if river_anomaly is not None else factory()
    training_time = model.fit(train.values)
    val_model = model.clone_fitted() if isinstance(model, RiverHSTBaseline) else copy.deepcopy(model)
    if isinstance(val_model, LagIForestBaseline):
        val_scores, val_time = val_model.score_stream(train.values, validation.values)
    else:
        val_scores, val_time = val_model.score_stream(validation.values)
    threshold, tuning = tune_score_threshold(validation.labels, val_scores)

    test_model = model.clone_fitted() if isinstance(model, RiverHSTBaseline) else copy.deepcopy(model)
    if isinstance(test_model, LagIForestBaseline):
        test_scores, test_time = test_model.score_stream(train.values + validation.values, test.values)
    else:
        # Replay validation observations to evolve state, discarding their
        # predictions. State updates still occur strictly after each score.
        test_model.score_stream(validation.values)
        test_scores, test_time = test_model.score_stream(test.values)
    return {
        "scores": test_scores,
        "alerts": test_scores >= threshold,
        "threshold": threshold,
        "training_time_s": training_time,
        "validation_inference_time_s": val_time,
        "inference_time_s": test_time,
        "tuning": tuning,
    }


def strict_series_report(truth, pred, scores):
    strict = compute_point_metrics(truth, pred, scores)
    events = event_window_metrics(truth, pred)
    strict.update({key: events[key] for key in (
        "event_precision", "event_recall", "event_f1", "event_tp", "event_fp", "event_fn",
        "false_alert_count",
    )})
    strict["_truth"] = np.asarray(truth, dtype=bool).tolist()
    strict["_pred"] = np.asarray(pred, dtype=bool).tolist()
    strict["_scores"] = np.asarray(scores, dtype=float).tolist()
    return strict


def main():
    parser = argparse.ArgumentParser(
        description="Strict-causal NAB/Yahoo comparison with EWMA, River HST, and Lag_IForest baselines."
    )
    parser.add_argument("--dataset", choices=["nab", "yahoo"], required=True)
    parser.add_argument("--max-series", type=int, default=None, help="Optional smoke-test limit")
    parser.add_argument("--window-size", type=int, default=200)
    parser.add_argument("--candidate-size", type=int, default=50)
    parser.add_argument("--val-fraction", type=float, default=0.2)
    parser.add_argument("--test-fraction", type=float, default=0.3)
    parser.add_argument("--output-dir", default="outputs/report comparison")
    args = parser.parse_args()

    river_anomaly = require_river()
    series_list = existing.load_dataset(
        args.dataset, max_series=args.max_series, use_windows=args.dataset == "nab"
    )
    if not series_list:
        raise RuntimeError(f"No series loaded for {args.dataset!r}")
    output_dir = Path(args.output_dir) / args.dataset
    output_dir.mkdir(parents=True, exist_ok=True)

    base = DetectorConfig(
        window_size=args.window_size,
        candidate_size=args.candidate_size,
        freeze_window=args.window_size,
        hold_steps=args.candidate_size,
    )
    tau_grid = [1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
    rows = []
    detector_times = {name: {"training_time_s": 0.0, "inference_time_s": 0.0} for name in VARIANT_ORDER}

    for series in tqdm(series_list, desc=f"{args.dataset.upper()} report benchmark", unit="series"):
        train, validation, test = temporal_split(series, args.val_fraction, args.test_fraction)
        context = TimeSeries(points=train.points + validation.points, name=series.name, metadata=dict(series.metadata))
        per_variant = {}
        thresholds = {}

        # Existing A-D: same threshold grid and strict prefix replay as the
        # established temporal benchmark. E retains configured defaults.
        for name, detector_cls in STATISTICAL.items():
            if name in existing.VARIANTS:
                best_tau, _ = tune_threshold(
                    detector_cls, base, validation, tau_grid=tau_grid,
                    metric="f1", warmup_series=train,
                )
                thresholds[name] = best_tau
                config = dataclasses.replace(base, tau=best_tau)
            elif name == "E":
                config = base
                thresholds[name] = config.e_tau_high
            else:
                # Search E2's documented parameter grid on this validation
                # block, always warming from the preceding training prefix.
                config = tune_e2_config(train, validation, base)
                thresholds[name] = {"tau_moderate": config.e2_tau_moderate,
                                    "tau_extreme": config.e2_tau_extreme}
            fit_elapsed = 0.0
            result = _run_detector_from_prefix(detector_cls, config, context, test)
            truth = np.asarray(test.labels, dtype=bool)
            pred = np.asarray(result.alerts, dtype=bool)
            scores = np.asarray(result.scores, dtype=float)
            per_variant[name] = strict_series_report(truth, pred, scores)
            per_variant[name]["literature_aligned"] = literature_series_metrics(truth, pred, scores)
            per_variant[name]["runtime"] = {"training_time_s": fit_elapsed, "inference_time_s": result.runtime_s}
            detector_times[name]["training_time_s"] += fit_elapsed
            detector_times[name]["inference_time_s"] += result.runtime_s

        # LSTM: train split fit, validation residual calibration, then test
        # after replaying the historical prefix, matching the established run.
        lstm = LSTMAnomalyDetector(**existing.LSTM_SETTINGS)
        lstm_train_time = lstm.fit(train.values)
        val_result = lstm.score(validation.values, context=train.values)
        normal_scores = [score for score, label in zip(val_result.scores, validation.labels) if not label]
        lstm_tau = lstm.threshold_from(normal_scores, target_fpr=0.01)
        lstm_result = lstm.score(test.values, tau=lstm_tau, context=context.values)
        lstm_truth = np.asarray(test.labels, dtype=bool)
        per_variant["LSTM"] = strict_series_report(lstm_truth, lstm_result.alerts, lstm_result.scores)
        per_variant["LSTM"]["literature_aligned"] = literature_series_metrics(lstm_truth, lstm_result.alerts, lstm_result.scores)
        per_variant["LSTM"]["runtime"] = {"training_time_s": lstm_train_time, "inference_time_s": lstm_result.inference_time_s}
        thresholds["LSTM"] = lstm_tau
        detector_times["LSTM"]["training_time_s"] += lstm_train_time
        detector_times["LSTM"]["inference_time_s"] += lstm_result.inference_time_s

        for name, factory in (
            ("EWMA", lambda *_: EWMABaseline()),
            ("River_HST", lambda river: RiverHSTBaseline(river)),
            ("Lag_IForest", lambda *_: LagIForestBaseline()),
        ):
            baseline = score_baseline_on_split(factory, train, validation, test, river_anomaly=river_anomaly if name == "River_HST" else None)
            truth = np.asarray(test.labels, dtype=bool)
            per_variant[name] = strict_series_report(truth, baseline["alerts"], baseline["scores"])
            per_variant[name]["literature_aligned"] = literature_series_metrics(truth, baseline["alerts"], baseline["scores"])
            per_variant[name]["runtime"] = {
                "training_time_s": baseline["training_time_s"],
                "validation_inference_time_s": baseline["validation_inference_time_s"],
                "inference_time_s": baseline["inference_time_s"],
            }
            per_variant[name]["validation_tuning"] = baseline["tuning"]
            thresholds[name] = baseline["threshold"]
            detector_times[name]["training_time_s"] += baseline["training_time_s"]
            detector_times[name]["inference_time_s"] += baseline["inference_time_s"]

        rows.append({
            "series": series.name,
            "length": len(series),
            "train_length": len(train),
            "validation_length": len(validation),
            "test_length": len(test),
            "train_anomalies": int(sum(train.labels)),
            "validation_anomalies": int(sum(validation.labels)),
            "test_anomalies": int(sum(test.labels)),
            "thresholds": thresholds,
            "metrics": per_variant,
        })

    # Collect arrays from private fields, then strip them before serialization.
    pooled = aggregate_report_metrics(rows)
    for row in rows:
        for metrics in row["metrics"].values():
            metrics.pop("_truth", None)
            metrics.pop("_pred", None)
            metrics.pop("_scores", None)

    summary = {
        "dataset": args.dataset,
        "protocol": "per-series chronological train/validation/test; thresholds chosen without test labels; score before online update",
        "variants": VARIANT_ORDER,
        "series": rows,
        "pooled": pooled,
        "runtime": detector_times,
        "complexity": {
            "EWMA": {"training": "O(N_train)", "inference": "O(1) per point", "memory": "O(1)", "settings": {"alpha": 0.05}},
            "River_HST": {"training": "O(N_train * n_trees * height)", "inference": "O(n_trees * height) per point", "memory": "O(n_trees * 2^height)", "settings": {"n_trees": 25, "height": 8, "window_size": 250}},
            "Lag_IForest": {"training": "O(T * n_estimators * subsample * log(subsample))", "inference": "O(n_estimators * log(subsample)) per point", "memory": "O(n_estimators * subsample + lookback)", "settings": {"lookback": 10, "n_estimators": 100}},
            "statistical_variants": "detector-specific online cost; no learned fitting phase",
            "LSTM": {"training": "O(E*N*L*H^2)", "inference": "O(L*H^2) per point"},
        },
    }
    json_path = output_dir / "report_benchmark_summary.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    pooled_path = output_dir / "pooled_metrics.json"
    pooled_path.write_text(json.dumps(pooled, indent=2), encoding="utf-8")
    (output_dir / "complexity_timing.json").write_text(
        json.dumps({"runtime": detector_times, "complexity": summary["complexity"]}, indent=2),
        encoding="utf-8",
    )
    print_tables(args.dataset, pooled, detector_times)
    print(f"Report saved to: {json_path.resolve()}")
    print(f"Pooled metrics saved to: {pooled_path.resolve()}")


def tune_e2_config(train, validation, base):
    """Bounded E2 validation grid using its existing point/event/FPR rule."""
    from drift_anomaly.pipeline.metrics import pooled_point_metrics

    combinations = [
        (floor, lag, auto, k, low, high)
        for floor in (0.1, 0.5)
        for lag, auto in ((1, False), (24, True))
        for k in (2, 3)
        for low, high in ((2.0, 5.0), (3.0, 8.0))
    ]
    best = None
    context = train
    for floor, lag, auto, k, low, high in tqdm(combinations, desc="E2 validation tuning", leave=False):
        cfg = dataclasses.replace(
            base, e_scale_floor_ratio=floor, e_seasonal_lag=lag, e_auto_lag=auto,
            e2_persistence_k=k, e2_tau_moderate=low, e2_tau_extreme=high,
        )
        result = _run_detector_from_prefix(VersionE2, cfg, context, validation)
        truth, pred = np.asarray(validation.labels, bool), np.asarray(result.alerts, bool)
        point_f1 = pooled_point_metrics(truth, pred)["f1"]
        event_f1 = event_window_metrics(truth, pred)["event_f1"] if truth.any() else 0.0
        fpr = float(np.mean(pred)) if not truth.any() and len(pred) else 0.0
        eligible = fpr <= base.e2_validation_zero_event_fpr_cap
        objective = 0.5 * point_f1 + 0.5 * event_f1
        candidate = (eligible, objective, -fpr, cfg)
        if best is None or candidate[:3] > best[:3]:
            best = candidate
    if best is None:
        return base
    if not best[0]:
        logger.warning("E2 validation FPR cap unsatisfied; selected lowest-FPR fallback.")
    return best[3]


def aggregate_report_metrics(rows):
    """Aggregate strict and literature metrics from per-series raw vectors."""
    pooled = {}
    for name in VARIANT_ORDER:
        truth_by_series, pred_by_series, scores_by_series = {}, {}, {}
        for row in rows:
            item = row["metrics"][name]
            truth_by_series[row["series"]] = np.asarray(item["_truth"], dtype=bool)
            pred_by_series[row["series"]] = np.asarray(item["_pred"], dtype=bool)
            scores_by_series[row["series"]] = np.asarray(item["_scores"], dtype=float)
        strict = evaluate_pointwise_macro({
            key: (truth_by_series[key], pred_by_series[key]) for key in truth_by_series
        })
        event_f1 = [
            event_window_metrics(truth_by_series[key], pred_by_series[key])["event_f1"]
            for key in truth_by_series if truth_by_series[key].any()
        ]
        literature = aggregate_literature_metrics({
            key: (truth_by_series[key], pred_by_series[key], scores_by_series[key])
            for key in truth_by_series
        })
        pooled[name] = {
            "pointwise_micro": strict["micro"],
            "event_window_macro_f1": float(np.mean(event_f1)) if event_f1 else 0.0,
            "zero_event_point_fpr": strict["micro"]["zero_event_point_fpr"],
            "zero_event_mean_fpr": strict["zero_event_series_mean_fpr"],
            "zero_event_false_alert_count": strict["zero_event_series_false_alerts"],
            "pointwise_macro_positive_series_f1": strict["macro_f1_positive_series"],
            "literature_aligned": literature,
        }
    return pooled


def print_tables(dataset, pooled, runtime):
    print("\n" + "=" * 104)
    print(f"STRICT CAUSAL BENCHMARK SUMMARY ({dataset.upper()})")
    print(f"{'Variant':<14}{'Point F1':>12}{'Event F1':>12}{'Zero FPR point':>18}{'Zero FPR mean':>17}{'Macro F1+':>14}")
    print("-" * 87)
    for name in VARIANT_ORDER:
        item = pooled[name]
        print(f"{name:<14}{item['pointwise_micro']['f1']:>12.4f}{item['event_window_macro_f1']:>12.4f}"
              f"{item['zero_event_point_fpr']:>18.4f}{item['zero_event_mean_fpr']:>17.4f}"
              f"{item['pointwise_macro_positive_series_f1']:>14.4f}")
    print("\nLITERATURE-ALIGNED METRICS (POINT-ADJUSTED & ORACLE-BEST)")
    print(f"{'Variant':<14}{'PA-F1 micro':>14}{'PA-F1 macro+':>16}{'PR-AUC+':>12}{'ROC-AUC+':>12}{'Oracle F1+':>13}{'Oracle PA-F1+':>15}")
    print("-" * 96)
    for name in VARIANT_ORDER:
        item = pooled[name]["literature_aligned"]
        print(f"{name:<14}{item['pa_micro_f1']:>14.4f}{item['pa_macro_positive_series']['pa_f1']:>16.4f}"
              f"{item['positive_series_mean_pr_auc']:>12.4f}{item['positive_series_mean_roc_auc']:>12.4f}"
              f"{item['oracle_best_f1_macro_positive_series']:>13.4f}"
              f"{item['oracle_best_pa_f1_macro_positive_series']:>15.4f}")
    print("\nTRAINING / INFERENCE TIME (seconds)")
    print(f"{'Variant':<14}{'Training':>15}{'Inference':>17}")
    for name in VARIANT_ORDER:
        print(f"{name:<14}{runtime[name]['training_time_s']:>15.3f}{runtime[name]['inference_time_s']:>17.3f}")
    print("=" * 104)


if __name__ == "__main__":
    main()
