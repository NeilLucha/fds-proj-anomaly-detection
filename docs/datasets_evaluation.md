# Datasets and Evaluation Guide

## 1. Supported Benchmarks

### 1.1 Synthetic Benchmark Suite (`SyntheticGenerator`)
Fully deterministic data generator (`seed=42`) creating canonical anomaly & drift scenarios:
- `stationary`: Pure Gaussian process with no anomalies or changepoints.
- `step_shift`: Sudden level shift in mean with isolated spike anomalies.
- `variance_shift`: Sudden change in standard deviation.
- `spike_burst`: Concentrated burst of consecutive anomalous spikes (non-drift regime).
- `isolated_spikes`: Point anomalies in a stationary background.
- `compound`: Sequential composition of stationary, step-shift, variance-shift, and burst regimes.

### 1.2 Numenta Anomaly Benchmark (NAB)
- Automated download via GitHub archive (`numenta/NAB`).
- Ground-truth anomaly labels parsed from `labels/combined_labels.json`.
- Standardized into unified `TimeSeriesPoint(timestamp, value, is_anomaly, change_point)`.

### 1.3 Yahoo Webscope S5
- Covers benchmarks A1, A2, A3, and A4 (especially A4 for change-points + anomalies).
- Requires academic access agreement from Yahoo Research.
- Automatic verification via `drift-anomaly check-yahoo` with clear extraction path instructions (`data/yahoo/` or `data/raw/yahoo_s5/`).

---

## 2. Leak-Free Validation & Evaluation Methodology

```
Temporal Order: [-------- Train (60%) --------][---- Validation (20%) ----][---- Test (20%) ----]
                                                          │                           │
                                                    Tuning tau (F1)             Final Evaluation
                                                    No Test Leakage             Strict Out-of-Sample
```

- **Strict Non-Shuffled Temporal Splitting**: Time series order is preserved exactly.
- **Threshold Tuning**: $\tau \in [1.5, 5.0]$ is selected to maximize $F_1$ on validation outputs after replaying the training prefix to establish detector state.
- **Out-of-Sample Testing**: The tuned detector is replayed through train and validation, then scores the held-out test interval with its causal state intact.
- **Split Diagnostics**: Per-series reports include anomaly counts in train, validation, and test; thresholds are weakly identified when validation has no positives.

---

## 3. Reported Metrics

For benchmark summaries, `pointwise_fpr_fp_over_fp_plus_tn` is pooled pointwise `FP/(FP+TN)` across all series. `zero_event_fpr` and `zero_event_point_fpr` are pooled pointwise `FP/(FP+TN)` restricted to points from series with no positive labels. `zero_event_mean_fpr` (also `zero_event_series_mean_fpr` in the JSON summary) is the unweighted mean of the per-series `FP/(FP+TN)` on those zero-event series. The pooled rate weights longer series more; the mean rate gives every series equal weight. Neither is alert-cluster precision.

The optional literature-aligned section reports point-adjusted (PA) precision/recall/F1 at the same validation-selected threshold, micro and positive-series macro PA-F1, per-positive-series Average Precision and ROC-AUC, and per-series oracle best unadjusted F1 and PA-F1. Point adjustment fills a complete contiguous true anomaly window after any one point in the window is detected. Strict metrics do not do this. Oracle scores use test labels to sweep thresholds and are reported only as retrospective upper bounds. PA is included for comparability, with the known overestimation caveat from Kim et al., AAAI 2022 ([paper](https://ojs.aaai.org/index.php/AAAI/article/view/20680)).

## Standalone report comparison

`scripts/run_report_benchmarks.py` compares the established A-D, E, E2, and LSTM variants against EWMA residual, River Half-Space Trees, and causal lag-window Isolation Forest on NAB or Yahoo S5. All series use chronological train/validation/test splits. Added-baseline thresholds maximize `0.5 * validation Point F1 + 0.5 * validation Event F1` subject to a 0.10 false-positive rate over validation negative points; the report marks a lowest-FPR fallback if no threshold meets the cap. EWMA and HST score before updating their online state; HST scaling is derived from training values and clipped to fixed limits. Lag_IForest uses only training lag windows to fit and past values to construct every validation/test feature. It does not refit on validation or test.

Install River with `python -m pip install river` (or `python -m pip install -e ".[report]"`); the script fails clearly if it is missing. Outputs default to `outputs/report comparison/<dataset>/` and include `report_benchmark_summary.json`, standalone `pooled_metrics.json`, and `complexity_timing.json` with strict metrics, literature-aligned metrics, runtime, and complexity. Use `--max-series 1` for a smoke check; omit it for the complete benchmark.

### Point-Level Anomaly Detection
- **Precision**: $\frac{\text{TP}}{\text{TP} + \text{FP}}$
- **Recall**: $\frac{\text{TP}}{\text{TP} + \text{FN}}$
- **F1 Score**: $2 \cdot \frac{\text{Precision} \cdot \text{Recall}}{\text{Precision} + \text{Recall}}$
- **ROC-AUC**: Area under ROC curve on continuous scores $S_t$.
- **PR-AUC**: Area under Precision-Recall curve.

### Drift & Telemetry Diagnostics
- `n_drift_triggers`: Total number of drift detection events.
- `false_drift_rate`: Drift triggers occurring far from true changepoints.
- `adaptation_accepted`: Count of post-drift candidate windows accepted into baseline.
- `adaptation_rejected`: Count of candidate windows rejected by the contamination gate.
- `runtime_s`: Wall-clock execution time per series.
