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
