# drift-anomaly

> **Drift-aware anomaly detection for univariate time series using robust statistics and adaptive baselines.**

[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://python.org)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Version](https://img.shields.io/badge/version-0.1.0-green.svg)](pyproject.toml)

---

## Table of Contents

1. [Overview](#overview)
2. [Mathematical Summary](#mathematical-summary)
3. [Architecture](#architecture)
4. [Quickstart](#quickstart)
5. [Dataset Notes](#dataset-notes)
6. [Detectors and Comparison Baselines](#detectors-and-comparison-baselines)
7. [Evaluation Methodology](#evaluation-methodology)
8. [Configuration](#configuration)
9. [CLI Reference](#cli-reference)
10. [Testing](#testing)

---

## Overview

`drift-anomaly` detects point anomalies in real-valued, potentially non-stationary time series. It is designed around three guiding principles:

| Principle | Implementation |
|-----------|----------------|
| **Causality** | Every anomaly score at time `t` uses only data strictly prior to `t`. No look-ahead bias. |
| **Robustness** | The baseline is characterised by the *median* and *MAD*, not mean/std. Outliers do not corrupt the reference window. |
| **Drift awareness** | The reference window adapts when concept drift is detected (ADWIN or Page-Hinkley), so the scorer tracks a shifting distribution. |

Three algorithmic variants (A, B, C) explore different trade-offs between contamination safety, simplicity, and reference-window stability.

Benchmarks are supported on:
- **NAB** (Numenta Anomaly Benchmark) — auto-downloaded from GitHub
- **Yahoo Webscope S5** — manually downloaded, four sub-benchmarks (A1–A4)
- **Synthetic scenarios** — stationary, step-shift, variance-shift, spike-burst, isolated-spikes, compound

---

## Mathematical Summary

### Notation

| Symbol | Meaning |
|--------|---------|
| `x_t` | Observation at time `t` |
| `W_t^ref` | Reference window of the last `L` observations **strictly before** `t` |
| `m_t` | Rolling median of `W_t^ref` |
| `MAD_t` | Median Absolute Deviation of `W_t^ref` around `m_t` |
| `σ̂_t` | Robust scale estimate |
| `S_t` | Anomaly score |
| `τ` | Alert threshold (multiples of `σ̂`) |

### Rolling Median

```
m_t = median( x_{t-L}, ..., x_{t-1} )
```

The median is computed over the last `L = window_size` observations **prior** to the current step.

### MAD (Median Absolute Deviation)

```
MAD_t = median( |x_i - m_t| )   for i in {t-L, ..., t-1}
```

**Gaussian consistency factor (1.4826):** Under Gaussian data, `E[MAD] = σ / 1.4826`, so multiplying by `1.4826` makes MAD a consistent estimator of the standard deviation. The factor is `1 / Φ⁻¹(0.75)` where `Φ` is the standard normal CDF, because `Φ⁻¹(0.75) ≈ 0.6745`.

### Robust Scale Estimate

```
σ̂_t = 1.4826 · MAD_t + ε
```

The additive `ε = 1e-6` guards against division-by-zero when the window is constant.

### Anomaly Score

```
S_t = |x_t - m_t| / σ̂_t
```

An alert fires when `S_t > τ`. With Gaussian data, `τ = 3` corresponds to the classic "3-sigma rule" (99.7% coverage).

### Threshold Selection

`τ` is tuned on the validation split to maximise F1 score. A grid of candidates `τ ∈ {0.5, 1.0, 1.5, ..., 10.0}` is evaluated; the one yielding the best validation F1 is used for the held-out test set.

### ADWIN Drift Condition

ADWIN maintains an adaptive window and splits it at every possible position. For a split with sub-window sizes `n₀` (older) and `n₁` (newer):

```
Drift if:  |μ̂₀ - μ̂₁| > ε_cut

where:  ε_cut = sqrt( (1/(2·n₀) + 1/(2·n₁)) · ln(4·W²/δ) )
        W = n₀ + n₁  (total window size)
        δ = adwin_delta  (false-positive probability)
```

Memory is `O(log W)` via an *exponential histogram*: each bucket level `k` can hold at most `max_buckets = 5` entries, and overflow merges the two oldest into level `k+1`.

### Page-Hinkley Test

```
z̄_t = (1/t) · Σ z_i          (running mean)
U_t = Σ (z_i - z̄_t - δ)      (cumulative upward sum)
U_min = min_{i≤t} U_i

Drift (upward): U_t - U_min > λ
Drift (downward, two-sided): L_max - L_t > λ
  where  L_t = Σ -(z_i - z̄_t - δ)
```

Parameters: `λ = ph_lambda` (threshold), `δ = ph_delta` (minimum expected shift magnitude).

### Contamination Gate (Version A)

Before adopting a candidate window as the new baseline, two checks are applied. Candidate outliers are measured around the candidate's own median, so a clean level shift is not automatically rejected:

1. **Internal anomaly contamination**: `frac_anomalous = mean(|cand_i - m_candidate| / (1.4826*MAD_candidate + ε) > τ_int)`  
   Reject if `frac_anomalous > ρ_max`.

2. **Volatility gate**: `mad_ratio = MAD_candidate / (MAD_old + ε)`  
   Reject if `mad_ratio > κ_max`.

Both gates must pass for the candidate to be accepted.

---

## Architecture

```
fds-proj-anomaly-detection/
├── pyproject.toml                    # Build config, dependencies, CLI entry-point
├── README.md                         # This file
├── COMMANDS.md                       # Copy-pasteable command reference
├── LICENSE
├── configs/
│   ├── synthetic.yaml                # Config for synthetic benchmark
│   ├── nab.yaml                      # Config for NAB benchmark
│   └── yahoo.yaml                    # Config for Yahoo S5 benchmark
├── data/
│   ├── nab/                          # Auto-downloaded NAB data (gitignored)
│   └── yahoo/                        # Manually downloaded Yahoo data (gitignored)
├── docs/
│   ├── algorithm_math.md             # Full mathematical specification
│   ├── datasets_evaluation.md        # Dataset guide and evaluation methodology
│   └── tuning_ablation.md            # Hyperparameter tuning and ablation guide
├── outputs/                          # Reports, plots, CSVs (gitignored)
├── scripts/
│   └── run_full_experiment.py        # One-shot full experiment runner
├── src/
│   └── drift_anomaly/
│       ├── __init__.py               # Package init; exports key classes
│       ├── types.py                  # Dataclasses: TimeSeries, DetectorConfig, etc.
│       ├── statistics.py             # MAD, sigma_hat, anomaly_score, candidate gate
│       ├── drift.py                  # ADWINDetector, PageHinkleyDetector, factory
│       ├── cli.py                    # Click CLI: all drift-anomaly sub-commands
│       ├── data/
│       │   ├── __init__.py           # Exports SyntheticGenerator, NABLoader, YahooS5Loader
│       │   ├── schema.py             # normalize_to_schema, from_dataframe, to_dataframe
│       │   ├── synthetic.py          # SyntheticGenerator: 6 deterministic scenarios
│       │   ├── nab.py                # NABLoader: auto-download + label loading
│       │   └── yahoo_s5.py           # YahooS5Loader: A1-A4 benchmark reader
│       ├── detectors/
│       │   ├── __init__.py           # Exports VersionA, VersionB, VersionC
│       │   ├── base.py               # BaseDetector ABC
│       │   ├── version_a_cautious_rolling.py   # Version A: contamination-gated rolling
│       │   ├── version_b_rolling_adwin.py      # Version B: simple rolling + ADWIN
│       │   └── version_c_frozen_delayed.py     # Version C: frozen-window + delayed adapt
│       ├── pipeline/
│       │   ├── __init__.py           # Exports EvalPipeline, tune_threshold
│       │   ├── evaluate.py           # EvalPipeline: run, metrics, splits
│       │   └── tune.py               # tune_threshold: grid search on validation split
│       └── reporting/
│           ├── __init__.py
│           └── plots.py              # plot_scores, plot_roc, plot_pr, summary_table
└── tests/
    ├── test_statistics.py            # Unit tests for statistics.py
    ├── test_drift.py                 # Unit tests for ADWIN and Page-Hinkley
    ├── test_detectors.py             # Integration tests for all detector variants
    ├── test_data.py                  # Tests for data loaders and schema
    └── test_pipeline.py              # End-to-end pipeline and tuning tests
```

---

## Quickstart

### Prerequisites

- Python 3.10 or later
- `pip` or `pipx`

### Install

```bash
# 1. Clone the repository
git clone https://github.com/your-org/fds-proj-anomaly-detection.git
cd fds-proj-anomaly-detection

# 2. Install in editable mode with dev dependencies
pip install -e ".[dev]"

# 3. Generate synthetic data and run a smoke test
drift-anomaly generate-synthetic
drift-anomaly smoke-test

# 4. Run all four variants on synthetic data
drift-anomaly evaluate-all --dataset synthetic --config configs/synthetic.yaml

# 5. Generate a full report with plots
drift-anomaly report --output-dir outputs/
```

After `smoke-test` you should see output like:

```
[smoke-test] Variant A  F1=0.82  ROC-AUC=0.91
[smoke-test] Variant B  F1=0.79  ROC-AUC=0.89
[smoke-test] Variant C  F1=0.80  ROC-AUC=0.90
Smoke test PASSED.
```

---

## Dataset Notes

### NAB (Numenta Anomaly Benchmark)

NAB contains ~58 real-world and synthetic time series spanning server metrics, IoT sensors, and financial data.

**Auto-download** (no account required):

```bash
drift-anomaly download-nab
# Downloads ~12 MB from https://github.com/numenta/NAB to data/nab/
```

Labels are stored in `data/nab/labels/combined_labels.json` as timestamp lists per file.  
The loader converts them to per-point boolean labels with a configurable tolerance window.

### Yahoo Webscope S5

The Yahoo S5 benchmark requires a **free academic license**.

**Manual download steps:**

1. Visit https://webscope.sandbox.yahoo.com/ → Datasets → S5 (Anomaly Detection)
2. Request access with your institutional email (usually approved within 24 h)
3. Download the ZIP, extract to `data/yahoo/` so the structure is:
   ```
   data/yahoo/
   ├── A1Benchmark/   real_*.csv
   ├── A2Benchmark/   synthetic_*.csv
   ├── A3Benchmark/   A3Benchmark.csv  (or per-file)
   └── A4Benchmark/   A4Benchmark.csv
   ```
4. Verify: `drift-anomaly check-yahoo`

**Sub-benchmarks:**

| Benchmark | Type | Series | Description |
|-----------|------|--------|-------------|
| A1 | Real | 67 | Yahoo server metrics with known anomalies |
| A2 | Synthetic | 100 | Synthetic with additive outliers |
| A3 | Synthetic | 100 | Synthetic with change points |
| A4 | Synthetic | 100 | Synthetic with seasonal patterns + anomalies |

### Synthetic Generator

Six deterministic scenarios (seed=42 by default):

| Scenario | Description | Anomaly Type |
|----------|-------------|-------------|
| `stationary` | Gaussian noise, no drift | Isolated spikes |
| `step_shift` | Abrupt mean shift mid-series | Transition zone |
| `variance_shift` | Abrupt variance increase | High-spread segment |
| `spike_burst` | Burst of anomalous spikes | Clustered |
| `isolated_spikes` | Sparse point anomalies | Isolated |
| `compound` | Step + variance + spikes | Mixed |

Generate all scenarios:

```bash
drift-anomaly generate-synthetic
# Saves to data/synthetic/*.csv
```

---

## Detectors and Comparison Baselines

The project includes seven statistical or learned detectors and three additional reference baselines. Every online detector scores the current point using past observations only. The report benchmark compares all ten on NAB and Yahoo S5.

| Model | Family | Scoring and adaptation | Fit cost |
|---|---|---|---|
| **A** | Statistical, contamination-gated rolling | Rolling robust baseline; drift candidate updates pass anomaly-fraction and dispersion gates. | No learned fit |
| **B** | Statistical, simple rolling | Continuously updates a rolling robust baseline; ADWIN is used for drift telemetry. | No learned fit |
| **C** | Statistical, frozen/delayed | Keeps the initial robust baseline until a drift candidate is accepted in immediate or delayed mode. | No learned fit |
| **D** | Statistical, adaptive innovation | Scores causal innovations and volatility; clipped updates limit the effect of large residuals. | No learned fit |
| **E** | Statistical, causal residual | Uses a MAD/IQR scale floor, local trend and optional seasonal-lag residuals, persistence, and drift-gated reset. | No learned fit |
| **E2** | Statistical, dual-tier trigger | Extends E with immediate alerts for extreme scores and persistence for moderate scores. | No learned fit |
| **LSTM** | Neural forecast model | Forecasts the next point from a lookback window; threshold is calibrated using validation residuals. | Trained on the training split |
| **EWMA** | Online statistical baseline | Scores a standardized residual from causal exponentially weighted mean/variance, then updates state. | O(N) streaming initialization |
| **River_HST** | Online tree anomaly baseline | Half-Space Trees score each point before `learn_one`; feature scaling and limits use training data only. | O(N * trees * height) |
| **Lag_IForest** | Batch tree baseline | Fits Isolation Forest on training-only lag vectors `[x_(t-1), ..., x_(t-L)]`; model stays fixed during validation/test scoring. | O(T * trees * subsample * log(subsample)) |

### Statistical detector details

- **A - Cautious Rolling Baseline:** After drift, gathers an `M`-point candidate. It accepts the update only if candidate contamination and dispersion stay within configured `rho_max` and `kappa_max` limits.
- **B - Simple Rolling + ADWIN:** Uses the last `L` observations continuously and has no candidate contamination gate. This keeps the method simple but can absorb anomaly bursts into its baseline.
- **C - Frozen-Window Detector:** Anchors its baseline on the initial `L0` points. It can refresh immediately or wait `H` steps and validate a candidate before changing the reference.
- **D - Clipped Robust Innovation Tracker:** Scores `r_t = x_t - m_(t-1)` against prior robust scales. Clipped EW updates reduce the effect of large residuals on the baseline.
- **E - Causal Robust Residual:** Uses `max(1.4826*MAD, alpha*IQR, sigma_floor)`, a local linear predictor, and an optional warm-up-selected seasonal lag. It uses causal score calibration and persistence filtering.
- **E2 - Dual-Tier Spike + Persistence:** Alerts immediately above an extreme threshold; moderate scores require at least two threshold crossings in the recent K steps. Its parameters are selected on validation data with a false-positive constraint.

### Report comparison protocol

`scripts/run_report_benchmarks.py` adds EWMA, River HST, and Lag_IForest without replacing or modifying A-D, E, E2, or LSTM. It uses chronological train/validation/test splits. EWMA and HST update only after scoring the current observation; Lag_IForest uses past-only lag features and never fits on validation or test data. Added-baseline thresholds are selected on validation data using `0.5 * Point F1 + 0.5 * Event F1`, subject to a 0.10 validation negative-point FPR cap; the report marks a lowest-FPR fallback when no candidate meets the cap.

The original benchmark also supports `--series-split`: A-D share a validation-selected `tau`, while E/E2 use bounded validation searches and LSTM uses fixed architecture settings with validation residual calibration. In temporal mode, A-D tune per series, E uses configured values, and E2 selects its settings from validation data. These searches are intentionally limited rather than exhaustive; see [COMMANDS.md](COMMANDS.md) for the exact commands and tuning spaces. Progress bars show threshold selection, E/E2 validation tuning, test-series scoring, and LSTM epochs.

---

## Evaluation Methodology

### Temporal Split

Data are split **in time order** (no shuffling) to avoid look-ahead bias:

```
|<---------- train 60% ---------->|<-- val 20% -->|<-- test 20% -->|
```

- **Train**: replayed to warm-start detector and drift-monitor state.
- **Validation**: replayed after train and used for threshold tuning (`τ` selection); only validation labels score the candidate thresholds.
- **Test**: scored after replaying train and validation, preserving online state. Test labels are not used during tuning.
- The real benchmark runner also supports `--series-split`: a deterministic, anomaly-stratified 30% series holdout for one shared threshold per variant. This evaluates transfer across series; each held-out series is processed from its own start.
- Reports include pooled micro pointwise F1, macro F1 over positive test series, event/window F1, and separate false-alert counts/rates for zero-event series. The event metric counts each contiguous true-label run as one event and groups false alerts separated by at most five points.

### Leak-Free Tuning

Threshold `τ` is selected by maximising F1 on the **validation split only**. The test split is not inspected until a single final evaluation run.

### Reproducible Full-Dataset Results

Older output folders contain runs with different variants, data subsets, and protocols, so they are not a consistent comparison set. Run the ordered workflow in [RUN_ALL_EVALUATIONS.md](RUN_ALL_EVALUATIONS.md) to tune/train and evaluate A-D, E, E2, and LSTM on synthetic, NAB, and Yahoo S5 using the same series-holdout protocol. The resulting comparable reports are written to `outputs/final_comparison/<dataset>/`.

For the report comparison, run `python scripts/run_report_benchmarks.py --dataset nab` or `--dataset yahoo`. This standalone script retains A-D, E, E2, and LSTM, and adds EWMA, River Half-Space Trees (`River_HST`), and causal lag-window Isolation Forest (`Lag_IForest`). Added baselines tune with `0.5 * validation Point F1 + 0.5 * validation Event F1`, subject to a 0.10 validation negative-point FPR cap and an explicitly reported lowest-FPR fallback. It writes formatted strict/literature-aligned tables plus `report_benchmark_summary.json`, `pooled_metrics.json`, and `complexity_timing.json` under `outputs/report comparison/<dataset>/`. Install River with `python -m pip install river` (or `python -m pip install -e ".[report]"`). HST scores each point before learning it; Lag_IForest fits only on training lag windows.

Yahoo S5 is available locally under `data/raw/yahoo_s5/`; `drift-anomaly check-yahoo --data-root data/raw/yahoo_s5` verifies the A1-A4 directories before evaluation.

### Latest Report Comparison Results

The following values are from the saved report comparison pooled metrics, not a new benchmark run. NAB includes 58 series and Yahoo Webscope S5 includes 367 series. The report runner used chronological 50% train, 20% validation, and 30% test splits. Added baselines select thresholds on validation data; oracle rows below are retrospective test-label threshold sweeps and are not deployable results.

#### NAB: Strict Causal Metrics

| Pipeline | Point F1 | Event F1 macro+ | Zero-event FPR pooled | Zero-event FPR mean | Macro F1+ |
|---|---:|---:|---:|---:|---:|
| A | 0.1209 | 0.1341 | 0.1236 | 0.1439 | 0.1512 |
| B | 0.0803 | 0.1523 | 0.0588 | 0.0815 | 0.0834 |
| C | 0.2093 | 0.1510 | 0.3166 | 0.2905 | 0.2664 |
| D | 0.2788 | 0.1197 | 0.2260 | 0.2708 | 0.2971 |
| E | 0.0864 | 0.2268 | 0.1220 | 0.1605 | 0.0965 |
| E2 | 0.0917 | 0.2506 | 0.1203 | 0.1540 | 0.0946 |
| LSTM | 0.1412 | 0.2268 | 0.0181 | 0.0203 | 0.1224 |
| EWMA | 0.0310 | 0.3594 | 0.0068 | 0.0108 | 0.0380 |
| River_HST | 0.1481 | 0.3725 | 0.0132 | 0.0170 | 0.1448 |
| Lag_IForest | 0.2078 | 0.4233 | 0.0153 | 0.0214 | 0.1777 |

#### NAB: Literature-Aligned Metrics

| Pipeline | PA-F1 micro | PA-F1 macro+ | PR-AUC / AP+ | ROC-AUC+ | Oracle F1+ | Oracle PA-F1+ |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.6094 | 0.6397 | 0.2589 | 0.4972 | 0.3621 | 0.8576 |
| B | 0.7477 | 0.6810 | 0.2401 | 0.4902 | 0.3570 | 0.8770 |
| C | 0.3995 | 0.5452 | 0.2843 | 0.5334 | 0.3854 | 0.8915 |
| D | 0.4688 | 0.5899 | 0.3017 | 0.5676 | 0.4032 | 0.9091 |
| E | 0.5003 | 0.5206 | 0.2440 | 0.4918 | 0.3689 | 0.8948 |
| E2 | 0.4916 | 0.5118 | 0.2455 | 0.4852 | 0.3680 | 0.8832 |
| LSTM | 0.8065 | 0.7734 | 0.2767 | 0.5366 | 0.3763 | 0.9441 |
| EWMA | 0.7502 | 0.7131 | 0.2182 | 0.4855 | 0.3590 | 0.9322 |
| River_HST | 0.6741 | 0.5974 | 0.3088 | 0.5493 | 0.4029 | 0.8980 |
| Lag_IForest | 0.6974 | 0.6363 | 0.3529 | 0.5664 | 0.4306 | 0.8861 |

#### Yahoo S5: Strict Causal Metrics

| Pipeline | Point F1 | Event F1 macro+ | Zero-event FPR pooled | Zero-event FPR mean | Macro F1+ |
|---|---:|---:|---:|---:|---:|
| A | 0.0858 | 0.1464 | 0.0420 | 0.0420 | 0.0997 |
| B | 0.1027 | 0.1078 | 0.0179 | 0.0180 | 0.0806 |
| C | 0.0625 | 0.2567 | 0.2430 | 0.2459 | 0.1683 |
| D | 0.1147 | 0.3050 | 0.0724 | 0.0724 | 0.2621 |
| E | 0.1843 | 0.0767 | 0.0013 | 0.0013 | 0.0606 |
| E2 | 0.1726 | 0.0611 | 0.0008 | 0.0008 | 0.0418 |
| LSTM | 0.0503 | 0.3625 | 0.3042 | 0.3063 | 0.2684 |
| EWMA | 0.2288 | 0.4275 | 0.0045 | 0.0045 | 0.3652 |
| River_HST | 0.1688 | 0.2814 | 0.0173 | 0.0157 | 0.2446 |
| Lag_IForest | 0.0447 | 0.1601 | 0.2459 | 0.2476 | 0.0815 |

#### Yahoo S5: Literature-Aligned Metrics

| Pipeline | PA-F1 micro | PA-F1 macro+ | PR-AUC / AP+ | ROC-AUC+ | Oracle F1+ | Oracle PA-F1+ |
|---|---:|---:|---:|---:|---:|---:|
| A | 0.1298 | 0.1147 | 0.2072 | 0.5444 | 0.2640 | 0.2830 |
| B | 0.2147 | 0.1050 | 0.1269 | 0.5161 | 0.1678 | 0.1988 |
| C | 0.0658 | 0.1707 | 0.4302 | 0.8483 | 0.4965 | 0.5145 |
| D | 0.1185 | 0.2648 | 0.5888 | 0.8858 | 0.6437 | 0.6581 |
| E | 0.3418 | 0.0803 | 0.1583 | 0.5236 | 0.2007 | 0.2216 |
| E2 | 0.3452 | 0.0643 | 0.1137 | 0.5226 | 0.1656 | 0.1953 |
| LSTM | 0.0556 | 0.2827 | 0.5796 | 0.8480 | 0.6288 | 0.6542 |
| EWMA | 0.5044 | 0.4388 | 0.5783 | 0.9069 | 0.6380 | 0.7150 |
| River_HST | 0.2086 | 0.2584 | 0.4052 | 0.7922 | 0.4753 | 0.4886 |
| Lag_IForest | 0.0519 | 0.0960 | 0.0886 | 0.5556 | 0.1202 | 0.1546 |

Strict metrics favor different pipelines across measures: Lag_IForest has the highest NAB Event F1, while D leads NAB Point F1; EWMA leads Yahoo strict Point F1, Event F1, and Macro F1+. Point adjustment and oracle scores are intentionally separate because they can substantially exceed strict thresholded performance. The oracle values use test labels to select thresholds and should be treated only as upper-bound diagnostics.

The source artifacts are [NAB pooled metrics](<outputs/report comparison/nab/NAB_pooled_metrics.json>), [Yahoo pooled metrics](<outputs/report comparison/yahoo/Yahoo_pooled_metrics.json>), and the per-series [NAB report](<outputs/report comparison/nab/report_benchmark_summary.json>) / [Yahoo report](<outputs/report comparison/yahoo/report_benchmark_summary.json>).

### Metrics

| Metric | Formula | Notes |
|--------|---------|-------|
| Precision | `TP / (TP + FP)` | Point-wise |
| Recall | `TP / (TP + FN)` | Point-wise |
| F1 | `2·P·R / (P + R)` | Harmonic mean |
| ROC-AUC | Area under ROC curve | Score-based, threshold-free |
| PR-AUC | Area under PR curve | Better for imbalanced datasets |
| `false_drift_rate` | `# drift events not near a change-point / total drift events` | Measures spurious drift triggers |

`pointwise_fpr_fp_over_fp_plus_tn` is the pooled pointwise rate over all series. `zero_event_fpr` and `zero_event_point_fpr` are the pooled pointwise `FP/(FP+TN)` restricted to zero-event series. `zero_event_mean_fpr` is the unweighted mean of that rate calculated separately for each zero-event series; it gives series equal weight regardless of length.

The benchmark prints a second **LITERATURE-ALIGNED METRICS (POINT-ADJUSTED & ORACLE-BEST)** table. PA-F1 fills an entire contiguous true anomaly segment when at least one detector alert lands inside it, then computes ordinary pointwise F1; strict Point F1 scores the raw predictions without filling missed points. PA scores use the same validation-tuned threshold and are provided only for comparison with published literature. Per-series oracle best F1 / PA-F1 sweep thresholds on test scores and labels, so they are diagnostic upper bounds, not deployable results. PR-AUC in this table is Average Precision (AP); AUC values are averaged over positive test series containing both classes. Point adjustment can inflate apparent performance, as discussed by Kim et al. ([AAAI 2022](https://ojs.aaai.org/index.php/AAAI/article/view/20680)). See [the metric specification](docs/algorithm_math.md#5-optional-literature-aligned-metrics).

### Drift Diagnostics

`false_drift_rate` is computed with a tolerance window of ±20 steps: a drift event is considered *true* if any ground-truth `change_point` occurs within that window.

---

## Configuration

Each benchmark has a YAML config file under `configs/`. Example:

```yaml
# configs/synthetic.yaml
detector:
  window_size: 200       # L
  candidate_size: 50     # M
  tau: 3.0               # anomaly threshold (tuned on validation)
  epsilon: 1.0e-6
  drift_detector: adwin
  adwin_delta: 0.002
  ph_lambda: 50.0
  ph_delta: 0.005
  tau_cand: 3.0          # Version A gate threshold
  rho_max: 0.3           # Version A: max anomalous fraction
  kappa_max: 3.0         # Version A: max MAD ratio
  freeze_window: 200     # L0 (Version C)
  adaptation_mode: delayed
  hold_steps: 50         # H (Version C delayed)
  cooldown: 20
  feed_scores: false

evaluation:
  train_frac: 0.6
  val_frac: 0.2
  tau_grid: [0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 10.0]
  change_point_tolerance: 20

paths:
  data_root: data/synthetic
  output_dir: outputs/synthetic
```

To override a parameter without editing the YAML, use the `--set` flag (dot notation):

```bash
drift-anomaly evaluate --variant A --dataset synthetic \
  --config configs/synthetic.yaml \
  --set detector.tau=2.5 \
  --set detector.adwin_delta=0.001
```

---

## CLI Reference

```
drift-anomaly [OPTIONS] COMMAND [ARGS]...

Global options:
  --log-level TEXT    Logging level: DEBUG, INFO, WARNING, ERROR  [default: INFO]
  --version           Show version and exit.
  --help              Show help.

Commands:
  download-nab        Download NAB dataset from GitHub to data/nab/
  check-yahoo         Verify Yahoo S5 data is present and readable
  generate-synthetic  Generate all synthetic scenarios to data/synthetic/
  smoke-test          Quick sanity check: run all variants on step_shift
  evaluate            Run one variant on one dataset, print metrics
  evaluate-all        Run all four variants on one dataset, compare
  tune                Tune threshold τ on validation split for one variant
  report              Generate full HTML/PDF report with plots
```

For detailed help on any sub-command:

```bash
drift-anomaly evaluate --help
drift-anomaly tune --help
```

---

## Testing

The test suite uses `pytest` with `pytest-cov`.

### Run All Tests

```bash
pytest tests/ -v
```

### Run with Coverage

```bash
pytest tests/ -v --cov=src/drift_anomaly --cov-report=html
# Open htmlcov/index.html in a browser
```

### Run Specific Modules

```bash
pytest tests/test_statistics.py -v    # Core statistical primitives
pytest tests/test_drift.py -v         # ADWIN and Page-Hinkley detectors
pytest tests/test_detectors.py -v     # All three detector variants (integration)
pytest tests/test_data.py -v          # Data loaders and schema conversion
pytest tests/test_pipeline.py -v      # End-to-end pipeline and tuning
```

### Expected Coverage

A passing run should report ≥ 85% line coverage across all source modules.

---


## License

MIT — see [LICENSE](LICENSE) for details.
