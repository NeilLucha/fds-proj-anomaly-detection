# Training, Tuning, and Evaluation Command Guide

This guide provides an end-to-end explanation of the commands used to run training (threshold & parameter tuning) and evaluation across all algorithmic variants and datasets (Synthetic, NAB, and Yahoo S5). It details **what each command does**, **how the training/evaluation workflow executes internally**, and **the guarantees enforced against data leakage**.

---

## 1. End-to-End Workflow Architecture

The sequential anomaly detection workflow operates in three strictly decoupled temporal phases:

```
Full Stream {x_1, ..., x_N}
┌───────────────────────────────┬───────────────────────────┬───────────────────────────┐
│     Train Split (60%)         │   Validation Split (20%)  │      Test Split (20%)     │
│  Initial Burn-In & Warm-Up    │   Threshold (tau) Tuning  │    Out-of-Sample Eval     │
│   (Zero Labels Consulted)     │   (Validation Labels Only)│  (Strict Pre-fitted Tau)  │
└───────────────────────────────┴───────────────────────────┴───────────────────────────┘
```

1. **Burn-in / Train Phase**: The detector establishes its initial reference baseline ($W^{ref}$ of size $L$ or $W_0$ of size $L_0$).
2. **Validation / Tuning Phase**: Grid-search over candidate anomaly thresholds $\tau \in \{1.5, 2.0, \dots, 5.0\}$ is executed on the validation slice. Ground-truth validation labels are used **only** to score the objective metric ($F_1$).
3. **Out-of-Sample Test Phase**: The optimal threshold $\tau^*$ is frozen. The detector streams through the held-out test series sequentially. When scoring point $x_t$, only prior observations $x_1, \dots, x_{t-1}$ are used. Test labels are never accessed until scoring is complete.

---

## 2. Real Datasets End-to-End Runner

A dedicated runner script executes the full tuning, evaluation, telemetry logging, and plot generation across real benchmarks:

```bash
# Run on NAB (Numenta Anomaly Benchmark) with automatic download
python scripts/run_real_benchmarks.py --dataset nab --max-series 10 --window-size 200 --candidate-size 50
python scripts/run_real_benchmarks.py --dataset nab --series-split --output-dir outputs/pooled

# Run on Yahoo Webscope S5 (requires manual placement in data/raw/yahoo_s5/)
python scripts/run_real_benchmarks.py --dataset yahoo --max-series 10 --window-size 200 --candidate-size 50
```

### What This Command Does Internally:
1. **Dataset Ingestion**: Verifies whether cached data exists. If NAB is missing, automatically downloads and extracts the official GitHub repository (`labels/combined_labels.json` and time series CSVs).
2. **Schema Normalization**: Parses CSV records into the unified internal schema `TimeSeriesPoint(timestamp, value, is_anomaly, change_point)`.
3. **Temporal Partitioning**: Splits each series chronologically (default 60% train, 20% validation, 20% test) with **no random shuffling**.
4. **Validation Tuning**: Temporal mode executes `tune_threshold()` on each series' validation interval after replaying its training prefix. `--series-split` instead selects a deterministic stratified 30% series holdout and maximizes pooled validation pointwise F1 for one shared threshold per variant.
5. **Causal Test Evaluation**: Temporal mode replays train and validation with the selected threshold before scoring the held-out interval. Series-holdout mode processes each held-out series from its beginning. Both modes record:
  - Point-level metrics: Precision, Recall, $F_1$, ROC-AUC, PR-AUC.
  - Pooled pointwise micro and positive-series macro F1, contiguous-window event F1, and separate false alerts/FPR for zero-event test series.
  - Train, validation, and test anomaly counts, so empty-positive splits are visible.
   - Drift diagnostics: Trigger count, candidate window acceptance/rejection count, runtime.
6. **Artifact & Plot Generation**:
   - `evaluation_summary.json`: Detailed per-series metrics and optimal $\tau^*$ values.
   - `metrics_variant_{A,B,C}.csv`: Tabular per-series metric rows for statistical analysis.
   - `variant_comparison.png`: Multi-metric grouped bar charts comparing Variants A, B, C, and D.
   - `plots/<series>_variant_{A,B,C}.png`: Three-panel diagnostic visualization showing:
     1. Raw signal with ground-truth anomalies and detected drift events.
     2. Sequential anomaly score $S_t$ and threshold $\tau$.
     3. Binary alert predictions $\hat{y}_t$ versus ground truth $y_t$.

---

## 3. Modular CLI Command Reference

The command-line interface `drift-anomaly` exposes individual stages for reproducible experimentation:

### 3.1 Data Preparation

#### Download NAB Benchmark
```bash
drift-anomaly download-nab --data-root data/nab
```
- **Action**: Downloads the complete Numenta Anomaly Benchmark repository (~45MB) from GitHub and unpacks labels and series into `data/nab/NAB-master/`. Skips download if already cached.

#### Check Yahoo S5 Availability
```bash
drift-anomaly check-yahoo --data-root data/yahoo
```
- **Action**: Checks if Yahoo S5 benchmark directories (`A1Benchmark` through `A4Benchmark`) are present. If missing, prints license agreement and extraction instructions.

#### Generate Synthetic Benchmarks
```bash
drift-anomaly generate-synthetic --output-dir data/synthetic --n-points 1000 --seed 42
```
- **Action**: Generates 6 deterministic synthetic series (`stationary`, `step_shift`, `variance_shift`, `spike_burst`, `isolated_spikes`, `compound`) demonstrating distinct drift and anomaly regimes.

---

### 3.2 Quick Sanity & Smoke Testing

```bash
drift-anomaly smoke-test --n-points 400
```
- **Action**: Instantiates all four variants (`VersionA`, `VersionB`, `VersionC`, `VersionD`) on a synthetic step-shift series, runs sequential streaming, verifies that scores and alerts are finite, and confirms zero unhandled exceptions.

---

### 3.3 Threshold Tuning (Validation Only)

```bash
drift-anomaly tune --variant A --dataset synthetic --config configs/synthetic.yaml --output-dir outputs/
```
- **Options**:
  - `--variant`: `A` (cautious rolling), `B` (rolling ablation), `C` (frozen delayed).
  - `--dataset`: `synthetic`, `nab`, or `yahoo`.
  - `--config`: Path to YAML configuration specifying window sizes and drift detector parameters.
- **Action**: Extracts the validation split (middle 20%), performs a grid search over $\tau \in [1.5, 5.0]$, computes validation $F_1$, and saves the results to `outputs/tune_<variant>_<dataset>.json`. **Zero test points or test labels are ever exposed.**

---

### 3.4 Out-of-Sample Evaluation

#### Single Variant Evaluation
```bash
drift-anomaly evaluate --variant A --dataset synthetic --config configs/synthetic.yaml --output-dir outputs/
```
- **Action**: Evaluates the specified variant on the test slice (last 20%) using pre-fitted parameters. Outputs Precision, Recall, $F_1$, ROC-AUC, PR-AUC, drift trigger count, and runtime to terminal and saves `outputs/eval_<variant>_<dataset>.json`.

#### Multi-Variant Comparative Evaluation
```bash
drift-anomaly evaluate-all --dataset synthetic --config configs/synthetic.yaml --output-dir outputs/
```
- **Action**: Evaluates Variants A, B, C, and D under identical data splits, outputs a comparative summary table in the terminal, and records the combined results in `outputs/eval_all_<dataset>.json`.

---

### 3.5 Aggregation & Reporting

```bash
# Human-readable table
drift-anomaly report --output-dir outputs/ --format text

# Markdown table
drift-anomaly report --output-dir outputs/ --format markdown
```
- **Action**: Scans all `eval_*.json` files in `--output-dir`, parses metrics across all runs, and generates an aggregated tabular report.

---

## 4. How Training and Evaluation Work Internally

### 4.1 Causal Robust Statistics
Given causal window $W_t^{ref} = (x_{t-L}, \dots, x_{t-1})$:
- Location: $m_t = \text{median}(W_t^{ref})$
- Scale: $\text{MAD}_t = \text{median}(\{|x_i - m_t| : x_i \in W_t^{ref}\})$
- Gaussian Consistent Scale: $\hat{\sigma}_t = 1.4826 \cdot \text{MAD}_t + \epsilon$
- Standardized Score: $S_t = |x_t - m_t| / \hat{\sigma}_t$
- Alert: $\hat{y}_t = \mathbf{1}\{S_t > \tau\}$

### 4.2 Online Drift Monitoring
- **ADWIN**: Dynamically monitors subwindows $W_0, W_1$ with counts $n_0, n_1$. Signals drift when $|\hat{\mu}_0 - \hat{\mu}_1| > \sqrt{\frac{1}{2m} \ln\left(\frac{4 |W|^2}{\delta}\right)}$ where $m = (\frac{1}{n_0} + \frac{1}{n_1})^{-1}$.
- **Page-Hinkley**: Monitors cumulative deviations $U_t = \sum (z_i - \bar{z}_t - \delta_{\text{PH}})$, firing when $U_t - \min U_k > \lambda_{\text{PH}}$.

### 4.3 Candidate Contamination Gate (Version A)
Upon drift trigger at step $t_{\text{drift}}$, Version A pauses baseline updates and collects candidate window $C = (x_{t+1}, \dots, x_{t+M})$.
It scores candidate points against the **pre-drift baseline**:
$$s_i^{(C)} = \frac{|x_i - m^{base}|}{1.4826 \cdot \text{MAD}^{base} + \epsilon}$$
The candidate window is **rejected** if:
1. Outlier fraction: $\frac{1}{M}\sum \mathbf{1}\{s_i^{(C)} > \tau_{\text{cand}}\} > \rho_{\text{max}}$ (burst of anomalies), OR
2. Scale expansion: $\frac{\text{MAD}_C + \epsilon}{\text{MAD}^{base} + \epsilon} > \kappa_{\text{max}}$ (excessive noise inflation).

If rejected, the baseline is retained without contamination. If accepted, $(m^{base}, \text{MAD}^{base}) \leftarrow (\text{median}(C), \text{MAD}(C))$.

---

## 5. Verification Commands

Run the full automated test suite to ensure environment integrity:

```bash
# Run all 52 unit and integration tests
pytest tests/ -v

# Run test suite with statement coverage report
pytest tests/ -v --cov=drift_anomaly --cov-report=term-missing

# Run end-to-end synthetic experiment runner
python scripts/run_full_experiment.py

# Run end-to-end real dataset benchmark runner
python scripts/run_real_benchmarks.py --dataset nab --max-series 5
```
