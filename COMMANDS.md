# Command Reference

> Every command is copy-pasteable. Run from the repository root after installing with `pip install -e ".[dev]"`.

---

## 1. Environment Setup

```bash
# Install package + all dev dependencies in editable mode
pip install -e ".[dev]"

# Verify installation
drift-anomaly --version
python -c "import drift_anomaly; print(drift_anomaly.__version__)"
```

---

## 2. Download / Prepare Data

```bash
# Auto-download NAB (~12 MB) from GitHub
drift-anomaly download-nab

# Verify Yahoo S5 data is present (manual download required — see README.md)
drift-anomaly check-yahoo

# Generate all synthetic scenarios (stationary, step_shift, variance_shift,
# spike_burst, isolated_spikes, compound) to data/synthetic/
drift-anomaly generate-synthetic
```

---

## 3. Smoke Test (Quick Sanity Check)

Runs all four variants on the `step_shift` synthetic series and prints summary metrics. Should complete in < 10 s.

```bash
drift-anomaly smoke-test
```

---

## 4. Run All Four Variants

```bash
# Synthetic data
drift-anomaly evaluate-all --dataset synthetic --config configs/synthetic.yaml

# NAB (requires download)
drift-anomaly evaluate-all --dataset nab --config configs/nab.yaml

# Yahoo S5 (requires manual download)
drift-anomaly evaluate-all --dataset yahoo --config configs/yahoo.yaml
```

Full real-benchmark run (all available series, default):

```bash
python scripts/run_real_benchmarks.py --dataset nab --series-split --output-dir outputs/benchmark_v4
python scripts/run_real_benchmarks.py --dataset yahoo --series-split --output-dir outputs/benchmark_v4
```

`--series-split` tunes one shared threshold per variant on a deterministic stratified 30% series holdout and reports pooled pointwise and event/window metrics plus zero-event false-alert rates. Yahoo S5 must be downloaded under its academic license first. The benchmark script also supports temporal within-series tuning when `--series-split` is omitted.

Both commands include a small PyTorch LSTM forecasting baseline and emit `complexity_timing.json` with measured fit/inference time, parameter count, and rough asymptotic cost. PyTorch is optional for the core package; install it separately to run this comparison.

---

## 5. Tune Threshold (Validation Only)

Grid-searches `τ` on the validation split (never touches test data).

```bash
drift-anomaly tune --variant A --dataset synthetic --config configs/synthetic.yaml
drift-anomaly tune --variant B --dataset synthetic --config configs/synthetic.yaml
drift-anomaly tune --variant C --dataset synthetic --config configs/synthetic.yaml
drift-anomaly tune --variant D --dataset synthetic --config configs/synthetic.yaml
```

Tune on NAB:

```bash
drift-anomaly tune --variant A --dataset nab --config configs/nab.yaml
drift-anomaly tune --variant B --dataset nab --config configs/nab.yaml
drift-anomaly tune --variant C --dataset nab --config configs/nab.yaml
```

---

## 6. Evaluate Each Variant (Test Split)

Uses the tuned `τ` and reports final metrics on the held-out test split.

```bash
# Synthetic
drift-anomaly evaluate --variant A --dataset synthetic --config configs/synthetic.yaml
drift-anomaly evaluate --variant B --dataset synthetic --config configs/synthetic.yaml
drift-anomaly evaluate --variant C --dataset synthetic --config configs/synthetic.yaml

# NAB
drift-anomaly evaluate --variant A --dataset nab --config configs/nab.yaml
drift-anomaly evaluate --variant B --dataset nab --config configs/nab.yaml
drift-anomaly evaluate --variant C --dataset nab --config configs/nab.yaml

# Yahoo S5
drift-anomaly evaluate --variant A --dataset yahoo --config configs/yahoo.yaml
drift-anomaly evaluate --variant B --dataset yahoo --config configs/yahoo.yaml
drift-anomaly evaluate --variant C --dataset yahoo --config configs/yahoo.yaml
```

---

## 7. Override Config Parameters

Use `--set key=value` (dot notation) without editing the YAML:

```bash
drift-anomaly evaluate --variant A --dataset synthetic \
  --config configs/synthetic.yaml \
  --set detector.tau=2.5 \
  --set detector.adwin_delta=0.001 \
  --set detector.window_size=300
```

---

## 8. Generate Full Report with Plots

Produces HTML report, score time-series plots, ROC/PR curves, and a Markdown summary table.

```bash
drift-anomaly report --output-dir outputs/

# Specify dataset
drift-anomaly report --dataset nab --output-dir outputs/nab/
drift-anomaly report --dataset yahoo --output-dir outputs/yahoo/
drift-anomaly report --dataset synthetic --output-dir outputs/synthetic/
```

---

## 9. Run Test Suite

```bash
# All tests, verbose
pytest tests/ -v

# With HTML coverage report
pytest tests/ -v --cov=src/drift_anomaly --cov-report=html

# Fast (no coverage overhead)
pytest tests/ -v --no-cov
```

---

## 10. Run Specific Test Modules

```bash
pytest tests/test_statistics.py -v    # MAD, sigma_hat, anomaly_score, candidate gate
pytest tests/test_drift.py -v         # ADWINDetector, PageHinkleyDetector
pytest tests/test_detectors.py -v     # Version A / B / C integration
pytest tests/test_data.py -v          # Loaders, schema normalisation
pytest tests/test_pipeline.py -v      # EvalPipeline, tune_threshold end-to-end
```

---

## 11. Run with Verbose Logging

```bash
drift-anomaly --log-level DEBUG evaluate --variant A --dataset synthetic --config configs/synthetic.yaml

# Page-Hinkley instead of ADWIN
drift-anomaly --log-level DEBUG evaluate --variant B --dataset synthetic \
  --config configs/synthetic.yaml \
  --set detector.drift_detector=page-hinkley
```

---

## 12. One-Shot Full Experiment

Runs the complete pipeline (NAB download → synthetic generation → evaluate-all → report):

```bash
python scripts/run_full_experiment.py

# Skip NAB (useful if network is unavailable)
python scripts/run_full_experiment.py --skip-nab

# Custom output directory
python scripts/run_full_experiment.py --output-dir results/experiment_01/
```

---

## 13. Python API (Quick Reference)

```python
from drift_anomaly import DetectorConfig, TimeSeries
from drift_anomaly.detectors import VersionA, VersionB, VersionC
from drift_anomaly.data import load_series

# Load a synthetic series
ts = load_series("step_shift", dataset="synthetic")

# Configure a detector
cfg = DetectorConfig(window_size=200, tau=3.0, adwin_delta=0.002)

# Run Version A
det = VersionA(cfg)
result = det.run(ts)

print(f"F1={result.alerts}")          # list[bool]
print(f"Drift events: {result.drift_events}")
```

---

## 14. Inspect Available Series

```bash
# List all NAB series
python -c "from drift_anomaly.data import NABLoader; l=NABLoader(); print('\n'.join(l.list_series()))"

# List all Yahoo series
python -c "from drift_anomaly.data import YahooS5Loader; l=YahooS5Loader(); print('\n'.join(l.list_series()))"
```
