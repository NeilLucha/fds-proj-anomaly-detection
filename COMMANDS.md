# Commands

Run commands from the repository root. For the full ordered workflow across synthetic, NAB, and Yahoo, see [RUN_ALL_EVALUATIONS.md](RUN_ALL_EVALUATIONS.md).

## Setup and checks

```powershell
python -m pip install -e ".[dev]"
python -m pip install torch
python -m pytest -q
drift-anomaly smoke-test
```

## Data

```powershell
drift-anomaly download-nab --data-root data/raw/nab
drift-anomaly check-yahoo --data-root data/raw/yahoo_s5
drift-anomaly generate-synthetic --output-dir data/synthetic
```

Yahoo S5 must already be licensed and extracted. `configs/yahoo.yaml` and the default Yahoo loader use `data/raw/yahoo_s5`.

## Full comparison: tune/train and test A-D, E, E2, and LSTM

```powershell
python scripts/run_real_benchmarks.py --dataset synthetic --series-split --output-dir outputs/final_comparison
python scripts/run_real_benchmarks.py --dataset nab --series-split --output-dir outputs/final_comparison
python scripts/run_real_benchmarks.py --dataset yahoo --series-split --output-dir outputs/final_comparison
```

Each run performs validation threshold selection, E and E2 validation ablations, LSTM fitting and validation threshold calibration, then scores held-out test data. Results go under `outputs/final_comparison/<dataset>/`, including per-variant CSVs, pooled strict and literature-aligned metrics, timing/complexity, a JSON summary, and plots. The terminal prints a strict summary followed by `LITERATURE-ALIGNED METRICS (POINT-ADJUSTED & ORACLE-BEST)`. Oracle metrics use test labels for threshold sweeps and are diagnostic upper bounds only. Progress bars show tuning, ablation, test series, and LSTM epochs.

## Report comparison with reference baselines

Install the optional River dependency and run one command per dataset:

```powershell
python -m pip install river
python scripts/run_report_benchmarks.py --dataset nab --output-dir "outputs/report comparison"
python scripts/run_report_benchmarks.py --dataset yahoo --output-dir "outputs/report comparison"
```

This standalone runner compares A-D, E, E2, LSTM, EWMA, `River_HST`, and `Lag_IForest`. Added-baseline thresholds maximize `0.5 * validation Point F1 + 0.5 * validation Event F1` under a 0.10 validation negative-point FPR cap; reports mark a lowest-FPR fallback if needed. Add `--max-series 1` for a small end-to-end smoke run. If River is unavailable, the script stops with an install message instead of omitting HST. Reports are saved under `outputs/report comparison/<dataset>/` as `report_benchmark_summary.json` (series and pooled results), `pooled_metrics.json`, and `complexity_timing.json`.

With `--series-split`, A-D share a validation-selected `tau` from `[1.5, 2.0, ..., 5.0]`. E and E2 search bounded validation grids over scale floor, lag, persistence, and trigger thresholds. E2 maximizes `0.5 * point F1 + 0.5 * positive-series macro Event F1`, with a zero-event macro-FPR cap of 0.10; a lowest-FPR fallback is reported if no candidate meets that cap. LSTM settings are fixed (lookback 24, hidden width 16, five epochs, learning rate 0.01); its threshold uses the 99th percentile of normal validation residuals. Other settings come from the command line or config and are not searched.

## Single-series CLI

```powershell
drift-anomaly tune --variant E2 --dataset yahoo --config configs/yahoo.yaml
drift-anomaly evaluate --variant E2 --dataset yahoo --config configs/yahoo.yaml
drift-anomaly evaluate-all --dataset yahoo --config configs/yahoo.yaml
```

`tune` searches only `tau` for one series. `evaluate` scores one series with the supplied config after replaying its earlier prefix. `evaluate-all` compares statistical variants A-D, E, and E2; use the full benchmark script for multi-series comparison and LSTM.
