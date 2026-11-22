# Run All Evaluations

Run these PowerShell commands from the repository root, in order. Each benchmark command tunes/calibrates on its validation split, fits the LSTM, and evaluates on held-out test data. Wait for each command to finish before starting the next.

## 1. Install dependencies

```powershell
python -m pip install -e ".[dev]"
python -m pip install torch
python -m pip install river
```

## 2. Verify code and local Yahoo data

```powershell
python -m pytest -q
drift-anomaly smoke-test
drift-anomaly check-yahoo --data-root data/raw/yahoo_s5
```

## 3. Prepare NAB

```powershell
drift-anomaly download-nab --data-root data/raw/nab
```

## 4. Tune/train and evaluate every pipeline on each dataset

Each run compares A, B, C, D, E, E2, and LSTM. Do not add `--max-series` for a complete benchmark.

```powershell
python scripts/run_real_benchmarks.py --dataset synthetic --series-split --output-dir outputs/final_comparison
python scripts/run_real_benchmarks.py --dataset nab --series-split --output-dir outputs/final_comparison
python scripts/run_real_benchmarks.py --dataset yahoo --series-split --output-dir outputs/final_comparison
```

The results for each dataset are saved separately in `outputs/final_comparison/synthetic/`, `outputs/final_comparison/nab/`, and `outputs/final_comparison/yahoo/`. Each folder contains the evaluation summary, per-variant metrics, pooled strict and literature-aligned metrics, complexity/timing report, and plots. The terminal prints the strict comparison first, followed by `LITERATURE-ALIGNED METRICS (POINT-ADJUSTED & ORACLE-BEST)`. Oracle metrics sweep test thresholds and are diagnostic upper bounds, not tuned test results. The progress bars track threshold tuning, E/E2 validation ablations, test-series evaluation, and LSTM epochs.

NAB is downloaded by the command above. Yahoo S5 must be present under `data/raw/yahoo_s5/` with `A1Benchmark/` through `A4Benchmark/`; this guide assumes its usage terms have already been accepted.

The benchmark's tuning is intentionally bounded: A-D tune `tau` on validation data; E and E2 search bounded validation grids in series-split mode. E2 selects by `0.5 * point F1 + 0.5 * positive-series macro Event F1`, subject to a 0.10 zero-event macro-FPR cap; the summary marks a lowest-FPR fallback if no candidate meets the cap. LSTM uses fixed architecture/training settings and calibrates its alert threshold from validation residuals. Window and drift-detector settings are supplied through CLI/config values, not exhaustively optimized. See `COMMANDS.md` for details.

## 5. Report comparison with additional reference baselines

After the A-D/E/E2/LSTM comparisons, run these full NAB and Yahoo report benchmarks. Each uses the same chronological train/validation/test split and includes EWMA, River HST, and Lag_IForest:

```powershell
python scripts/run_report_benchmarks.py --dataset nab --output-dir "outputs/report comparison"
python scripts/run_report_benchmarks.py --dataset yahoo --output-dir "outputs/report comparison"
```

The formatted terminal output has strict and literature-aligned metric tables plus training/inference times. Added baseline thresholds use the shared validation objective `0.5 * Point F1 + 0.5 * Event F1` under a 0.10 negative-point FPR cap; reports identify a lowest-FPR fallback if necessary. Each dataset folder contains `report_benchmark_summary.json`, `pooled_metrics.json`, and `complexity_timing.json`. Add `--max-series 1` to smoke-check the integration before a full run. River is required so the comparison never silently drops HST.
