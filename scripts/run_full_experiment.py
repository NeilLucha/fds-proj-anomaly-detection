#!/usr/bin/env python3
"""
scripts/run_full_experiment.py
==============================

Automated end-to-end experiment runner:
1. Generates synthetic benchmarks
2. Evaluates all detector variants on synthetic series
3. Runs threshold tuning on validation splits
4. Generates summary evaluation reports and metrics tables
"""
from __future__ import annotations

import sys
import subprocess
from pathlib import Path


def run_cmd(cmd: list[str], desc: str) -> None:
    print(f"\n[RUNNING] {desc}...")
    print(f"Command: {' '.join(cmd)}")
    res = subprocess.run(cmd, capture_output=True, text=True)
    if res.returncode != 0:
        print(f"[ERROR] Failed: {desc}")
        print(res.stderr)
        sys.exit(res.returncode)
    else:
        if res.stdout.strip():
            print(res.stdout.strip())
        print(f"[OK] Completed: {desc}")


def main() -> None:
    root = Path(__file__).resolve().parent.parent
    py = sys.executable

    # 1. Quick sanity check
    run_cmd([py, "-m", "drift_anomaly.cli", "smoke-test"], "CLI Smoke Test")

    # 2. Generate synthetic data
    run_cmd(
        [py, "-m", "drift_anomaly.cli", "generate-synthetic", "--output-dir", "data/synthetic"],
        "Generate Synthetic Benchmarks"
    )

    # 3. Tune threshold on synthetic validation split
    run_cmd(
        [py, "-m", "drift_anomaly.cli", "tune", "--variant", "A", "--dataset", "synthetic", "--config", "configs/synthetic.yaml"],
        "Tune Threshold on Validation Split (Variant A)"
    )

    # 4. Evaluate all variants on synthetic series
    run_cmd(
        [py, "-m", "drift_anomaly.cli", "evaluate-all", "--dataset", "synthetic", "--config", "configs/synthetic.yaml"],
        "Evaluate All Variants on Synthetic Benchmark"
    )

    # 5. Generate aggregate report
    run_cmd(
        [py, "-m", "drift_anomaly.cli", "report", "--output-dir", "outputs/"],
        "Generate Summary Report"
    )

    print("\n✓ Full experiment suite completed successfully.")


if __name__ == "__main__":
    main()
