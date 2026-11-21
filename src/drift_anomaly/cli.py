"""
drift_anomaly.cli
=================

Command-line interface for the drift-aware anomaly detection library.

Usage
-----
::

    drift-anomaly download-nab
    drift-anomaly check-yahoo
    drift-anomaly generate-synthetic
    drift-anomaly tune       --variant A --dataset synthetic --config configs/synthetic.yaml
    drift-anomaly evaluate   --variant A --dataset synthetic --config configs/synthetic.yaml
    drift-anomaly evaluate-all --dataset synthetic --config configs/synthetic.yaml
    drift-anomaly report     --output-dir outputs/
    drift-anomaly smoke-test
"""
from __future__ import annotations

import logging
import sys
import time
from pathlib import Path
from typing import Optional

import click
import yaml
from tqdm.auto import tqdm

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Logging setup helper
# ---------------------------------------------------------------------------

def _setup_logging(verbose: bool = False) -> None:
    level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
        datefmt="%H:%M:%S",
        level=level,
        stream=sys.stderr,
    )


def _load_config(config_path: Optional[str]) -> dict:
    """Load a YAML config file and return it as a dict."""
    if config_path is None:
        return {}
    p = Path(config_path)
    if not p.exists():
        raise click.ClickException(f"Config file not found: {p}")
    with p.open() as fh:
        data = yaml.safe_load(fh) or {}
    logger.debug("Loaded config from %s: %s", p, data)
    return data


def _config_to_detector_config(cfg: dict):
    """Convert a YAML config dict to a DetectorConfig object."""
    from drift_anomaly.types import DetectorConfig

    allowed = {f.name for f in DetectorConfig.__dataclass_fields__.values()}  # type: ignore[union-attr]
    kwargs = {k: v for k, v in cfg.items() if k in allowed}
    return DetectorConfig(**kwargs)


# ---------------------------------------------------------------------------
# CLI group
# ---------------------------------------------------------------------------

@click.group()
@click.option("--verbose", "-v", is_flag=True, default=False, help="Enable debug logging.")
@click.pass_context
def cli(ctx: click.Context, verbose: bool) -> None:
    """Drift-aware anomaly detection toolkit.

    Use the sub-commands below to download data, tune thresholds, evaluate
    detectors, and generate reports.
    """
    ctx.ensure_object(dict)
    ctx.obj["verbose"] = verbose
    _setup_logging(verbose)


# ---------------------------------------------------------------------------
# download-nab
# ---------------------------------------------------------------------------

@cli.command("download-nab")
@click.option(
    "--data-root",
    default="data/nab",
    show_default=True,
    help="Directory to download NAB data into.",
)
@click.pass_context
def download_nab(ctx: click.Context, data_root: str) -> None:
    """Download the Numenta Anomaly Benchmark (NAB) dataset from GitHub."""
    _setup_logging(ctx.obj.get("verbose", False))
    from drift_anomaly.data.nab import NABLoader

    click.echo(f"Downloading NAB dataset to '{data_root}' …")
    loader = NABLoader(data_root=data_root)
    try:
        loader.download()
        click.echo(click.style("✓ NAB download complete.", fg="green"))
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(f"NAB download failed: {exc}") from exc


# ---------------------------------------------------------------------------
# check-yahoo
# ---------------------------------------------------------------------------

@cli.command("check-yahoo")
@click.option(
    "--data-root",
    default="data/raw/yahoo_s5",
    show_default=True,
    help="Root directory where Yahoo S5 data should be placed.",
)
@click.pass_context
def check_yahoo(ctx: click.Context, data_root: str) -> None:
    """Check whether the Yahoo S5 benchmark dataset is available locally."""
    _setup_logging(ctx.obj.get("verbose", False))
    from drift_anomaly.data.yahoo_s5 import YahooS5Loader

    loader = YahooS5Loader(data_root=data_root)
    available = loader.is_available()
    if available:
        click.echo(click.style(f"✓ Yahoo S5 data found at '{data_root}'.", fg="green"))
    else:
        click.echo(
            click.style("✗ Yahoo S5 data NOT found.", fg="yellow")
            + "\n\nThe Yahoo Webscope S5 dataset requires a manual academic license download:\n"
            "  1. Visit https://webscope.sandbox.yahoo.com/catalog.php?datatype=s\n"
            "  2. Request access and download 'ydata-labeled-time-series-anomalies-v1_0.zip'\n"
            f"  3. Extract it into: {Path(data_root).resolve()}\n"
        )
        sys.exit(1)


# ---------------------------------------------------------------------------
# generate-synthetic
# ---------------------------------------------------------------------------

@cli.command("generate-synthetic")
@click.option(
    "--output-dir",
    default="data/synthetic",
    show_default=True,
    help="Directory to save generated CSV files.",
)
@click.option(
    "--n-points",
    default=1000,
    show_default=True,
    help="Number of observations per series.",
)
@click.option(
    "--seed",
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.pass_context
def generate_synthetic(
    ctx: click.Context, output_dir: str, n_points: int, seed: int
) -> None:
    """Generate synthetic benchmark time series and save them to disk."""
    _setup_logging(ctx.obj.get("verbose", False))
    import pandas as pd
    from drift_anomaly.data.synthetic import SyntheticGenerator
    from drift_anomaly.data.schema import to_dataframe

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    gen = SyntheticGenerator(seed=seed)
    series_methods = [
        "stationary",
        "step_shift",
        "variance_shift",
        "spike_burst",
        "isolated_spikes",
        "compound",
    ]

    click.echo(f"Generating {len(series_methods)} synthetic series → '{out}'")
    for name in series_methods:
        method = getattr(gen, name)
        try:
            ts = method(n=n_points)
        except TypeError:
            ts = method()
        df = to_dataframe(ts)
        fp = out / f"{name}.csv"
        df.to_csv(fp, index=False)
        click.echo(f"  Saved {len(ts)} points → {fp}")

    click.echo(click.style("✓ Synthetic data generation complete.", fg="green"))


# ---------------------------------------------------------------------------
# tune
# ---------------------------------------------------------------------------

@cli.command("tune")
@click.option(
    "--variant",
    type=click.Choice(["A", "B", "C", "D", "E", "E2"], case_sensitive=True),
    default="A",
    show_default=True,
    help="Detector variant to tune.",
)
@click.option(
    "--dataset",
    type=click.Choice(["synthetic", "nab", "yahoo"], case_sensitive=False),
    default="synthetic",
    show_default=True,
    help="Dataset to use for tuning.",
)
@click.option(
    "--config",
    "config_path",
    default=None,
    help="Path to a YAML config file (overrides defaults).",
)
@click.option(
    "--series",
    default=None,
    help="Series name within the dataset (default: first available).",
)
@click.option(
    "--output-dir",
    default="outputs/",
    show_default=True,
    help="Directory to write tuning results.",
)
@click.pass_context
def tune(
    ctx: click.Context,
    variant: str,
    dataset: str,
    config_path: Optional[str],
    series: Optional[str],
    output_dir: str,
) -> None:
    """Tune the anomaly threshold τ on the validation split of a dataset."""
    _setup_logging(ctx.obj.get("verbose", False))
    from drift_anomaly.pipeline import tune_threshold, temporal_split
    from drift_anomaly.data import load_series
    import json

    cfg_dict = _load_config(config_path)
    # Allow config to override dataset/variant
    dataset = cfg_dict.pop("dataset", dataset)
    variant = cfg_dict.pop("variant", variant)

    det_cfg = _config_to_detector_config(cfg_dict)

    # Resolve series name
    _series_name = series or _default_series(dataset)
    click.echo(
        f"Tuning variant {variant} on dataset='{dataset}', series='{_series_name}'"
    )

    ts = _load_ts(dataset, _series_name, cfg_dict)
    val_fraction = cfg_dict.get("val_fraction", 0.2)
    test_fraction = cfg_dict.get("test_fraction", 0.2)
    train_ts, val_ts, _ = temporal_split(ts, val_fraction=val_fraction, test_fraction=test_fraction)

    from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2
    detector_map = {"A": VersionA, "B": VersionB, "C": VersionC, "D": VersionD, "E": VersionE, "E2": VersionE2}
    det_cls = detector_map.get(variant, VersionA)

    best_tau, results = tune_threshold(
        DetectorClass=det_cls,
        config=det_cfg,
        val_series=val_ts,
        warmup_series=train_ts,
    )

    best_f1 = results.get(best_tau).f1 if (results and best_tau in results) else 0.0
    click.echo(f"Best τ = {best_tau:.4f}  |  Val F1 = {best_f1:.4f}")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"tune_{variant}_{dataset}.json"
    import dataclasses
    serializable_results = {str(k): dataclasses.asdict(v) for k, v in results.items()}
    with result_file.open("w") as fh:
        json.dump({"best_tau": best_tau, "best_f1": best_f1, "grid": serializable_results}, fh, indent=2)
    click.echo(f"Results saved → {result_file}")


# ---------------------------------------------------------------------------
# evaluate
# ---------------------------------------------------------------------------

@cli.command("evaluate")
@click.option(
    "--variant",
    type=click.Choice(["A", "B", "C", "D", "E", "E2"], case_sensitive=True),
    default="A",
    show_default=True,
)
@click.option(
    "--dataset",
    type=click.Choice(["synthetic", "nab", "yahoo"], case_sensitive=False),
    default="synthetic",
    show_default=True,
)
@click.option("--config", "config_path", default=None, help="Path to YAML config.")
@click.option("--series", default=None, help="Series name within dataset.")
@click.option("--output-dir", default="outputs/", show_default=True)
@click.pass_context
def evaluate(
    ctx: click.Context,
    variant: str,
    dataset: str,
    config_path: Optional[str],
    series: Optional[str],
    output_dir: str,
) -> None:
    """Evaluate a single detector variant on a dataset and report metrics."""
    _setup_logging(ctx.obj.get("verbose", False))
    from drift_anomaly.pipeline.tune import _run_detector_from_prefix, temporal_split
    from drift_anomaly.pipeline.metrics import compute_metrics
    from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2
    from drift_anomaly.data import load_series
    from drift_anomaly.types import TimeSeries
    import json

    cfg_dict = _load_config(config_path)
    dataset = cfg_dict.pop("dataset", dataset)
    variant = cfg_dict.pop("variant", variant)
    det_cfg = _config_to_detector_config(cfg_dict)

    _series_name = series or _default_series(dataset)
    click.echo(
        f"Evaluating variant {variant} on dataset='{dataset}', series='{_series_name}'"
    )

    ts = _load_ts(dataset, _series_name, cfg_dict)
    val_fraction = cfg_dict.get("val_fraction", 0.2)
    test_fraction = cfg_dict.get("test_fraction", 0.2)
    train_ts, val_ts, test_ts = temporal_split(ts, val_fraction=val_fraction, test_fraction=test_fraction)
    context_ts = TimeSeries(points=train_ts.points + val_ts.points, name=ts.name,
                            metadata=dict(ts.metadata))

    detector_map = {"A": VersionA, "B": VersionB, "C": VersionC, "D": VersionD, "E": VersionE, "E2": VersionE2}
    det_cls = detector_map.get(variant, VersionA)

    t0 = time.perf_counter()
    result = _run_detector_from_prefix(det_cls, det_cfg, context_ts, test_ts, show_progress=True)
    metrics = compute_metrics(result, test_ts)
    elapsed = time.perf_counter() - t0

    click.echo("\n── Evaluation Results ─────────────────────────")
    click.echo(f"  Precision : {metrics.precision:.4f}")
    click.echo(f"  Recall    : {metrics.recall:.4f}")
    click.echo(f"  F1        : {metrics.f1:.4f}")
    click.echo(f"  ROC-AUC   : {metrics.roc_auc:.4f}")
    click.echo(f"  PR-AUC    : {metrics.pr_auc:.4f}")
    click.echo(f"  Drift #   : {metrics.n_drift_triggers}")
    click.echo(f"  Runtime   : {elapsed:.2f}s")

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"eval_{variant}_{dataset}.json"
    import dataclasses
    with result_file.open("w") as fh:
        json.dump(dataclasses.asdict(metrics), fh, indent=2)
    click.echo(f"\nResults saved → {result_file}")


# ---------------------------------------------------------------------------
# evaluate-all
# ---------------------------------------------------------------------------

@cli.command("evaluate-all")
@click.option(
    "--dataset",
    type=click.Choice(["synthetic", "nab", "yahoo"], case_sensitive=False),
    default="synthetic",
    show_default=True,
)
@click.option("--config", "config_path", default=None, help="Path to YAML config.")
@click.option("--series", default=None, help="Series name within dataset.")
@click.option("--output-dir", default="outputs/", show_default=True)
@click.pass_context
def evaluate_all(
    ctx: click.Context,
    dataset: str,
    config_path: Optional[str],
    series: Optional[str],
    output_dir: str,
) -> None:
    """Evaluate statistical detector variants A through E2 and compare results."""
    _setup_logging(ctx.obj.get("verbose", False))
    from drift_anomaly.pipeline.tune import _run_detector_from_prefix, temporal_split
    from drift_anomaly.pipeline.metrics import compute_metrics
    from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2
    from drift_anomaly.types import TimeSeries
    import json, dataclasses

    cfg_dict = _load_config(config_path)
    dataset = cfg_dict.pop("dataset", dataset)
    det_cfg = _config_to_detector_config(cfg_dict)

    _series_name = series or _default_series(dataset)
    ts = _load_ts(dataset, _series_name, cfg_dict)
    val_fraction = cfg_dict.get("val_fraction", 0.2)
    test_fraction = cfg_dict.get("test_fraction", 0.2)
    train_ts, val_ts, test_ts = temporal_split(ts, val_fraction=val_fraction, test_fraction=test_fraction)
    context_ts = TimeSeries(points=train_ts.points + val_ts.points, name=ts.name,
                            metadata=dict(ts.metadata))

    detector_map = {"A": VersionA, "B": VersionB, "C": VersionC, "D": VersionD, "E": VersionE, "E2": VersionE2}
    all_results = {}
    for variant in tqdm(["A", "B", "C", "D", "E", "E2"], desc="Evaluating variants"):
        click.echo(f"Evaluating variant {variant} …")
        try:
            det_cls = detector_map[variant]
            result = _run_detector_from_prefix(det_cls, det_cfg, context_ts, test_ts, show_progress=True)
            all_results[variant] = dataclasses.asdict(compute_metrics(result, test_ts))
        except Exception as exc:  # noqa: BLE001
            click.echo(click.style(f"  ✗ Variant {variant} failed: {exc}", fg="red"))
            all_results[variant] = {"error": str(exc)}

    # Summary table
    click.echo("\n── All-Variants Summary ───────────────────────")
    click.echo(f"{'Variant':<10} {'Precision':<12} {'Recall':<10} {'F1':<10} {'ROC-AUC':<10}")
    for v, r in all_results.items():
        if "error" in r:
            click.echo(f"{v:<10} ERROR: {r['error']}")
        else:
            click.echo(
                f"{v:<10} {r.get('precision', 0):<12.4f} {r.get('recall', 0):<10.4f}"
                f" {r.get('f1', 0):<10.4f} {r.get('roc_auc', 0):<10.4f}"
            )

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)
    result_file = out / f"eval_all_{dataset}.json"
    with result_file.open("w") as fh:
        json.dump(all_results, fh, indent=2)
    click.echo(f"\nResults saved → {result_file}")


# ---------------------------------------------------------------------------
# report
# ---------------------------------------------------------------------------

@cli.command("report")
@click.option(
    "--output-dir",
    default="outputs/",
    show_default=True,
    help="Directory containing evaluation JSON files.",
)
@click.option(
    "--format",
    "fmt",
    type=click.Choice(["text", "markdown"], case_sensitive=False),
    default="text",
    show_default=True,
    help="Output format for the summary report.",
)
@click.pass_context
def report(ctx: click.Context, output_dir: str, fmt: str) -> None:
    """Aggregate all evaluation JSON files in --output-dir into a summary report."""
    _setup_logging(ctx.obj.get("verbose", False))
    import json

    out = Path(output_dir)
    if not out.exists():
        raise click.ClickException(f"Output directory not found: {out}")

    json_files = sorted(out.glob("eval_*.json"))
    if not json_files:
        click.echo(
            f"No evaluation result files (eval_*.json) found in '{out}'.\n"
            "Run 'drift-anomaly evaluate' or 'drift-anomaly evaluate-all' first."
        )
        return

    rows = []
    for fp in json_files:
        with fp.open() as fh:
            data = json.load(fh)
        # Could be a flat dict (single variant) or nested (all-variants)
        if isinstance(data, dict) and any(k in data for k in ["A", "B", "C"]):
            for variant, metrics in data.items():
                rows.append({"file": fp.name, "variant": variant, **metrics})
        else:
            rows.append({"file": fp.name, **data})

    if fmt == "markdown":
        click.echo("| File | Variant | Precision | Recall | F1 | ROC-AUC |")
        click.echo("|------|---------|-----------|--------|-----|---------|")
        for r in rows:
            click.echo(
                f"| {r.get('file','')} | {r.get('variant','')} "
                f"| {r.get('precision',float('nan')):.4f} "
                f"| {r.get('recall',float('nan')):.4f} "
                f"| {r.get('f1',float('nan')):.4f} "
                f"| {r.get('roc_auc',float('nan')):.4f} |"
            )
    else:
        click.echo(f"{'File':<30} {'Var':<6} {'Prec':<8} {'Rec':<8} {'F1':<8} {'ROC-AUC':<8}")
        click.echo("-" * 70)
        for r in rows:
            click.echo(
                f"{r.get('file',''):<30} {r.get('variant',''):<6}"
                f" {r.get('precision',float('nan')):<8.4f}"
                f" {r.get('recall',float('nan')):<8.4f}"
                f" {r.get('f1',float('nan')):<8.4f}"
                f" {r.get('roc_auc',float('nan')):<8.4f}"
            )


# ---------------------------------------------------------------------------
# smoke-test
# ---------------------------------------------------------------------------

@cli.command("smoke-test")
@click.option(
    "--n-points",
    default=400,
    show_default=True,
    help="Number of observations in the test series.",
)
@click.pass_context
def smoke_test(ctx: click.Context, n_points: int) -> None:
    """Quick end-to-end sanity check — runs variants A through E2 on synthetic data."""
    _setup_logging(ctx.obj.get("verbose", False))
    import math
    from drift_anomaly.data.synthetic import SyntheticGenerator
    from drift_anomaly.types import DetectorConfig
    from drift_anomaly.detectors import VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2

    click.echo(f"Running smoke test with n_points={n_points} …")
    gen = SyntheticGenerator(seed=0)
    ts = gen.step_shift(n=n_points)

    cfg = DetectorConfig(
        window_size=min(100, n_points // 4),
        candidate_size=min(30, n_points // 8),
        cooldown=10,
        hold_steps=20,
    )

    all_ok = True
    for Cls in tqdm([VersionA, VersionB, VersionC, VersionD, VersionE, VersionE2], desc="Smoke test"):
        name = Cls.__name__
        t0 = time.perf_counter()
        try:
            det = Cls(cfg)
            result = det.run(ts)
            n_alerts = sum(result.alerts)
            n_scores = sum(1 for s in result.scores if not math.isnan(s))
            elapsed = time.perf_counter() - t0
            click.echo(
                click.style(f"  ✓ {name}", fg="green")
                + f"  alerts={n_alerts}  valid_scores={n_scores}  "
                f"drift_events={len(result.drift_events)}  {elapsed:.3f}s"
            )
        except Exception as exc:  # noqa: BLE001
            click.echo(click.style(f"  ✗ {name}: {exc}", fg="red"))
            all_ok = False

    if all_ok:
        click.echo(click.style("\n✓ Smoke test PASSED.", fg="green", bold=True))
    else:
        click.echo(click.style("\n✗ Smoke test FAILED — see errors above.", fg="red", bold=True))
        sys.exit(1)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _default_series(dataset: str) -> str:
    """Return a sensible default series name for each dataset."""
    defaults = {
        "synthetic": "step_shift",
        "nab": "realKnownCause/machine_temperature_system_failure.csv",
        "yahoo": "A1/real_1.csv",
    }
    return defaults.get(dataset, "step_shift")


def _load_ts(dataset: str, series_name: str, cfg_dict: dict):
    """Load a TimeSeries, handling synthetic/nab/yahoo differences."""
    from drift_anomaly.data import load_series

    kwargs: dict = {}
    if dataset == "synthetic":
        kwargs["seed"] = cfg_dict.get("seed", 42)
        kwargs["n"] = cfg_dict.get("n_points", 1000)
    elif dataset == "nab":
        data_root = cfg_dict.get("data_root", "data/nab")
        kwargs["data_root"] = data_root
    elif dataset in ("yahoo", "yahoo_s5"):
        data_root = cfg_dict.get("data_root", "data/raw/yahoo_s5")
        kwargs["data_root"] = data_root

    try:
        return load_series(series_name, dataset=dataset, **kwargs)
    except Exception as exc:  # noqa: BLE001
        raise click.ClickException(
            f"Could not load series '{series_name}' from dataset '{dataset}': {exc}"
        ) from exc


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

def main() -> None:
    """Package entry-point called by the ``drift-anomaly`` console script."""
    # Click help and status messages include Unicode punctuation/icons. Use
    # UTF-8 on Windows consoles that default to the legacy cp1252 encoding.
    if hasattr(sys.stdout, "reconfigure"):
        try:
            sys.stdout.reconfigure(encoding="utf-8")
        except (AttributeError, OSError):
            pass
    cli()


if __name__ == "__main__":
    main()

