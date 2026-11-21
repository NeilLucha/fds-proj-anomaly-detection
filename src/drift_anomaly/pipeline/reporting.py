"""drift_anomaly.pipeline.reporting
====================================

Generate evaluation reports: JSON, CSV summaries, and optional plots.

All matplotlib work uses the ``Agg`` non-interactive backend so that
this module can be imported and used in headless / server environments
without a display.
"""
from __future__ import annotations

import csv
import dataclasses
import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import matplotlib
matplotlib.use("Agg")  # Must be set before importing pyplot.
import matplotlib.pyplot as plt
import numpy as np

from ..types import EvalMetrics

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# JSON serialisation helpers
# ---------------------------------------------------------------------------

def _metrics_to_dict(metrics: EvalMetrics) -> dict:
    """Convert an :class:`EvalMetrics` to a plain ``dict``."""
    return dataclasses.asdict(metrics)


def _safe_json_default(obj):
    """JSON encoder for numpy scalar types."""
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, (np.floating,)):
        return float(obj)
    if isinstance(obj, np.ndarray):
        return obj.tolist()
    raise TypeError(f"Object of type {type(obj)} is not JSON-serialisable.")


# ---------------------------------------------------------------------------
# Save helpers
# ---------------------------------------------------------------------------

def save_metrics_json(metrics_dict: Dict, output_path: Path) -> None:
    """Serialise *metrics_dict* to a JSON file at *output_path*.

    Parameters
    ----------
    metrics_dict : dict
        Arbitrary nested mapping; values may contain :class:`EvalMetrics`
        instances, plain dicts, or JSON-serialisable scalars.
    output_path : Path
        Destination file (parent directories are created automatically).
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # Convert EvalMetrics leaves to plain dicts.
    def _convert(obj):
        if isinstance(obj, EvalMetrics):
            return _metrics_to_dict(obj)
        if isinstance(obj, dict):
            return {k: _convert(v) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_convert(v) for v in obj]
        return obj

    serialisable = _convert(metrics_dict)

    with output_path.open("w", encoding="utf-8") as fh:
        json.dump(serialisable, fh, indent=2, default=_safe_json_default)

    logger.info("save_metrics_json: wrote %s", output_path)


def save_metrics_csv(
    metrics_list: List[EvalMetrics],
    output_path: Path,
    series_names: Optional[List[str]] = None,
    variant_name: str = "unknown",
) -> None:
    """Write *metrics_list* to a CSV file, one row per series.

    Parameters
    ----------
    metrics_list : list[EvalMetrics]
        Metrics to write.
    output_path : Path
        Destination CSV file.
    series_names : list[str] or None
        Names for each row.  Defaults to ``"series_0"``, ``"series_1"``, …
    variant_name : str
        Variant identifier written into the ``variant`` column.
    """
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    if series_names is None:
        series_names = [f"series_{i}" for i in range(len(metrics_list))]

    if len(series_names) != len(metrics_list):
        raise ValueError(
            "series_names and metrics_list must have the same length; "
            f"got {len(series_names)} vs {len(metrics_list)}."
        )

    fieldnames = ["series", "variant"] + [f.name for f in dataclasses.fields(EvalMetrics)]

    with output_path.open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=fieldnames)
        writer.writeheader()
        for name, m in zip(series_names, metrics_list):
            row = {"series": name, "variant": variant_name}
            row.update(_metrics_to_dict(m))
            writer.writerow(row)

    logger.info("save_metrics_csv: wrote %d rows to %s", len(metrics_list), output_path)


# ---------------------------------------------------------------------------
# Plot helpers
# ---------------------------------------------------------------------------

def plot_series_with_anomalies(
    values: np.ndarray,
    true_labels: np.ndarray,
    pred_labels: np.ndarray,
    scores: np.ndarray,
    drift_events: List[int],
    title: str = "Series",
    output_path: Optional[Path] = None,
) -> None:
    """Plot a time series alongside anomaly labels, scores, and drift markers.

    Parameters
    ----------
    values : np.ndarray
        Raw observation values.
    true_labels : np.ndarray
        Ground-truth binary anomaly labels.
    pred_labels : np.ndarray
        Predicted binary anomaly labels (threshold-based alerts).
    scores : np.ndarray
        Continuous anomaly scores.
    drift_events : list[int]
        Time indices of detected drift events.
    title : str
        Figure title.
    output_path : Path or None
        If given, save the figure here (PNG).  Otherwise display is
        attempted (but will silently fail under Agg).
    """
    values = np.asarray(values, dtype=float)
    true_labels = np.asarray(true_labels, dtype=bool)
    pred_labels = np.asarray(pred_labels, dtype=bool)
    scores = np.asarray(scores, dtype=float)

    n = len(values)
    t = np.arange(n)

    fig, axes = plt.subplots(3, 1, figsize=(14, 9), sharex=True)
    fig.suptitle(title, fontsize=13)

    # --- Top panel: raw series + anomaly markers ---
    ax0 = axes[0]
    ax0.plot(t, values, color="steelblue", linewidth=0.8, label="value")
    if true_labels.any():
        ax0.scatter(
            t[true_labels], values[true_labels],
            color="red", marker="x", s=30, zorder=5, label="true anomaly",
        )
    if pred_labels.any():
        ax0.scatter(
            t[pred_labels], values[pred_labels],
            color="orange", marker="o", s=15, zorder=4, alpha=0.7, label="pred anomaly",
        )
    for de in drift_events:
        if 0 <= de < n:
            ax0.axvline(de, color="purple", linestyle="--", linewidth=0.8, alpha=0.6)
    ax0.set_ylabel("value")
    ax0.legend(fontsize=8, loc="upper right")

    # --- Middle panel: anomaly score ---
    ax1 = axes[1]
    ax1.plot(t, scores, color="darkorange", linewidth=0.8, label="score")
    ax1.set_ylabel("anomaly score")
    ax1.legend(fontsize=8, loc="upper right")

    # --- Bottom panel: label agreement ---
    ax2 = axes[2]
    ax2.fill_between(t, true_labels.astype(int), alpha=0.4, color="red", label="true")
    ax2.fill_between(t, pred_labels.astype(int), alpha=0.4, color="orange", label="pred")
    for de in drift_events:
        if 0 <= de < n:
            ax2.axvline(de, color="purple", linestyle="--", linewidth=0.8, alpha=0.6)
    ax2.set_ylabel("anomaly (binary)")
    ax2.set_xlabel("time step")
    ax2.set_ylim(-0.05, 1.3)
    ax2.legend(fontsize=8, loc="upper right")

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("plot_series_with_anomalies: saved to %s", output_path)
    plt.close(fig)


def plot_metrics_comparison(
    metrics_by_variant: Dict[str, List[EvalMetrics]],
    output_path: Optional[Path] = None,
) -> None:
    """Bar chart comparing F1 / precision / recall across detector variants.

    Parameters
    ----------
    metrics_by_variant : dict[str, list[EvalMetrics]]
        Mapping ``{variant_name: [EvalMetrics per series]}``.
    output_path : Path or None
        Destination PNG file.  If ``None``, the figure is not saved.
    """
    if not metrics_by_variant:
        logger.warning("plot_metrics_comparison: empty metrics_by_variant; nothing to plot.")
        return

    variant_names = list(metrics_by_variant.keys())
    metric_keys = ["precision", "recall", "f1"]

    # Compute mean per variant per metric.
    means: Dict[str, Dict[str, float]] = {}
    for vname in variant_names:
        mlist = metrics_by_variant[vname]
        means[vname] = {}
        for mk in metric_keys:
            vals = [getattr(m, mk) for m in mlist]
            means[vname][mk] = float(np.mean(vals)) if vals else 0.0

    x = np.arange(len(metric_keys))
    width = 0.8 / max(len(variant_names), 1)

    fig, ax = plt.subplots(figsize=(9, 5))
    colors = plt.cm.tab10(np.linspace(0, 1, len(variant_names)))  # type: ignore[attr-defined]

    for i, (vname, color) in enumerate(zip(variant_names, colors)):
        vals = [means[vname][mk] for mk in metric_keys]
        offsets = x + (i - len(variant_names) / 2 + 0.5) * width
        bars = ax.bar(offsets, vals, width * 0.9, label=vname, color=color, alpha=0.85)
        for bar, val in zip(bars, vals):
            ax.text(
                bar.get_x() + bar.get_width() / 2,
                bar.get_height() + 0.01,
                f"{val:.3f}",
                ha="center", va="bottom", fontsize=7,
            )

    ax.set_xticks(x)
    ax.set_xticklabels([mk.upper() for mk in metric_keys])
    ax.set_ylim(0.0, 1.15)
    ax.set_ylabel("score (mean over series)")
    ax.set_title("Detector Variant Comparison")
    ax.legend(fontsize=9)
    ax.grid(axis="y", linestyle="--", alpha=0.4)

    plt.tight_layout()

    if output_path is not None:
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(output_path, dpi=120, bbox_inches="tight")
        logger.info("plot_metrics_comparison: saved to %s", output_path)
    plt.close(fig)


# ---------------------------------------------------------------------------
# Full report generation
# ---------------------------------------------------------------------------

def generate_report(
    metrics_by_variant: Dict[str, List[EvalMetrics]],
    output_dir: Path,
    make_plots: bool = True,
) -> None:
    """Write a full evaluation report to *output_dir*.

    Outputs
    -------
    ``<output_dir>/metrics.json``
        All metrics for all variants in JSON format.
    ``<output_dir>/metrics_<variant>.csv``
        One CSV per variant with one row per series.
    ``<output_dir>/plots/comparison.png``  *(if make_plots)*
        Bar chart comparing F1/precision/recall across variants.

    Parameters
    ----------
    metrics_by_variant : dict[str, list[EvalMetrics]]
        ``{variant_name: [EvalMetrics per series]}``.
    output_dir : Path
        Root directory for all output files.
    make_plots : bool
        Whether to generate matplotlib figures (default ``True``).
    """
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    if not metrics_by_variant:
        logger.warning("generate_report: metrics_by_variant is empty; nothing to write.")
        return

    # --- JSON ---------------------------------------------------------------
    json_path = output_dir / "metrics.json"
    # Build nested dict: {variant: {series_index: metrics_dict}}
    json_payload: dict = {}
    for vname, mlist in metrics_by_variant.items():
        json_payload[vname] = {
            f"series_{i}": _metrics_to_dict(m) for i, m in enumerate(mlist)
        }
    save_metrics_json(json_payload, json_path)

    # --- CSV ----------------------------------------------------------------
    for vname, mlist in metrics_by_variant.items():
        csv_path = output_dir / f"metrics_{vname}.csv"
        save_metrics_csv(mlist, csv_path, variant_name=vname)

    # --- Plots --------------------------------------------------------------
    if make_plots:
        plots_dir = output_dir / "plots"
        plots_dir.mkdir(parents=True, exist_ok=True)

        comparison_path = plots_dir / "comparison.png"
        try:
            plot_metrics_comparison(metrics_by_variant, output_path=comparison_path)
        except Exception as exc:  # noqa: BLE001
            logger.error(
                "generate_report: plot_metrics_comparison failed: %s", exc, exc_info=True
            )

    logger.info(
        "generate_report: completed. Files in %s", output_dir
    )
