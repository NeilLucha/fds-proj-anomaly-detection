"""
drift_anomaly.data
==================

Data loading and generation utilities for the drift-anomaly library.

Exports
-------
SyntheticGenerator
    Deterministic synthetic time-series generator (stationary, step-shift,
    variance-shift, spike-burst, isolated-spikes, compound).
NABLoader
    Loader for the Numenta Anomaly Benchmark (auto-downloadable from GitHub).
YahooS5Loader
    Loader for the Yahoo Webscope S5 anomaly benchmark (manual download
    required due to academic license).
load_series
    Convenience helper that loads a single named series from either dataset.
"""

from __future__ import annotations

from .synthetic import SyntheticGenerator
from .nab import NABLoader
from .yahoo_s5 import YahooS5Loader


def load_series(
    name: str,
    dataset: str = "synthetic",
    **kwargs,
):
    """Load a single time series by name from the requested dataset.

    Parameters
    ----------
    name : str
        Series identifier.  For ``dataset="synthetic"`` this is one of
        ``"stationary"``, ``"step_shift"``, ``"variance_shift"``,
        ``"spike_burst"``, ``"isolated_spikes"``, or ``"compound"``.
        For ``dataset="nab"`` this is a path fragment such as
        ``"realKnownCause/machine_temperature_system_failure.csv"``.
        For ``dataset="yahoo"`` this is ``"<benchmark>/<filename>"``, e.g.
        ``"A1/real_1.csv"``.
    dataset : str
        One of ``"synthetic"``, ``"nab"``, or ``"yahoo"``.
    **kwargs
        Extra keyword arguments forwarded to the underlying loader or
        generator method.

    Returns
    -------
    TimeSeries
        The requested series in unified schema.

    Raises
    ------
    ValueError
        If *dataset* is not recognised or *name* is invalid.
    """
    dataset = dataset.lower()

    if dataset == "synthetic":
        gen = SyntheticGenerator(seed=kwargs.pop("seed", 42))
        method = getattr(gen, name, None)
        if method is None:
            raise ValueError(
                f"Unknown synthetic series '{name}'. "
                f"Choose from: stationary, step_shift, variance_shift, "
                f"spike_burst, isolated_spikes, compound."
            )
        return method(**kwargs)

    if dataset == "nab":
        loader = NABLoader(data_root=kwargs.pop("data_root", None))
        return loader.load_series(name)

    if dataset in ("yahoo", "yahoo_s5"):
        loader = YahooS5Loader(data_root=kwargs.pop("data_root", None))
        parts = name.split("/", 1)
        if len(parts) != 2:
            raise ValueError(
                f"Yahoo series name must be '<benchmark>/<filename>', got '{name}'."
            )
        benchmark, filename = parts
        return loader.load_series(benchmark=benchmark, filename=filename)

    raise ValueError(
        f"Unknown dataset '{dataset}'. Choose from: synthetic, nab, yahoo."
    )


__all__ = [
    "SyntheticGenerator",
    "NABLoader",
    "YahooS5Loader",
    "load_series",
]
