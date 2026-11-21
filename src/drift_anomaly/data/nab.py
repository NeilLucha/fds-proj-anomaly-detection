"""
drift_anomaly.data.nab
======================

Loader for the **Numenta Anomaly Benchmark (NAB)**.

NAB is a public benchmark of real-world and synthetic time series with
ground-truth anomaly labels.  The repository is downloadable from GitHub
under a AGPL-3.0 / BSD license.

Reference
---------
Lavin & Ahmad (2015), "Evaluating Real-time Anomaly Detection Algorithms –
the Numenta Anomaly Benchmark", Proc. ICMLA.
GitHub: https://github.com/numenta/NAB
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd

from ..types import TimeSeries, TimeSeriesPoint
from .downloader import DatasetDownloader, download_file, extract_zip, get_data_dir
from .schema import normalize_to_schema

logger = logging.getLogger(__name__)


class NABLoader:
    """Load NAB time series into the unified :class:`~drift_anomaly.types.TimeSeries` schema.

    NAB is organised as a flat collection of CSV files grouped into named
    sub-directories (e.g. ``realKnownCause/``, ``artificialWithAnomaly/``).
    Ground-truth anomaly labels are stored in two JSON files at the
    repository root:

    - ``labels/combined_labels.json`` – point-exact anomaly timestamps.
    - ``labels/combined_windows.json`` – anomaly windows (start, end pairs).

    This loader uses *combined_labels.json* to populate ``is_anomaly`` and
    does **not** mark any change-points (NAB has no concept-drift labels).

    Parameters
    ----------
    data_root : Path, optional
        Directory where the NAB archive will be stored / expected.
        Defaults to ``<project>/data/raw/nab/``.

    Examples
    --------
    >>> loader = NABLoader()
    >>> loader.download()
    True
    >>> series_list = loader.load_all()
    >>> len(series_list) > 0
    True
    """

    NAB_GITHUB_URL: str = (
        "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
    )

    def __init__(self, data_root: Optional[Union[Path, str]] = None) -> None:
        if data_root is not None:
            self.data_root: Path = Path(data_root).resolve()
        else:
            self.data_root = (get_data_dir() / "nab").resolve()
        logger.debug("NABLoader: data_root='%s'", self.data_root)

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    @property
    def _nab_root(self) -> Path:
        """Return the extracted NAB root (``NAB-master/``)."""
        candidate = self.data_root / "NAB-master"
        return candidate

    @property
    def _data_subdir(self) -> Path:
        return self._nab_root / "data"

    @property
    def _labels_dir(self) -> Path:
        return self._nab_root / "labels"

    def _load_combined_labels(self) -> Dict[str, List[str]]:
        """Return the combined_labels.json dict keyed by relative path."""
        path = self._labels_dir / "combined_labels.json"
        if not path.exists():
            raise FileNotFoundError(
                f"NAB labels file not found at '{path}'. "
                "Call download() first or point data_root at the NAB directory."
            )
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    def _load_combined_windows(self) -> Dict[str, List[List[str]]]:
        """Return the combined_windows.json dict keyed by relative path."""
        path = self._labels_dir / "combined_windows.json"
        if not path.exists():
            logger.warning(
                "NAB windows file not found at '%s'; window labels unavailable.", path
            )
            return {}
        with open(path, "r", encoding="utf-8") as fh:
            return json.load(fh)

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return True if NAB data directory and labels are present locally."""
        return self._nab_root.exists() and self._data_subdir.exists() and (self._labels_dir / "combined_labels.json").exists()

    def download(self) -> bool:
        """Download NAB from GitHub if not already present.

        Returns
        -------
        bool
            ``True`` if data was freshly downloaded, ``False`` if already
            present.

        Raises
        ------
        requests.HTTPError
            If the GitHub request fails.
        """
        if self._nab_root.exists() and self._data_subdir.exists():
            logger.info("NABLoader.download: data already present at '%s'.", self._nab_root)
            return False

        self.data_root.mkdir(parents=True, exist_ok=True)
        zip_path = self.data_root / "nab_master.zip"

        logger.info("NABLoader.download: downloading NAB from GitHub …")
        downloaded = download_file(self.NAB_GITHUB_URL, zip_path)

        if zip_path.exists():
            extract_zip(zip_path, self.data_root)
            zip_path.unlink()
            logger.info("NABLoader.download: extracted and cleaned up zip.")

        return downloaded

    def list_available(self) -> List[str]:
        """Return relative paths (as strings) of all CSV files in the NAB dataset.

        Returns
        -------
        list of str
            Paths relative to the ``data/`` sub-directory, e.g.
            ``"realKnownCause/machine_temperature_system_failure.csv"``.

        Raises
        ------
        FileNotFoundError
            If the NAB data directory is missing.
        """
        if not self._data_subdir.exists():
            raise FileNotFoundError(
                f"NAB data directory not found: '{self._data_subdir}'. "
                "Call NABLoader.download() first."
            )
        files = sorted(self._data_subdir.rglob("*.csv"))
        relative = [str(f.relative_to(self._data_subdir)) for f in files]
        logger.debug("NABLoader.list_available: %d series found.", len(relative))
        return relative

    def load_series(self, filename: str, use_windows: bool = False) -> TimeSeries:
        """Load a single NAB series by its relative filename.

        Parameters
        ----------
        filename : str
            Relative path of the CSV file under the NAB ``data/`` directory,
            e.g. ``"realKnownCause/machine_temperature_system_failure.csv"``.
        use_windows : bool, default False
            If True, mark all points inside the anomaly windows from
            *combined_windows.json* as anomalies (window-aware labeling).
            If False, only mark point-exact anomaly timestamps from
            *combined_labels.json*.

        Returns
        -------
        TimeSeries
            Populated with anomaly labels from *combined_labels.json* or *combined_windows.json*.

        Raises
        ------
        FileNotFoundError
            If the specified CSV or the labels JSON is missing.
        """
        # Normalise path separators
        filename_norm = filename.replace("\\", "/")
        csv_path = self._data_subdir / Path(*filename_norm.split("/"))

        if not csv_path.exists():
            raise FileNotFoundError(
                f"NAB series CSV not found: '{csv_path}'. "
                f"Check filename or call download() first."
            )

        logger.debug("NABLoader.load_series: reading '%s'", csv_path)
        df = pd.read_csv(csv_path)

        # NAB CSVs always have 'timestamp' and 'value' columns
        if "timestamp" not in df.columns or "value" not in df.columns:
            raise ValueError(
                f"Unexpected CSV schema in '{csv_path}': "
                f"expected 'timestamp' and 'value' columns, got {list(df.columns)}."
            )

        # Parse timestamps → Unix epoch seconds
        df["dt"] = pd.to_datetime(df["timestamp"])
        timestamps = df["dt"].astype("int64").values / 1e9
        values = df["value"].astype(float).values
        n = len(values)

        json_key = filename_norm
        if use_windows:
            combined_windows = self._load_combined_windows()
            windows = combined_windows.get(json_key, [])
            is_anomaly_mask = np.zeros(n, dtype=bool)
            for w_start, w_end in windows:
                t0 = pd.Timestamp(w_start)
                t1 = pd.Timestamp(w_end)
                mask = (df["dt"] >= t0) & (df["dt"] <= t1)
                is_anomaly_mask |= mask.values
            is_anomaly = list(is_anomaly_mask)
            n_json_entries = len(windows)
        else:
            # Build anomaly labels from combined_labels.json
            combined_labels = self._load_combined_labels()
            # Key in JSON is a normalised relative path with forward slashes
            anomaly_timestamps_str: List[str] = combined_labels.get(json_key, [])
            # Match normalized UTC instants rather than floating epoch seconds:
            # parsed timestamps can differ by sub-second/rounding representation.
            anomaly_set = set()
            for ts_str in anomaly_timestamps_str:
                try:
                    anomaly_set.add(pd.Timestamp(ts_str).value)
                except Exception as exc:
                    logger.warning(
                        "load_series: could not parse anomaly timestamp '%s': %s",
                        ts_str, exc,
                    )

            is_anomaly = [int(t) in anomaly_set for t in df["dt"].astype("int64").values]
            n_json_entries = len(anomaly_timestamps_str)

        logger.info(
            "NABLoader.load_series: '%s' – %d points, %d anomalies (use_windows=%s).",
            filename_norm, n, sum(is_anomaly), use_windows,
        )

        series_name = csv_path.stem
        return normalize_to_schema(
            timestamps=timestamps,
            values=values,
            labels=is_anomaly,
            change_points=None,
            name=series_name,
            metadata={
                "source": "NAB",
                "filename": filename_norm,
                "use_windows": use_windows,
                "n_anomaly_entries_in_json": n_json_entries,
            },
        )

    def load_all(self, use_windows: bool = False) -> List[TimeSeries]:
        """Load every available NAB series with ground-truth labels.

        Parameters
        ----------
        use_windows : bool, default False
            Whether to use anomaly windows from *combined_windows.json*.

        Returns
        -------
        list of TimeSeries
            All series in the NAB dataset, in sorted filename order.
        """
        filenames = self.list_available()
        series_list: List[TimeSeries] = []

        for fname in filenames:
            try:
                ts = self.load_series(fname, use_windows=use_windows)
                series_list.append(ts)
            except Exception as exc:
                logger.error(
                    "NABLoader.load_all: failed to load '%s': %s", fname, exc
                )
        return series_list

        logger.info(
            "NABLoader.load_all: loaded %d / %d series.",
            len(series_list), len(filenames),
        )
        return series_list
