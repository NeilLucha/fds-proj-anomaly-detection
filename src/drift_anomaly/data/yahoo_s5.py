"""
drift_anomaly.data.yahoo_s5
============================

Loader for the **Yahoo Webscope S5** anomaly detection benchmark.

.. important::

   Yahoo S5 requires a manual download under an academic agreement.
   See :meth:`YahooS5Loader.print_setup_instructions` for step-by-step
   instructions, or simply call any loader method and the instructions
   will be printed automatically when the data is absent.

Dataset structure
-----------------
After extraction the directory should look like::

    data/raw/yahoo_s5/
        A1Benchmark/   – real production traffic (is_anomaly column)
        A2Benchmark/   – synthetic anomalies (is_anomaly column)
        A3Benchmark/   – synthetic + changepoints (is_anomaly, changepoint)
        A4Benchmark/   – synthetic + anomalies + changepoints

Benchmarks
----------
A1 : Real
    47 real Yahoo production metric time series with manually labelled
    point anomalies.
A2 : Synthetic anomalies
    100 synthetic series; ``is_anomaly`` marks point anomalies.
A3 : Synthetic with change-points
    100 synthetic series; ``changepoint`` column marks regime changes.
A4 : Synthetic mixed
    100 synthetic series with both point anomalies and change-points.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd

from ..types import TimeSeries, TimeSeriesPoint
from .downloader import get_data_dir
from .schema import normalize_to_schema

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Column name mappings per benchmark
# ---------------------------------------------------------------------------

#: Expected CSV column names for each benchmark.
_BENCHMARK_COLS: Dict[str, Dict[str, Optional[str]]] = {
    "A1": {"timestamp": "timestamps", "value": "value",
           "label": "is_anomaly", "change_point": None},
    "A2": {"timestamp": "timestamps", "value": "value",
           "label": "is_anomaly", "change_point": None},
    "A3": {"timestamp": "timestamps", "value": "value",
           "label": "is_anomaly", "change_point": "changepoint"},
    "A4": {"timestamp": "timestamps", "value": "value",
           "label": "is_anomaly", "change_point": "changepoint"},
}

#: Directory names for each benchmark key.
_BENCHMARK_DIRS: Dict[str, str] = {
    "A1": "A1Benchmark",
    "A2": "A2Benchmark",
    "A3": "A3Benchmark",
    "A4": "A4Benchmark",
}


class YahooS5Loader:
    """Load Yahoo Webscope S5 anomaly benchmark data.

    Parameters
    ----------
    data_root : Path, optional
        Root directory containing the extracted Yahoo S5 data, i.e. the
        parent of the ``A1Benchmark/``, ``A2Benchmark/``, … directories.
        Defaults to ``<project>/data/raw/yahoo_s5/``.

    Examples
    --------
    >>> loader = YahooS5Loader()
    >>> if loader.is_available():
    ...     series = loader.load_benchmark("A1")
    ... else:
    ...     loader.print_setup_instructions()
    """

    def __init__(self, data_root: Optional[Union[Path, str]] = None) -> None:
        if data_root is not None:
            self.data_root: Path = Path(data_root).resolve()
        else:
            self.data_root = (get_data_dir() / "yahoo_s5").resolve()
        logger.debug("YahooS5Loader: data_root='%s'", self.data_root)

    # ------------------------------------------------------------------
    # Availability check
    # ------------------------------------------------------------------

    def is_available(self) -> bool:
        """Return ``True`` if at least one benchmark directory is present.

        Returns
        -------
        bool
        """
        for bm_dir in _BENCHMARK_DIRS.values():
            if (self.data_root / bm_dir).is_dir():
                logger.debug(
                    "YahooS5Loader.is_available: found '%s' – data present.",
                    bm_dir,
                )
                return True
        logger.debug(
            "YahooS5Loader.is_available: no benchmark directories found under '%s'.",
            self.data_root,
        )
        return False

    # ------------------------------------------------------------------
    # Setup instructions
    # ------------------------------------------------------------------

    def print_setup_instructions(self) -> None:
        """Print clear step-by-step instructions for obtaining Yahoo S5.

        Yahoo S5 is freely available for academic use but requires
        registration and agreement to Yahoo's terms.  This method prints
        the instructions to stdout so users can easily follow them.
        """
        instructions = f"""
╔══════════════════════════════════════════════════════════════════════════════╗
║          Yahoo Webscope S5 — Manual Download Instructions                   ║
╚══════════════════════════════════════════════════════════════════════════════╝

The Yahoo S5 dataset requires academic registration and cannot be downloaded
automatically.  Please follow these steps:

  1. Register at:
       https://webscope.sandbox.yahoo.com/

  2. Search for "S5 - A Labeled Anomaly Detection Dataset" and request access.
     You will receive a download link after agreeing to the data usage terms.

  3. Download and extract the archive.  The extracted directory should contain:
       A1Benchmark/
       A2Benchmark/
       A3Benchmark/
       A4Benchmark/

  4. Place (or symlink) this directory at:
       {self.data_root}

     Expected final structure:
       {self.data_root}/A1Benchmark/real_1.csv
       {self.data_root}/A1Benchmark/real_2.csv
       …
       {self.data_root}/A4Benchmark/

  5. Re-run your code.  YahooS5Loader.is_available() will return True once
     the data is in place.

Benchmark descriptions:
  A1 – 47 real Yahoo production metric series with point anomaly labels.
  A2 – 100 synthetic series with injected point anomalies.
  A3 – 100 synthetic series with change-points (changepoint column).
  A4 – 100 synthetic series with both anomalies and change-points.
"""
        try:
            print(instructions)
        except UnicodeEncodeError:
            import sys
            sys.stdout.buffer.write(instructions.encode("utf-8", errors="replace"))
        logger.info("YahooS5Loader.print_setup_instructions: instructions printed.")

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _benchmark_dir(self, benchmark: str) -> Path:
        """Resolve and validate the directory for *benchmark*."""
        bm = benchmark.upper()
        if bm not in _BENCHMARK_DIRS:
            raise ValueError(
                f"Unknown benchmark '{benchmark}'. "
                f"Choose from: {list(_BENCHMARK_DIRS.keys())}."
            )
        return self.data_root / _BENCHMARK_DIRS[bm]

    def _col_map(self, benchmark: str) -> Dict[str, Optional[str]]:
        """Return the column-name map for *benchmark*."""
        return _BENCHMARK_COLS[benchmark.upper()]

    def _load_csv(
        self,
        csv_path: Path,
        benchmark: str,
        series_name: str,
    ) -> TimeSeries:
        """Read a single CSV and convert to :class:`TimeSeries`.

        Parameters
        ----------
        csv_path : Path
            Absolute path to the CSV file.
        benchmark : str
            Benchmark key (``"A1"``–``"A4"``).
        series_name : str
            Name to assign to the resulting :class:`TimeSeries`.

        Returns
        -------
        TimeSeries
        """
        bm = benchmark.upper()
        col = self._col_map(bm)

        logger.debug("YahooS5Loader._load_csv: reading '%s'", csv_path)
        df = pd.read_csv(csv_path)
        df.columns = [c.strip() for c in df.columns]  # strip accidental whitespace

        # --- timestamp ---------------------------------------------------
        ts_col = col["timestamp"]
        if ts_col and ts_col in df.columns:
            raw_ts = df[ts_col]
            if pd.api.types.is_integer_dtype(raw_ts) or pd.api.types.is_float_dtype(raw_ts):
                timestamps = raw_ts.astype(float).values
            else:
                timestamps = pd.to_datetime(raw_ts).astype("int64").values / 1e9
        else:
            # Fallback: use row index as timestamps
            timestamps = df.index.astype(float).values
            logger.warning(
                "_load_csv: timestamp column '%s' not found in '%s'; "
                "using row index.",
                ts_col, csv_path.name,
            )

        # --- value -------------------------------------------------------
        val_col = col["value"]
        if val_col and val_col in df.columns:
            values = df[val_col].astype(float).values
        else:
            raise KeyError(
                f"Value column '{val_col}' not found in '{csv_path}'. "
                f"Available columns: {list(df.columns)}."
            )

        # --- anomaly labels ----------------------------------------------
        label_col = col["label"]
        if label_col and label_col in df.columns:
            is_anomaly = df[label_col].astype(int).astype(bool).values
        else:
            is_anomaly = None
            logger.debug(
                "_load_csv: label column '%s' not found; marking all normal.",
                label_col,
            )

        # --- change-point labels -----------------------------------------
        cp_col = col["change_point"]
        if cp_col and cp_col in df.columns:
            change_points = df[cp_col].astype(int).astype(bool).values
        else:
            change_points = None

        n_anom = int(is_anomaly.sum()) if is_anomaly is not None else 0
        n_cp = int(change_points.sum()) if change_points is not None else 0
        logger.info(
            "YahooS5Loader._load_csv: '%s' – %d points, %d anomalies, %d change-points.",
            csv_path.name, len(values), n_anom, n_cp,
        )

        return normalize_to_schema(
            timestamps=timestamps,
            values=values,
            labels=is_anomaly,
            change_points=change_points,
            name=series_name,
            metadata={
                "source": "Yahoo S5",
                "benchmark": bm,
                "filename": csv_path.name,
            },
        )

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def load_benchmark(self, benchmark: str = "A1") -> List[TimeSeries]:
        """Load every series in a specific benchmark split.

        Parameters
        ----------
        benchmark : str
            One of ``"A1"``, ``"A2"``, ``"A3"``, ``"A4"`` (case-insensitive).

        Returns
        -------
        list of TimeSeries
            All series in the benchmark, sorted by filename.

        Raises
        ------
        RuntimeError
            If the data is not available.  :meth:`print_setup_instructions`
            is called automatically before the exception is raised.
        ValueError
            If *benchmark* is not a valid key.
        """
        if not self.is_available():
            self.print_setup_instructions()
            raise RuntimeError(
                f"Yahoo S5 data not found at '{self.data_root}'. "
                "See the instructions above."
            )

        bm = benchmark.upper()
        bm_dir = self._benchmark_dir(bm)

        if not bm_dir.is_dir():
            raise FileNotFoundError(
                f"Benchmark directory '{bm_dir}' not found. "
                f"Only these benchmarks are present: "
                f"{[b for b in _BENCHMARK_DIRS.values() if (self.data_root / b).is_dir()]}."
            )

        csv_files = sorted(bm_dir.glob("*.csv"))
        if not csv_files:
            raise FileNotFoundError(
                f"No CSV files found in '{bm_dir}'."
            )

        series_list: List[TimeSeries] = []
        for csv_path in csv_files:
            try:
                ts = self._load_csv(
                    csv_path=csv_path,
                    benchmark=bm,
                    series_name=csv_path.stem,
                )
                series_list.append(ts)
            except Exception as exc:
                logger.error(
                    "YahooS5Loader.load_benchmark: failed to load '%s': %s",
                    csv_path.name, exc,
                )

        logger.info(
            "YahooS5Loader.load_benchmark: loaded %d / %d series from %s.",
            len(series_list), len(csv_files), bm,
        )
        return series_list

    def load_series(self, benchmark: str, filename: str) -> TimeSeries:
        """Load a single series from a specified benchmark.

        Parameters
        ----------
        benchmark : str
            One of ``"A1"``–``"A4"`` (case-insensitive).
        filename : str
            CSV filename (with or without ``.csv`` extension), e.g.
            ``"real_1.csv"`` or ``"real_1"``.

        Returns
        -------
        TimeSeries

        Raises
        ------
        RuntimeError
            If the Yahoo S5 data is not installed.
        FileNotFoundError
            If the specified CSV file does not exist.
        """
        if not self.is_available():
            self.print_setup_instructions()
            raise RuntimeError(
                f"Yahoo S5 data not found at '{self.data_root}'."
            )

        bm = benchmark.upper()
        bm_dir = self._benchmark_dir(bm)

        # Normalise filename
        if not filename.endswith(".csv"):
            filename = filename + ".csv"
        csv_path = bm_dir / filename

        if not csv_path.exists():
            raise FileNotFoundError(
                f"Series file '{csv_path}' not found. "
                f"Use list_available('{bm}') to see what is available."
            )

        return self._load_csv(
            csv_path=csv_path,
            benchmark=bm,
            series_name=csv_path.stem,
        )

    def list_available(self, benchmark: Optional[str] = None) -> List[str]:
        """List available CSV filenames, optionally filtered by benchmark.

        Parameters
        ----------
        benchmark : str, optional
            If given (e.g. ``"A1"``), return only filenames from that
            benchmark.  Otherwise return filenames from all present
            benchmarks, prefixed with the benchmark key.

        Returns
        -------
        list of str
            Sorted list of ``"<benchmark>/<filename>.csv"`` strings when
            *benchmark* is *None*, or plain ``"<filename>.csv"`` strings
            when *benchmark* is given.
        """
        if not self.is_available():
            return []

        if benchmark is not None:
            bm = benchmark.upper()
            bm_dir = self._benchmark_dir(bm)
            if not bm_dir.is_dir():
                return []
            return sorted(f.name for f in bm_dir.glob("*.csv"))

        result: List[str] = []
        for bm_key, bm_dir_name in _BENCHMARK_DIRS.items():
            bm_dir = self.data_root / bm_dir_name
            if bm_dir.is_dir():
                result.extend(
                    f"{bm_key}/{f.name}" for f in sorted(bm_dir.glob("*.csv"))
                )
        return result
