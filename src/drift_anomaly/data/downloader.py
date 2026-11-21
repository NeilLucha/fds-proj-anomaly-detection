"""
drift_anomaly.data.downloader
==============================

Low-level download utilities shared by all dataset loaders.

Provides
--------
download_file(url, dest_path, chunk_size)
    Stream a URL to disk with a tqdm progress bar.  Skips if the file
    already exists.
extract_zip(zip_path, extract_to)
    Extract a ZIP archive to a destination directory.
get_data_dir()
    Resolve the canonical project-level ``data/raw/`` directory.
DatasetDownloader
    High-level orchestrator for all supported datasets (NAB, Yahoo S5).
"""

from __future__ import annotations

import logging
import os
import zipfile
from pathlib import Path
from typing import Optional

import requests
from tqdm import tqdm

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Module-level helpers
# ---------------------------------------------------------------------------

def download_file(
    url: str,
    dest_path: Path,
    chunk_size: int = 8192,
) -> bool:
    """Download *url* to *dest_path*, streaming with a progress bar.

    Parameters
    ----------
    url : str
        Remote URL to fetch.
    dest_path : Path
        Local file path to write to.  The parent directory must exist.
    chunk_size : int
        Number of bytes per chunk when streaming the response body.

    Returns
    -------
    bool
        ``True`` if the file was downloaded, ``False`` if it was already
        present on disk and the download was skipped.

    Raises
    ------
    requests.HTTPError
        If the server returns a non-2xx response.
    OSError
        If writing to *dest_path* fails.
    """
    if dest_path.exists():
        logger.info("download_file: '%s' already exists – skipping.", dest_path)
        return False

    dest_path.parent.mkdir(parents=True, exist_ok=True)
    logger.info("download_file: fetching '%s' → '%s'", url, dest_path)

    with requests.get(url, stream=True, timeout=120) as resp:
        resp.raise_for_status()
        total = int(resp.headers.get("Content-Length", 0)) or None
        with (
            open(dest_path, "wb") as fh,
            tqdm(
                desc=dest_path.name,
                total=total,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                leave=False,
            ) as bar,
        ):
            for chunk in resp.iter_content(chunk_size=chunk_size):
                if chunk:
                    fh.write(chunk)
                    bar.update(len(chunk))

    logger.info("download_file: saved %s bytes to '%s'.", dest_path.stat().st_size, dest_path)
    return True


def extract_zip(zip_path: Path, extract_to: Path) -> None:
    """Extract *zip_path* into *extract_to*, creating it if necessary.

    Parameters
    ----------
    zip_path : Path
        Path to the ZIP archive on disk.
    extract_to : Path
        Target directory.  Created if it does not exist.

    Raises
    ------
    FileNotFoundError
        If *zip_path* does not exist.
    zipfile.BadZipFile
        If *zip_path* is not a valid ZIP archive.
    """
    if not zip_path.exists():
        raise FileNotFoundError(f"ZIP archive not found: '{zip_path}'")

    extract_to.mkdir(parents=True, exist_ok=True)
    logger.info("extract_zip: extracting '%s' → '%s'", zip_path, extract_to)

    with zipfile.ZipFile(zip_path, "r") as zf:
        members = zf.namelist()
        logger.debug("extract_zip: %d members in archive.", len(members))
        zf.extractall(extract_to)

    logger.info("extract_zip: done.")


def get_data_dir() -> Path:
    """Return the project-level raw-data directory.

    Resolution order:

    1. ``DRIFT_ANOMALY_DATA_DIR`` environment variable (if set).
    2. The ``data/raw/`` directory relative to this file's project root
       (walking upward from this module until a ``pyproject.toml`` is found).
    3. ``./data/raw/`` relative to the current working directory as a
       last resort.

    Returns
    -------
    Path
        Resolved (but not necessarily existing) path to ``data/raw/``.
    """
    env_override = os.environ.get("DRIFT_ANOMALY_DATA_DIR")
    if env_override:
        p = Path(env_override).expanduser().resolve()
        logger.debug("get_data_dir: using env override '%s'.", p)
        return p

    # Walk upward to find pyproject.toml
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "pyproject.toml").exists():
            p = parent / "data" / "raw"
            logger.debug("get_data_dir: found project root at '%s'.", parent)
            return p

    # Fallback
    p = Path.cwd() / "data" / "raw"
    logger.debug("get_data_dir: falling back to cwd '%s'.", p)
    return p


# ---------------------------------------------------------------------------
# DatasetDownloader
# ---------------------------------------------------------------------------

class DatasetDownloader:
    """Orchestrate downloading / locating all supported datasets.

    Parameters
    ----------
    data_root : Path, optional
        Root directory for all raw datasets.  Defaults to :func:`get_data_dir`.

    Attributes
    ----------
    data_root : Path
        Base directory; subdirectories ``nab/`` and ``yahoo_s5/`` live here.
    """

    NAB_URL: str = (
        "https://github.com/numenta/NAB/archive/refs/heads/master.zip"
    )

    def __init__(self, data_root: Optional[Path] = None) -> None:
        self.data_root: Path = (data_root or get_data_dir()).resolve()
        logger.debug("DatasetDownloader: data_root='%s'", self.data_root)

    # ------------------------------------------------------------------
    # NAB
    # ------------------------------------------------------------------

    def download_nab(self) -> Path:
        """Download the NAB repository from GitHub if not already present.

        The archive is extracted to ``<data_root>/nab/``.

        Returns
        -------
        Path
            Path to the NAB data directory (``<data_root>/nab/``).

        Raises
        ------
        requests.HTTPError
            If the GitHub download fails.
        """
        nab_dir = self.data_root / "nab"
        sentinel = nab_dir / "NAB-master"

        if sentinel.exists():
            logger.info("download_nab: NAB already present at '%s'.", nab_dir)
            return nab_dir

        zip_dest = self.data_root / "nab_master.zip"
        try:
            download_file(self.NAB_URL, zip_dest)
            extract_zip(zip_dest, nab_dir)
        finally:
            # Clean up the ZIP regardless of success / failure
            if zip_dest.exists():
                zip_dest.unlink()
                logger.debug("download_nab: removed temporary zip '%s'.", zip_dest)

        logger.info("download_nab: NAB ready at '%s'.", nab_dir)
        return nab_dir

    # ------------------------------------------------------------------
    # Yahoo S5
    # ------------------------------------------------------------------

    def check_yahoo(self) -> Optional[Path]:
        """Check whether the Yahoo S5 dataset has been manually placed.

        Yahoo S5 requires academic registration and cannot be downloaded
        programmatically.  This method inspects ``<data_root>/yahoo_s5/``
        and returns its path if at least one benchmark directory is found,
        or *None* otherwise.

        Returns
        -------
        Path or None
            The ``yahoo_s5`` directory if the data is present, else *None*.
        """
        yahoo_dir = self.data_root / "yahoo_s5"
        benchmarks = ["A1Benchmark", "A2Benchmark", "A3Benchmark", "A4Benchmark"]
        found = [b for b in benchmarks if (yahoo_dir / b).is_dir()]

        if found:
            logger.info(
                "check_yahoo: found benchmark directories %s in '%s'.",
                found,
                yahoo_dir,
            )
            return yahoo_dir

        logger.warning(
            "check_yahoo: Yahoo S5 not found at '%s'. "
            "Register at https://webscope.sandbox.yahoo.com/ and place the "
            "extracted data there.",
            yahoo_dir,
        )
        return None
