"""
drift_anomaly.data.schema
=========================

Utilities for converting raw arrays and DataFrames into the unified
:class:`~drift_anomaly.types.TimeSeries` schema used throughout the library.

The canonical schema stores every observation as a
:class:`~drift_anomaly.types.TimeSeriesPoint` with four fields::

    timestamp   : float   – Unix epoch seconds (or monotonic integer index)
    value       : float   – scalar measurement
    is_anomaly  : bool    – ground-truth anomaly label
    change_point: bool    – ground-truth regime-change label
"""

from __future__ import annotations

import logging
from dataclasses import dataclass, field
from typing import Dict, Any, List, Optional, Sequence, Union

import numpy as np
import pandas as pd

from ..types import TimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Public conversion helper
# ---------------------------------------------------------------------------

def normalize_to_schema(
    timestamps: Union[Sequence[float], np.ndarray, pd.Series],
    values: Union[Sequence[float], np.ndarray, pd.Series],
    labels: Union[Sequence[bool], np.ndarray, pd.Series, None] = None,
    change_points: Union[Sequence[bool], np.ndarray, pd.Series, None] = None,
    name: str = "unnamed",
    metadata: Optional[Dict[str, Any]] = None,
) -> TimeSeries:
    """Convert raw arrays to a unified :class:`TimeSeries` object.

    Parameters
    ----------
    timestamps : array-like of float
        Monotonically increasing timestamps (Unix epoch seconds or plain
        integer indices).  Must have the same length as *values*.
    values : array-like of float
        Scalar observations.
    labels : array-like of bool, optional
        Ground-truth anomaly labels (``True`` = anomalous).  When *None*,
        all points are assumed normal (``False``).
    change_points : array-like of bool, optional
        Ground-truth regime-change labels.  When *None*, no change points
        are marked.
    name : str
        Human-readable name for the resulting :class:`TimeSeries`.
    metadata : dict, optional
        Arbitrary key-value pairs stored on the series (e.g. source, units).

    Returns
    -------
    TimeSeries
        A fully populated :class:`TimeSeries` in the unified schema.

    Raises
    ------
    ValueError
        If *timestamps* and *values* have different lengths, or if either
        *labels* or *change_points* (when provided) differ in length.

    Examples
    --------
    >>> import numpy as np
    >>> from drift_anomaly.data.schema import normalize_to_schema
    >>> ts = normalize_to_schema(
    ...     timestamps=np.arange(10, dtype=float),
    ...     values=np.random.randn(10),
    ... )
    >>> len(ts)
    10
    """
    ts_arr = np.asarray(timestamps, dtype=float)
    val_arr = np.asarray(values, dtype=float)

    n = len(ts_arr)
    if len(val_arr) != n:
        raise ValueError(
            f"timestamps and values must have equal length, "
            f"got {n} and {len(val_arr)}."
        )

    if labels is not None:
        lab_arr = np.asarray(labels, dtype=bool)
        if len(lab_arr) != n:
            raise ValueError(
                f"labels length ({len(lab_arr)}) must match timestamps length ({n})."
            )
    else:
        lab_arr = np.zeros(n, dtype=bool)

    if change_points is not None:
        cp_arr = np.asarray(change_points, dtype=bool)
        if len(cp_arr) != n:
            raise ValueError(
                f"change_points length ({len(cp_arr)}) must match "
                f"timestamps length ({n})."
            )
    else:
        cp_arr = np.zeros(n, dtype=bool)

    points: List[TimeSeriesPoint] = [
        TimeSeriesPoint(
            timestamp=float(ts_arr[i]),
            value=float(val_arr[i]),
            is_anomaly=bool(lab_arr[i]),
            change_point=bool(cp_arr[i]),
        )
        for i in range(n)
    ]

    logger.debug(
        "normalize_to_schema: created '%s' with %d points "
        "(%d anomalies, %d change-points)",
        name,
        n,
        int(lab_arr.sum()),
        int(cp_arr.sum()),
    )

    return TimeSeries(
        points=points,
        name=name,
        metadata=metadata or {},
    )


def from_dataframe(
    df: pd.DataFrame,
    timestamp_col: str = "timestamp",
    value_col: str = "value",
    label_col: Optional[str] = "is_anomaly",
    change_point_col: Optional[str] = "change_point",
    name: str = "unnamed",
    metadata: Optional[Dict[str, Any]] = None,
) -> TimeSeries:
    """Build a :class:`TimeSeries` from a :class:`pandas.DataFrame`.

    Parameters
    ----------
    df : pd.DataFrame
        Source data frame.  Must contain at least *timestamp_col* and
        *value_col* columns.
    timestamp_col : str
        Column name for timestamps.  If the column holds
        :class:`~pandas.Timestamp` objects they are converted to Unix
        epoch seconds automatically.
    value_col : str
        Column name for the scalar observations.
    label_col : str or None
        Column name for anomaly labels.  Pass *None* to skip (all normal).
    change_point_col : str or None
        Column name for change-point labels.  Pass *None* to skip.
    name : str
        Series name propagated to :class:`TimeSeries`.
    metadata : dict, optional
        Extra metadata stored on the returned :class:`TimeSeries`.

    Returns
    -------
    TimeSeries

    Raises
    ------
    KeyError
        If a required column is missing from *df*.
    """
    if timestamp_col not in df.columns:
        raise KeyError(f"Column '{timestamp_col}' not found in DataFrame.")
    if value_col not in df.columns:
        raise KeyError(f"Column '{value_col}' not found in DataFrame.")

    raw_ts = df[timestamp_col]
    # Convert pandas Timestamps to epoch seconds if needed
    if pd.api.types.is_datetime64_any_dtype(raw_ts):
        timestamps = raw_ts.astype("int64") / 1e9
    else:
        timestamps = raw_ts.astype(float)

    values = df[value_col].astype(float)

    labels: Optional[pd.Series] = None
    if label_col is not None and label_col in df.columns:
        labels = df[label_col].astype(bool)
    elif label_col is not None:
        logger.warning(
            "Label column '%s' not found in DataFrame; all points marked normal.",
            label_col,
        )

    change_pts: Optional[pd.Series] = None
    if change_point_col is not None and change_point_col in df.columns:
        change_pts = df[change_point_col].astype(bool)
    elif change_point_col is not None:
        logger.debug(
            "Change-point column '%s' not found; no change-points marked.",
            change_point_col,
        )

    return normalize_to_schema(
        timestamps=timestamps.values,
        values=values.values,
        labels=labels.values if labels is not None else None,
        change_points=change_pts.values if change_pts is not None else None,
        name=name,
        metadata=metadata,
    )


def to_dataframe(ts: TimeSeries) -> pd.DataFrame:
    """Convert a :class:`TimeSeries` back to a :class:`pandas.DataFrame`.

    Parameters
    ----------
    ts : TimeSeries
        Source series.

    Returns
    -------
    pd.DataFrame
        Columns: ``timestamp``, ``value``, ``is_anomaly``, ``change_point``.
    """
    records = [
        {
            "timestamp": p.timestamp,
            "value": p.value,
            "is_anomaly": p.is_anomaly,
            "change_point": p.change_point,
        }
        for p in ts.points
    ]
    return pd.DataFrame(records)
