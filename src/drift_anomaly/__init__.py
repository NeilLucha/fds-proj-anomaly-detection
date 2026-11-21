"""
drift_anomaly
=============

Drift-aware anomaly detection for time series using robust statistics
and adaptive baselines.

Three detector variants are provided:

* **Version A** – Cautious Rolling Baseline with Candidate-Window
  Contamination Gate
* **Version B** – Simplified Rolling Window + ADWIN
* **Version C** – Frozen-Window Detector with Delayed Drift Adaptation

Example
-------
>>> from drift_anomaly import DetectorConfig, TimeSeries
>>> from drift_anomaly.detectors import VersionA, VersionB, VersionC
"""

__version__ = "0.1.0"

from drift_anomaly.types import (
    TimeSeriesPoint,
    TimeSeries,
    DetectorConfig,
    StepOutput,
    DetectorResult,
    EvalMetrics,
)
from drift_anomaly.detectors.version_a_cautious_rolling import VersionA
from drift_anomaly.detectors.version_b_rolling_adwin import VersionB
from drift_anomaly.detectors.version_c_frozen_delayed import VersionC

__all__ = [
    "__version__",
    "TimeSeriesPoint",
    "TimeSeries",
    "DetectorConfig",
    "StepOutput",
    "DetectorResult",
    "EvalMetrics",
    "VersionA",
    "VersionB",
    "VersionC",
]
