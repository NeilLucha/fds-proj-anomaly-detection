"""
drift_anomaly.types
===================

Dataclasses for the entire drift-anomaly library.  All fields are typed;
default values match the paper notation.

Naming conventions mirror the algorithm description:
  L  = window_size
  M  = candidate_size
  L0 = freeze_window
  H  = hold_steps
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


# ---------------------------------------------------------------------------
# Time-series primitives
# ---------------------------------------------------------------------------

@dataclass
class TimeSeriesPoint:
    """A single observation in a labelled time series.

    Attributes
    ----------
    timestamp : float
        Unix epoch seconds (or any monotonically increasing index).
    value : float
        The observed scalar measurement.
    is_anomaly : bool
        Ground-truth label — True if this point is anomalous.
    change_point : bool
        Ground-truth label — True if a regime change begins here.
    """

    timestamp: float
    value: float
    is_anomaly: bool = False
    change_point: bool = False


@dataclass
class TimeSeries:
    """An ordered collection of :class:`TimeSeriesPoint` objects.

    Attributes
    ----------
    points : list[TimeSeriesPoint]
        Observations in chronological order.
    name : str
        Human-readable identifier for the series.
    metadata : dict[str, Any]
        Arbitrary key-value pairs (source, units, sampling_rate, …).
    """

    points: List[TimeSeriesPoint]
    name: str = "unnamed"
    metadata: Dict[str, Any] = field(default_factory=dict)

    def __len__(self) -> int:  # noqa: D105
        return len(self.points)

    @property
    def values(self) -> List[float]:
        """Return a plain Python list of observation values."""
        return [p.value for p in self.points]

    @property
    def labels(self) -> List[bool]:
        """Return a plain Python list of is_anomaly ground-truth labels."""
        return [p.is_anomaly for p in self.points]

    @property
    def change_points(self) -> List[bool]:
        """Return a plain Python list of change_point ground-truth labels."""
        return [p.change_point for p in self.points]

    @property
    def timestamps(self) -> List[float]:
        """Return a list of observation timestamps."""
        return [p.timestamp for p in self.points]

    @property
    def is_anomaly(self) -> List[bool]:
        """Return a list of boolean is_anomaly labels."""
        return [p.is_anomaly for p in self.points]

    @property
    def change_point(self) -> List[bool]:
        """Return a list of boolean change_point labels."""
        return [p.change_point for p in self.points]


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class DetectorConfig:
    """Hyper-parameters shared by all detector variants.

    Parameters
    ----------
    window_size : int
        ``L`` — size of the rolling baseline window.
    candidate_size : int
        ``M`` — size of the candidate window evaluated after drift.
    tau : float
        Anomaly threshold expressed as multiples of ``sigma_hat``.
    epsilon : float
        Additive floor added to ``sigma_hat`` to avoid division by zero.
    drift_detector : str
        Which drift-detection algorithm to use.  One of ``"adwin"`` or
        ``"page-hinkley"``.
    adwin_delta : float
        ADWIN false-positive probability parameter ``δ``.
    ph_lambda : float
        Page-Hinkley threshold ``λ``.
    ph_delta : float
        Page-Hinkley sensitivity parameter ``δ`` (minimum expected shift).
    tau_cand : float
        Anomaly threshold used *inside* the candidate-window contamination
        gate (Version A).
    rho_max : float
        Maximum allowed fraction of anomalous points inside a candidate
        window before it is rejected (contamination gate).
    kappa_max : float
        Maximum allowed ratio ``MAD_candidate / MAD_baseline`` before a
        candidate window is rejected (volatility gate).
    freeze_window : int
        ``L0`` — size of the initial frozen reference window (Version C).
    adaptation_mode : str
        ``"immediate"`` or ``"delayed"`` (Version C).
    hold_steps : int
        ``H`` — number of steps to accumulate a candidate window after
        drift is detected before attempting adaptation (Version C delayed).
    cooldown : int
        Minimum number of steps between consecutive drift-trigger events.
    feed_scores : bool
        If ``True``, feed anomaly scores (rather than raw values) to the
        drift detector.
    """

    # Baseline window
    window_size: int = 200          # L
    candidate_size: int = 50        # M

    # Anomaly detection threshold
    tau: float = 3.0
    epsilon: float = 1e-6

    # Drift detector selection
    drift_detector: str = "adwin"   # "adwin" | "page-hinkley"

    # ADWIN hyper-parameters
    adwin_delta: float = 0.002

    # Page-Hinkley hyper-parameters
    ph_lambda: float = 50.0
    ph_delta: float = 0.005

    # Candidate-window contamination gate (Version A)
    tau_cand: float = 3.0
    tau_int: float = 3.5
    e_scale_floor_ratio: float = 0.25
    e_sigma_floor: float = 1e-3
    e_tau_low: float = 2.5
    e_tau_high: float = 7.0
    e_persistence_k: int = 2
    e_seasonal_lag: int = 24
    e_auto_lag: bool = False
    e_max_lag: int = 48
    e_online_calibration: bool = True
    e_calibration_quantile: float = 0.99
    e2_tau_moderate: float = 2.5
    e2_tau_extreme: float = 7.0
    e2_persistence_k: int = 3
    e2_alert_hold_steps: int = 0
    e2_validation_zero_event_fpr_cap: float = 0.10
    rho_max: float = 0.3
    kappa_max: float = 3.0

    # Frozen-window detector (Version C)
    freeze_window: int = 200        # L0
    adaptation_mode: str = "delayed"
    hold_steps: int = 50            # H

    # Shared
    cooldown: int = 20
    feed_scores: bool = False


# ---------------------------------------------------------------------------
# Step-level and run-level output
# ---------------------------------------------------------------------------

@dataclass
class StepOutput:
    """Result produced by a detector for a single time step ``t``.

    Attributes
    ----------
    t : int
        Zero-based time index.
    value : float
        The raw observation ``x_t``.
    score : float
        Anomaly score computed from the *prior* baseline.
    alert : bool
        ``True`` when ``score > tau``.
    drift_detected : bool
        ``True`` when the drift detector fired at this step.
    baseline_updated : bool
        ``True`` when the baseline statistics were updated at this step.
    baseline_m : float
        Baseline median ``m`` **after** any update at this step.
    baseline_mad : float
        Baseline MAD **after** any update at this step.
    """

    t: int
    value: float
    score: float
    alert: bool
    drift_detected: bool
    baseline_updated: bool
    baseline_m: float
    baseline_mad: float


@dataclass
class DetectorResult:
    """Aggregated output for an entire time series run.

    Attributes
    ----------
    series_name : str
        Name of the input :class:`TimeSeries`.
    variant : str
        Detector variant identifier (``"A"``, ``"B"``, or ``"C"``).
    scores : list[float]
        Anomaly score for every time step.
    alerts : list[bool]
        Alert flag for every time step.
    drift_events : list[int]
        Time indices at which drift was detected.
    adaptation_events : list[int]
        Time indices at which the baseline was updated after drift.
    runtime_s : float
        Wall-clock run time in seconds.
    config : DetectorConfig
        The configuration used for this run.
    """

    series_name: str
    variant: str
    scores: List[float]
    alerts: List[bool]
    drift_events: List[int]
    adaptation_events: List[int]
    runtime_s: float
    config: DetectorConfig


# ---------------------------------------------------------------------------
# Evaluation metrics
# ---------------------------------------------------------------------------

@dataclass
class EvalMetrics:
    """Evaluation metrics comparing detector output to ground-truth labels.

    Attributes
    ----------
    precision : float
        Point-wise precision of anomaly alerts vs ground-truth labels.
    recall : float
        Point-wise recall of anomaly alerts vs ground-truth labels.
    f1 : float
        Harmonic mean of precision and recall.
    roc_auc : float
        Area under the ROC curve computed from anomaly scores.
    pr_auc : float
        Area under the precision-recall curve.
    n_drift_triggers : int
        Total number of drift events fired by the detector.
    false_drift_rate : float
        Fraction of drift events that do *not* coincide with a ground-truth
        change point (within a tolerance window).
    adaptation_accepted : int
        Number of candidate windows accepted as new baseline.
    adaptation_rejected : int
        Number of candidate windows rejected by the contamination gate.
    runtime_s : float
        Wall-clock run time of the detection run, in seconds.
    """

    precision: float
    recall: float
    f1: float
    roc_auc: float
    pr_auc: float
    n_drift_triggers: int
    false_drift_rate: float
    adaptation_accepted: int
    adaptation_rejected: int
    runtime_s: float
