"""
drift_anomaly.statistics
========================

Robust statistical primitives used throughout the anomaly-detection
pipeline.

All functions operate on plain :class:`numpy.ndarray` objects and are
intentionally free of Pandas so they can be called in tight, causal loops
without overhead.

Naming follows the algorithm description:
  m       = rolling median
  MAD     = median absolute deviation
  sigma   = robust scale estimate (consistency-corrected MAD + epsilon)
  s_t     = anomaly score at time t
"""
from __future__ import annotations

import logging
from typing import Tuple

import numpy as np

logger = logging.getLogger(__name__)

# 1 / Phi^{-1}(0.75) ≈ 1.4826  — makes MAD a consistent estimator of σ
# for Gaussian data.
GAUSSIAN_CONSISTENCY_FACTOR: float = 1.4826


# ---------------------------------------------------------------------------
# Core estimators
# ---------------------------------------------------------------------------

def rolling_median(window: np.ndarray) -> float:
    """Compute the median of *window*.

    Parameters
    ----------
    window : np.ndarray
        One-dimensional array of recent observations (past values only).

    Returns
    -------
    float
        Median value of *window*.

    Notes
    -----
    Uses :func:`numpy.median`, which is *O(n log n)*.  For the window sizes
    used in practice (L ≤ 1 000) this is fast enough.
    """
    if window.size == 0:
        raise ValueError("rolling_median: window must be non-empty")
    return float(np.median(window))


def mad(window: np.ndarray, center: float) -> float:
    """Compute the Median Absolute Deviation (MAD) around *center*.

    .. math::
        \\text{MAD} = \\operatorname{median}(|x_i - \\text{center}|)

    Parameters
    ----------
    window : np.ndarray
        One-dimensional array of recent observations.
    center : float
        The central value (usually the median) around which deviations are
        measured.

    Returns
    -------
    float
        The raw MAD (not yet scaled by the consistency factor).
    """
    if window.size == 0:
        raise ValueError("mad: window must be non-empty")
    return float(np.median(np.abs(window - center)))


def sigma_hat(mad_val: float, epsilon: float = 1e-6) -> float:
    """Consistency-corrected robust scale estimate.

    .. math::
        \\hat{\\sigma} = 1.4826 \\cdot \\text{MAD} + \\epsilon

    The additive ``epsilon`` guards against a zero denominator when the
    window is constant (e.g. a step-function segment all equal).

    Parameters
    ----------
    mad_val : float
        Raw MAD value.
    epsilon : float
        Small positive floor added to prevent zero scale.

    Returns
    -------
    float
        The robust scale estimate.
    """
    return GAUSSIAN_CONSISTENCY_FACTOR * mad_val + epsilon


def anomaly_score(x: float, m: float, sigma: float) -> float:
    """Compute the normalised anomaly score for a single observation.

    .. math::
        s_t = \\frac{|x_t - m|}{\\hat{\\sigma}}

    Parameters
    ----------
    x : float
        Current observation ``x_t``.
    m : float
        Baseline median computed from *past* observations only.
    sigma : float
        Robust scale estimate (``sigma_hat``) from past observations.

    Returns
    -------
    float
        Non-negative anomaly score.  Scores above a threshold ``τ``
        indicate anomalies.
    """
    return abs(x - m) / sigma


def compute_window_stats(
    window: np.ndarray,
    epsilon: float = 1e-6,
) -> Tuple[float, float, float]:
    """Compute median, MAD, and robust sigma for *window* in one call.

    Parameters
    ----------
    window : np.ndarray
        One-dimensional array of *past* observations (causal window).
    epsilon : float
        Floor added to MAD when computing ``sigma_hat``.

    Returns
    -------
    tuple[float, float, float]
        ``(median, mad_val, sigma)`` where

        * ``median`` is the window median,
        * ``mad_val`` is the raw MAD around ``median``, and
        * ``sigma`` is ``sigma_hat(mad_val, epsilon)``.

    Raises
    ------
    ValueError
        If *window* is empty.
    """
    if window.size == 0:
        raise ValueError("compute_window_stats: window must be non-empty")
    m = rolling_median(window)
    mad_val = mad(window, m)
    sig = sigma_hat(mad_val, epsilon)
    return m, mad_val, sig


# ---------------------------------------------------------------------------
# Candidate-window contamination gate
# ---------------------------------------------------------------------------

def check_candidate_window(
    candidate: np.ndarray,
    base_m: float,
    base_mad: float,
    tau_cand: float,
    rho_max: float,
    kappa_max: float,
    epsilon: float,
    tau_int: float | None = None,
) -> Tuple[bool, float, float, float]:
    """Evaluate whether a candidate window is clean enough to adopt as the
    new baseline.

    Two rejection criteria are applied (the *contamination gate*):

    1. **Internal anomaly contamination**: reject if the fraction of
       candidate points far from the candidate's own median exceeds
       ``rho_max``. This permits clean location shifts.

    2. **Volatility gate**: reject if the ratio
       ``MAD_candidate / MAD_baseline > kappa_max``.

    If both gates are passed the candidate is *accepted* and its statistics
    can replace the current baseline.

    Parameters
    ----------
    candidate : np.ndarray
        Array of recent observations collected during the hold phase.
    base_m : float
        Baseline median *before* the potential update (from past window).
    base_mad : float
        Baseline MAD *before* the potential update.
    tau_cand : float
        Anomaly threshold used *inside* the contamination check.
    rho_max : float
        Maximum tolerated fraction of anomalous points (0 < rho_max < 1).
    kappa_max : float
        Maximum tolerated ``MAD_candidate / MAD_baseline`` ratio.
    epsilon : float
        Floor used to compute ``sigma_hat`` for the contamination check.

    Returns
    -------
    tuple[bool, float, float, float]
        ``(accepted, frac_anomalous, mad_ratio, candidate_m)`` where

        * ``accepted`` — whether the candidate passed both gates,
        * ``frac_anomalous`` — fraction of candidate points flagged as
          anomalous against the *old* baseline,
        * ``mad_ratio`` — ``MAD_candidate / (MAD_baseline + epsilon)``,
        * ``candidate_m`` — median of the candidate window.

    Notes
    -----
    Causality is preserved: *base_m* and *base_mad* must come from
    observations *prior* to the candidate window.
    """
    if candidate.size == 0:
        logger.warning("check_candidate_window: empty candidate — rejecting")
        return False, 0.0, 0.0, 0.0

    # --- internal contamination and volatility checks ------------------
    candidate_m = float(np.median(candidate))
    candidate_mad = float(np.median(np.abs(candidate - candidate_m)))
    candidate_sigma = sigma_hat(candidate_mad, epsilon)
    scores = np.abs(candidate - candidate_m) / candidate_sigma
    internal_threshold = tau_cand if tau_int is None else tau_int
    frac_anomalous = float(np.mean(scores > internal_threshold))
    mad_ratio = candidate_mad / (base_mad + epsilon)

    # --- decision -------------------------------------------------------
    accepted = (frac_anomalous <= rho_max) and (mad_ratio <= kappa_max)

    if not accepted:
        logger.debug(
            "Candidate rejected: frac_anomalous=%.3f (max %.3f), "
            "mad_ratio=%.3f (max %.3f)",
            frac_anomalous, rho_max, mad_ratio, kappa_max,
        )

    return accepted, frac_anomalous, mad_ratio, candidate_m
