"""
drift_anomaly.detectors.version_b_rolling_adwin
===============================================

**Version B — Simplified Rolling Window + ADWIN**

Algorithm overview
------------------
1. **Warm-up phase** (``t < L``): accumulate the first ``L`` observations.
   Score every point as 0, raise no alerts.

2. **Normal operation** (``t >= L``):

   a. Compute baseline statistics ``(m, MAD, σ)`` from the rolling window
      ``x[t-L : t]`` (past ``L`` observations — strictly causal).
   b. Compute anomaly score ``s_t = |x_t - m| / σ``.
   c. Feed ``s_t`` (or ``x_t`` if ``feed_scores=False``) to the ADWIN
      drift detector.
   d. If ADWIN fires *and* the cooldown has expired:

      * Mark ``drift_detected = True`` for this step.
      * Reset ADWIN.
      * Optionally reset the rolling window to only the ``M`` most recent
        observations so the baseline quickly adapts to the new regime.

Design rationale
----------------
Version B is intentionally simpler than Version A: there is no candidate
accumulation phase and no contamination gate.  Drift is handled by letting
ADWIN trim its own window; after a drift event the rolling window is
partially flushed to accelerate baseline adaptation.

Causality
---------
Baseline statistics at step ``t`` are always derived from
``x[t-L : t]``.  ``x_t`` is appended *after* scoring.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque

import numpy as np

from ..drift import DriftDetector, make_drift_detector
from ..statistics import anomaly_score, compute_window_stats
from ..types import DetectorConfig, StepOutput
from .base import BaseDetector

logger = logging.getLogger(__name__)


class VersionB(BaseDetector):
    """Simplified Rolling Window + ADWIN detector.

    Parameters
    ----------
    config : DetectorConfig
        Shared configuration object.

    Attributes
    ----------
    name : str
        ``"B"``
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._drift_detector: DriftDetector = self._make_detector()
        L = config.window_size
        self._window: Deque[float] = deque(maxlen=L)
        self._baseline_m: float = 0.0
        self._baseline_mad: float = 0.0
        self._baseline_sigma: float = config.epsilon
        self._cooldown_remaining: int = 0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D102
        return "B"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all internal state."""
        super().reset()
        cfg = self.config
        self._drift_detector = self._make_detector()
        self._window = deque(maxlen=cfg.window_size)
        self._baseline_m = 0.0
        self._baseline_mad = 0.0
        self._baseline_sigma = cfg.epsilon
        self._cooldown_remaining = 0

    # ------------------------------------------------------------------
    # Core step
    # ------------------------------------------------------------------

    def step(self, x: float, t: int) -> StepOutput:
        """Process observation ``x_t`` at time index ``t``.

        Parameters
        ----------
        x : float
            Current observation.
        t : int
            Zero-based time index.

        Returns
        -------
        StepOutput
            Score, alert, drift flag, and updated baseline statistics.
        """
        cfg = self.config
        drift_detected = False
        baseline_updated = False

        # ---- Warm-up phase --------------------------------------------
        if len(self._window) < cfg.window_size:
            self._window.append(x)
            score = 0.0
            alert = False
            if len(self._window) >= 2:
                arr = np.array(self._window, dtype=float)
                m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
                self._baseline_m = m
                self._baseline_mad = mad_val
                self._baseline_sigma = sig
            return StepOutput(
                t=t,
                value=x,
                score=score,
                alert=alert,
                drift_detected=False,
                baseline_updated=False,
                baseline_m=self._baseline_m,
                baseline_mad=self._baseline_mad,
            )

        # ---- Score using CURRENT (pre-append) baseline ----------------
        score = anomaly_score(x, self._baseline_m, self._baseline_sigma)
        alert = score > cfg.tau

        # ---- Cooldown counter -----------------------------------------
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ---- Update drift detector ------------------------------------
        feed_val = score if cfg.feed_scores else x
        drift_now = self._drift_detector.update(feed_val)

        if drift_now and self._cooldown_remaining == 0:
            drift_detected = True
            logger.info(
                "t=%d VersionB: drift detected → flushing window to last M=%d obs",
                t, cfg.candidate_size,
            )
            # Reset detector
            self._drift_detector.reset()
            # Partial window flush: keep only the last M observations
            # (most recent regime observations)
            M = cfg.candidate_size
            recent: list[float] = list(self._window)[-M:] + [x]
            self._window = deque(recent, maxlen=cfg.window_size)
            arr = np.array(self._window, dtype=float)
            m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
            self._baseline_m = m
            self._baseline_mad = mad_val
            self._baseline_sigma = sig
            baseline_updated = True
            self._cooldown_remaining = cfg.cooldown
            return StepOutput(
                t=t,
                value=x,
                score=score,
                alert=alert,
                drift_detected=True,
                baseline_updated=True,
                baseline_m=self._baseline_m,
                baseline_mad=self._baseline_mad,
            )

        # ---- No drift: slide window and update baseline ----------------
        self._window.append(x)
        arr = np.array(self._window, dtype=float)
        m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
        self._baseline_m = m
        self._baseline_mad = mad_val
        self._baseline_sigma = sig

        return StepOutput(
            t=t,
            value=x,
            score=score,
            alert=alert,
            drift_detected=False,
            baseline_updated=False,
            baseline_m=self._baseline_m,
            baseline_mad=self._baseline_mad,
        )

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _make_detector(self) -> DriftDetector:
        cfg = self.config
        return make_drift_detector(
            cfg.drift_detector,
            adwin_delta=cfg.adwin_delta,
            ph_lambda=cfg.ph_lambda,
            ph_delta=cfg.ph_delta,
        )
