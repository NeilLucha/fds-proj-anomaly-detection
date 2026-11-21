"""
drift_anomaly.detectors.version_a_cautious_rolling
===================================================

**Version A — Cautious Rolling Baseline with Candidate-Window
Contamination Gate**

Algorithm overview
------------------
1. **Warm-up phase** (``t < L``): accumulate the first ``L`` observations.
   Score every point as 0, raise no alerts, produce no drift signals.

2. **Normal operation** (``t >= L``):

   a. Compute baseline statistics ``(m, MAD, σ)`` from the rolling window
      ``x[t-L : t]`` (strictly past — no data leakage).
   b. Compute anomaly score ``s_t = |x_t - m| / σ``.
   c. Feed ``s_t`` (or ``x_t`` if ``feed_scores=False``) to the drift
      detector.
   d. If drift is detected and the cooldown has expired, enter the
      **candidate-accumulation phase**.

3. **Candidate-accumulation phase**: collect the next ``M`` observations
   into a candidate buffer.  During this phase:

   * Scoring continues using the *frozen* pre-drift baseline.
   * The drift detector is paused (not updated) to avoid re-triggering.

4. **Contamination gate**: once ``M`` observations have been collected,
   call :func:`~drift_anomaly.statistics.check_candidate_window` with the
   frozen baseline statistics.

   * **Accepted**: baseline is updated to the candidate window's statistics.
     The rolling window is reset to the candidate buffer.  Cooldown timer
     restarts.
   * **Rejected**: the candidate is discarded, the original baseline is
     kept, and the drift detector is re-initialised.

Causality
---------
Baseline statistics at step ``t`` are always computed from
``x[t-L : t]``, i.e., strictly past data.  ``x_t`` never influences its
own score.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, Optional

import numpy as np

from ..drift import DriftDetector, make_drift_detector
from ..statistics import (
    anomaly_score,
    check_candidate_window,
    compute_window_stats,
)
from ..types import DetectorConfig, StepOutput
from .base import BaseDetector

logger = logging.getLogger(__name__)


class VersionA(BaseDetector):
    """Cautious Rolling Baseline with Candidate-Window Contamination Gate.

    Parameters
    ----------
    config : DetectorConfig
        Shared configuration object.

    Attributes
    ----------
    name : str
        ``"A"``
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._drift_detector: DriftDetector = self._make_detector()

        # Rolling window (up to L observations)
        L = config.window_size
        self._window: Deque[float] = deque(maxlen=L)

        # Frozen baseline (set when drift is first detected)
        self._baseline_m: float = 0.0
        self._baseline_mad: float = 0.0
        self._baseline_sigma: float = config.epsilon

        # State
        self._in_candidate_phase: bool = False
        self._candidate_buffer: list[float] = []
        self._cooldown_remaining: int = 0
        self._last_drift_t: int = -config.cooldown - 1

        # Frozen baseline snapshot for candidate gate
        self._frozen_m: float = 0.0
        self._frozen_mad: float = 0.0

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D102
        return "A"

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
        self._in_candidate_phase = False
        self._candidate_buffer = []
        self._cooldown_remaining = 0
        self._last_drift_t = -cfg.cooldown - 1
        self._frozen_m = 0.0
        self._frozen_mad = 0.0

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
            # During warm-up we cannot score yet
            score = 0.0
            alert = False
            # Update baseline from growing window (causal: x already appended
            # but we only use the window BEFORE scoring was possible)
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

        # ---- Score using CURRENT (pre-update) baseline ----------------
        score = anomaly_score(x, self._baseline_m, self._baseline_sigma)
        alert = score > cfg.tau

        # ---- Cooldown counter -----------------------------------------
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ---- Candidate-accumulation phase -----------------------------
        if self._in_candidate_phase:
            self._candidate_buffer.append(x)

            if len(self._candidate_buffer) >= cfg.candidate_size:
                # Gate evaluation
                candidate_arr = np.array(self._candidate_buffer, dtype=float)
                accepted, frac, ratio, cand_m = check_candidate_window(
                    candidate_arr,
                    self._frozen_m,
                    self._frozen_mad,
                    cfg.tau_cand,
                    cfg.rho_max,
                    cfg.kappa_max,
                    cfg.epsilon,
                    tau_int=cfg.tau_int,
                )
                self._in_candidate_phase = False
                self._candidate_buffer = []

                if accepted:
                    logger.info(
                        "t=%d VersionA: candidate accepted "
                        "(frac=%.3f, mad_ratio=%.3f) → baseline updated",
                        t, frac, ratio,
                    )
                    # Reset rolling window to candidate observations
                    self._window = deque(candidate_arr.tolist(), maxlen=cfg.window_size)
                    arr = np.array(self._window, dtype=float)
                    m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
                    self._baseline_m = m
                    self._baseline_mad = mad_val
                    self._baseline_sigma = sig
                    baseline_updated = True
                    # Restart cooldown
                    self._cooldown_remaining = cfg.cooldown
                    # Re-init drift detector from scratch
                    self._drift_detector = self._make_detector()
                else:
                    logger.info(
                        "t=%d VersionA: candidate rejected "
                        "(frac=%.3f, mad_ratio=%.3f) → keep old baseline",
                        t, frac, ratio,
                    )
                    # Keep old baseline; re-init drift detector
                    self._drift_detector = self._make_detector()

            # Slide window and return (no drift update during candidate phase)
            self._window.append(x)
            return StepOutput(
                t=t,
                value=x,
                score=score,
                alert=alert,
                drift_detected=False,
                baseline_updated=baseline_updated,
                baseline_m=self._baseline_m,
                baseline_mad=self._baseline_mad,
            )

        # ---- Normal operation: update drift detector -------------------
        feed_val = score if cfg.feed_scores else x
        drift_now = self._drift_detector.update(feed_val)

        if drift_now and self._cooldown_remaining == 0:
            drift_detected = True
            logger.info(
                "t=%d VersionA: drift detected → entering candidate phase",
                t,
            )
            # Freeze the current baseline for the contamination gate
            self._frozen_m = self._baseline_m
            self._frozen_mad = self._baseline_mad
            # Enter candidate-accumulation phase
            self._in_candidate_phase = True
            self._candidate_buffer = [x]  # x starts the candidate buffer
            self._cooldown_remaining = cfg.cooldown
            # Slide rolling window (x included, baseline stats will update
            # after candidate phase)
            self._window.append(x)
            return StepOutput(
                t=t,
                value=x,
                score=score,
                alert=alert,
                drift_detected=True,
                baseline_updated=False,
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
