"""
drift_anomaly.detectors.version_c_frozen_delayed
=================================================

**Version C — Frozen-Window Detector with Delayed Drift Adaptation**

Algorithm overview
------------------
1. **Initialisation** (``t < L0``): accumulate the first ``L0 = freeze_window``
   observations.  Scores are ``0``.  When ``t = L0 - 1``, the initial
   frozen baseline is computed and locked.

2. **Normal operation** (``t >= L0``):

   a. Compute anomaly score ``s_t = |x_t - m_frozen| / σ_frozen`` using
      the **frozen** baseline — no rolling updates to the reference.
   b. Feed ``s_t`` (or ``x_t`` if ``feed_scores=False``) to the drift
      detector.
   c. If drift is detected and cooldown has expired → begin the hold phase.

3. **Hold phase** (``adaptation_mode = "delayed"``):

   * Accumulate the next ``H`` observations in a hold buffer.
   * Scoring still uses the frozen baseline.
   * Drift detector is paused (not updated).
   * After ``H`` steps the contamination gate is applied to the hold
     buffer.
     - **Accepted**: baseline is updated to the hold-buffer statistics.
       Cooldown restarts.  New frozen baseline is set.
     - **Rejected**: hold buffer is discarded; baseline remains frozen.

4. **Immediate mode** (``adaptation_mode = "immediate"``):

   * On drift detection, update the baseline immediately from the last
     ``L0`` observations (re-freeze) without a hold phase.
   * This mode trades caution for speed of adaptation.

Causality
---------
In both modes the baseline used to score ``x_t`` is computed from
observations **prior** to ``t``.  ``x_t`` never influences its own score.
"""
from __future__ import annotations

import logging
from collections import deque
from typing import Deque, List, Optional

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


class VersionC(BaseDetector):
    """Frozen-Window Detector with Delayed (or Immediate) Drift Adaptation.

    Parameters
    ----------
    config : DetectorConfig
        Shared configuration.  Relevant fields:

        * ``freeze_window`` (L0) — initial reference window size.
        * ``adaptation_mode`` — ``"delayed"`` or ``"immediate"``.
        * ``hold_steps`` (H) — hold buffer length in delayed mode.
        * ``candidate_size`` (M) — alias for hold buffer in immediate mode.
        * ``tau_cand``, ``rho_max``, ``kappa_max`` — contamination gate params.
        * ``cooldown`` — minimum steps between consecutive drift events.
        * ``drift_detector`` — ``"adwin"`` or ``"page-hinkley"``.

    Attributes
    ----------
    name : str
        ``"C"``
    """

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        L0 = config.freeze_window
        self._init_buffer: List[float] = []          # used during warm-up
        self._frozen_m: float = 0.0
        self._frozen_mad: float = 0.0
        self._frozen_sigma: float = config.epsilon
        self._drift_detector: DriftDetector = self._make_detector()
        self._cooldown_remaining: int = 0
        self._in_hold_phase: bool = False
        self._hold_buffer: List[float] = []
        self._recent_window: Deque[float] = deque(maxlen=L0)

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D102
        return "C"

    # ------------------------------------------------------------------
    # Reset
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Clear all state."""
        super().reset()
        cfg = self.config
        self._init_buffer = []
        self._frozen_m = 0.0
        self._frozen_mad = 0.0
        self._frozen_sigma = cfg.epsilon
        self._drift_detector = self._make_detector()
        self._cooldown_remaining = 0
        self._in_hold_phase = False
        self._hold_buffer = []
        self._recent_window = deque(maxlen=cfg.freeze_window)

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
        L0 = cfg.freeze_window

        # ---- Warm-up phase --------------------------------------------
        if len(self._init_buffer) < L0:
            self._init_buffer.append(x)
            self._recent_window.append(x)

            if len(self._init_buffer) == L0:
                # Freeze the initial baseline
                arr = np.array(self._init_buffer, dtype=float)
                m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
                self._frozen_m = m
                self._frozen_mad = mad_val
                self._frozen_sigma = sig
                logger.info(
                    "t=%d VersionC: initial baseline frozen "
                    "(m=%.4f, MAD=%.4f, σ=%.4f)",
                    t, m, mad_val, sig,
                )

            return StepOutput(
                t=t,
                value=x,
                score=0.0,
                alert=False,
                drift_detected=False,
                baseline_updated=False,
                baseline_m=self._frozen_m,
                baseline_mad=self._frozen_mad,
            )

        # ---- Compute score with FROZEN baseline -----------------------
        score = anomaly_score(x, self._frozen_m, self._frozen_sigma)
        alert = score > cfg.tau

        # Keep a sliding window of recent observations (for immediate mode)
        self._recent_window.append(x)

        # ---- Cooldown counter -----------------------------------------
        if self._cooldown_remaining > 0:
            self._cooldown_remaining -= 1

        # ---- Hold phase (delayed adaptation) --------------------------
        if self._in_hold_phase:
            self._hold_buffer.append(x)
            H = cfg.hold_steps

            if len(self._hold_buffer) >= H:
                # Apply contamination gate
                hold_arr = np.array(self._hold_buffer, dtype=float)
                accepted, frac, ratio, cand_m = check_candidate_window(
                    hold_arr,
                    self._frozen_m,
                    self._frozen_mad,
                    cfg.tau_cand,
                    cfg.rho_max,
                    cfg.kappa_max,
                    cfg.epsilon,
                    tau_int=cfg.tau_int,
                )
                self._in_hold_phase = False
                self._hold_buffer = []

                if accepted:
                    logger.info(
                        "t=%d VersionC (delayed): hold buffer accepted "
                        "(frac=%.3f, mad_ratio=%.3f) → refreezing baseline",
                        t, frac, ratio,
                    )
                    arr = np.array(hold_arr, dtype=float)
                    m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
                    self._frozen_m = m
                    self._frozen_mad = mad_val
                    self._frozen_sigma = sig
                    # Reset recent window to hold buffer content
                    self._recent_window = deque(hold_arr.tolist(), maxlen=L0)
                    self._cooldown_remaining = cfg.cooldown
                    self._drift_detector = self._make_detector()
                    return StepOutput(
                        t=t,
                        value=x,
                        score=score,
                        alert=alert,
                        drift_detected=False,
                        baseline_updated=True,
                        baseline_m=self._frozen_m,
                        baseline_mad=self._frozen_mad,
                    )
                else:
                    logger.info(
                        "t=%d VersionC (delayed): hold buffer rejected "
                        "(frac=%.3f, mad_ratio=%.3f) → keep frozen baseline",
                        t, frac, ratio,
                    )
                    self._drift_detector = self._make_detector()
                    return StepOutput(
                        t=t,
                        value=x,
                        score=score,
                        alert=alert,
                        drift_detected=False,
                        baseline_updated=False,
                        baseline_m=self._frozen_m,
                        baseline_mad=self._frozen_mad,
                    )

            # Still accumulating hold buffer
            return StepOutput(
                t=t,
                value=x,
                score=score,
                alert=alert,
                drift_detected=False,
                baseline_updated=False,
                baseline_m=self._frozen_m,
                baseline_mad=self._frozen_mad,
            )

        # ---- Normal operation: update drift detector ------------------
        feed_val = score if cfg.feed_scores else x
        drift_now = self._drift_detector.update(feed_val)

        if drift_now and self._cooldown_remaining == 0:
            logger.info("t=%d VersionC: drift detected", t)

            if cfg.adaptation_mode == "immediate":
                # Immediate mode: re-freeze from the last L0 observations
                arr = np.array(list(self._recent_window), dtype=float)
                if arr.size >= 2:
                    m, mad_val, sig = compute_window_stats(arr, cfg.epsilon)
                    self._frozen_m = m
                    self._frozen_mad = mad_val
                    self._frozen_sigma = sig
                    logger.info(
                        "t=%d VersionC (immediate): baseline updated "
                        "(m=%.4f, MAD=%.4f)",
                        t, m, mad_val,
                    )
                    baseline_updated = True
                else:
                    baseline_updated = False
                self._drift_detector.reset()
                self._cooldown_remaining = cfg.cooldown
                return StepOutput(
                    t=t,
                    value=x,
                    score=score,
                    alert=alert,
                    drift_detected=True,
                    baseline_updated=baseline_updated,
                    baseline_m=self._frozen_m,
                    baseline_mad=self._frozen_mad,
                )
            else:
                # Delayed mode: start hold phase
                self._in_hold_phase = True
                self._hold_buffer = [x]
                self._cooldown_remaining = cfg.cooldown
                return StepOutput(
                    t=t,
                    value=x,
                    score=score,
                    alert=alert,
                    drift_detected=True,
                    baseline_updated=False,
                    baseline_m=self._frozen_m,
                    baseline_mad=self._frozen_mad,
                )

        # ---- No drift --------------------------------------------------
        return StepOutput(
            t=t,
            value=x,
            score=score,
            alert=alert,
            drift_detected=False,
            baseline_updated=False,
            baseline_m=self._frozen_m,
            baseline_mad=self._frozen_mad,
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
