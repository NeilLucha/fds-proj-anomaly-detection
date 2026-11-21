"""Version E: causal robust residual, seasonality and persistence detector."""
from __future__ import annotations

from collections import deque

import numpy as np

from ..drift import make_drift_detector
from ..types import DetectorConfig, StepOutput
from .base import BaseDetector


class VersionE(BaseDetector):
    """Robust scale floor plus trend/seasonal residuals and alert persistence."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self.reset()

    @property
    def name(self) -> str:
        return "E2" if self.dual_tier else "E"

    dual_tier = False

    def reset(self) -> None:
        super().reset()
        cfg = self.config
        self._window = deque(maxlen=max(2, cfg.window_size))
        self._score_history = deque(maxlen=max(50, cfg.window_size))
        persistence_k = cfg.e2_persistence_k if self.dual_tier else cfg.e_persistence_k
        self._recent_flags = deque(maxlen=max(2, persistence_k))
        self._alert_hold_remaining = 0
        self._detector = make_drift_detector(
            cfg.drift_detector, adwin_delta=cfg.adwin_delta,
            ph_lambda=cfg.ph_lambda, ph_delta=cfg.ph_delta,
        )
        self._location = 0.0
        self._scale = max(cfg.e_sigma_floor, cfg.epsilon)
        self._last_drift = -cfg.cooldown - 1
        self._seasonal_lag = max(2, cfg.e_seasonal_lag)
        self._lag_selected = False

    def step(self, x: float, t: int) -> StepOutput:
        cfg = self.config
        x = float(x)
        if len(self._window) < max(2, cfg.window_size):
            self._window.append(x)
            arr = np.asarray(self._window, dtype=float)
            self._location = float(np.median(arr))
            if len(arr) > 1:
                self._scale = self._robust_scale(arr)
            if cfg.e_auto_lag and len(arr) >= 8 and not self._lag_selected:
                self._seasonal_lag = self._select_lag(arr)
                self._lag_selected = True
            return StepOutput(t, x, 0.0, False, False, False, self._location, self._scale)

        prior = np.asarray(self._window, dtype=float)
        median = float(np.median(prior))
        mad = float(np.median(np.abs(prior - median)))
        q25, q75 = np.quantile(prior, [0.25, 0.75])
        iqr = float(q75 - q25)
        scale = max(1.4826 * mad, cfg.e_scale_floor_ratio * iqr, cfg.e_sigma_floor)

        # Robust local linear predictor, fit only to the trailing causal window.
        nfit = min(len(prior), max(8, cfg.window_size // 3))
        y = prior[-nfit:]
        if nfit >= 3:
            # Closed-form least-squares slope for x=0..n-1, avoiding a
            # per-step polyfit while remaining strictly causal.
            xbar = (nfit - 1) / 2.0
            ybar = float(np.mean(y))
            denom = nfit * (nfit * nfit - 1) / 12.0
            slope = float(np.dot(np.arange(nfit, dtype=float) - xbar, y - ybar) / denom)
            level_prediction = ybar + slope * ((nfit - 1) - xbar + 1.0)
        else:
            level_prediction = median
        level_score = abs(x - level_prediction) / scale

        lag_score = 0.0
        lag = self._seasonal_lag
        if len(prior) >= lag:
            seasonal_resid = prior[lag:] - prior[:-lag]
            smed = float(np.median(seasonal_resid))
            smad = float(np.median(np.abs(seasonal_resid - smed)))
            seasonal_scale = max(1.4826 * smad, cfg.e_scale_floor_ratio * iqr, cfg.e_sigma_floor)
            lag_score = abs((x - prior[-lag]) - smed) / seasonal_scale

        score = max(level_score, lag_score)
        # Calibrate alert thresholds from prior accepted (non-alert) scores.
        # The current score is deliberately absent, so calibration is causal.
        calibration = 1.0
        tau_low_base = cfg.e2_tau_moderate if self.dual_tier else cfg.e_tau_low
        tau_high_base = cfg.e2_tau_extreme if self.dual_tier else cfg.e_tau_high
        if cfg.e_online_calibration and len(self._score_history) >= 20:
            baseline_q = float(np.quantile(self._score_history, cfg.e_calibration_quantile))
            calibration = max(1.0, baseline_q / max(tau_low_base, cfg.epsilon))
        low, high = tau_low_base * calibration, tau_high_base * calibration
        if self.dual_tier:
            # E2 fires immediately for a single extreme shock; moderate scores
            # need two exceedances in the preceding/current K observations.
            self._recent_flags.append(score > low)
            k = min(max(2, cfg.e2_persistence_k), len(self._recent_flags))
            persistent = sum(list(self._recent_flags)[-k:]) >= 2
            triggered = score > high or persistent
            if triggered:
                self._alert_hold_remaining = max(0, cfg.e2_alert_hold_steps)
            elif self._alert_hold_remaining > 0:
                triggered = True
                self._alert_hold_remaining -= 1
            alert = triggered
        else:
            extreme = score >= high
            self._recent_flags.append(score >= low)
            k = min(max(1, cfg.e_persistence_k), len(self._recent_flags))
            persistent = k >= 2 and sum(list(self._recent_flags)[-k:]) >= 2
            alert = extreme or persistent

        clipped_residual = float(np.clip(x - level_prediction, -3.0 * scale, 3.0 * scale))
        drift = self._detector.update(clipped_residual)
        if drift and t - self._last_drift >= cfg.cooldown:
            self._last_drift = t
            # Update location only from a dispersion-compatible recent block.
            recent = np.asarray(list(self._window)[-cfg.candidate_size:], dtype=float)
            if recent.size:
                rmed = float(np.median(recent))
                rmad = float(np.median(np.abs(recent - rmed)))
                if rmad <= max(cfg.kappa_max * mad, cfg.e_sigma_floor):
                    self._location = rmed
                    self._window.clear()
                    self._window.extend(recent.tolist())
                    updated = True
                else:
                    updated = False
            else:
                updated = False
            self._detector = make_drift_detector(
                cfg.drift_detector, adwin_delta=cfg.adwin_delta,
                ph_lambda=cfg.ph_lambda, ph_delta=cfg.ph_delta,
            )
        else:
            updated = False

        # The current score is fixed before this update, keeping causality.
        self._window.append(x)
        if not alert and score < low:
            self._score_history.append(score)
        return StepOutput(t, x, score, bool(alert), bool(drift), updated,
                          float(self._location if updated else median), float(scale))

    def _robust_scale(self, arr: np.ndarray) -> float:
        cfg = self.config
        med = float(np.median(arr))
        mad = float(np.median(np.abs(arr - med)))
        q25, q75 = np.quantile(arr, [0.25, 0.75])
        return max(1.4826 * mad, cfg.e_scale_floor_ratio * float(q75 - q25), cfg.e_sigma_floor)

    def _select_lag(self, values: np.ndarray) -> int:
        cfg = self.config
        x = np.asarray(values, dtype=float)
        x = x - np.median(x)
        max_lag = min(max(2, cfg.e_max_lag), max(2, len(x) // 2))
        denom = float(np.dot(x, x))
        if denom <= cfg.epsilon:
            return max(2, cfg.e_seasonal_lag)
        scores = [float(np.dot(x[:-lag], x[lag:]) / denom) for lag in range(2, max_lag + 1)]
        return int(np.argmax(scores) + 2) if scores else max(2, cfg.e_seasonal_lag)


class VersionE2(VersionE):
    """Version E with immediate extreme-spike and persistent moderate tiers."""

    dual_tier = True
