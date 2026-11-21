"""Version D: robust local prediction with clipped scale adaptation.

The detector predicts the next value with a causal exponentially weighted
location and tracks residual scale with clipped absolute residuals. Scores
combine an innovation shock with a one-sided volatility shock. Residuals past
the configured clipping threshold do not move the location or scale state.
"""
from __future__ import annotations

from collections import deque

import numpy as np

from ..drift import DriftDetector, make_drift_detector
from ..statistics import anomaly_score
from ..types import DetectorConfig, StepOutput
from .base import BaseDetector


class VersionD(BaseDetector):
    """Causal robust adaptive detector with fast location and slow scale."""

    def __init__(self, config: DetectorConfig) -> None:
        super().__init__(config)
        self._residuals: deque[float] = deque(maxlen=config.window_size)
        self._location = 0.0
        self._scale = config.epsilon
        self._volatility_reference = config.epsilon
        self._drift: DriftDetector = self._make_detector()
        self._cooldown = 0
        self._initialized = False

    @property
    def name(self) -> str:
        return "D"

    def _make_detector(self) -> DriftDetector:
        cfg = self.config
        return make_drift_detector(
            cfg.drift_detector,
            adwin_delta=cfg.adwin_delta,
            ph_lambda=cfg.ph_lambda,
            ph_delta=cfg.ph_delta,
        )

    def reset(self) -> None:
        super().reset()
        self._residuals = deque(maxlen=self.config.window_size)
        self._location = 0.0
        self._scale = self.config.epsilon
        self._volatility_reference = self.config.epsilon
        self._drift = self._make_detector()
        self._cooldown = 0
        self._initialized = False

    def step(self, x: float, t: int) -> StepOutput:
        cfg = self.config
        if not self._initialized:
            self._residuals.append(float(x))
            self._location = float(np.median(self._residuals))
            if len(self._residuals) >= 2:
                mad = float(np.median(np.abs(np.asarray(self._residuals) - self._location)))
                self._scale = max(1.4826 * mad, cfg.epsilon)
            if len(self._residuals) >= cfg.window_size:
                self._initialized = True
                self._volatility_reference = self._scale
            return StepOutput(t, x, 0.0, False, False, False, self._location, self._scale)

        # Predict from state that only includes observations before t.
        residual = float(x) - self._location
        innovation = abs(residual) / max(self._scale, cfg.epsilon)
        # Volatility channel compares current absolute innovation with the
        # robust residual-scale reference; the max avoids double-counting.
        abs_resid = abs(residual)
        volatility_score = abs_resid / max(self._volatility_reference, cfg.epsilon)
        score = max(innovation, volatility_score)
        alert = score > cfg.tau

        drift = False
        updated = False
        feed = innovation if cfg.feed_scores else float(x)
        drift_now = self._drift.update(feed)
        if drift_now and self._cooldown == 0:
            drift = True
            self._drift = self._make_detector()
            self._cooldown = cfg.cooldown
            # Reset location from a robust recent batch; retain old scale so
            # a sudden variance burst cannot instantly normalize itself.
            recent = np.asarray(list(self._residuals)[-cfg.candidate_size :])
            if recent.size:
                self._location = float(np.median(recent))
                updated = True
        else:
            # Huber-like clipping: only ordinary residuals update location.
            if innovation <= cfg.tau_cand:
                alpha = 2.0 / (max(2, cfg.window_size) + 1.0)
                self._location += alpha * float(np.clip(residual, -cfg.tau_cand * self._scale,
                                                        cfg.tau_cand * self._scale))
                clipped_abs = min(abs_resid, cfg.tau_cand * self._scale)
                beta = 1.0 / max(2, cfg.window_size)
                self._scale = max((1.0 - beta) * self._scale + beta * 1.2533 * clipped_abs,
                                  cfg.epsilon)
                self._volatility_reference = max(
                    (1.0 - beta) * self._volatility_reference + beta * self._scale,
                    cfg.epsilon,
                )
                self._residuals.append(float(x))
            if self._cooldown > 0:
                self._cooldown -= 1

        return StepOutput(t, x, score, alert, drift, updated, self._location, self._scale)
