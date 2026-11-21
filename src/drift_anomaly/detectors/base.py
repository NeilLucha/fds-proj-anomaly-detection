"""
drift_anomaly.detectors.base
============================

Abstract base class for all detector variants.

Every variant inherits :class:`BaseDetector` and must implement:

* :meth:`step` — causal single-step update and scoring.
* :attr:`name` — short identifier string.

The :meth:`run` method is fully implemented here: it iterates over all
time steps, calls :meth:`step`, collects outputs, measures wall-clock time,
and returns a :class:`DetectorResult`.
"""
from __future__ import annotations

import logging
import time
from abc import ABC, abstractmethod
from typing import List, Optional

import numpy as np

from ..types import (
    DetectorConfig,
    DetectorResult,
    StepOutput,
    TimeSeries,
)

logger = logging.getLogger(__name__)


class BaseDetector(ABC):
    """Abstract base for drift-aware anomaly detectors.

    Parameters
    ----------
    config : DetectorConfig
        Hyper-parameters controlling detection behaviour.

    Attributes
    ----------
    config : DetectorConfig
        Stored configuration (read-only by convention).
    """

    def __init__(self, config: DetectorConfig) -> None:
        self.config: DetectorConfig = config

    # ------------------------------------------------------------------
    # Abstract interface
    # ------------------------------------------------------------------

    @abstractmethod
    def step(self, x: float, t: int) -> StepOutput:
        """Process a single observation and return step-level results.

        **Causality contract**: when computing the score for ``x_t`` the
        implementation may only use observations ``x_0 … x_{t-1}`` that
        have already been processed via previous :meth:`step` calls.
        ``x_t`` itself must **not** influence the baseline statistics used
        to score it.

        Parameters
        ----------
        x : float
            The current observation ``x_t``.
        t : int
            Zero-based time index.

        Returns
        -------
        StepOutput
            Score, alert flag, drift flag, and updated baseline statistics.
        """

    @property
    @abstractmethod
    def name(self) -> str:
        """Short human-readable name for this detector variant."""

    # ------------------------------------------------------------------
    # Concrete helpers
    # ------------------------------------------------------------------

    def reset(self) -> None:
        """Reset internal state so the detector can be re-run from scratch.

        Sub-classes should override and call ``super().reset()`` if they
        maintain additional mutable state.
        """
        logger.debug("%s: reset called", self.name)

    def run(self, series: TimeSeries) -> DetectorResult:
        """Run the detector over an entire :class:`TimeSeries`.

        Iterates through every point in *series*, calls :meth:`step` for
        each, and aggregates the results into a :class:`DetectorResult`.

        Parameters
        ----------
        series : TimeSeries
            The input time series.  Only :attr:`TimeSeries.points` values
            are used; ground-truth labels are **not** accessed here.

        Returns
        -------
        DetectorResult
            Aggregated scores, alerts, drift events, and timing.

        Notes
        -----
        * The detector is **not** reset before running; call :meth:`reset`
          explicitly if you want a fresh start.
        * Wall-clock time is measured with :func:`time.perf_counter`.
        """
        self.reset()

        scores: List[float] = []
        alerts: List[bool] = []
        drift_events: List[int] = []
        adaptation_events: List[int] = []

        logger.info(
            "%s: starting run on series '%s' (%d points)",
            self.name, series.name, len(series),
        )

        t_start = time.perf_counter()

        for t, point in enumerate(series.points):
            out: StepOutput = self.step(point.value, t)
            scores.append(out.score)
            alerts.append(out.alert)
            if out.drift_detected:
                drift_events.append(t)
            if out.baseline_updated:
                adaptation_events.append(t)

        t_end = time.perf_counter()
        runtime_s = t_end - t_start

        logger.info(
            "%s: finished in %.3f s — %d alerts, %d drift events, "
            "%d adaptations",
            self.name, runtime_s, sum(alerts), len(drift_events),
            len(adaptation_events),
        )

        return DetectorResult(
            series_name=series.name,
            variant=self.name,
            scores=scores,
            alerts=alerts,
            drift_events=drift_events,
            adaptation_events=adaptation_events,
            runtime_s=runtime_s,
            config=self.config,
        )
