"""
drift_anomaly.data.synthetic
============================

Deterministic synthetic time-series generator for benchmarking and unit tests.

All methods on :class:`SyntheticGenerator` accept a *seed* at construction
time so that every run with the same seed produces identical data, enabling
fully reproducible experiments.

Patterns available
------------------
stationary       – pure Gaussian noise, no anomalies or drift
step_shift       – mean step-change at a specified index + isolated spikes
variance_shift   – variance step-change at a specified index + isolated spikes
spike_burst      – sustained anomalous burst region in an otherwise normal series
isolated_spikes  – point anomalies at explicit positions
compound         – four-segment series combining all patterns
"""

from __future__ import annotations

import logging
from typing import List, Optional

import numpy as np

from ..types import TimeSeries, TimeSeriesPoint

logger = logging.getLogger(__name__)


class SyntheticGenerator:
    """Generate labelled synthetic univariate time series with a fixed seed.

    Parameters
    ----------
    seed : int
        Seed for :class:`numpy.random.Generator`.  Using the same seed
        guarantees identical outputs across runs.

    Examples
    --------
    >>> gen = SyntheticGenerator(seed=0)
    >>> ts = gen.stationary(n=200)
    >>> len(ts)
    200
    """

    def __init__(self, seed: int = 42) -> None:
        self.seed = seed
        self._rng = np.random.default_rng(seed)

    # ------------------------------------------------------------------
    # Private helpers
    # ------------------------------------------------------------------

    def _fresh_rng(self) -> np.random.Generator:
        """Return a fresh Generator seeded from *self.seed*.

        Each public method calls this so that the output of method A is
        independent of whether method B was called first.
        """
        return np.random.default_rng(self.seed)

    @staticmethod
    def _make_timestamps(n: int) -> np.ndarray:
        """Return integer timestamps 0, 1, …, n-1 as floats."""
        return np.arange(n, dtype=float)

    # ------------------------------------------------------------------
    # Public generators
    # ------------------------------------------------------------------

    def stationary(
        self,
        n: int = 1000,
        mu: float = 0.0,
        sigma: float = 1.0,
    ) -> TimeSeries:
        """Generate a purely stationary Gaussian series with no anomalies.

        Parameters
        ----------
        n : int
            Number of time steps.
        mu : float
            Mean of the Gaussian distribution.
        sigma : float
            Standard deviation of the Gaussian distribution.

        Returns
        -------
        TimeSeries
            All points have ``is_anomaly=False`` and ``change_point=False``.
        """
        rng = self._fresh_rng()
        values = rng.normal(loc=mu, scale=sigma, size=n)
        timestamps = self._make_timestamps(n)

        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=False,
                change_point=False,
            )
            for i in range(n)
        ]

        logger.debug("stationary: n=%d, mu=%.2f, sigma=%.2f", n, mu, sigma)
        return TimeSeries(
            points=points,
            name="synthetic_stationary",
            metadata={"generator": "SyntheticGenerator", "pattern": "stationary",
                      "mu": mu, "sigma": sigma, "seed": self.seed},
        )

    def step_shift(
        self,
        n: int = 1000,
        shift_at: Optional[int] = None,
        mu_before: float = 0.0,
        mu_after: float = 5.0,
        sigma: float = 1.0,
        n_anomalies: int = 10,
    ) -> TimeSeries:
        """Generate a mean step-shift series with isolated spike anomalies.

        A change-point label (``change_point=True``) is placed at index
        *shift_at*.  Additionally *n_anomalies* random spikes of magnitude
        ``±6·sigma`` are injected at positions that do *not* coincide with
        the change-point index; those points carry ``is_anomaly=True``.

        Parameters
        ----------
        n : int
            Total length of the series.
        shift_at : Optional[int]
            Index at which the mean shifts. Defaults to n // 2.
        mu_before : float
            Mean of the first segment (indices ``[0, shift_at)``).
        mu_after : float
            Mean of the second segment (indices ``[shift_at, n)``).
        sigma : float
            Gaussian noise standard deviation applied throughout.
        n_anomalies : int
            Number of isolated spike anomalies to inject.

        Returns
        -------
        TimeSeries
        """
        if shift_at is None:
            shift_at = n // 2
        if not 0 < shift_at < n:
            raise ValueError(f"shift_at={shift_at} must be in (0, {n}).")

        rng = self._fresh_rng()
        values = np.empty(n, dtype=float)
        values[:shift_at] = rng.normal(loc=mu_before, scale=sigma, size=shift_at)
        values[shift_at:] = rng.normal(loc=mu_after, scale=sigma, size=n - shift_at)

        is_anomaly = np.zeros(n, dtype=bool)
        change_point = np.zeros(n, dtype=bool)
        change_point[shift_at] = True

        # Inject spikes at positions away from shift_at
        candidates = [i for i in range(n) if i != shift_at]
        spike_positions = rng.choice(candidates, size=min(n_anomalies, len(candidates)),
                                     replace=False)
        for pos in spike_positions:
            sign = rng.choice([-1.0, 1.0])
            baseline = mu_before if pos < shift_at else mu_after
            values[pos] = baseline + sign * 6.0 * sigma
            is_anomaly[pos] = True

        timestamps = self._make_timestamps(n)
        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=bool(is_anomaly[i]),
                change_point=bool(change_point[i]),
            )
            for i in range(n)
        ]

        logger.debug(
            "step_shift: n=%d, shift_at=%d, mu_before=%.2f, mu_after=%.2f, "
            "n_anomalies=%d",
            n, shift_at, mu_before, mu_after, n_anomalies,
        )
        return TimeSeries(
            points=points,
            name="synthetic_step_shift",
            metadata={
                "generator": "SyntheticGenerator",
                "pattern": "step_shift",
                "shift_at": shift_at,
                "mu_before": mu_before,
                "mu_after": mu_after,
                "sigma": sigma,
                "n_anomalies": n_anomalies,
                "seed": self.seed,
            },
        )

    def variance_shift(
        self,
        n: int = 1000,
        shift_at: Optional[int] = None,
        mu: float = 0.0,
        sigma_before: float = 1.0,
        sigma_after: float = 4.0,
        n_anomalies: int = 10,
    ) -> TimeSeries:
        """Generate a variance step-shift series with isolated spike anomalies.

        The mean is constant at *mu* throughout; only the standard deviation
        changes at *shift_at*.  A change-point label is placed at *shift_at*
        and *n_anomalies* spikes are injected (``is_anomaly=True``).

        Parameters
        ----------
        n : int
            Total length of the series.
        shift_at : Optional[int]
            Index at which the standard deviation shifts. Defaults to n // 2.
        mu : float
            Constant mean throughout the series.
        sigma_before : float
            Standard deviation of the first segment.
        sigma_after : float
            Standard deviation of the second segment.
        n_anomalies : int
            Number of isolated spike anomalies to inject.

        Returns
        -------
        TimeSeries
        """
        if shift_at is None:
            shift_at = n // 2
        if not 0 < shift_at < n:
            raise ValueError(f"shift_at={shift_at} must be in (0, {n}).")

        rng = self._fresh_rng()
        values = np.empty(n, dtype=float)
        values[:shift_at] = rng.normal(loc=mu, scale=sigma_before, size=shift_at)
        values[shift_at:] = rng.normal(loc=mu, scale=sigma_after, size=n - shift_at)

        is_anomaly = np.zeros(n, dtype=bool)
        change_point = np.zeros(n, dtype=bool)
        change_point[shift_at] = True

        candidates = [i for i in range(n) if i != shift_at]
        spike_positions = rng.choice(candidates, size=min(n_anomalies, len(candidates)),
                                     replace=False)
        for pos in spike_positions:
            sign = rng.choice([-1.0, 1.0])
            spike_sigma = sigma_before if pos < shift_at else sigma_after
            values[pos] = mu + sign * 6.0 * spike_sigma
            is_anomaly[pos] = True

        timestamps = self._make_timestamps(n)
        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=bool(is_anomaly[i]),
                change_point=bool(change_point[i]),
            )
            for i in range(n)
        ]

        logger.debug(
            "variance_shift: n=%d, shift_at=%d, sigma_before=%.2f, sigma_after=%.2f",
            n, shift_at, sigma_before, sigma_after,
        )
        return TimeSeries(
            points=points,
            name="synthetic_variance_shift",
            metadata={
                "generator": "SyntheticGenerator",
                "pattern": "variance_shift",
                "shift_at": shift_at,
                "mu": mu,
                "sigma_before": sigma_before,
                "sigma_after": sigma_after,
                "n_anomalies": n_anomalies,
                "seed": self.seed,
            },
        )

    def spike_burst(
        self,
        n: int = 1000,
        burst_start: Optional[int] = None,
        burst_end: Optional[int] = None,
        spike_magnitude: float = 8.0,
        sigma: float = 1.0,
    ) -> TimeSeries:
        """Generate a series with a sustained spike-burst region.

        Every point inside ``[burst_start, burst_end)`` is treated as an
        anomaly (``is_anomaly=True``).  There is **no** concept-drift in this
        pattern — no change-point labels are set.  Outside the burst the
        series is stationary Gaussian noise.

        Parameters
        ----------
        n : int
            Total length of the series.
        burst_start : Optional[int]
            First index of the burst region (inclusive). Defaults to int(n * 0.4).
        burst_end : Optional[int]
            Last index of the burst region (exclusive). Defaults to int(n * 0.45).
        spike_magnitude : float
            Additive magnitude of burst spikes (alternating sign).
        sigma : float
            Gaussian noise standard deviation of the background process.

        Returns
        -------
        TimeSeries
        """
        if burst_start is None:
            burst_start = int(n * 0.4)
        if burst_end is None:
            burst_end = int(n * 0.45)
        if not 0 <= burst_start < burst_end <= n:
            raise ValueError(
                f"burst_start={burst_start} and burst_end={burst_end} "
                f"must satisfy 0 <= burst_start < burst_end <= {n}."
            )

        rng = self._fresh_rng()
        values = rng.normal(loc=0.0, scale=sigma, size=n)

        is_anomaly = np.zeros(n, dtype=bool)
        for i in range(burst_start, burst_end):
            sign = 1.0 if (i - burst_start) % 2 == 0 else -1.0
            values[i] = sign * spike_magnitude + rng.normal(0.0, sigma * 0.1)
            is_anomaly[i] = True

        timestamps = self._make_timestamps(n)
        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=bool(is_anomaly[i]),
                change_point=False,
            )
            for i in range(n)
        ]

        logger.debug(
            "spike_burst: n=%d, burst=[%d,%d), magnitude=%.2f",
            n, burst_start, burst_end, spike_magnitude,
        )
        return TimeSeries(
            points=points,
            name="synthetic_spike_burst",
            metadata={
                "generator": "SyntheticGenerator",
                "pattern": "spike_burst",
                "burst_start": burst_start,
                "burst_end": burst_end,
                "spike_magnitude": spike_magnitude,
                "sigma": sigma,
                "seed": self.seed,
            },
        )

    def isolated_spikes(
        self,
        n: int = 1000,
        spike_positions: Optional[List[int]] = None,
        spike_magnitude: float = 8.0,
        sigma: float = 1.0,
    ) -> TimeSeries:
        """Generate a stationary series with isolated point anomalies.

        Parameters
        ----------
        n : int
            Total length of the series.
        spike_positions : list of int, optional
            Explicit 0-based indices where spikes are placed.  When *None*,
            10 random positions are chosen deterministically from *self.seed*.
        spike_magnitude : float
            Additive magnitude of the spike (alternating sign per spike).
        sigma : float
            Gaussian noise standard deviation of the background process.

        Returns
        -------
        TimeSeries
        """
        rng = self._fresh_rng()
        values = rng.normal(loc=0.0, scale=sigma, size=n)

        if spike_positions is None:
            spike_positions = rng.choice(n, size=10, replace=False).tolist()

        is_anomaly = np.zeros(n, dtype=bool)
        for k, pos in enumerate(spike_positions):
            if not 0 <= pos < n:
                raise ValueError(
                    f"spike_positions[{k}]={pos} is out of range [0, {n})."
                )
            sign = 1.0 if k % 2 == 0 else -1.0
            values[pos] = sign * spike_magnitude
            is_anomaly[pos] = True

        timestamps = self._make_timestamps(n)
        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=bool(is_anomaly[i]),
                change_point=False,
            )
            for i in range(n)
        ]

        logger.debug(
            "isolated_spikes: n=%d, %d spikes, magnitude=%.2f",
            n, len(spike_positions), spike_magnitude,
        )
        return TimeSeries(
            points=points,
            name="synthetic_isolated_spikes",
            metadata={
                "generator": "SyntheticGenerator",
                "pattern": "isolated_spikes",
                "spike_positions": list(spike_positions),
                "spike_magnitude": spike_magnitude,
                "sigma": sigma,
                "seed": self.seed,
            },
        )

    def compound(self, n: int = 2000) -> TimeSeries:
        """Generate a complex multi-regime series combining all patterns.

        The series is divided into four equal segments of length ``n // 4``::

            Segment 0: stationary  (mu=0,  sigma=1)
            Segment 1: step shift  (mu changes from 0 → 5)
            Segment 2: variance shift (sigma changes from 1 → 3)
            Segment 3: spike burst  (burst spans full segment)

        Change-point labels are placed at the boundaries between segments 0→1
        and 1→2.  Isolated spikes (3 per segment) are injected in segments
        0 and 2.  The burst in segment 3 marks every point as anomalous.

        Parameters
        ----------
        n : int
            Total length.  Will be rounded down to the nearest multiple of 4.

        Returns
        -------
        TimeSeries
        """
        seg = (n // 4)
        n = seg * 4  # ensure exact divisibility

        rng = self._fresh_rng()

        values = np.empty(n, dtype=float)
        is_anomaly = np.zeros(n, dtype=bool)
        change_point = np.zeros(n, dtype=bool)

        # Segment 0: stationary mu=0, sigma=1
        values[0:seg] = rng.normal(0.0, 1.0, size=seg)

        # Segment 1: mean shift to mu=5, sigma=1; change-point at boundary
        values[seg:2 * seg] = rng.normal(5.0, 1.0, size=seg)
        change_point[seg] = True

        # Segment 2: variance shift sigma=3 (mu still 5); change-point at boundary
        values[2 * seg:3 * seg] = rng.normal(5.0, 3.0, size=seg)
        change_point[2 * seg] = True

        # Segment 3: spike burst (every point anomalous)
        values[3 * seg:4 * seg] = rng.normal(0.0, 1.0, size=seg)
        burst_magnitude = 8.0
        for i in range(3 * seg, 4 * seg):
            sign = 1.0 if (i - 3 * seg) % 2 == 0 else -1.0
            values[i] = sign * burst_magnitude + rng.normal(0.0, 0.1)
            is_anomaly[i] = True

        # Inject isolated spikes in segment 0 (3 spikes)
        s0_candidates = list(range(0, seg))
        spike_pos_s0 = rng.choice(s0_candidates, size=3, replace=False)
        for k, pos in enumerate(spike_pos_s0):
            sign = 1.0 if k % 2 == 0 else -1.0
            values[pos] = sign * 8.0
            is_anomaly[pos] = True

        # Inject isolated spikes in segment 2 (3 spikes)
        s2_candidates = list(range(2 * seg, 3 * seg))
        spike_pos_s2 = rng.choice(s2_candidates, size=3, replace=False)
        for k, pos in enumerate(spike_pos_s2):
            sign = 1.0 if k % 2 == 0 else -1.0
            values[pos] = 5.0 + sign * 18.0  # 6 * sigma_after=3
            is_anomaly[pos] = True

        timestamps = self._make_timestamps(n)
        points = [
            TimeSeriesPoint(
                timestamp=float(timestamps[i]),
                value=float(values[i]),
                is_anomaly=bool(is_anomaly[i]),
                change_point=bool(change_point[i]),
            )
            for i in range(n)
        ]

        logger.debug(
            "compound: n=%d (4×%d), change_points=%s, anomalies=%d",
            n,
            seg,
            [seg, 2 * seg],
            int(is_anomaly.sum()),
        )
        return TimeSeries(
            points=points,
            name="synthetic_compound",
            metadata={
                "generator": "SyntheticGenerator",
                "pattern": "compound",
                "segment_length": seg,
                "change_points": [seg, 2 * seg],
                "seed": self.seed,
            },
        )
