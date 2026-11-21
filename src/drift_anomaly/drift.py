"""
drift_anomaly.drift
===================

Online drift-detection algorithms.

Both :class:`ADWINDetector` and :class:`PageHinkleyDetector` implement the
:class:`DriftDetector` abstract interface and can be swapped in any detector
variant that calls ``drift_detector.update(value)``.

No third-party drift-detection libraries (river, scikit-multiflow, …) are
used — everything is implemented from scratch with NumPy.
"""
from __future__ import annotations

import logging
import math
from abc import ABC, abstractmethod
from collections import deque
from typing import Deque, List, Tuple

import numpy as np

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Abstract base
# ---------------------------------------------------------------------------

class DriftDetector(ABC):
    """Abstract interface for online drift detectors.

    Sub-classes must implement :meth:`update`, :meth:`reset`, and the
    :attr:`name` property.
    """

    @abstractmethod
    def update(self, value: float) -> bool:
        """Ingest one new observation and return whether drift is detected.

        Parameters
        ----------
        value : float
            The latest observation (raw measurement or anomaly score).

        Returns
        -------
        bool
            ``True`` if drift is detected at this step.
        """

    @abstractmethod
    def reset(self) -> None:
        """Reset internal state as if no data had been seen yet."""

    @property
    @abstractmethod
    def name(self) -> str:
        """Human-readable identifier for this detector."""


# ---------------------------------------------------------------------------
# ADWIN — Adaptive Windowing
# ---------------------------------------------------------------------------

class ADWINDetector(DriftDetector):
    """ADWIN (Adaptive Windowing) drift detector.

    Maintains an adaptive window whose length shrinks whenever a
    statistically significant difference is found between two consecutive
    sub-windows.  The implementation follows the original paper:

        Bifet & Gavaldà, "Learning from Time-Changing Data with Adaptive
        Windowing", SDM 2007.

    Data are stored in an *exponential histogram* — a list of ``(total,
    count)`` *buckets* where the ``k``-th bucket can hold at most
    ``2^k`` values.  When a bucket overflows it is merged with its
    neighbour, keeping memory usage logarithmic in the window size.

    Parameters
    ----------
    delta : float
        False-positive probability (confidence parameter).  Smaller values
        → fewer false positives but slower detection.  Typical: 0.002.
    max_buckets : int
        Maximum number of buckets at each row of the exponential histogram.
        Controls the compression rate.  Default 5 (same as the reference
        implementation).
    min_window : int
        Minimum total window size before drift detection is attempted.

    Notes
    -----
    Drift condition for a split at position ``n1 | n2`` (counts in each
    sub-window, with ``W = n1 + n2``):

    .. math::
        |\\hat{\\mu}_0 - \\hat{\\mu}_1| > \\varepsilon_{\\text{cut}}

    where

    .. math::
        \\varepsilon_{\\text{cut}} =
            \\sqrt{\\frac{1}{2m_0} + \\frac{1}{2m_1}}
            \\cdot \\sqrt{\\ln\\!\\left(\\frac{4 W^2}{\\delta}\\right)}

    with ``m0 = n1``, ``m1 = n2``.
    """

    def __init__(
        self,
        delta: float = 0.002,
        max_buckets: int = 5,
        min_window: int = 20,
    ) -> None:
        self._delta = delta
        self._max_buckets = max_buckets
        self._min_window = min_window
        # Each element is a deque of (bucket_sum, bucket_count)
        # _buckets[k] holds buckets of size 2^k
        self._buckets: List[Deque[Tuple[float, int]]] = []
        self._total: float = 0.0   # sum of all values in window
        self._width: int = 0       # total number of values in window
        self._drift_detected: bool = False

    # ------------------------------------------------------------------
    # DriftDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D102
        return "adwin"

    def reset(self) -> None:
        """Clear all buckets and counters."""
        self._buckets = []
        self._total = 0.0
        self._width = 0
        self._drift_detected = False
        logger.debug("ADWINDetector reset")

    def update(self, value: float) -> bool:
        """Add *value* and detect drift.

        Returns
        -------
        bool
            ``True`` if drift is detected (and the window was trimmed).
        """
        self._drift_detected = False
        self._insert(value)
        self._compress_buckets()
        if self._width >= self._min_window:
            self._drift_detected = self._detect_change()
        return self._drift_detected

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _insert(self, value: float) -> None:
        """Add a new size-1 bucket for *value* at level 0."""
        if not self._buckets:
            self._buckets.append(deque())
        self._buckets[0].appendleft((value, 1))
        self._total += value
        self._width += 1

    def _compress_buckets(self) -> None:
        """Merge overflow buckets upward through the exponential histogram.

        A level-``k`` bucket can hold at most ``max_buckets`` entries.
        When a level has ``max_buckets + 1`` entries the two *oldest*
        (rightmost) are merged into one level-``(k+1)`` entry.
        """
        for k in range(len(self._buckets)):
            if len(self._buckets[k]) <= self._max_buckets:
                break  # no overflow at this level → done
            # Merge the two oldest buckets at level k
            if k + 1 >= len(self._buckets):
                self._buckets.append(deque())
            b1 = self._buckets[k].pop()   # oldest
            b2 = self._buckets[k].pop()   # second oldest
            merged = (b1[0] + b2[0], b1[1] + b2[1])
            self._buckets[k + 1].appendleft(merged)

    def _detect_change(self) -> bool:
        """Scan all possible splits and return True if any shows drift."""
        # Build a flat ordered list of (bucket_sum, bucket_count)
        # from newest (index 0) to oldest.
        all_buckets: List[Tuple[float, int]] = []
        for level_buckets in self._buckets:
            all_buckets.extend(level_buckets)  # newest first within each level

        W = self._width
        total_sum = self._total

        # Sliding totals for the *right* (older) sub-window
        right_sum: float = 0.0
        right_count: int = 0
        drift_found = False

        for bucket_sum, bucket_count in all_buckets:
            right_sum += bucket_sum
            right_count += bucket_count
            left_count = W - right_count
            left_sum = total_sum - right_sum

            if left_count <= 0:
                continue

            mu_right = right_sum / right_count
            mu_left = left_sum / left_count

            # Hoeffding bound  ε_cut
            # In ADWIN, data in [0, 1] uses eps_cut = sqrt(1/(2m) * ln(4/delta))
            # For general or unbounded data, scale by variance or bound.
            # Here m_harmonic = 1/(2*m0) + 1/(2*m1) where m0=right_count, m1=left_count.
            # Notice 1/(2*m) where m = (1/m0 + 1/m1)^(-1).
            # If values have sample variance, Hoeffding bound needs range or standard deviation.
            # In standard ADWIN (Bifet & Gavalda 2007), m = 1 / (1/n0 + 1/n1)
            # eps_cut = sqrt( (1 / (2*m)) * ln(4*W / delta) )
            # Also note standard ADWIN has m = (1/n0 + 1/n1)^(-1).
            # In the current code: m_harmonic = 1/(2*n0) + 1/(2*n1) = 1/(2*m).
            # But notice: ln(4 * W^2 / delta) is used, and without range/variance scaling,
            # for standard normal data N(0,1), standard deviations can cause false positives if
            # min_subwindow is too small (e.g. subwindow of size 1 or 2).
            # A split should require both subwindows to have at least min_subwindow elements!
            min_sub = max(5, self._min_window // 4)
            if left_count < min_sub or right_count < min_sub:
                continue

            m_harmonic = 1.0 / (2.0 * right_count) + 1.0 / (2.0 * left_count)
            eps_cut = math.sqrt(
                m_harmonic * math.log(4.0 * W * W / self._delta)
            )

            if abs(mu_left - mu_right) > eps_cut:
                # Drift detected — drop the older sub-window
                logger.debug(
                    "ADWIN drift: |%.4f - %.4f| = %.4f > eps=%.4f "
                    "(W=%d, left=%d, right=%d)",
                    mu_left, mu_right, abs(mu_left - mu_right), eps_cut,
                    W, left_count, right_count,
                )
                self._drop_old_buckets(right_count)
                drift_found = True
                break  # restart scan after trim

        return drift_found

    def _drop_old_buckets(self, n_to_drop: int) -> None:
        """Remove the *n_to_drop* oldest values from the bucket list."""
        dropped_sum = 0.0
        dropped_count = 0
        for k in range(len(self._buckets) - 1, -1, -1):
            level = self._buckets[k]
            while level and dropped_count < n_to_drop:
                b_sum, b_count = level.pop()   # oldest at right
                dropped_sum += b_sum
                dropped_count += b_count
        self._total -= dropped_sum
        self._width -= dropped_count
        # Clean up empty levels from the end
        while self._buckets and not self._buckets[-1]:
            self._buckets.pop()

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def width(self) -> int:
        """Current adaptive window size (number of stored values)."""
        return self._width

    @property
    def mean(self) -> float:
        """Current window mean (``0.0`` if window is empty)."""
        if self._width == 0:
            return 0.0
        return self._total / self._width


# ---------------------------------------------------------------------------
# Page-Hinkley test
# ---------------------------------------------------------------------------

class PageHinkleyDetector(DriftDetector):
    """Two-sided Page-Hinkley test for online drift detection.

    The Page-Hinkley (PH) test monitors a cumulative sum statistic and
    fires when the statistic deviates too far from its running minimum
    (upward shift) or maximum (downward shift when ``two_sided=True``).

    Update equations (upward shift):

    .. math::
        \\bar{z}_t &= \\frac{1}{t} \\sum_{i=1}^{t} z_i \\\\
        U_t        &= \\sum_{i=1}^{t} (z_i - \\bar{z}_t - \\delta) \\\\
        \\text{drift} &\\Leftrightarrow U_t - \\min_{i \\le t} U_i > \\lambda

    For a downward shift the signs are flipped.

    Parameters
    ----------
    lambda_ : float
        Detection threshold ``λ``.  Larger values → fewer false positives
        but slower detection.
    delta : float
        Minimum expected shift magnitude ``δ``.  Prevents triggering on
        very small, stable drifts.
    two_sided : bool
        If ``True`` (default), detect both upward *and* downward shifts.
    min_instances : int
        Minimum number of observations before the test fires.
    """

    def __init__(
        self,
        lambda_: float = 50.0,
        delta: float = 0.005,
        two_sided: bool = True,
        min_instances: int = 30,
    ) -> None:
        self._lambda = lambda_
        self._delta = delta
        self._two_sided = two_sided
        self._min_instances = min_instances
        self._reset_state()

    # ------------------------------------------------------------------
    # DriftDetector interface
    # ------------------------------------------------------------------

    @property
    def name(self) -> str:  # noqa: D102
        return "page-hinkley"

    def reset(self) -> None:
        """Reset all running statistics."""
        self._reset_state()
        logger.debug("PageHinkleyDetector reset")

    def update(self, value: float) -> bool:
        """Ingest *value* and return whether drift is detected.

        Returns
        -------
        bool
            ``True`` if the PH statistic exceeds the threshold.
        """
        self._n += 1
        self._sum += value
        mean = self._sum / self._n

        # Upward PH statistic
        self._U += value - mean - self._delta
        self._U_min = min(self._U_min, self._U)

        # Downward PH statistic:
        # Detects when value is consistently below mean - delta.
        # Deviations: (mean - value - delta).
        # We accumulate T_t = sum( (mean - value) - delta ).
        # Drift when T_t - min(T) > lambda.
        if self._two_sided:
            self._L += (mean - value) - self._delta
            self._L_min = min(self._L_min, self._L)

        if self._n < self._min_instances:
            return False

        ph_up = self._U - self._U_min
        if self._two_sided:
            ph_down = self._L - self._L_min
        else:
            ph_down = 0.0

        drift = bool(ph_up > self._lambda or ph_down > self._lambda)
        if drift:
            logger.debug(
                "Page-Hinkley drift: ph_up=%.3f, ph_down=%.3f, λ=%.1f (n=%d)",
                ph_up, ph_down, self._lambda, self._n,
            )
            # Reset cumulative sums on drift trigger to avoid infinite re-triggering on stationary post-drift
            self._U = 0.0
            self._U_min = 0.0
            if self._two_sided:
                self._L = 0.0
                self._L_min = 0.0
        return drift

    # ------------------------------------------------------------------
    # Private
    # ------------------------------------------------------------------

    def _reset_state(self) -> None:
        self._n: int = 0
        self._sum: float = 0.0
        self._U: float = 0.0
        self._U_min: float = 0.0
        self._L: float = 0.0
        self._L_min: float = 0.0

    # ------------------------------------------------------------------
    # Convenience properties
    # ------------------------------------------------------------------

    @property
    def ph_up(self) -> float:
        """Current upward Page-Hinkley statistic value."""
        return self._U - self._U_min

    @property
    def ph_down(self) -> float:
        """Current downward Page-Hinkley statistic value (two-sided only)."""
        if not self._two_sided:
            return 0.0
        return self._L_max - self._L


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------

def make_drift_detector(
    name: str,
    adwin_delta: float = 0.002,
    ph_lambda: float = 50.0,
    ph_delta: float = 0.005,
) -> DriftDetector:
    """Instantiate a drift detector by name.

    Parameters
    ----------
    name : str
        ``"adwin"`` or ``"page-hinkley"``.
    adwin_delta : float
        Passed to :class:`ADWINDetector`.
    ph_lambda : float
        Passed to :class:`PageHinkleyDetector`.
    ph_delta : float
        Passed to :class:`PageHinkleyDetector`.

    Returns
    -------
    DriftDetector
        A freshly created drift detector.

    Raises
    ------
    ValueError
        If *name* is not recognised.
    """
    if name == "adwin":
        return ADWINDetector(delta=adwin_delta)
    elif name in ("page-hinkley", "ph"):
        return PageHinkleyDetector(lambda_=ph_lambda, delta=ph_delta)
    else:
        raise ValueError(
            f"Unknown drift detector '{name}'. Choose 'adwin' or 'page-hinkley'."
        )
