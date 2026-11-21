"""
tests/test_statistics.py
========================

Unit tests for drift_anomaly.statistics.
All tests are self-contained: no network access, no external data files.
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from drift_anomaly.statistics import (
    GAUSSIAN_CONSISTENCY_FACTOR,
    rolling_median,
    mad,
    sigma_hat,
    anomaly_score,
    compute_window_stats,
)


# ---------------------------------------------------------------------------
# test_mad_basic
# ---------------------------------------------------------------------------

class TestMadBasic:
    """The MAD of [1, 2, 3, 4, 5] around its median (3) is 1.0."""

    def test_mad_basic(self):
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        center = rolling_median(window)       # 3.0
        result = mad(window, center)
        assert math.isclose(result, 1.0, rel_tol=1e-9), (
            f"Expected MAD=1.0, got {result}"
        )

    def test_mad_constant_window(self):
        """MAD of a constant window is 0."""
        window = np.full(10, 7.0)
        center = rolling_median(window)
        assert mad(window, center) == 0.0

    def test_mad_two_values(self):
        """MAD of [0, 2] around 1 is 1."""
        window = np.array([0.0, 2.0])
        result = mad(window, 1.0)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_mad_negative_values(self):
        """MAD works with negative values."""
        window = np.array([-3.0, -1.0, 0.0, 1.0, 3.0])
        center = rolling_median(window)       # 0.0
        result = mad(window, center)
        assert math.isclose(result, 1.0, rel_tol=1e-9)

    def test_rolling_median_odd(self):
        window = np.array([5.0, 1.0, 3.0])
        assert rolling_median(window) == 3.0

    def test_rolling_median_even(self):
        window = np.array([1.0, 2.0, 3.0, 4.0])
        assert math.isclose(rolling_median(window), 2.5, rel_tol=1e-9)

    def test_rolling_median_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            rolling_median(np.array([]))

    def test_mad_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            mad(np.array([]), 0.0)


# ---------------------------------------------------------------------------
# test_sigma_hat_gaussian_factor
# ---------------------------------------------------------------------------

class TestSigmaHatGaussianFactor:
    """GAUSSIAN_CONSISTENCY_FACTOR must be 1.4826."""

    def test_gaussian_factor_value(self):
        assert math.isclose(GAUSSIAN_CONSISTENCY_FACTOR, 1.4826, rel_tol=1e-4), (
            f"Expected 1.4826, got {GAUSSIAN_CONSISTENCY_FACTOR}"
        )

    def test_sigma_hat_formula(self):
        """sigma_hat(mad_val) == 1.4826 * mad_val + epsilon."""
        mad_val = 2.0
        eps = 1e-6
        expected = GAUSSIAN_CONSISTENCY_FACTOR * mad_val + eps
        assert math.isclose(sigma_hat(mad_val, eps), expected, rel_tol=1e-9)

    def test_sigma_hat_default_epsilon(self):
        """Default epsilon is 1e-6."""
        result = sigma_hat(1.0)
        assert math.isclose(result, GAUSSIAN_CONSISTENCY_FACTOR * 1.0 + 1e-6, rel_tol=1e-9)

    def test_sigma_hat_always_positive(self):
        """sigma_hat is always strictly positive."""
        for mad_val in [0.0, 0.001, 1.0, 100.0]:
            assert sigma_hat(mad_val) > 0.0


# ---------------------------------------------------------------------------
# test_anomaly_score_zero_at_median
# ---------------------------------------------------------------------------

class TestAnomalyScoreZeroAtMedian:
    """The anomaly score of a point equal to the median should be 0."""

    def test_score_at_median_is_zero(self):
        window = np.arange(1.0, 11.0)      # 1..10
        m = rolling_median(window)          # 5.5
        mad_val = mad(window, m)
        sig = sigma_hat(mad_val)
        score = anomaly_score(m, m, sig)
        assert math.isclose(score, 0.0, abs_tol=1e-12), (
            f"Expected 0, got {score}"
        )

    def test_score_above_median_positive(self):
        window = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 10.0])
        m = rolling_median(window)
        mad_val = mad(window, m)
        sig = sigma_hat(mad_val)
        # x = 10 is far from median 0 -> score >> 0
        score = anomaly_score(10.0, m, sig)
        assert score > 0.0

    def test_score_symmetry(self):
        """Score is symmetric: anomaly_score(m+d) == anomaly_score(m-d)."""
        m = 5.0
        sig = 2.0
        d = 3.0
        assert math.isclose(
            anomaly_score(m + d, m, sig),
            anomaly_score(m - d, m, sig),
            rel_tol=1e-9,
        )


# ---------------------------------------------------------------------------
# test_sigma_hat_epsilon_guard
# ---------------------------------------------------------------------------

class TestSigmaHatEpsilonGuard:
    """When MAD=0 (constant window), sigma_hat should equal epsilon."""

    def test_mad_zero_gives_epsilon(self):
        eps = 1e-6
        result = sigma_hat(0.0, eps)
        assert math.isclose(result, eps, rel_tol=1e-9), (
            f"Expected sigma_hat=epsilon={eps}, got {result}"
        )

    def test_no_division_by_zero(self):
        """anomaly_score should not raise ZeroDivisionError for constant window."""
        window = np.full(50, 3.0)
        m = rolling_median(window)
        mad_val = mad(window, m)
        sig = sigma_hat(mad_val, epsilon=1e-6)
        # Should not raise
        score = anomaly_score(3.0, m, sig)
        assert math.isfinite(score)

    def test_custom_epsilon(self):
        eps = 0.01
        assert math.isclose(sigma_hat(0.0, eps), eps, rel_tol=1e-9)


# ---------------------------------------------------------------------------
# test_compute_window_stats_shape
# ---------------------------------------------------------------------------

class TestComputeWindowStatsShape:
    """compute_window_stats returns a 3-tuple (median, mad, sigma)."""

    def test_returns_three_values(self):
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        result = compute_window_stats(window)
        assert len(result) == 3

    def test_all_finite(self):
        rng = np.random.default_rng(42)
        window = rng.normal(0, 1, size=100)
        m, mv, sig = compute_window_stats(window)
        assert math.isfinite(m)
        assert math.isfinite(mv)
        assert math.isfinite(sig)

    def test_sigma_gt_mad(self):
        """sigma_hat > mad_val because of the consistency factor and epsilon."""
        window = np.array([1.0, 2.0, 3.0, 4.0, 5.0])
        _, mv, sig = compute_window_stats(window)
        assert sig > mv

    def test_median_in_range(self):
        window = np.linspace(0, 10, 11)   # 0, 1, …, 10
        m, _, _ = compute_window_stats(window)
        assert 0 <= m <= 10

    def test_raises_on_empty(self):
        with pytest.raises(ValueError, match="non-empty"):
            compute_window_stats(np.array([]))

    def test_single_element(self):
        """Single-element window: median == value, mad == 0, sigma == epsilon."""
        m, mv, sig = compute_window_stats(np.array([7.0]))
        assert math.isclose(m, 7.0, rel_tol=1e-9)
        assert math.isclose(mv, 0.0, abs_tol=1e-15)
        assert math.isclose(sig, 1e-6, rel_tol=1e-6)
