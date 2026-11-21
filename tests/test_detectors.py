"""
tests/test_detectors.py
=======================

Integration and unit tests for detector variants:
- Version A (cautious rolling baseline with contamination gate)
- Version B (simplified rolling window + ADWIN baseline ablation)
- Version C (frozen-window with immediate and delayed adaptation)
"""
from __future__ import annotations

import math
import numpy as np
import pytest

from drift_anomaly.data.synthetic import SyntheticGenerator
from drift_anomaly.detectors.version_a_cautious_rolling import VersionA
from drift_anomaly.detectors.version_b_rolling_adwin import VersionB
from drift_anomaly.detectors.version_c_frozen_delayed import VersionC
from drift_anomaly.types import DetectorConfig, TimeSeriesPoint, TimeSeries


@pytest.fixture
def gen() -> SyntheticGenerator:
    return SyntheticGenerator(seed=42)


@pytest.fixture
def default_config() -> DetectorConfig:
    return DetectorConfig(
        window_size=50,
        candidate_size=20,
        tau=3.0,
        epsilon=1e-6,
        drift_detector="adwin",
        adwin_delta=0.01,
        ph_lambda=30.0,
        ph_delta=0.01,
        tau_cand=3.0,
        rho_max=0.3,
        kappa_max=3.0,
        freeze_window=50,
        adaptation_mode="delayed",
        hold_steps=20,
        cooldown=10,
    )


def test_version_a_causality(gen: SyntheticGenerator, default_config: DetectorConfig):
    """Verify that scoring at step t does not depend on future data."""
    ts = gen.stationary(n=120)
    det = VersionA(default_config)
    res = det.run(ts)
    assert len(res.scores) == len(ts)
    assert len(res.alerts) == len(ts)
    for i, s in enumerate(res.scores):
        if not math.isnan(s):
            assert s >= 0.0


def test_version_a_burn_in(default_config: DetectorConfig):
    """First observations should score without error."""
    det = VersionA(default_config)
    L = default_config.window_size
    pts = [TimeSeriesPoint(timestamp=float(i), value=1.0) for i in range(L + 10)]
    ts = TimeSeries(points=pts, name="burn_in")
    res = det.run(ts)
    assert len(res.scores) == len(pts)


def test_version_a_spike_burst_rejection(gen: SyntheticGenerator, default_config: DetectorConfig):
    """A contaminated candidate window (e.g. spike burst) must be handled by the gate."""
    ts = gen.spike_burst(n=400, burst_start=150, burst_end=190, spike_magnitude=10.0)
    det = VersionA(default_config)
    res = det.run(ts)
    assert len(res.scores) == 400


def test_version_a_step_shift_adaptation(gen: SyntheticGenerator, default_config: DetectorConfig):
    """A persistent step shift should adapt the baseline."""
    ts = gen.step_shift(n=500, shift_at=200, mu_before=0.0, mu_after=8.0, n_anomalies=0)
    det = VersionA(default_config)
    res = det.run(ts)
    assert len(res.adaptation_events) >= 1, "Persistent step shift should adapt baseline"


def test_version_b_basic_run(gen: SyntheticGenerator, default_config: DetectorConfig):
    """Version B runs smoothly on stationary and shifted data."""
    ts = gen.step_shift(n=300, shift_at=150, mu_before=0.0, mu_after=6.0, n_anomalies=2)
    det = VersionB(default_config)
    res = det.run(ts)
    assert len(res.scores) == len(ts)
    assert res.variant == "B"
    assert len(res.drift_events) >= 1


def test_version_c_immediate_mode(gen: SyntheticGenerator):
    """Version C immediate mode updates baseline upon drift trigger."""
    cfg = DetectorConfig(
        window_size=50,
        freeze_window=50,
        adaptation_mode="immediate",
        drift_detector="adwin",
        adwin_delta=0.01,
        cooldown=10,
    )
    det = VersionC(cfg)
    ts = gen.step_shift(n=400, shift_at=150, mu_before=0.0, mu_after=8.0, n_anomalies=0)
    res = det.run(ts)
    assert res.variant == "C"
    assert len(res.adaptation_events) >= 1


def test_version_c_delayed_mode_transient(gen: SyntheticGenerator):
    """Version C delayed mode holds before updating."""
    cfg = DetectorConfig(
        window_size=50,
        freeze_window=50,
        adaptation_mode="delayed",
        hold_steps=30,
        drift_detector="adwin",
        adwin_delta=0.01,
        cooldown=10,
    )
    det = VersionC(cfg)
    ts = gen.spike_burst(n=400, burst_start=150, burst_end=170, spike_magnitude=8.0)
    res = det.run(ts)
    assert res.variant == "C"
