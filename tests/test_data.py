"""
tests/test_data.py
==================

Unit tests for data generation, schema normalization, and dataset loaders.
"""
from __future__ import annotations

from pathlib import Path
import numpy as np
import pytest

from drift_anomaly.data.schema import normalize_to_schema
from drift_anomaly.data.synthetic import SyntheticGenerator
from drift_anomaly.data.nab import NABLoader
from drift_anomaly.data.yahoo_s5 import YahooS5Loader
from drift_anomaly.types import TimeSeries


def test_synthetic_deterministic():
    """Same seed must produce identical time series."""
    gen1 = SyntheticGenerator(seed=123)
    gen2 = SyntheticGenerator(seed=123)
    s1 = gen1.step_shift(n=200, shift_at=100)
    s2 = gen2.step_shift(n=200, shift_at=100)
    np.testing.assert_array_equal(s1.values, s2.values)
    np.testing.assert_array_equal(s1.is_anomaly, s2.is_anomaly)
    np.testing.assert_array_equal(s1.change_point, s2.change_point)


def test_synthetic_step_shift_labels():
    """Verify change-point index and anomaly properties."""
    gen = SyntheticGenerator(seed=42)
    s = gen.step_shift(n=300, shift_at=150, n_anomalies=5)
    assert len(s) == 300
    assert s.change_point[150] == True
    assert np.sum(s.change_point) == 1
    assert np.sum(s.is_anomaly) == 5
    assert not s.is_anomaly[150]  # Isolated spike should not clash with changepoint


def test_synthetic_spike_burst_labels():
    """Burst region should have is_anomaly=True and change_point=False."""
    gen = SyntheticGenerator(seed=42)
    s = gen.spike_burst(n=300, burst_start=100, burst_end=130)
    assert np.sum(s.is_anomaly) == 30
    assert np.all(s.is_anomaly[100:130])
    assert not np.any(s.change_point)


def test_schema_normalization():
    """Ensure normalize_to_schema properly populates TimeSeriesPoint."""
    timestamps = [0.0, 1.0, 2.0, 3.0]
    values = [10.0, 12.0, 100.0, 11.0]
    labels = [False, False, True, False]
    cp = [False, False, False, True]
    ts = normalize_to_schema(timestamps, values, labels, cp, name="test_schema")
    assert isinstance(ts, TimeSeries)
    assert len(ts) == 4
    assert ts.is_anomaly[2] == True
    assert ts.change_point[3] == True
    assert ts.values[2] == 100.0


def test_yahoo_loader_unavailable(tmp_path: Path):
    """Yahoo loader should handle missing data cleanly and print instructions."""
    loader = YahooS5Loader(data_root=tmp_path / "empty_yahoo")
    assert loader.is_available() is False
    loader.print_setup_instructions()


def test_yahoo_huggingface_header_aliases_and_aggregate_skip(tmp_path: Path):
    """Parse HF A3/A4 header names and ignore their combined wide CSVs."""
    loader = YahooS5Loader(data_root=tmp_path)
    a3 = tmp_path / "A3Benchmark" / "A3Benchmark-TS1.csv"
    a3.parent.mkdir(parents=True)
    a3.write_text(
        "timestamps,value,anomaly,changepoint,trend\n"
        "100,1.0,0,0,0\n101,9.0,1,1,0\n",
        encoding="utf-8",
    )
    (a3.parent / "A3Benchmark_all.csv").write_text(
        "timestamps,A3Benchmark-TS1\n100,1\n", encoding="utf-8"
    )
    series = loader.load_benchmark("A3")
    assert len(series) == 1
    assert series[0].timestamps == [100.0, 101.0]
    assert series[0].labels == [False, True]
    assert series[0].change_points == [False, True]


def test_yahoo_huggingface_a1_timestamp_header(tmp_path: Path):
    loader = YahooS5Loader(data_root=tmp_path)
    path = tmp_path / "A1Benchmark" / "real_1.csv"
    path.parent.mkdir(parents=True)
    path.write_text(
        "timestamp,value,is_anomaly\n1,2.0,0\n2,8.0,1\n", encoding="utf-8"
    )
    series = loader.load_benchmark("A1")
    assert len(series) == 1
    assert series[0].timestamps == [1.0, 2.0]
    assert series[0].labels == [False, True]
