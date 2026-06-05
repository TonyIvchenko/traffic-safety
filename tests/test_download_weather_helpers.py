from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import download_weather as dw

YEARS = [2018, 2019, 2020, 2021, 2022, 2023]


def test_coverage_fraction_full_partial_and_none():
    assert dw.coverage_fraction(20100101, 20251231, YEARS) == pytest.approx(1.0)
    assert dw.coverage_fraction(20000101, 20051231, YEARS) == 0.0
    partial = dw.coverage_fraction(20200101, 20251231, YEARS)
    assert 0.0 < partial < 1.0


def test_haversine_km_vectorized():
    zero = dw.haversine_km(0.0, 0.0, np.array([0.0]), np.array([0.0]))
    assert zero[0] == pytest.approx(0.0, abs=1e-9)

    one_degree = dw.haversine_km(0.0, 0.0, np.array([1.0]), np.array([0.0]))
    expected = np.radians(1.0) * 6371.0088
    assert one_degree[0] == pytest.approx(expected, rel=1e-6)


def test_aggregate_climatology_averages_and_counts():
    hourly = pd.DataFrame(
        {
            "station_index": [0, 0, 1],
            "month": [1, 1, 1],
            "hour_of_week": [5, 5, 5],
            "temp_c": [10.0, 20.0, 4.0],
            "dewpoint_c": [5.0, 7.0, 1.0],
            "relative_humidity_pct": [60.0, 80.0, 50.0],
            "wind_speed_mps": [2.0, 4.0, 6.0],
            "wet_hour": [0.0, 1.0, 1.0],
        }
    )

    climatology = dw.aggregate_climatology(hourly)

    assert len(climatology) == 2  # (station 0, m1, h5) and (station 1, m1, h5)
    station0 = climatology[climatology["station_index"] == 0].iloc[0]
    assert station0["temp_c"] == pytest.approx(15.0)
    assert station0["wet_hour"] == pytest.approx(0.5)
    assert int(station0["obs_count"]) == 2
    assert climatology["station_index"].dtype == np.int16
    assert climatology["month"].dtype == np.int8
    assert climatology["hour_of_week"].dtype == np.int16
    assert climatology["obs_count"].dtype == np.int32
