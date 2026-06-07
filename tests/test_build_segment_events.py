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

import build_segment_events as bse


def test_fahrenheit_to_celsius_handles_none_and_nan():
    assert bse.fahrenheit_to_celsius(None) is None
    assert bse.fahrenheit_to_celsius(np.nan) is None
    assert bse.fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
    assert bse.fahrenheit_to_celsius(212.0) == pytest.approx(100.0)


def test_mph_to_mps_handles_none_and_nan():
    assert bse.mph_to_mps(None) is None
    assert bse.mph_to_mps(np.nan) is None
    assert bse.mph_to_mps(10.0) == pytest.approx(10 * 0.44704)


def test_wet_hour_from_row_uses_precip_then_condition():
    assert bse.wet_hour_from_row(0.2, "Clear") == 1.0  # measurable precip
    assert bse.wet_hour_from_row(0.0, "Light Rain") == 1.0  # condition keyword
    assert bse.wet_hour_from_row(np.nan, "Snow") == 1.0
    assert bse.wet_hour_from_row(0.0, "Fair") == 0.0
    assert bse.wet_hour_from_row(None, None) == 0.0


def test_state_abbreviation_map_is_consistent():
    assert bse.STATE_ABBR_TO_FIPS["CA"] == "06"
    assert bse.STATE_ABBR_TO_FIPS["NY"] == "36"
    assert bse.STATE_ABBR_TO_FIPS["DC"] == "11"
    assert len(bse.STATE_ABBR_TO_FIPS) == 51  # 50 states + DC


def test_normalize_chunk_filters_and_maps_state():
    chunk = pd.DataFrame(
        {
            "Start_Lat": [34.05, 5.0, 40.7],
            "Start_Lng": [-118.25, -118.0, -74.0],
            "Severity": [2, 3, 2],
            "Temperature(F)": [70.0, 70.0, 50.0],
            "Humidity(%)": [50.0, 50.0, 60.0],
            "Wind_Speed(mph)": [5.0, 5.0, 8.0],
            "Precipitation(in)": [0.0, 0.0, 0.1],
            "Start_Time": ["2022-01-01 10:00:00", "2022-01-01 11:00:00", "2022-01-01 12:00:00"],
            "State": ["CA", "CA", "ZZ"],
        }
    )

    result = bse.normalize_chunk(chunk)

    # Row 2 is out of latitude bounds; row 3 has an unknown state abbreviation.
    assert len(result) == 1
    assert result.iloc[0]["state_fips"] == "06"
    assert result.iloc[0]["Start_Lat"] == pytest.approx(34.05)
