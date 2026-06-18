from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import refresh_segment_tiles as rst


def test_wet_hour_from_summary_keywords_and_probability():
    assert rst.wet_hour_from_summary("Snow showers", None) == 1.0
    assert rst.wet_hour_from_summary("Clear", None) == 0.0
    assert rst.wet_hour_from_summary("Clear", 45.0) == 1.0
    assert rst.wet_hour_from_summary("Clear", 10.0) == 0.0
    assert rst.wet_hour_from_summary(None, None) == 0.0


def test_quantitative_value_variants():
    assert rst.quantitative_value(None) is None
    assert rst.quantitative_value(7) == 7.0
    assert rst.quantitative_value({"value": 3.5}) == 3.5
    assert rst.quantitative_value({"value": None}) is None
    assert rst.quantitative_value("text") is None


def test_humidity_value_explicit_and_computed():
    assert rst.humidity_value(150.0, 20.0, 10.0) == 100.0
    assert rst.humidity_value(-5.0, 20.0, 10.0) == 0.0
    # When humidity is absent it is derived from temperature/dewpoint.
    assert rst.humidity_value(None, 20.0, 20.0) == pytest.approx(100.0, abs=1e-2)


def test_parse_wind_direction_numeric_cardinal_and_calm():
    assert rst.parse_wind_direction_degrees(None) is None
    assert rst.parse_wind_direction_degrees(45.0) == 45.0
    assert rst.parse_wind_direction_degrees(370.0) == pytest.approx(10.0)
    assert rst.parse_wind_direction_degrees("calm") is None
    assert rst.parse_wind_direction_degrees("VRB") is None
    assert rst.parse_wind_direction_degrees("NE") == 45.0
    assert rst.parse_wind_direction_degrees("n") == 0.0
    assert rst.parse_wind_direction_degrees("ESE") == 112.5
    assert rst.parse_wind_direction_degrees("280") == pytest.approx(280.0)
    assert rst.parse_wind_direction_degrees("not-a-direction") is None
