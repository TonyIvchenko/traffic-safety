from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model_support as ms


def test_approximate_utc_offset_hours_buckets():
    assert ms.approximate_utc_offset_hours(21.3, -157.8) == -10  # Hawaii
    assert ms.approximate_utc_offset_hours(61.2, -149.9) == -9  # Alaska
    assert ms.approximate_utc_offset_hours(34.0, -118.2) == -8  # Pacific
    assert ms.approximate_utc_offset_hours(39.7, -104.9) == -7  # Mountain
    assert ms.approximate_utc_offset_hours(41.8, -87.6) == -6  # Central
    assert ms.approximate_utc_offset_hours(40.7, -74.0) == -5  # Eastern


def test_relative_humidity_saturates_when_temp_equals_dewpoint():
    rh = ms.relative_humidity_from_temp_dewpoint(20.0, 20.0)
    assert rh.shape == ()
    assert float(rh) == pytest.approx(100.0, abs=1e-3)


def test_relative_humidity_is_lower_for_drier_air():
    rh = float(ms.relative_humidity_from_temp_dewpoint(30.0, 5.0))
    assert 0.0 <= rh < 50.0


def test_relative_humidity_is_clipped_to_unit_range():
    rh = ms.relative_humidity_from_temp_dewpoint(
        np.array([10.0, 25.0], dtype=np.float32),
        np.array([25.0, -40.0], dtype=np.float32),
    )
    assert np.all(rh >= 0.0)
    assert np.all(rh <= 100.0)


def test_parse_wind_speed_handles_units_and_missing_values():
    assert ms.parse_wind_speed_string_mps(None) is None
    assert ms.parse_wind_speed_string_mps("calm") is None
    mph = ms.parse_wind_speed_string_mps("10 mph")
    assert mph == pytest.approx(10 * 0.44704, rel=1e-6)
    kmh = ms.parse_wind_speed_string_mps("36 km/h")
    assert kmh == pytest.approx(36 / 3.6, rel=1e-6)
    # Ranges should resolve to the maximum reported speed.
    assert ms.parse_wind_speed_string_mps("5 to 15 mph") == pytest.approx(
        15 * 0.44704, rel=1e-6
    )


def test_temperature_and_pressure_conversions():
    assert ms.fahrenheit_to_celsius(None) is None
    assert ms.fahrenheit_to_celsius(32.0) == pytest.approx(0.0)
    assert ms.fahrenheit_to_celsius(212.0) == pytest.approx(100.0)
    assert ms.pascal_to_hpa(None) is None
    assert ms.pascal_to_hpa(101325.0) == pytest.approx(1013.25)


def test_build_feature_matrix_shape_and_dtype():
    features = ms.build_feature_matrix(
        latitudes=[34.0, 41.8],
        longitudes=[-118.2, -87.6],
        hour_of_week=[17, 100],
        months=[9, 1],
        totals=[120.0, 0.0],
        same_hour=[6.0, 0.0],
        temp_c=[18.0, -2.0],
        dewpoint_c=[10.0, -6.0],
        relative_humidity_pct=[60.0, 80.0],
        wind_speed_mps=[4.0, 8.0],
        wet_hour=[0.0, 1.0],
    )
    assert features.shape == (2, 16)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()


def test_lookup_weather_climatology_fills_nan_with_defaults():
    cube = np.full((1, 12, 168, 3), np.nan, dtype=np.float32)
    cube[0, 0, 0, :] = [np.nan, 5.0, np.nan]
    defaults = [1.0, 2.0, 3.0]

    weather = ms.lookup_weather_climatology(
        weather_cube=cube,
        station_indices=0,
        months=1,
        hour_of_week=0,
        weather_defaults=defaults,
    )

    assert weather.shape == (1, 3)
    assert weather[0, 0] == pytest.approx(1.0)  # NaN -> default
    assert weather[0, 1] == pytest.approx(5.0)  # kept
    assert weather[0, 2] == pytest.approx(3.0)  # NaN -> default


def test_lookup_weather_climatology_requires_4d_cube():
    with pytest.raises(ValueError):
        ms.lookup_weather_climatology(
            weather_cube=np.zeros((2, 3), dtype=np.float32),
            station_indices=0,
            months=1,
            hour_of_week=0,
            weather_defaults=[0.0, 0.0, 0.0],
        )
