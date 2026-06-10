from __future__ import annotations

from pathlib import Path
import sys
import warnings

import numpy as np
import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_model as tm


def _climatology(rows: list[dict]) -> pd.DataFrame:
    base_columns = ["station_index", "month", "hour_of_week", *tm.WEATHER_FEATURE_NAMES]
    return pd.DataFrame(rows, columns=base_columns)


def test_weather_defaults_average_features_and_zero_fill_nan():
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 0,
             "temp_c": 10.0, "dewpoint_c": 4.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": 2.0, "wet_hour": 0.0},
            {"station_index": 0, "month": 1, "hour_of_week": 1,
             "temp_c": 20.0, "dewpoint_c": 6.0, "relative_humidity_pct": 80.0,
             "wind_speed_mps": 4.0, "wet_hour": 1.0},
        ]
    )

    defaults = tm.weather_defaults_from_climatology(climatology)

    assert defaults.dtype == np.float32
    assert defaults.shape == (len(tm.WEATHER_FEATURE_NAMES),)
    assert defaults[0] == pytest.approx(15.0)  # temp mean
    assert defaults[3] == pytest.approx(3.0)  # wind mean


def test_weather_defaults_replace_all_nan_column_with_zero():
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 0,
             "temp_c": np.nan, "dewpoint_c": 4.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": 2.0, "wet_hour": 0.0},
        ]
    )
    defaults = tm.weather_defaults_from_climatology(climatology)
    assert defaults[0] == 0.0  # all-NaN temp column collapses to 0.0


def test_build_weather_cube_fills_gaps_and_uses_defaults():
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 5,
             "temp_c": 10.0, "dewpoint_c": 5.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": np.nan, "wet_hour": 0.0},
        ]
    )
    weather_defaults = np.array([0.0, 0.0, 0.0, 7.0, 0.0], dtype=np.float32)

    cube = tm.build_weather_cube(climatology, weather_defaults)

    assert cube.shape == (1, 12, 24 * 7, len(tm.WEATHER_FEATURE_NAMES))
    assert cube.dtype == np.float32
    # The observed temperature is preserved at its month/hour slot.
    assert cube[0, 0, 5, 0] == pytest.approx(10.0)
    # Gaps are back-filled, so no NaNs survive.
    assert not np.isnan(cube).any()
    # Wind has no observations anywhere, so it falls back to the provided default.
    assert np.allclose(cube[:, :, :, 3], 7.0)


def test_build_weather_cube_does_not_emit_runtime_warnings():
    # Sparse climatology used to leak "Mean of empty slice" RuntimeWarnings.
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 5,
             "temp_c": 10.0, "dewpoint_c": 5.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": np.nan, "wet_hour": 0.0},
        ]
    )
    weather_defaults = np.zeros(len(tm.WEATHER_FEATURE_NAMES), dtype=np.float32)

    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        tm.build_weather_cube(climatology, weather_defaults)
