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


def _candidate_cells() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "cell_id": ["a", "b"],
            "center_lat": [34.0, 41.0],
            "center_lon": [-118.0, -87.0],
            "station_index": [0, 1],
        }
    )


def test_build_context_summarizes_history_and_geometry():
    history = pd.DataFrame(
        {
            "cell_id": ["a", "a", "b"],
            "hour_of_week": [5, 5, 10],
        }
    )

    context = tm.build_context(history, _candidate_cells())

    assert context.lat_by_cell["a"] == pytest.approx(34.0)
    assert context.lon_by_cell["b"] == pytest.approx(-87.0)
    assert context.total_by_cell["a"] == 2
    assert context.hour_by_cell[("a", 5)] == 2
    assert context.hour_by_cell[("b", 10)] == 1


def test_sample_negatives_is_deterministic_and_well_formed():
    candidate_cells = _candidate_cells()
    first = tm.sample_negatives(2022, 50, candidate_cells, np.random.default_rng(7))
    second = tm.sample_negatives(2022, 50, candidate_cells, np.random.default_rng(7))

    assert len(first) == 50
    assert list(first.columns) == [
        "cell_id", "station_index", "month", "day", "hour", "hour_of_week"
    ]
    # Same seed -> identical samples.
    pd.testing.assert_frame_equal(first, second)
    assert set(first["cell_id"]).issubset({"a", "b"})
    assert first["hour_of_week"].between(0, 167).all()
    assert first["month"].between(1, 12).all()
    assert first["hour"].between(0, 23).all()
