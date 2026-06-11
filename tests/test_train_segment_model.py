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

import train_segment_model as tsm


def _climatology(rows: list[dict]) -> pd.DataFrame:
    columns = ["station_index", "month", "hour_of_week", *tsm.WEATHER_COLUMNS]
    return pd.DataFrame(rows, columns=columns)


def test_build_weather_cube_returns_cube_and_defaults():
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 5,
             "temp_c": 10.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": np.nan, "wet_hour": 0.0},
        ]
    )

    cube, defaults = tsm.build_weather_cube(climatology)

    assert cube.shape == (1, 12, 24 * 7, len(tsm.WEATHER_COLUMNS))
    assert cube.dtype == np.float32
    assert defaults.shape == (len(tsm.WEATHER_COLUMNS),)
    # Observed temperature is preserved at its slot.
    assert cube[0, 0, 5, 0] == pytest.approx(10.0)
    # No NaNs survive the back-fill.
    assert not np.isnan(cube).any()
    # Wind is unobserved everywhere, so its all-NaN mean collapses to a 0.0 default.
    assert defaults[2] == pytest.approx(0.0)
    assert np.allclose(cube[:, :, :, 2], 0.0)


def test_build_weather_cube_sizes_station_axis_to_max_index():
    climatology = _climatology(
        [
            {"station_index": 3, "month": 2, "hour_of_week": 10,
             "temp_c": 5.0, "relative_humidity_pct": 70.0,
             "wind_speed_mps": 3.0, "wet_hour": 1.0},
        ]
    )

    cube, _ = tsm.build_weather_cube(climatology)

    # Station axis spans 0..max(station_index).
    assert cube.shape[0] == 4
    assert cube[3, 1, 10, 0] == pytest.approx(5.0)


def test_build_weather_cube_does_not_emit_runtime_warnings():
    climatology = _climatology(
        [
            {"station_index": 0, "month": 1, "hour_of_week": 5,
             "temp_c": 10.0, "relative_humidity_pct": 60.0,
             "wind_speed_mps": np.nan, "wet_hour": 0.0},
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("error", category=RuntimeWarning)
        tsm.build_weather_cube(climatology)
