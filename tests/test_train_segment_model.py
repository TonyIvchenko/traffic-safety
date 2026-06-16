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


def test_sample_negatives_is_deterministic_and_in_range():
    roads = pd.DataFrame({"segment_idx": np.arange(5, dtype=np.int32)})
    first = tsm.sample_negatives(roads, 40, 2022, np.random.default_rng(11))
    second = tsm.sample_negatives(roads, 40, 2022, np.random.default_rng(11))

    assert len(first) == 40
    assert list(first.columns) == ["segment_idx", "month", "hour_of_week"]
    pd.testing.assert_frame_equal(first, second)
    assert first["segment_idx"].between(0, 4).all()
    assert first["hour_of_week"].between(0, 167).all()
    assert first["month"].between(1, 12).all()


def test_build_context_counts_history():
    history = pd.DataFrame({"segment_idx": [0, 0, 1], "hour_of_week": [5, 5, 10]})

    context = tsm.build_context(history)

    assert context.total_by_idx[0] == 2
    assert context.hour_by_idx[(0, 5)] == 2
    assert context.hour_by_idx[(1, 10)] == 1


def test_build_design_matrix_combines_static_and_temporal_blocks():
    static_features = np.arange(12, dtype=np.float32).reshape(4, 3)
    context = tsm.SegmentContext(
        total_by_idx={0: 7, 2: 0},
        hour_by_idx={(0, 5): 3},
    )
    rows = pd.DataFrame(
        {
            "segment_idx": [0, 2],
            "hour_of_week": [5, 40],
            "month": [1, 6],
            "temp_c": [10.0, -2.0],
            "relative_humidity_pct": [60.0, 80.0],
            "wind_speed_mps": [3.0, 7.0],
            "wet_hour": [0.0, 1.0],
        }
    )

    matrix = tsm.build_design_matrix(rows, static_features, context)

    # 3 static columns + 13 temporal feature columns.
    assert matrix.shape == (2, 3 + 13)
    assert matrix.dtype == np.float32
    assert np.isfinite(matrix).all()
    # Static block is taken from the rows' segment indices, unchanged.
    assert np.allclose(matrix[:, :3], static_features[[0, 2]])


def test_attach_positive_weather_fills_only_missing_values():
    # Climatology cube: station 0, month 1, hour_of_week 5 -> one value per feature.
    cube = np.zeros((1, 12, 24 * 7, len(tsm.WEATHER_COLUMNS)), dtype=np.float32)
    cube[0, 0, 5, :] = [100.0, 200.0, 300.0, 400.0]
    station_indices = np.array([0, 0], dtype=np.int16)
    weather_defaults = np.zeros(len(tsm.WEATHER_COLUMNS), dtype=np.float32)

    positives = pd.DataFrame(
        {
            "segment_idx": [0, 1],
            "month": [1, 1],
            "hour_of_week": [5, 5],
            "temp_c": [np.nan, 15.0],
            "relative_humidity_pct": [22.0, np.nan],
            "wind_speed_mps": [np.nan, 5.0],
            "wet_hour": [1.0, np.nan],
        }
    )

    filled = tsm.attach_positive_weather(positives, station_indices, cube, weather_defaults)

    # Observed values survive; NaNs are replaced by the climatology lookup.
    assert filled["temp_c"].tolist() == [100.0, 15.0]
    assert filled["relative_humidity_pct"].tolist() == [22.0, 200.0]
    assert filled["wind_speed_mps"].tolist() == [300.0, 5.0]
    assert filled["wet_hour"].tolist() == [1.0, 400.0]
