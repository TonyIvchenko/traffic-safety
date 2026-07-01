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

import build_hotspot_report as bhr


def _roads() -> pd.DataFrame:
    # Corridor A (CA, linearid L1): two segments, 8 events total.
    # Corridor B (CA, linearid L2): one short segment, 5 events (higher density).
    # Corridor C (NY, linearid L3): one segment, 1 event (below min_events).
    return pd.DataFrame(
        {
            "state_fips": ["06", "06", "06", "36"],
            "linearid": ["L1", "L1", "L2", "L3"],
            "segment_idx": [0, 1, 2, 3],
            "fullname": ["Main St", "", "Short Rd", "Empty Ln"],
            "length_km": [2.0, 2.0, 0.5, 1.0],
            "center_lat": [34.0, 34.1, 35.0, 40.7],
            "center_lon": [-118.0, -118.1, -119.0, -74.0],
            "coords_json": [
                "[[-118.0, 34.0], [-118.05, 34.05]]",
                "[[-118.05, 34.05], [-118.1, 34.1]]",
                "[[-119.0, 35.0], [-119.01, 35.01]]",
                "[[-74.0, 40.7], [-74.01, 40.71]]",
            ],
        }
    )


def _counts() -> tuple[np.ndarray, np.ndarray]:
    total = np.array([5.0, 3.0, 5.0, 1.0], dtype=np.float32)
    hours = np.zeros((4, 168), dtype=np.float32)
    hours[0, 17] = 4.0  # corridor A peaks Mon 17:00
    hours[1, 17] = 2.0
    hours[2, 100] = 5.0  # corridor B peaks Fri 04:00
    return total, hours


def test_build_corridor_table_aggregates_and_ranks_by_events():
    total, hours = _counts()
    table = bhr.build_corridor_table(_roads(), total, hours, top_n=10, min_events=3)

    assert list(table["rank"]) == [1, 2]
    top = table.iloc[0]
    assert top["linearid"] == "L1"  # 8 events beats 5
    assert top["name"] == "Main St"  # first non-empty name wins
    assert top["segments"] == 2
    assert top["length_km"] == pytest.approx(4.0)
    assert top["historical_events"] == pytest.approx(8.0)
    assert top["events_per_km"] == pytest.approx(2.0)
    assert top["peak_hour_of_week"] == 17
    assert top["peak_hour_label"] == "Mon 17:00"
    assert top["peak_hour_events"] == pytest.approx(6.0)


def test_build_corridor_table_rank_by_density_and_min_events():
    total, hours = _counts()
    table = bhr.build_corridor_table(
        _roads(), total, hours, top_n=10, min_events=3, rank_by="events_per_km"
    )
    # Short Rd: 5 events / 0.5 km = 10 per km, beats Main St's 2 per km.
    assert table.iloc[0]["linearid"] == "L2"
    # The 1-event NY corridor is filtered out everywhere.
    assert "L3" not in set(table["linearid"])


def test_build_corridor_table_forecast_and_top_n():
    total, hours = _counts()
    forecast = np.array([0.2, 0.9, 0.4, 0.1], dtype=np.float32)
    table = bhr.build_corridor_table(
        _roads(), total, hours, forecast, top_n=1, min_events=3
    )
    assert len(table) == 1
    assert table.iloc[0]["forecast_risk_max"] == pytest.approx(0.9)  # max over members

    with pytest.raises(ValueError):
        bhr.build_corridor_table(_roads(), total, hours, rank_by="nonsense")


def test_corridors_geojson_builds_multilinestrings():
    total, hours = _counts()
    roads = _roads()
    table = bhr.build_corridor_table(roads, total, hours, top_n=10, min_events=3)
    payload = bhr.corridors_geojson(table, roads)

    assert payload["type"] == "FeatureCollection"
    assert payload["count"] == 2
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "MultiLineString"
    assert len(feature["geometry"]["coordinates"]) == 2  # both L1 member segments
    assert feature["properties"]["name"] == "Main St"
    assert feature["properties"]["rank"] == 1


def test_corridors_geojson_empty_table():
    payload = bhr.corridors_geojson(pd.DataFrame(), _roads())
    assert payload == {"type": "FeatureCollection", "count": 0, "features": []}
