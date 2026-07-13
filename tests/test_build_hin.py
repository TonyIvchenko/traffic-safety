from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_hin as bh


def _segments() -> pd.DataFrame:
    # Two short E-W segments ~1 degree of longitude apart, both in state 06.
    return pd.DataFrame(
        {
            "segment_id": ["west", "east"],
            "state_fips": ["06", "06"],
            "center_lat": [34.0, 34.0],
            "center_lon": [-118.0, -117.0],
            "coords_json": [
                "[[-118.01, 34.0], [-117.99, 34.0]]",
                "[[-117.01, 34.0], [-116.99, 34.0]]",
            ],
        }
    ).reset_index(drop=True)


def test_state_fips_from_code_zero_pads():
    assert bh.state_fips_from_code(6) == "06"   # FARS numeric FIPS for California
    assert bh.state_fips_from_code(12) == "12"
    assert bh.state_fips_from_code("6") == "06"


def test_build_state_indexes_groups_by_state():
    coords, state_indices, state_trees = bh.build_state_indexes(_segments())
    assert len(coords) == 2
    assert list(state_indices) == ["06"]
    assert len(state_indices["06"]) == 2
    assert "06" in state_trees


def test_match_crashes_picks_nearest_segment():
    segments = _segments()
    coords, state_indices, state_trees = bh.build_state_indexes(segments)
    crashes = pd.DataFrame(
        {
            "lat": [34.0, 34.0],
            "lon": [-118.0, -117.0],  # right on the west segment, then the east one
            "state_fips": ["06", "06"],
        }
    )
    matched = bh.match_crashes(crashes, coords, state_indices, state_trees, max_match_km=0.25)
    assert list(matched) == [0, 1]


def test_match_crashes_drops_points_beyond_radius():
    segments = _segments()
    coords, state_indices, state_trees = bh.build_state_indexes(segments)
    # ~11 km north of any road: outside a 0.25 km match radius.
    crashes = pd.DataFrame({"lat": [34.1], "lon": [-118.0], "state_fips": ["06"]})
    matched = bh.match_crashes(crashes, coords, state_indices, state_trees, max_match_km=0.25)
    assert list(matched) == [-1]
    # A generous radius does match it.
    loose = bh.match_crashes(crashes, coords, state_indices, state_trees, max_match_km=25.0)
    assert list(loose) == [0]


def test_match_crashes_ignores_unknown_state():
    segments = _segments()
    coords, state_indices, state_trees = bh.build_state_indexes(segments)
    crashes = pd.DataFrame({"lat": [40.7], "lon": [-74.0], "state_fips": ["36"]})
    matched = bh.match_crashes(crashes, coords, state_indices, state_trees)
    assert list(matched) == [-1]


def test_match_crashes_is_row_aligned():
    segments = _segments()
    coords, state_indices, state_trees = bh.build_state_indexes(segments)
    crashes = pd.DataFrame(
        {
            "lat": [34.1, 34.0, 34.0],
            "lon": [-118.0, -117.0, -118.0],
            "state_fips": ["06", "06", "06"],
        }
    )
    matched = bh.match_crashes(crashes, coords, state_indices, state_trees, max_match_km=0.25)
    assert list(matched) == [-1, 1, 0]  # unmatched, east, west - in input order
    assert len(matched) == len(crashes)
