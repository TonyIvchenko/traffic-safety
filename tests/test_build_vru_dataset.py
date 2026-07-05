from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_vru_dataset as bvd


def test_vru_case_stats_counts_ped_and_cyc():
    person = pd.DataFrame(
        {
            "ST_CASE": [1, 1, 1, 2, 2, 3, 4, 4],
            # case1: driver + pedestrian + bicyclist; case2: 2 pedestrians;
            # case3: driver only (no VRU); case4: personal-conveyance + other-pedalcyclist
            "PER_TYP": [1, 5, 6, 5, 5, 1, 8, 7],
        }
    )
    stats = bvd.vru_case_stats(person)

    # Case 3 has no VRU and must be absent.
    assert set(stats.index) == {1, 2, 4}
    assert stats.loc[1, "ped_count"] == 1 and stats.loc[1, "cyc_count"] == 1
    assert stats.loc[1, "vru_count"] == 2
    assert stats.loc[2, "ped_count"] == 2 and stats.loc[2, "cyc_count"] == 0
    # PER_TYP 8 counts as pedestrian, 7 as cyclist.
    assert stats.loc[4, "ped_count"] == 1 and stats.loc[4, "cyc_count"] == 1


def test_vru_case_stats_handles_no_vru():
    person = pd.DataFrame({"ST_CASE": [1, 2], "PER_TYP": [1, 2]})
    stats = bvd.vru_case_stats(person)
    assert stats.empty
    assert list(stats.columns) == ["ped_count", "cyc_count", "vru_count"]


def _raw_accidents() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "ST_CASE": [1, 2, 3],
            "STATE": [6, 6, 36],
            "STATENAME": ["California", "California", "New York"],
            "YEAR": [2022, 2022, 2022],
            "MONTH": [9, 9, 13],  # case 3 has an invalid month -> dropped
            "DAY": [6, 7, 8],
            "DAY_WEEK": [3, 4, 5],  # FARS Sun=1..Sat=7
            "HOUR": [17, 2, 9],
            "LATITUDE": [34.05, 34.10, 40.7],
            "LONGITUD": [-118.24, -118.30, -74.0],
            "RUR_URB": [2, 2, 1],
            "FUNC_SYS": [1, 2, 3],
            "LGT_COND": [1, 2, 1],
        }
    )


def test_clean_accident_frame_bounds_and_derived_columns():
    cleaned = bvd.clean_accident_frame(_raw_accidents())
    # Invalid-month row dropped.
    assert set(cleaned["ST_CASE"]) == {1, 2}
    row = cleaned.set_index("ST_CASE").loc[1]
    # FARS DAY_WEEK=3 (Tuesday) -> Monday-first day_of_week=2.
    assert row["day_of_week"] == 2
    assert row["hour_of_week"] == (2 - 1) * 24 + 17
    assert isinstance(row["cell_id"], str) and row["cell_id"]


def test_build_vru_events_inner_joins_and_labels():
    cleaned = bvd.clean_accident_frame(_raw_accidents())
    stats = pd.DataFrame(
        {"ped_count": [1], "cyc_count": [0], "vru_count": [1]}, index=pd.Index([1], name="ST_CASE")
    )
    events = bvd.build_vru_events(cleaned, stats)
    # Only ST_CASE 1 has VRU involvement and survives the inner join.
    assert len(events) == 1
    assert events.iloc[0]["ped_count"] == 1
    assert events.iloc[0]["state_code"] == 6
    assert "vru_count" in events.columns
    assert "light_code" in events.columns
