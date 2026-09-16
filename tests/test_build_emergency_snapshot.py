from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_emergency_snapshot as bes
import facilities

FAKE_REGIONS = [
    {"id": "alpha", "name": "Alpha", "bbox": [33.9, 34.2, -118.4, -118.0]},
    {"id": "beta", "name": "Beta", "bbox": [40.6, 40.9, -74.1, -73.9]},
]

# alpha has full coverage; beta has none.
FAKE_FACILITIES = facilities.FacilityStore.from_records(
    [
        {"id": "h", "kind": "hospital", "lat": 34.05, "lon": -118.2},
        {"id": "s", "kind": "emergency_shelter", "lat": 34.05, "lon": -118.2},
        {"id": "f", "kind": "fire_station", "lat": 34.05, "lon": -118.2},
    ]
)


def _profiles():
    # alpha: low-and-varying (safe at frame 0); beta: rising (peak at end).
    return {
        "alpha": [0.10 + (how % 24) / 1000.0 for how in range(168)],
        "beta": [0.10 + how / 1000.0 for how in range(168)],
    }


def test_region_facility_coverage():
    coverage = bes.region_facility_coverage(FAKE_FACILITIES, FAKE_REGIONS[0])
    assert coverage["counts"] == {"hospital": 1, "fire_station": 1, "emergency_shelter": 1}
    assert coverage["total"] == 3
    # beta has no facilities in its bbox.
    assert bes.region_facility_coverage(FAKE_FACILITIES, FAKE_REGIONS[1])["total"] == 0


def test_build_readiness_table_ranks_least_ready_first():
    table = bes.build_readiness_table(
        _profiles(), FAKE_FACILITIES, day_of_week=1, hour=0, month=1, region_catalog=FAKE_REGIONS
    )
    assert table["count"] == 2
    assert table["frame_idx"] == 0 and table["frame_label"] == "Mon 00:00"
    # beta (no hospital -> limited) ranks before alpha (full coverage).
    assert table["regions"][0]["region_id"] == "beta"
    assert table["regions"][0]["readiness"]["rating"] == "limited"
    assert table["regions"][1]["region_id"] == "alpha"
    assert table["regions"][1]["facility_coverage"]["total"] == 3


def test_build_readiness_table_skips_unknown_region():
    table = bes.build_readiness_table(
        {"ghost": [0.5] * 168}, FAKE_FACILITIES, region_catalog=FAKE_REGIONS
    )
    assert table["count"] == 0


def test_readiness_geojson():
    table = bes.build_readiness_table(
        _profiles(), FAKE_FACILITIES, region_catalog=FAKE_REGIONS
    )
    geojson = bes.readiness_geojson(table)
    assert geojson["type"] == "FeatureCollection"
    assert len(geojson["features"]) == 2
    props = geojson["features"][0]["properties"]
    assert {"region_id", "readiness", "level", "facility_total"} <= set(props)
    assert geojson["features"][0]["geometry"]["type"] == "Point"
