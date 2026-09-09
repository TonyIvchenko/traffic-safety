from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import facilities


RECORDS = [
    {"id": "h1", "name": "Hosp One", "kind": "hospital", "lat": 34.00, "lon": -118.00, "capacity": 400},
    {"id": "s1", "name": "Shelter One", "kind": "emergency_shelter", "lat": 34.05, "lon": -118.05},
    {"id": "f1", "name": "Fire One", "kind": "fire_station", "lat": 34.10, "lon": -118.10},
    {"id": "h2", "name": "Hosp Two", "kind": "hospital", "lat": 40.71, "lon": -74.01},
    {"id": "bad", "name": "No coords", "kind": "hospital"},  # dropped
    {"id": "bad2", "name": "Bad coords", "kind": "hospital", "lat": 999, "lon": 0},  # dropped
]


def _store():
    return facilities.FacilityStore.from_records(RECORDS)


def test_from_records_cleans_and_skips_malformed():
    store = _store()
    assert len(store) == 4  # two bad records dropped
    assert set(store.kinds()) == {"hospital", "emergency_shelter", "fire_station"}


def test_nearest_with_kind_filter():
    store = _store()
    nearest = store.nearest(34.01, -118.01, kind="hospital", k=1)
    assert len(nearest) == 1
    assert nearest[0]["id"] == "h1"
    assert "distance_km" in nearest[0]
    # Without a kind filter, the nearest of any kind may be a non-hospital.
    any_nearest = store.nearest(34.06, -118.06, k=1)[0]
    assert any_nearest["id"] == "s1"


def test_within_radius_and_bbox():
    store = _store()
    near = store.within_radius(34.01, -118.01, 20.0)
    assert {f["id"] for f in near} == {"h1", "s1", "f1"}  # h2 (NYC) excluded
    box = store.within_bbox((33.9, 34.06, -118.11, -118.04), kind="emergency_shelter")
    assert [f["id"] for f in box] == ["s1"]
    assert store.within_bbox("nonsense") == []


def test_capacity_and_trauma_normalized():
    store = facilities.FacilityStore.from_records(
        [{"id": "x", "kind": "hospital", "lat": 1, "lon": 1, "capacity": float("nan"),
          "trauma_center": True}]
    )
    facility = store.all()[0]
    assert facility["capacity"] is None  # NaN -> None
    assert facility["trauma_center"] is True


def test_load_default_uses_reference_json(monkeypatch):
    monkeypatch.delenv(facilities.FACILITIES_PATH_ENV, raising=False)
    # No processed parquet in a clean checkout -> curated reference fallback.
    store = facilities.load_facility_store()
    assert len(store) >= 15
    assert "hospital" in store.kinds()


def test_env_override_json(tmp_path, monkeypatch):
    path = tmp_path / "facilities.json"
    path.write_text(json.dumps({"facilities": RECORDS}), encoding="utf-8")
    monkeypatch.setenv(facilities.FACILITIES_PATH_ENV, str(path))
    store = facilities.load_facility_store()
    assert len(store) == 4


def test_from_parquet_round_trip(tmp_path):
    path = tmp_path / "facilities.parquet"
    pd.DataFrame(RECORDS[:4]).to_parquet(path, index=False)
    store = facilities.FacilityStore.from_parquet(path)
    assert len(store) == 4
    assert store.nearest(40.71, -74.01, kind="hospital", k=1)[0]["id"] == "h2"


def test_degrades_on_missing_and_corrupt(tmp_path):
    assert len(facilities.FacilityStore.from_json(tmp_path / "nope.json")) == 0
    assert len(facilities.FacilityStore.from_parquet(tmp_path / "nope.parquet")) == 0
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert len(facilities.FacilityStore.from_json(corrupt)) == 0
