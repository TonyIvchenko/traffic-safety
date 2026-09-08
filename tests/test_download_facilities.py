from __future__ import annotations

import json
from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import download_facilities as df

REFERENCE_PATH = REPO_DIR / "data" / "reference" / "critical_facilities.json"


def test_classify_kind():
    assert df.classify_kind("hospital") == "hospital"
    assert df.classify_kind("GENERAL ACUTE CARE HOSPITAL") == "hospital"
    assert df.classify_kind("FIRE STATION") == "fire_station"
    assert df.classify_kind("Emergency Shelter") == "emergency_shelter"
    assert df.classify_kind("post office") is None
    assert df.classify_kind("post office", default="hospital") == "hospital"
    assert df.classify_kind("post office", default="not_a_kind") is None


def test_normalize_hifld_style_feature():
    feature = {
        "type": "Feature",
        "geometry": {"type": "Point", "coordinates": [-118.3805, 34.0754]},
        "properties": {
            "ID": "H123", "NAME": "Example Medical Center", "TYPE": "GENERAL ACUTE CARE",
            "BEDS": "450", "STATE": "CA",
        },
    }
    record = df.normalize_facility(feature)
    assert record["id"] == "H123"
    assert record["name"] == "Example Medical Center"
    assert record["kind"] == "hospital"
    assert record["lat"] == 34.0754 and record["lon"] == -118.3805  # [lon,lat] order
    assert record["capacity"] == 450
    assert record["state"] == "CA"


def test_normalize_osm_style_node_with_default_kind():
    node = {"properties": {"name": "Station 5", "amenity": "fire_station", "lat": 40.71, "lon": -74.01}}
    record = df.normalize_facility(node)
    assert record["kind"] == "fire_station"
    assert record["lat"] == 40.71 and record["lon"] == -74.01
    # id falls back to a slug when no identifier is present.
    assert record["id"].startswith("station_5_")


def test_normalize_flags_trauma_center():
    feature = {
        "geometry": {"coordinates": [-80.2126, 25.7907]},
        "properties": {"NAME": "Trauma Hosp", "TYPE": "Level I Trauma Center"},
    }
    record = df.normalize_facility(feature)
    assert record["kind"] == "hospital"
    assert record["trauma_center"] is True


def test_normalize_rejects_bad_coords_and_unknown_kind():
    assert df.normalize_facility({"properties": {"NAME": "No coords", "TYPE": "hospital"}}) is None
    assert df.normalize_facility(
        {"geometry": {"coordinates": [999, 999]}, "properties": {"TYPE": "hospital"}}
    ) is None
    assert df.normalize_facility(
        {"geometry": {"coordinates": [-118, 34]}, "properties": {"NAME": "Mystery", "TYPE": "zoo"}}
    ) is None
    assert df.normalize_facility("nonsense") is None


def test_normalize_features_dedupes_and_frames():
    features = [
        {"geometry": {"coordinates": [-118.0, 34.0]},
         "properties": {"ID": "1", "NAME": "A", "TYPE": "hospital"}},
        {"geometry": {"coordinates": [-118.0, 34.0]},
         "properties": {"ID": "1", "NAME": "A dup", "TYPE": "hospital"}},  # same id -> dropped
        {"properties": {"TYPE": "hospital"}},  # no coords -> dropped
        {"geometry": {"coordinates": [-74.0, 40.0]},
         "properties": {"ID": "2", "NAME": "B", "amenity": "fire_station"}},
    ]
    frame = df.normalize_features(features)
    assert list(frame["id"]) == ["1", "2"]
    assert list(frame.columns) == df.FACILITY_COLUMNS


def test_reference_dataset_is_valid():
    payload = json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))
    facilities = payload["facilities"]
    assert len(facilities) >= 15
    ids = set()
    for facility in facilities:
        assert {"id", "name", "kind", "lat", "lon"} <= set(facility)
        assert facility["kind"] in df.FACILITY_KINDS
        assert -90 <= facility["lat"] <= 90 and -180 <= facility["lon"] <= 180
        assert facility["id"] not in ids  # ids unique
        ids.add(facility["id"])
    # All three kinds represented.
    kinds = {facility["kind"] for facility in facilities}
    assert kinds == set(df.FACILITY_KINDS)
