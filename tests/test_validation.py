from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import validation


GOOD = [
    {"id": "a", "name": "Alpha", "kind": "hospital", "lat": 34.0, "lon": -118.0, "capacity": 400},
    {"id": "b", "name": "Beta", "kind": "fire_station", "lat": 40.0, "lon": -74.0},
]


def test_valid_records_pass():
    report = validation.validate_records(GOOD, validation.FACILITY_SCHEMA)
    assert report["ok"] is True
    assert report["record_count"] == 2 and report["valid_records"] == 2
    assert report["issue_count"] == 0 and report["codes"] == {}


def test_missing_required_and_type_and_range():
    records = [
        {"name": "no id", "kind": "hospital", "lat": 34.0, "lon": -118.0},  # missing id
        {"id": "x", "name": "bad kind", "kind": "zoo", "lat": 34.0, "lon": -118.0},  # choice
        {"id": "y", "name": "bad lat", "kind": "hospital", "lat": 999.0, "lon": -118.0},  # range
        {"id": "z", "name": "bad type", "kind": "hospital", "lat": "north", "lon": -118.0},  # type
    ]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["ok"] is False
    assert report["codes"]["missing"] == 1
    assert report["codes"]["choice"] == 1
    assert report["codes"]["range"] == 1
    assert report["codes"]["type"] == 1
    assert report["valid_records"] == 0


def test_duplicate_ids():
    records = [
        {"id": "dup", "name": "A", "kind": "hospital", "lat": 34.0, "lon": -118.0},
        {"id": "dup", "name": "B", "kind": "hospital", "lat": 35.0, "lon": -119.0},
    ]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["codes"]["duplicate"] == 1
    # First occurrence is valid; the duplicate record carries the issue.
    assert report["valid_records"] == 1


def test_non_dict_record_and_nan():
    records = [
        "not a record",
        {"id": "n", "name": "NaN lat", "kind": "hospital", "lat": float("nan"), "lon": -118.0},
    ]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["codes"]["not_a_record"] == 1
    assert report["codes"]["not_finite"] == 1


def test_bool_not_accepted_as_int():
    records = [{"id": "b", "name": "B", "kind": "hospital", "lat": 34.0, "lon": -118.0, "capacity": True}]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["codes"]["type"] == 1  # True is not a valid int capacity


def test_max_issues_truncation():
    records = [{"kind": "zoo"} for _ in range(50)]  # each: missing id/name/lat/lon + bad choice
    report = validation.validate_records(records, validation.FACILITY_SCHEMA, max_issues=5)
    assert report["issue_count"] == 5
    assert report["issues_truncated"] is True


def test_validate_dataset_dispatch():
    report = validation.validate_dataset("critical_facilities", GOOD)
    assert report["ok"] is True and report["dataset"] == "critical_facilities"
    assert validation.validate_dataset("CRITICAL_FACILITIES", GOOD)["ok"] is True  # case-insensitive
    assert validation.validate_dataset("unknown_dataset", GOOD) is None


def test_empty_records():
    report = validation.validate_records([], validation.FACILITY_SCHEMA)
    assert report["ok"] is True and report["record_count"] == 0


def test_overflowing_int_is_not_finite_not_a_crash():
    # A JSON integer too large to become a float must degrade, not raise.
    records = [{"id": "x", "name": "X", "kind": "hospital", "lat": 0.0, "lon": 0.0, "capacity": 10**400}]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["codes"]["not_finite"] == 1
    assert report["ok"] is False
