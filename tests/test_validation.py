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


HIN_GOOD = [
    {"segment_id": "s1", "fullname": "Main St", "mtfcc": "S1200", "length_km": 1.5,
     "center_lat": 34.0, "center_lon": -118.0, "fatal_crashes": 3.0, "hin_rank": 1, "rur_urb": 2.0},
    {"segment_id": "s2", "center_lat": 40.0, "center_lon": -74.0},
]


def test_hin_schema_valid_records():
    report = validation.validate_records(HIN_GOOD, validation.HIN_SEGMENT_SCHEMA)
    assert report["ok"] is True and report["valid_records"] == 2


def test_hin_schema_catches_problems():
    records = [
        {"center_lat": 34.0, "center_lon": -118.0},  # missing segment_id
        {"segment_id": "a", "center_lat": 999.0, "center_lon": -118.0},  # lat out of range
        {"segment_id": "a", "center_lat": 34.0, "center_lon": -118.0},  # duplicate id
    ]
    report = validation.validate_records(records, validation.HIN_SEGMENT_SCHEMA)
    assert report["codes"]["missing"] >= 1  # segment_id (and center_lat on first row? no, present)
    assert report["codes"]["range"] == 1
    assert report["codes"]["duplicate"] == 1


def test_hin_numeric_fields_accept_floats_from_parquet():
    # rur_urb/hin_rank as floats (as parquet may store them) must not be flagged.
    records = [{"segment_id": "s", "center_lat": 34.0, "center_lon": -118.0,
                "rur_urb": 2.0, "hin_rank": 5.0, "fatal_crashes": 0.0}]
    report = validation.validate_records(records, validation.HIN_SEGMENT_SCHEMA)
    assert report["ok"] is True


def test_validate_dataset_dispatch_hin():
    assert validation.validate_dataset("high_injury_network", HIN_GOOD)["ok"] is True


def test_countermeasures_shares_segment_schema():
    # Countermeasures validate against the same HIN segment file/shape.
    assert validation.schema_for("countermeasures") is validation.HIN_SEGMENT_SCHEMA
    assert validation.validate_dataset("countermeasures", HIN_GOOD)["ok"] is True


EQUITY_GOOD = [
    {"segment_id": "e1", "tract_geoid": "06037", "svi_percentile": 0.82,
     "svi_category": "very_high", "disadvantaged": True, "in_equity_index": 1,
     "risk": 0.4, "crashes": 5.0, "center_lat": 34.0, "center_lon": -118.0},
    {"segment_id": "e2", "svi_category": "low", "disadvantaged": 0,
     "center_lat": 40.0, "center_lon": -74.0},
]


def test_equity_schema_valid_records():
    report = validation.validate_records(EQUITY_GOOD, validation.EQUITY_OVERLAY_SCHEMA)
    assert report["ok"] is True and report["valid_records"] == 2  # 0/1/True all accepted for bool flags


def test_equity_schema_catches_problems():
    records = [
        {"segment_id": "e", "svi_category": "catastrophic", "center_lat": 34.0, "center_lon": -118.0},  # choice
        {"segment_id": "f", "svi_percentile": 1.5, "center_lat": 34.0, "center_lon": -118.0},  # range
        {"center_lat": 34.0, "center_lon": -118.0},  # missing segment_id
    ]
    report = validation.validate_records(records, validation.EQUITY_OVERLAY_SCHEMA)
    assert report["codes"]["choice"] == 1
    assert report["codes"]["range"] == 1
    assert report["codes"]["missing"] == 1


def test_validate_dataset_dispatch_equity():
    assert validation.validate_dataset("equity_overlay", EQUITY_GOOD)["ok"] is True
    assert "equity_overlay" in validation.SCHEMAS


def test_overflowing_int_is_not_finite_not_a_crash():
    # A JSON integer too large to become a float must degrade, not raise.
    records = [{"id": "x", "name": "X", "kind": "hospital", "lat": 0.0, "lon": 0.0, "capacity": 10**400}]
    report = validation.validate_records(records, validation.FACILITY_SCHEMA)
    assert report["codes"]["not_finite"] == 1
    assert report["ok"] is False
