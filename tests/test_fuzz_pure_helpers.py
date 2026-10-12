"""Dependency-free property/fuzz tests for the pure helpers most exposed to
untrusted input: the data validator (POST /v1/validation/data) and the relative
advisory core. Seeded, so failures reproduce. Each helper must never raise on any
input and must return a JSON-standard result (no NaN/Infinity).
"""

from __future__ import annotations

import json
from pathlib import Path
import random
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import advisory
import countermeasures
import facilities
import geo_math
import validation

LEVELS = {"Low", "Moderate", "Elevated", "High", "Extreme"}

# A palette of hostile values: wrong types, non-finite, out-of-range, a huge int
# (float() overflows), nested, and plausible-but-odd strings.
HOSTILE = [
    None, True, False, 0, 1, -1, 2, 10**400, -(10**400),
    0.0, 0.5, 1.0, 1.5, -0.3, float("nan"), float("inf"), -float("inf"),
    "", "x", "hospital", "zoo", "low", [0.1], {"a": 1},
]


def _value(rng):
    return rng.choice(HOSTILE)


def _assert_json_standard(obj):
    json.dumps(obj, allow_nan=False)  # raises if a bare NaN/Infinity leaked


def test_fuzz_validate_records_never_raises():
    rng = random.Random(1234)
    schemas = [
        validation.FACILITY_SCHEMA,
        validation.HIN_SEGMENT_SCHEMA,
        validation.EQUITY_OVERLAY_SCHEMA,
    ]
    for schema in schemas:
        fields = [spec["name"] for spec in schema]
        for _ in range(400):
            count = rng.randint(0, 6)
            records = [
                {field: _value(rng) for field in fields if rng.random() > 0.3}
                for _ in range(count)
            ]
            if records and rng.random() < 0.2:
                records[rng.randrange(len(records))] = rng.choice([None, 5, "str", [1]])
            report = validation.validate_records(records, schema)
            assert isinstance(report["ok"], bool)
            assert report["record_count"] == len(records)
            assert report["issue_count"] == len(report["issues"])
            assert sum(report["codes"].values()) == report["issue_count"]
            _assert_json_standard(report)


def test_fuzz_relative_advisory_never_raises():
    rng = random.Random(99)
    for _ in range(2000):
        reference = [_value(rng) for _ in range(rng.randint(0, 10))]
        result = advisory.relative_advisory(_value(rng), reference)
        assert result["level"] in LEVELS
        assert 0.0 <= result["risk_score"] <= 1.0
        assert result["basis"] in {"relative_to_local_weekly_normal", "absolute"}
        assert isinstance(result["message"], str) and result["message"]
        _assert_json_standard(result)


def test_fuzz_advisory_primitives_never_raise():
    rng = random.Random(7)
    for _ in range(2000):
        level = advisory.advisory_level(_value(rng))
        assert level["name"] in LEVELS
        pct = advisory.percentile_of(_value(rng), [_value(rng) for _ in range(rng.randint(0, 6))])
        assert pct is None or 0.0 <= pct <= 1.0
        block = advisory.advisory(_value(rng), drivers=[_value(rng), _value(rng)])
        assert 0.0 <= block["risk_score"] <= 1.0
        _assert_json_standard(block)


def test_fuzz_facility_store_never_raises():
    # A dropped-in facilities file can carry hostile values (incl. a huge int that
    # overflows float()); loading and querying must degrade, not crash.
    rng = random.Random(555)
    fields = ["id", "name", "kind", "lat", "lon", "state", "capacity", "trauma_center"]
    for _ in range(400):
        records = [
            {field: _value(rng) for field in fields if rng.random() > 0.3}
            for _ in range(rng.randint(0, 6))
        ]
        store = facilities.FacilityStore.from_records(records)
        got = store.all()
        _assert_json_standard(got)
        _assert_json_standard(store.nearest(34.0, -118.0, k=3))
        _assert_json_standard(store.within_radius(34.0, -118.0, 50.0))
        _assert_json_standard(store.within_bbox((33.0, 35.0, -119.0, -117.0)))


def test_fuzz_nearest_segment_never_raises():
    rng = random.Random(777)
    for _ in range(400):
        segments = {
            f"s{i}": {"segment_id": f"s{i}", "center_lat": _value(rng), "center_lon": _value(rng)}
            for i in range(rng.randint(0, 6))
        }
        store = countermeasures.CountermeasureStore(segments)
        result = store.nearest_segment(34.05, -118.24, max_km=rng.choice([None, 10.0]))
        assert result is None or set(result) == {"segment_id", "distance_km"}


def test_fuzz_geo_math_points_never_raises():
    rng = random.Random(888)
    for _ in range(1000):
        points = []
        for i in range(rng.randint(0, 6)):
            if rng.random() < 0.7:
                points.append({"lat": _value(rng), "lon": _value(rng), "id": i})
            else:
                points.append((_value(rng), _value(rng)))
        _assert_json_standard(geo_math.nearest_k(34.0, -118.0, points, k=rng.randint(0, 4)))
        _assert_json_standard(geo_math.within_radius_km(34.0, -118.0, points, 50.0))
