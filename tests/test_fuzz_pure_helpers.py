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
