"""Schema-driven data-quality validation for dropped-in datasets.

A small, dependency-free validator: describe a dataset's columns as field specs
(type, required, range, allowed values, uniqueness) and :func:`validate_records`
returns a structured report of every issue — missing required fields, wrong
types, out-of-range values, disallowed values, and duplicates. Used to vet
operator-supplied files (e.g. a replacement critical-facilities feed) before they
reach the serving stores.

Pure and defensive: non-dict records are reported, not fatal; unknown field types
skip type-checking rather than raising.
"""

from __future__ import annotations

import math

# --------------------------------------------------------------------------- #
# Field specs and dataset schemas
# --------------------------------------------------------------------------- #

FACILITY_SCHEMA = (
    {"name": "id", "type": "str", "required": True, "unique": True},
    {"name": "name", "type": "str", "required": True},
    {
        "name": "kind", "type": "str", "required": True,
        "choices": ["hospital", "fire_station", "emergency_shelter"],
    },
    {"name": "lat", "type": "float", "required": True, "min": -90.0, "max": 90.0},
    {"name": "lon", "type": "float", "required": True, "min": -180.0, "max": 180.0},
    {"name": "capacity", "type": "int", "required": False, "min": 0},
    {"name": "trauma_center", "type": "bool", "required": False},
)

SCHEMAS = {
    "critical_facilities": FACILITY_SCHEMA,
}


def schema_for(dataset_id) -> tuple | None:
    """The validation schema for a dataset id (case-insensitive), or None."""
    return SCHEMAS.get(str(dataset_id).strip().lower())


# --------------------------------------------------------------------------- #
# Validation
# --------------------------------------------------------------------------- #


def _is_type(value, type_name: str) -> bool:
    if type_name == "str":
        return isinstance(value, str)
    if type_name == "bool":
        return isinstance(value, bool)
    if type_name == "int":
        return isinstance(value, int) and not isinstance(value, bool)
    if type_name == "float":
        return isinstance(value, (int, float)) and not isinstance(value, bool)
    return True  # unknown type -> no type constraint


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def validate_records(records, schema, *, max_issues: int | None = 1000) -> dict:
    """Validate ``records`` against ``schema`` (an iterable of field specs).

    Returns ``{ok, record_count, valid_records, issue_count, issues, codes,
    issues_truncated}``. Each issue is ``{index, field, code, detail}`` with code
    in {not_a_record, missing, type, range, choice, duplicate, not_finite}.
    """
    specs = list(schema or [])
    issues: list[dict] = []
    seen_unique: dict[str, set] = {spec["name"]: set() for spec in specs if spec.get("unique")}
    truncated = False
    valid_records = 0

    def _add(index, field, code, detail) -> None:
        nonlocal truncated
        if max_issues is not None and len(issues) >= max_issues:
            truncated = True
            return
        issues.append({"index": index, "field": field, "code": code, "detail": detail})

    record_list = list(records or [])
    for index, record in enumerate(record_list):
        before = len(issues)
        if not isinstance(record, dict):
            _add(index, None, "not_a_record", f"expected an object, got {type(record).__name__}")
            continue

        for spec in specs:
            name = spec["name"]
            present = name in record and record[name] is not None
            if not present:
                if spec.get("required"):
                    _add(index, name, "missing", "required field is missing or null")
                continue

            value = record[name]
            type_name = spec.get("type")
            if type_name and not _is_type(value, type_name):
                _add(index, name, "type", f"expected {type_name}, got {type(value).__name__}")
                continue  # further checks assume the right type

            if type_name in ("int", "float"):
                if not _finite(value):
                    _add(index, name, "not_finite", "numeric value is NaN or infinite")
                    continue
                number = float(value)
                if spec.get("min") is not None and number < spec["min"]:
                    _add(index, name, "range", f"{number} < min {spec['min']}")
                if spec.get("max") is not None and number > spec["max"]:
                    _add(index, name, "range", f"{number} > max {spec['max']}")

            choices = spec.get("choices")
            if choices is not None and value not in choices:
                _add(index, name, "choice", f"{value!r} not in {choices}")

            if spec.get("unique"):
                bucket = seen_unique[name]
                if value in bucket:
                    _add(index, name, "duplicate", f"duplicate {name}: {value!r}")
                else:
                    bucket.add(value)

        if len(issues) == before:
            valid_records += 1

    codes: dict[str, int] = {}
    for issue in issues:
        codes[issue["code"]] = codes.get(issue["code"], 0) + 1

    return {
        "ok": len(issues) == 0,
        "record_count": len(record_list),
        "valid_records": valid_records,
        "issue_count": len(issues),
        "issues": issues,
        "codes": codes,
        "issues_truncated": truncated,
    }


def validate_dataset(dataset_id, records, *, max_issues: int | None = 1000) -> dict | None:
    """Validate records for a known dataset id; None if no schema is registered."""
    schema = schema_for(dataset_id)
    if schema is None:
        return None
    report = validate_records(records, schema, max_issues=max_issues)
    report["dataset"] = str(dataset_id).strip().lower()
    return report
