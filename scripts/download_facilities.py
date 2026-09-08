"""Download and normalize critical facilities (hospitals, fire stations, shelters).

Reads an authoritative feed as GeoJSON — HIFLD Hospitals / Fire Stations, or an
OpenStreetMap Overpass export of ``amenity=hospital|fire_station|shelter`` — and
normalizes each feature to the facility schema used by the emergency endpoints
(see data/reference/critical_facilities.md), writing a processed parquet.

The download/write step needs the raw feed and geopandas-free parsing, so ``main``
is not run in CI; the pure ``classify_kind`` / ``normalize_facility`` /
``normalize_features`` helpers are unit-tested. With no feed available the runtime
falls back to the curated data/reference/critical_facilities.json.

    python scripts/download_facilities.py --input data/raw/facilities/hospitals.geojson --kind hospital

Output: data/processed/critical_facilities.parquet
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import re
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import CRITICAL_FACILITIES_PATH, FACILITIES_RAW_DIR, ensure_dirs

FACILITY_KINDS = ("hospital", "fire_station", "emergency_shelter")
FACILITY_COLUMNS = ["id", "name", "kind", "lat", "lon", "state", "capacity", "trauma_center"]

_KIND_KEYWORDS = (
    ("fire_station", ("fire station", "fire_station", "firehouse", "fire dept")),
    ("emergency_shelter", ("shelter", "evacuation", "emergency_shelter")),
    ("hospital", ("hospital", "acute care", "critical access", "medical center", "trauma")),
)


def classify_kind(raw_kind, default=None) -> str | None:
    """Map a raw type/amenity string to a facility kind, or ``default``/None."""
    text = str(raw_kind or "").strip().lower()
    if text in FACILITY_KINDS:
        return text
    for kind, keywords in _KIND_KEYWORDS:
        if any(keyword in text for keyword in keywords):
            return kind
    return default if default in FACILITY_KINDS else None


def _first(props: dict, *keys):
    for key in keys:
        if key in props and props[key] not in (None, ""):
            return props[key]
    return None


def _slug(name: str, lat: float, lon: float) -> str:
    base = re.sub(r"[^a-z0-9]+", "_", str(name).strip().lower()).strip("_") or "facility"
    return f"{base}_{round(lat, 4)}_{round(lon, 4)}"


def _to_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def normalize_facility(feature, *, default_kind=None) -> dict | None:
    """Normalize a GeoJSON Feature (or bare properties dict) to a facility record.

    Returns None when coordinates are missing/invalid or the kind can't be
    resolved. GeoJSON coordinates are ``[lon, lat]``; ``LATITUDE``/``LONGITUDE``
    (or ``lat``/``lon``) props are the fallback.
    """
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties") if isinstance(feature.get("properties"), dict) else feature
    geometry = feature.get("geometry") if isinstance(feature.get("geometry"), dict) else {}
    coords = geometry.get("coordinates")

    if isinstance(coords, (list, tuple)) and len(coords) >= 2:
        lon, lat = _to_float(coords[0]), _to_float(coords[1])
    else:
        lat = _to_float(_first(props, "LATITUDE", "latitude", "lat", "Y"))
        lon = _to_float(_first(props, "LONGITUDE", "longitude", "lon", "X"))
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None

    kind = classify_kind(
        _first(props, "kind", "TYPE", "type", "amenity", "FACILITY_TYPE"), default_kind
    )
    if kind is None:
        return None

    name = str(_first(props, "NAME", "name", "FACILITY", "FACNAME") or "Unknown").strip()
    identifier = _first(props, "id", "ID", "OBJECTID", "GLOBALID", "gid")
    trauma_raw = str(_first(props, "TRAUMA", "trauma", "TYPE", "type") or "").lower()

    return {
        "id": str(identifier) if identifier is not None else _slug(name, lat, lon),
        "name": name,
        "kind": kind,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "state": _first(props, "STATE", "state", "ST"),
        "capacity": _to_int(_first(props, "BEDS", "capacity", "CAPACITY", "POP")),
        "trauma_center": "trauma" in trauma_raw,
    }


def _to_int(value):
    number = _to_float(value)
    if number is None or number < 0:
        return None
    return int(number)


def normalize_features(features, *, default_kind=None) -> pd.DataFrame:
    """Normalize an iterable of features to a de-duplicated facilities DataFrame."""
    rows = []
    seen = set()
    for feature in features or []:
        record = normalize_facility(feature, default_kind=default_kind)
        if record is None or record["id"] in seen:
            continue
        seen.add(record["id"])
        rows.append(record)
    return pd.DataFrame(rows, columns=FACILITY_COLUMNS)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--input", type=Path, required=True, help="GeoJSON feed to normalize")
    parser.add_argument(
        "--kind", choices=FACILITY_KINDS, default=None, help="default kind if not in the feed"
    )
    parser.add_argument("--output", type=Path, default=CRITICAL_FACILITIES_PATH)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    payload = json.loads(Path(args.input).read_text(encoding="utf-8"))
    features = payload.get("features", payload) if isinstance(payload, dict) else payload
    facilities = normalize_features(features, default_kind=args.kind)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    facilities.to_parquet(args.output, index=False)
    print(f"wrote {len(facilities)} facilities to {args.output}")


if __name__ == "__main__":
    main()
