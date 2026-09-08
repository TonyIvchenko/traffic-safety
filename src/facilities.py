"""Critical-facilities store for the emergency / evacuation endpoints.

Loads points of safety (hospitals, fire stations, emergency shelters) and answers
spatial queries — nearest-k, within-bbox, within-radius — optionally filtered by
kind. Source resolution (first that works):

1. ``TRAFFIC_SAFETY_FACILITIES_PATH`` or an explicit path (``.parquet`` or ``.json``),
2. the processed ``data/processed/critical_facilities.parquet`` if present,
3. the curated ``data/reference/critical_facilities.json`` fallback.

Degrade-not-crash: a missing/corrupt source, or malformed records within it,
yield an empty store / skipped records rather than raising.
"""

from __future__ import annotations

from functools import lru_cache
import json
import math
import os
from pathlib import Path

import geo_math

FACILITIES_PATH_ENV = "TRAFFIC_SAFETY_FACILITIES_PATH"
REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_FACILITIES_PARQUET = REPO_DIR / "data" / "processed" / "critical_facilities.parquet"
REFERENCE_FACILITIES_JSON = REPO_DIR / "data" / "reference" / "critical_facilities.json"

FACILITY_FIELDS = ("id", "name", "kind", "lat", "lon", "state", "capacity", "trauma_center")
FACILITY_KINDS = ("hospital", "fire_station", "emergency_shelter")


def _to_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) else number


def _to_int(value):
    number = _to_float(value)
    if number is None or not math.isfinite(number) or number < 0:
        return None
    return int(number)


def _clean_facility(raw) -> dict | None:
    """Validate/normalize one facility record, or None if unusable."""
    if not isinstance(raw, dict):
        return None
    lat = _to_float(raw.get("lat"))
    lon = _to_float(raw.get("lon"))
    if lat is None or lon is None or not (-90.0 <= lat <= 90.0) or not (-180.0 <= lon <= 180.0):
        return None
    kind = str(raw.get("kind") or "").strip().lower() or "unknown"
    state = raw.get("state")
    return {
        "id": str(raw.get("id") or f"{kind}_{round(lat, 4)}_{round(lon, 4)}"),
        "name": str(raw.get("name") or "Unknown"),
        "kind": kind,
        "lat": round(lat, 6),
        "lon": round(lon, 6),
        "state": str(state) if state not in (None, "") else None,
        "capacity": _to_int(raw.get("capacity")),
        "trauma_center": bool(raw.get("trauma_center", False)),
    }


class FacilityStore:
    """Read-only spatial accessor over a list of facility records."""

    def __init__(self, facilities: list[dict]) -> None:
        self._facilities = facilities

    def __len__(self) -> int:
        return len(self._facilities)

    def kinds(self) -> list[str]:
        return sorted({facility["kind"] for facility in self._facilities})

    def _filtered(self, kind=None) -> list[dict]:
        if kind is None:
            return self._facilities
        wanted = str(kind).strip().lower()
        return [facility for facility in self._facilities if facility["kind"] == wanted]

    def all(self, *, kind=None) -> list[dict]:
        return [dict(facility) for facility in self._filtered(kind)]

    def nearest(self, lat, lon, *, kind=None, k: int = 1) -> list[dict]:
        return geo_math.nearest_k(lat, lon, self._filtered(kind), k=k)

    def within_radius(self, lat, lon, radius_km, *, kind=None) -> list[dict]:
        return geo_math.within_radius_km(lat, lon, self._filtered(kind), radius_km)

    def within_bbox(self, bbox, *, kind=None) -> list[dict]:
        try:
            min_lat, max_lat, min_lon, max_lon = (float(value) for value in bbox)
        except (TypeError, ValueError):
            return []
        hits = [
            dict(facility)
            for facility in self._filtered(kind)
            if min_lat <= facility["lat"] <= max_lat and min_lon <= facility["lon"] <= max_lon
        ]
        return hits

    @classmethod
    def from_records(cls, records) -> "FacilityStore":
        cleaned = [c for c in (_clean_facility(record) for record in (records or [])) if c]
        return cls(cleaned)

    @classmethod
    def from_json(cls, path) -> "FacilityStore":
        path = Path(path)
        if not path.exists():
            return cls([])
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            return cls([])
        records = payload.get("facilities") if isinstance(payload, dict) else payload
        if not isinstance(records, list):
            return cls([])
        return cls.from_records(records)

    @classmethod
    def from_parquet(cls, path) -> "FacilityStore":
        path = Path(path)
        if not path.exists():
            return cls([])
        try:
            import pandas as pd

            frame = pd.read_parquet(path)
        except (OSError, ValueError, ImportError):
            return cls([])
        return cls.from_records(frame.to_dict("records"))


@lru_cache(maxsize=4)
def _load_store_cached(path_str: str) -> FacilityStore:
    path = Path(path_str)
    if path.suffix == ".json":
        return FacilityStore.from_json(path)
    return FacilityStore.from_parquet(path)


def load_facility_store(path=None) -> FacilityStore:
    """The facilities store from an explicit path, the env override, or the fallbacks."""
    resolved = path or os.environ.get(FACILITIES_PATH_ENV)
    if resolved:
        return _load_store_cached(str(resolved))
    if DEFAULT_FACILITIES_PARQUET.exists():
        return _load_store_cached(str(DEFAULT_FACILITIES_PARQUET))
    return _load_store_cached(str(REFERENCE_FACILITIES_JSON))
