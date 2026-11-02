"""Traffic-exposure normalization via FHWA HPMS AADT.

Crash counts and raw risk reflect how many vehicles pass as much as how dangerous
a road is. Normalizing by *exposure* — vehicle-miles traveled (VMT = AADT x
length) — yields crashes per million VMT, the FHWA-standard denominator, so a
quiet road with a few crashes can rank riskier than a busy freeway with many.

AADT (annual average daily traffic) is loaded from, in order:
1. ``TRAFFIC_SAFETY_AADT_PATH`` (``.parquet`` or ``.json``),
2. the processed ``data/processed/aadt_segments.parquet`` if present,
3. the curated reference ``data/reference/aadt_segments.json``.

Pure and defensive: malformed inputs degrade to ``None``/empty rather than
raising, matching the degrade-not-crash contract of the live API surface.
"""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path

import geo_math

AADT_PATH_ENV = "TRAFFIC_SAFETY_AADT_PATH"
REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_AADT_PARQUET = REPO_DIR / "data" / "processed" / "aadt_segments.parquet"
REFERENCE_AADT_JSON = REPO_DIR / "data" / "reference" / "aadt_segments.json"

MILES_PER_KM = 0.621371
MVMT = 1_000_000.0
DEFAULT_MATCH_RADIUS_KM = 3.0

# Fallback AADT by functional class when no nearby HPMS segment is mapped. Rough
# nationwide order-of-magnitude medians — screening defaults, not measurements.
FUNCTIONAL_CLASS_AADT = {
    "interstate": 65000,
    "freeway_expressway": 45000,
    "principal_arterial": 25000,
    "minor_arterial": 12000,
    "collector": 5000,
    "local": 1000,
}

# TIGER MTFCC -> functional class (coarse screening map for segments that carry an
# MTFCC but no AADT of their own).
MTFCC_TO_CLASS = {
    "S1100": "freeway_expressway",  # primary road (interstate / US highway)
    "S1200": "principal_arterial",  # secondary road (state highway)
    "S1400": "local",               # local neighborhood road
    "S1630": "collector",           # ramp
    "S1640": "local",               # service drive
}

# HPMS F_SYSTEM code -> functional class (used by the offline index builder).
HPMS_F_SYSTEM_TO_CLASS = {
    1: "interstate",
    2: "freeway_expressway",
    3: "principal_arterial",
    4: "minor_arterial",
    5: "collector",
    6: "collector",
    7: "local",
}


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def km_to_miles(length_km):
    value = _finite_float(length_km)
    return None if value is None else value * MILES_PER_KM


def annual_vmt(aadt, length_km, *, days: float = 365.0):
    """Annual vehicle-miles traveled on a segment: ``AADT x length(mi) x days``.

    Returns ``None`` for missing/non-finite/negative inputs.
    """
    a = _finite_float(aadt)
    length_mi = km_to_miles(length_km)
    d = _finite_float(days)
    if a is None or length_mi is None or d is None:
        return None
    if a < 0 or length_mi < 0 or d < 0:
        return None
    return a * length_mi * d


def crashes_per_mvmt(crashes, aadt, length_km, *, years: float = 1.0):
    """Crashes per million VMT — the exposure-normalized rate.

    Returns ``None`` when exposure is zero/unknown (so callers fall back to a
    non-normalized reading instead of dividing by zero).
    """
    c = _finite_float(crashes)
    y = _finite_float(years)
    if c is None or c < 0 or y is None or y <= 0:
        return None
    vmt = annual_vmt(aadt, length_km)
    if vmt is None or vmt <= 0:
        return None
    return c / (vmt * y) * MVMT


def normalize_class(functional_class) -> str | None:
    """Canonical functional-class key, or None when unrecognized."""
    if not functional_class:
        return None
    key = str(functional_class).strip().lower().replace(" ", "_").replace("-", "_")
    return key if key in FUNCTIONAL_CLASS_AADT else None


def functional_class_for(*, functional_class=None, mtfcc=None) -> str | None:
    """Resolve a functional class from an explicit class or a TIGER MTFCC."""
    resolved = normalize_class(functional_class)
    if resolved is not None:
        return resolved
    if mtfcc:
        return MTFCC_TO_CLASS.get(str(mtfcc).strip().upper())
    return None


def default_aadt(functional_class) -> int | None:
    """The screening-default AADT for a functional class, or None."""
    return FUNCTIONAL_CLASS_AADT.get(normalize_class(functional_class) or "")


def _clean_record(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    lat = _finite_float(raw.get("lat"))
    lon = _finite_float(raw.get("lon"))
    aadt = _finite_float(raw.get("aadt"))
    if lat is None or lon is None or aadt is None or aadt < 0:
        return None
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "lat": lat,
        "lon": lon,
        "aadt": aadt,
        "functional_class": normalize_class(raw.get("functional_class")),
        "length_km": _finite_float(raw.get("length_km")),
        "state": raw.get("state"),
        "year": raw.get("year"),
    }


class AADTStore:
    """Nearest-segment AADT lookup with a functional-class default fallback."""

    def __init__(self, records) -> None:
        self._records = [r for r in (_clean_record(x) for x in (records or [])) if r]

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[dict]:
        return [dict(record) for record in self._records]

    def nearest(self, lat, lon, *, max_km: float = DEFAULT_MATCH_RADIUS_KM) -> dict | None:
        hits = geo_math.within_radius_km(lat, lon, self._records, max_km)
        return hits[0] if hits else None

    def lookup(
        self,
        lat,
        lon,
        *,
        functional_class=None,
        mtfcc=None,
        max_km: float = DEFAULT_MATCH_RADIUS_KM,
    ) -> dict:
        """AADT for a location: a nearby mapped segment, else a class default.

        ``basis`` is ``"hpms_segment"`` (a matched segment), ``"class_default"``
        (a functional-class median), or ``"unavailable"`` (neither is known).
        """
        match = self.nearest(lat, lon, max_km=max_km)
        if match is not None:
            return {
                "aadt": match["aadt"],
                "basis": "hpms_segment",
                "distance_km": match.get("distance_km"),
                "functional_class": match.get("functional_class"),
                "segment_name": match.get("name"),
                "source_year": match.get("year"),
            }
        resolved_class = functional_class_for(functional_class=functional_class, mtfcc=mtfcc)
        fallback = default_aadt(resolved_class) if resolved_class else None
        if fallback is not None:
            return {
                "aadt": float(fallback),
                "basis": "class_default",
                "distance_km": None,
                "functional_class": resolved_class,
                "segment_name": None,
                "source_year": None,
            }
        return {
            "aadt": None,
            "basis": "unavailable",
            "distance_km": None,
            "functional_class": resolved_class,
            "segment_name": None,
            "source_year": None,
        }

    @classmethod
    def from_records(cls, records) -> "AADTStore":
        return cls(records)

    @classmethod
    def from_json(cls, path) -> "AADTStore":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls([])
        records = payload.get("segments") if isinstance(payload, dict) else payload
        return cls(records or [])

    @classmethod
    def from_parquet(cls, path) -> "AADTStore":
        import pandas as pd

        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - a missing/corrupt index yields an empty store
            return cls([])
        return cls(frame.to_dict(orient="records"))


@functools.lru_cache(maxsize=4)
def _load_store_cached(path_str: str) -> AADTStore:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return AADTStore.from_parquet(path)
    return AADTStore.from_json(path)


def load_aadt_store(path=None) -> AADTStore:
    """Process-wide AADT store, resolving env/processed/reference in order."""
    if path is not None:
        return _load_store_cached(str(Path(path)))
    env_path = os.getenv(AADT_PATH_ENV)
    if env_path:
        return _load_store_cached(str(Path(env_path)))
    if DEFAULT_AADT_PARQUET.exists():
        return _load_store_cached(str(DEFAULT_AADT_PARQUET))
    return _load_store_cached(str(REFERENCE_AADT_JSON))
