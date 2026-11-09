"""Traffic-speed risk features from NPMRDS-style speed data.

The National Performance Management Research Data Set (NPMRDS, via FHWA/RITIS)
publishes observed vs free-flow speeds on the National Highway System. Speed is a
safety signal at *both* extremes: heavy congestion drives rear-end / sideswipe
crashes, and travel well above free-flow raises crash severity. This derives a
screening congestion ratio and a U-shaped speed-risk index for a location.

Loads from ``TRAFFIC_SAFETY_NPMRDS_PATH`` (``.parquet``/``.json``), the processed
``data/processed/npmrds_segments.parquet``, or the curated reference
``data/reference/npmrds_segments.json``. Pure and defensive: bad inputs degrade
to ``None``/neutral rather than raising. The index is a screening heuristic, not
a calibrated crash-modification factor.
"""

from __future__ import annotations

import functools
import json
import math
import os
from pathlib import Path

import geo_math

NPMRDS_PATH_ENV = "TRAFFIC_SAFETY_NPMRDS_PATH"
REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_NPMRDS_PARQUET = REPO_DIR / "data" / "processed" / "npmrds_segments.parquet"
REFERENCE_NPMRDS_JSON = REPO_DIR / "data" / "reference" / "npmrds_segments.json"
DEFAULT_MATCH_RADIUS_KM = 3.0

# Congestion-ratio (observed / free-flow) thresholds and the screening speed-risk
# multiplier each maps to. U-shaped: both heavy congestion and speeding elevate.
HEAVY_CONGESTION_RATIO = 0.50
MODERATE_CONGESTION_RATIO = 0.75
SPEEDING_RATIO = 1.15
HEAVY_CONGESTION_RISK = 1.25
MODERATE_CONGESTION_RISK = 1.12
SPEEDING_RISK = 1.20
NEUTRAL_RISK = 1.0

PEAK_HOURS = set(range(7, 10)) | set(range(16, 19))  # AM 7-9, PM 16-18 local


def _finite_float(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def congestion_ratio(observed_speed, reference_speed):
    """Observed / free-flow speed. <1 is congested, ~1 free-flow. None if invalid."""
    observed = _finite_float(observed_speed)
    reference = _finite_float(reference_speed)
    if observed is None or reference is None or reference <= 0 or observed < 0:
        return None
    return observed / reference


def speed_risk_index(observed_speed, reference_speed) -> float:
    """A screening crash-risk multiplier from the congestion ratio (U-shaped).

    Heavy congestion and travel well above free-flow both return an elevated
    factor; near free-flow returns 1.0. Unknown speeds return 1.0 (neutral).
    """
    ratio = congestion_ratio(observed_speed, reference_speed)
    if ratio is None:
        return NEUTRAL_RISK
    if ratio < HEAVY_CONGESTION_RATIO:
        return HEAVY_CONGESTION_RISK
    if ratio < MODERATE_CONGESTION_RATIO:
        return MODERATE_CONGESTION_RISK
    if ratio > SPEEDING_RATIO:
        return SPEEDING_RISK
    return NEUTRAL_RISK


def period_for_hour(hour) -> str:
    value = _finite_float(hour)
    if value is None:
        return "peak"
    return "peak" if int(value) % 24 in PEAK_HOURS else "offpeak"


def _clean_record(raw) -> dict | None:
    if not isinstance(raw, dict):
        return None
    lat = _finite_float(raw.get("lat"))
    lon = _finite_float(raw.get("lon"))
    reference = _finite_float(raw.get("reference_speed_mph"))
    if lat is None or lon is None or reference is None or reference <= 0:
        return None
    peak = _finite_float(raw.get("peak_speed_mph"))
    offpeak = _finite_float(raw.get("offpeak_speed_mph"))
    return {
        "id": raw.get("id"),
        "name": raw.get("name"),
        "lat": lat,
        "lon": lon,
        "reference_speed_mph": reference,
        "peak_speed_mph": peak,
        "offpeak_speed_mph": offpeak,
        "road_class": raw.get("road_class"),
        "year": raw.get("year"),
    }


class TrafficSpeedStore:
    """Nearest-segment speed lookup with a screening speed-risk index."""

    def __init__(self, records) -> None:
        self._records = [r for r in (_clean_record(x) for x in (records or [])) if r]

    def __len__(self) -> int:
        return len(self._records)

    def all(self) -> list[dict]:
        return [dict(record) for record in self._records]

    def nearest(self, lat, lon, *, max_km: float = DEFAULT_MATCH_RADIUS_KM) -> dict | None:
        hits = geo_math.within_radius_km(lat, lon, self._records, max_km)
        return hits[0] if hits else None

    def lookup(self, lat, lon, *, period: str = "peak", max_km: float = DEFAULT_MATCH_RADIUS_KM) -> dict:
        """Speed context + risk index at a point for a time period.

        ``basis`` is ``"npmrds_segment"`` when a nearby segment matched, else
        ``"unavailable"`` (neutral index, speeds None).
        """
        period = "peak" if str(period).strip().lower() != "offpeak" else "offpeak"
        match = self.nearest(lat, lon, max_km=max_km)
        if match is None:
            return {
                "basis": "unavailable",
                "reference_speed_mph": None,
                "observed_speed_mph": None,
                "period": period,
                "congestion_ratio": None,
                "speed_risk_index": NEUTRAL_RISK,
                "distance_km": None,
                "segment_name": None,
            }
        reference = match["reference_speed_mph"]
        observed = match["peak_speed_mph"] if period == "peak" else match["offpeak_speed_mph"]
        ratio = congestion_ratio(observed, reference)
        return {
            "basis": "npmrds_segment",
            "reference_speed_mph": reference,
            "observed_speed_mph": observed,
            "period": period,
            "congestion_ratio": round(ratio, 4) if ratio is not None else None,
            "speed_risk_index": speed_risk_index(observed, reference),
            "distance_km": match.get("distance_km"),
            "segment_name": match.get("name"),
        }

    @classmethod
    def from_records(cls, records) -> "TrafficSpeedStore":
        return cls(records)

    @classmethod
    def from_json(cls, path) -> "TrafficSpeedStore":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls([])
        records = payload.get("segments") if isinstance(payload, dict) else payload
        return cls(records or [])

    @classmethod
    def from_parquet(cls, path) -> "TrafficSpeedStore":
        import pandas as pd

        try:
            frame = pd.read_parquet(path)
        except Exception:  # noqa: BLE001 - a missing/corrupt index yields an empty store
            return cls([])
        return cls(frame.to_dict(orient="records"))


@functools.lru_cache(maxsize=4)
def _load_store_cached(path_str: str) -> TrafficSpeedStore:
    path = Path(path_str)
    if path.suffix == ".parquet":
        return TrafficSpeedStore.from_parquet(path)
    return TrafficSpeedStore.from_json(path)


def load_traffic_speed_store(path=None) -> TrafficSpeedStore:
    """Process-wide speed store, resolving env/processed/reference in order."""
    if path is not None:
        return _load_store_cached(str(Path(path)))
    env_path = os.getenv(NPMRDS_PATH_ENV)
    if env_path:
        return _load_store_cached(str(Path(env_path)))
    if DEFAULT_NPMRDS_PARQUET.exists():
        return _load_store_cached(str(DEFAULT_NPMRDS_PARQUET))
    return _load_store_cached(str(REFERENCE_NPMRDS_JSON))
