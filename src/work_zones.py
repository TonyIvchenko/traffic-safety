"""Work-zone hazard layer from WZDx (Work Zone Data Exchange) feeds.

Construction and maintenance zones raise crash risk (lane shifts, narrowed
lanes, sudden stops). WZDx is USDOT's open spec for publishing active work zones
as GeoJSON ``RoadEvent`` feature collections. This parses such a feed (v4 nested
``core_details`` or an older flat layout), filters to events active at a given
time, and answers point queries with a screening risk multiplier.

The feed is loaded from ``TRAFFIC_SAFETY_WZDX_PATH`` or the curated reference
``data/reference/work_zones.json``. Pure and defensive: malformed features and
feeds degrade to empty/neutral rather than raising. WZDx coverage is uneven, so a
"no work zone" answer means "none in this feed", not "none on the ground".
"""

from __future__ import annotations

from datetime import datetime, timezone
import functools
import json
import os
from pathlib import Path

import geo_math

WZDX_PATH_ENV = "TRAFFIC_SAFETY_WZDX_PATH"
REPO_DIR = Path(__file__).resolve().parents[1]
REFERENCE_WZDX_JSON = REPO_DIR / "data" / "reference" / "work_zones.json"
DEFAULT_RADIUS_KM = 1.5

# Screening crash-risk multiplier by WZDx vehicle_impact (how much of the roadway
# is disrupted). Order-of-magnitude judgement, not a calibrated CMF.
IMPACT_RISK = {
    "all-lanes-closed": 1.5,
    "alternating-one-way": 1.4,
    "some-lanes-closed": 1.3,
    "all-lanes-open-shift-left": 1.15,
    "all-lanes-open-shift-right": 1.15,
    "all-lanes-open": 1.1,
    "unknown": 1.2,
}
DEFAULT_IMPACT_RISK = 1.2


def _parse_dt(value):
    if not value:
        return None
    try:
        text = str(value).strip().replace("Z", "+00:00")
        parsed = datetime.fromisoformat(text)
    except (ValueError, TypeError):
        return None
    return parsed if parsed.tzinfo else parsed.replace(tzinfo=timezone.utc)


def _flatten_coords(coords) -> list[tuple[float, float]]:
    """Flatten GeoJSON coordinates (Point/LineString/MultiLineString) to (lon, lat)."""
    if not isinstance(coords, (list, tuple)) or not coords:
        return []
    if isinstance(coords[0], (int, float)) and len(coords) >= 2:
        return [(coords[0], coords[1])]
    points: list[tuple[float, float]] = []
    for item in coords:
        points.extend(_flatten_coords(item))
    return points


def _representative_point(geometry):
    coords = geometry.get("coordinates") if isinstance(geometry, dict) else None
    points = _flatten_coords(coords)
    if not points:
        return None
    lon, lat = points[len(points) // 2]  # midpoint of the event geometry
    try:
        return float(lat), float(lon)
    except (TypeError, ValueError):
        return None


def _normalize(feature) -> dict | None:
    if not isinstance(feature, dict):
        return None
    props = feature.get("properties") or {}
    # WZDx v4 nests descriptive fields under core_details; older feeds are flat.
    core = props.get("core_details") if isinstance(props.get("core_details"), dict) else props
    point = _representative_point(feature.get("geometry"))
    if point is None:
        return None
    lat, lon = point

    road_names = core.get("road_names") or props.get("road_names") or []
    if isinstance(road_names, str):
        road_names = [road_names]
    impact = str(props.get("vehicle_impact") or core.get("vehicle_impact") or "unknown").strip().lower()
    start = _parse_dt(props.get("start_date") or core.get("start_date"))
    end = _parse_dt(props.get("end_date") or core.get("end_date"))
    return {
        "id": str(feature.get("id") or props.get("id") or core.get("data_source_id") or ""),
        "lat": lat,
        "lon": lon,
        "road_names": [str(name) for name in road_names],
        "direction": core.get("direction") or props.get("direction"),
        "vehicle_impact": impact,
        "description": core.get("description") or props.get("description"),
        "start_date": start.isoformat() if start else None,
        "end_date": end.isoformat() if end else None,
        "_start": start,
        "_end": end,
    }


def _is_active(record: dict, when: datetime | None) -> bool:
    if when is None:
        return True
    start, end = record["_start"], record["_end"]
    if start is not None and when < start:
        return False
    if end is not None and when > end:
        return False
    return True


def _public(record: dict) -> dict:
    return {key: value for key, value in record.items() if not key.startswith("_")}


class WorkZoneStore:
    """Active-work-zone queries over a parsed WZDx feed."""

    def __init__(self, features) -> None:
        self._records = [r for r in (_normalize(f) for f in (features or [])) if r]

    def __len__(self) -> int:
        return len(self._records)

    def all(self, *, when: datetime | None = None) -> list[dict]:
        return [_public(r) for r in self._records if _is_active(r, when)]

    def near(self, lat, lon, radius_km: float = DEFAULT_RADIUS_KM, *, when=None) -> list[dict]:
        active = [r for r in self._records if _is_active(r, when)]
        hits = geo_math.within_radius_km(lat, lon, active, radius_km)
        return [_public(hit) for hit in hits]

    def point_impact(self, lat, lon, *, when=None, radius_km: float = DEFAULT_RADIUS_KM) -> dict:
        """Whether a point sits in/near an active work zone, with a risk multiplier.

        ``risk_multiplier`` is 1.0 (no effect) when nothing is nearby, else the
        worst nearby impact's multiplier — a screening factor for layering on top
        of the base crash risk.
        """
        nearby = self.near(lat, lon, radius_km, when=when)
        if not nearby:
            return {"in_work_zone": False, "count": 0, "nearest": None, "risk_multiplier": 1.0}
        multiplier = max(
            IMPACT_RISK.get(zone["vehicle_impact"], DEFAULT_IMPACT_RISK) for zone in nearby
        )
        return {
            "in_work_zone": True,
            "count": len(nearby),
            "nearest": nearby[0],
            "risk_multiplier": round(float(multiplier), 3),
        }

    @classmethod
    def from_records(cls, features) -> "WorkZoneStore":
        return cls(features)

    @classmethod
    def from_feature_collection(cls, payload) -> "WorkZoneStore":
        if isinstance(payload, dict):
            features = payload.get("features")
        elif isinstance(payload, list):
            features = payload
        else:
            features = None
        return cls(features or [])

    @classmethod
    def from_geojson(cls, path) -> "WorkZoneStore":
        path = Path(path)
        try:
            payload = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            return cls([])
        return cls.from_feature_collection(payload)


@functools.lru_cache(maxsize=4)
def _load_store_cached(path_str: str) -> WorkZoneStore:
    return WorkZoneStore.from_geojson(Path(path_str))


def load_work_zone_store(path=None) -> WorkZoneStore:
    """Process-wide work-zone store; ``TRAFFIC_SAFETY_WZDX_PATH`` overrides the path."""
    if path is not None:
        return _load_store_cached(str(Path(path)))
    env_path = os.getenv(WZDX_PATH_ENV)
    if env_path:
        return _load_store_cached(str(Path(env_path)))
    return _load_store_cached(str(REFERENCE_WZDX_JSON))


def utc_now() -> datetime:
    return datetime.now(timezone.utc)
