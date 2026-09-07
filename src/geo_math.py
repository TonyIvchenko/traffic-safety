"""Small great-circle helpers for emergency/evacuation geography.

Builds on :func:`segment_support.haversine_km` (great-circle distance) with the
projection and nearest-neighbour utilities the emergency endpoints need:
``destination_point`` (project along a bearing, for candidate evacuation
headings), ``bbox_around`` (a coarse spatial pre-filter), and ``nearest_k`` /
``within_radius_km`` (facility lookup). Pure and defensive: malformed points are
skipped rather than raising.
"""

from __future__ import annotations

import math

from segment_support import EARTH_RADIUS_KM, haversine_km

__all__ = [
    "EARTH_RADIUS_KM",
    "haversine_km",
    "destination_point",
    "bbox_around",
    "nearest_k",
    "within_radius_km",
]

# Approximate degrees-per-km near the surface (for the bbox pre-filter only).
_KM_PER_DEG_LAT = 110.574
_KM_PER_DEG_LON_EQ = 111.320


def destination_point(lat: float, lon: float, bearing_deg: float, distance_km: float) -> tuple[float, float]:
    """The point reached from ``(lat, lon)`` after ``distance_km`` along a bearing.

    ``bearing_deg`` is degrees clockwise from north. Returns ``(lat, lon)`` with
    longitude normalised to ``[-180, 180]``.
    """
    angular = float(distance_km) / EARTH_RADIUS_KM
    bearing = math.radians(float(bearing_deg))
    phi1 = math.radians(float(lat))
    lambda1 = math.radians(float(lon))

    phi2 = math.asin(
        math.sin(phi1) * math.cos(angular)
        + math.cos(phi1) * math.sin(angular) * math.cos(bearing)
    )
    lambda2 = lambda1 + math.atan2(
        math.sin(bearing) * math.sin(angular) * math.cos(phi1),
        math.cos(angular) - math.sin(phi1) * math.sin(phi2),
    )
    out_lat = math.degrees(phi2)
    out_lon = (math.degrees(lambda2) + 540.0) % 360.0 - 180.0
    return (out_lat, out_lon)


def bbox_around(lat: float, lon: float, radius_km: float) -> tuple[float, float, float, float]:
    """A ``(min_lat, max_lat, min_lon, max_lon)`` box enclosing a radius circle.

    A coarse over-approximation (clamped to valid lat/lon) for pre-filtering; use
    :func:`haversine_km` for the exact test.
    """
    radius = max(0.0, float(radius_km))
    dlat = radius / _KM_PER_DEG_LAT
    cos_lat = math.cos(math.radians(float(lat)))
    dlon = radius / (_KM_PER_DEG_LON_EQ * cos_lat) if abs(cos_lat) > 1e-9 else 180.0
    return (
        max(-90.0, float(lat) - dlat),
        min(90.0, float(lat) + dlat),
        max(-180.0, float(lon) - dlon),
        min(180.0, float(lon) + dlon),
    )


def _coord(point) -> tuple[float, float] | None:
    """Extract ``(lat, lon)`` from a ``{"lat","lon"}`` dict or a ``(lat, lon)`` pair."""
    if isinstance(point, dict):
        lat, lon = point.get("lat"), point.get("lon")
    elif isinstance(point, (list, tuple)) and len(point) >= 2:
        lat, lon = point[0], point[1]
    else:
        return None
    try:
        lat_f, lon_f = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    if math.isnan(lat_f) or math.isnan(lon_f):
        return None
    return (lat_f, lon_f)


def _annotated(point, coord: tuple[float, float], distance_km: float) -> dict:
    item = dict(point) if isinstance(point, dict) else {"lat": coord[0], "lon": coord[1]}
    item["distance_km"] = round(distance_km, 4)
    return item


def nearest_k(lat: float, lon: float, points, k: int = 1) -> list[dict]:
    """The ``k`` nearest points to ``(lat, lon)``, each annotated with ``distance_km``.

    Points may be ``{"lat","lon", ...}`` dicts (extra keys are preserved) or
    ``(lat, lon)`` pairs. Malformed points are skipped. Sorted nearest first.
    """
    scored = []
    for point in points or []:
        coord = _coord(point)
        if coord is None:
            continue
        scored.append((haversine_km(lat, lon, coord[0], coord[1]), coord, point))
    scored.sort(key=lambda triple: triple[0])
    limit = max(0, int(k))
    return [_annotated(point, coord, distance) for distance, coord, point in scored[:limit]]


def within_radius_km(lat: float, lon: float, points, radius_km: float) -> list[dict]:
    """All points within ``radius_km`` of ``(lat, lon)``, annotated and nearest first."""
    radius = max(0.0, float(radius_km))
    hits = []
    for point in points or []:
        coord = _coord(point)
        if coord is None:
            continue
        distance = haversine_km(lat, lon, coord[0], coord[1])
        if distance <= radius:
            hits.append(_annotated(point, coord, distance))
    hits.sort(key=lambda item: item["distance_km"])
    return hits
