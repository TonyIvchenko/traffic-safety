"""Evacuation-route scoring: rank the safest way out of a location.

Generates candidate destinations away from an origin (a compass fan, or explicit
safe points such as facilities/region exits), scores each origin->destination
route with an injected route scorer (``risk_eval.score_route`` bound to the
model), and ranks them so the least-risky egress is recommended.

Pure and deterministic given the injected ``score_route_fn`` — the module holds
no model or network state. ``score_route_fn(points)`` takes ``[(lon, lat), ...]``
(as :func:`risk_eval.score_route` does) and returns its summary dict.
"""

from __future__ import annotations

import geo_math

# Eight compass headings used to fan out candidate egress directions.
COMPASS_BEARINGS = (
    (0.0, "N"),
    (45.0, "NE"),
    (90.0, "E"),
    (135.0, "SE"),
    (180.0, "S"),
    (225.0, "SW"),
    (270.0, "W"),
    (315.0, "NW"),
)

DEFAULT_EVAC_DISTANCE_KM = 25.0

READINESS_KINDS = ("hospital", "emergency_shelter", "fire_station")


def assess_readiness(advisory_level_index, facility_counts) -> dict:
    """Evacuation-readiness rating from road conditions and facility coverage.

    ``advisory_level_index`` is the region's advisory level (1 Low .. 5 Extreme);
    ``facility_counts`` maps facility kind -> count within the region. "limited"
    when no hospital is mapped or conditions are Extreme; "good" when all three
    kinds are present and conditions are at most Moderate; else "moderate".
    """
    counts = facility_counts if isinstance(facility_counts, dict) else {}

    def _count(kind: str) -> int:
        try:
            return max(0, int(counts.get(kind, 0) or 0))
        except (TypeError, ValueError):
            return 0

    hospitals = _count("hospital")
    shelters = _count("emergency_shelter")
    fire = _count("fire_station")
    kinds_present = sum(1 for kind in READINESS_KINDS if _count(kind) > 0)

    try:
        level = int(advisory_level_index)
    except (TypeError, ValueError):
        level = 1
    level = min(5, max(1, level))

    reasons = []
    if hospitals == 0:
        reasons.append("no hospital mapped in region")
    if shelters == 0:
        reasons.append("no emergency shelter mapped in region")
    if fire == 0:
        reasons.append("no fire station mapped in region")
    if level >= 4:
        reasons.append("elevated road risk for this time")

    if hospitals == 0 or level >= 5:
        rating = "limited"
    elif kinds_present == 3 and level <= 2:
        rating = "good"
    else:
        rating = "moderate"

    return {
        "rating": rating,
        "kinds_present": kinds_present,
        "advisory_level_index": level,
        "reasons": reasons,
    }


def candidate_destinations(lat, lon, *, distance_km: float = DEFAULT_EVAC_DISTANCE_KM, bearings=None) -> list[dict]:
    """Egress destinations ``distance_km`` from ``(lat, lon)`` along each bearing.

    ``bearings`` is an iterable of ``(degrees, label)``; defaults to the 8 compass
    points. Each destination is ``{bearing_deg, compass, lat, lon}``.
    """
    headings = bearings if bearings is not None else COMPASS_BEARINGS
    destinations = []
    for bearing, label in headings:
        dest_lat, dest_lon = geo_math.destination_point(lat, lon, bearing, distance_km)
        destinations.append(
            {
                "bearing_deg": float(bearing),
                "compass": label,
                "lat": round(dest_lat, 6),
                "lon": round(dest_lon, 6),
            }
        )
    return destinations


def _rank_value(route: dict, key: str) -> float:
    value = route.get(key)
    return float(value) if isinstance(value, (int, float)) else float("inf")


def rank_evacuation_routes(
    origin_lat,
    origin_lon,
    destinations,
    score_route_fn,
    *,
    rank_by: str = "mean",
    include_steps: bool = True,
) -> dict:
    """Score each origin->destination route and rank them safest-first.

    ``rank_by`` is ``"mean"`` (default) or ``"max"`` route risk. Routes whose
    scorer returns a non-dict summary are skipped. The safest route is marked
    ``recommended`` and is ``recommended_index`` 0 in the returned (sorted) list.
    """
    key = "route_risk_score_max" if str(rank_by).strip().lower() == "max" else "route_risk_score_mean"
    routes = []
    for destination in destinations or []:
        points = [
            (float(origin_lon), float(origin_lat)),
            (float(destination["lon"]), float(destination["lat"])),
        ]
        summary = score_route_fn(points)
        if not isinstance(summary, dict):
            continue
        route = {
            "destination": destination,
            "distance_km": summary.get("distance_km"),
            "route_risk_score_mean": summary.get("route_risk_score_mean"),
            "route_risk_score_max": summary.get("route_risk_score_max"),
            "route_risk_level": summary.get("route_risk_level"),
            "high_risk_fraction": summary.get("high_risk_fraction"),
            "sample_count": summary.get("sample_count"),
        }
        if include_steps:
            route["steps"] = summary.get("steps")
        routes.append(route)

    routes.sort(key=lambda route: _rank_value(route, key))
    for index, route in enumerate(routes):
        route["rank"] = index + 1
        route["recommended"] = index == 0
    return {
        "origin": {"lat": float(origin_lat), "lon": float(origin_lon)},
        "rank_by": key,
        "count": len(routes),
        "recommended_index": 0 if routes else None,
        "routes": routes,
    }
