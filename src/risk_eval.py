"""Framework-free route risk scoring, shared by the API and the watch evaluator.

``score_route`` densifies a polyline, scores each sample with the injected
point predictors (deduplicating weather lookups by H3 cell), and aggregates a
route-level summary. HTTP concerns (status codes, GeoJSON, glare annotation)
stay in the API layer; provider/network errors propagate to the caller.
"""

from __future__ import annotations

import h3

from segment_support import densify_polyline, haversine_km, polyline_length_km

MAX_ROUTE_SAMPLES = 500
MIN_SPACING_KM = 0.05
MAX_SPACING_KM = 50.0
HIGH_RISK_LEVELS = {"high", "extreme"}


class RouteConfigError(ValueError):
    """Invalid route-scoring parameters (bad mode or spacing)."""


class RouteTooLongError(ValueError):
    """Route would need more samples than allowed at the requested spacing."""

    def __init__(self, estimated: int, max_samples: int, spacing_km: float):
        self.estimated = int(estimated)
        self.max_samples = int(max_samples)
        self.spacing_km = float(spacing_km)
        super().__init__(
            f"route needs ~{estimated} samples at {spacing_km} km spacing "
            f"(max {max_samples}); increase sample_spacing_km"
        )


def score_route(
    points: list[tuple[float, float]],
    *,
    mode: str,
    predict_point,
    predict_point_live,
    h3_resolution: int,
    sample_spacing_km: float = 2.0,
    day_of_week: int = 1,
    hour: int = 0,
    month: int = 1,
    forecast_hours: int = 0,
    provider: str = "auto",
    cell_cache: dict[str, dict] | None = None,
    max_samples: int = MAX_ROUTE_SAMPLES,
) -> dict:
    """Score a route given as ``[(lon, lat), ...]``; returns summary + steps.

    ``cell_cache`` may be shared across calls (e.g. candidate routes for the
    same trip) so overlapping cells reuse one weather lookup.
    """
    spacing = float(sample_spacing_km)
    if not (MIN_SPACING_KM <= spacing <= MAX_SPACING_KM):
        raise RouteConfigError(
            f"sample_spacing_km must be between {MIN_SPACING_KM} and {MAX_SPACING_KM} km"
        )
    mode_norm = str(mode).strip().lower()
    if mode_norm not in {"climatology", "live"}:
        raise RouteConfigError("mode must be 'climatology' or 'live'")

    estimated = int(polyline_length_km(points) / spacing) + len(points)
    if estimated > max_samples:
        raise RouteTooLongError(estimated, max_samples, spacing)

    dense = densify_polyline(points, max_edge_km=spacing)
    if len(dense) > max_samples:
        raise RouteTooLongError(len(dense), max_samples, spacing)

    cache = cell_cache if cell_cache is not None else {}
    steps: list[dict] = []
    cumulative = 0.0
    for index, (lon, lat) in enumerate(dense):
        if index > 0:
            prev_lon, prev_lat = dense[index - 1]
            cumulative += haversine_km(prev_lat, prev_lon, lat, lon)
        cell_id = h3.latlng_to_cell(float(lat), float(lon), h3_resolution)
        cached = cache.get(cell_id)
        if cached is None:
            if mode_norm == "live":
                cached = predict_point_live(
                    lat=lat,
                    lon=lon,
                    forecast_hours=forecast_hours,
                    provider=provider,
                )
            else:
                cached = predict_point(
                    lat=lat,
                    lon=lon,
                    day_of_week=day_of_week,
                    hour=hour,
                    month=month,
                )
            cache[cell_id] = cached
        steps.append(
            {
                "lat": float(lat),
                "lon": float(lon),
                "distance_km": round(float(cumulative), 4),
                "risk_score": float(cached["risk_score"]),
                "risk_level": cached["risk_level"],
                "cell_id": cell_id,
                "hazards": cached.get("hazards"),
            }
        )

    risk_scores = [step["risk_score"] for step in steps]
    riskiest = max(steps, key=lambda step: step["risk_score"])
    high_risk = sum(1 for step in steps if step["risk_level"] in HIGH_RISK_LEVELS)
    live_provider = None
    if mode_norm == "live" and cache:
        live_provider = next(iter(cache.values())).get("live_provider")

    return {
        "mode": mode_norm,
        "distance_km": round(float(cumulative), 4),
        "sample_count": len(steps),
        "sample_spacing_km": spacing,
        "route_risk_score_mean": sum(risk_scores) / len(risk_scores),
        "route_risk_score_max": riskiest["risk_score"],
        "route_risk_level": riskiest["risk_level"],
        "high_risk_fraction": high_risk / len(steps),
        "riskiest_point": riskiest,
        "steps": steps,
        "live_provider": live_provider,
    }
