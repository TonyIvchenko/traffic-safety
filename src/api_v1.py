"""Public, versioned ``/v1`` road-risk API.

The router is built from an injected :class:`V1Dependencies` bundle rather than
importing ``main`` directly, so this module has no dependency cycle with the web
app and stays straightforward to test. ``main`` constructs the dependencies from
its already-loaded model/overlay/provider objects and includes the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import h3
import requests

from live_weather import LiveWeatherProviderError
from segment_support import coords_from_json, densify_polyline, haversine_km, polyline_length_km
import segment_runtime

API_VERSION = "1.0"
RISK_LEVELS = ["low", "moderate", "high", "extreme"]
DEFAULT_RISK_THRESHOLDS = {"moderate": 0.10, "high": 0.25, "extreme": 0.45}
UNITS = {
    "temperature": "celsius",
    "wind_speed": "meters_per_second",
    "distance": "kilometers",
    "risk_score": "probability_0_1",
}

MAX_ROUTE_SAMPLES = 500
MAX_ROUTE_WAYPOINTS = 1000
MIN_SPACING_KM = 0.05
MAX_SPACING_KM = 50.0
HIGH_RISK_LEVELS = {"high", "extreme"}


class WeatherBlock(BaseModel):
    temp_c: float
    dewpoint_c: float
    relative_humidity_pct: float
    wind_speed_mps: float
    wet_hour: float
    summary: str


class PointRisk(BaseModel):
    model_version: str
    cell_id: str
    lat: float
    lon: float
    local_day_of_week: int
    local_hour: int
    month: int
    historical_cell_events: int
    historical_same_hour_events: int
    in_coverage: bool
    confidence: float
    risk_score: float
    risk_level: str
    weather_source: str
    weather: WeatherBlock
    live_provider: str | None = None
    live_provider_label: str | None = None
    target_timestamp_local: str | None = None
    forecast_hours: int | None = None


class HealthResponse(BaseModel):
    service: str
    status: str
    api_version: str
    model_version: str
    model_ready: bool


class HourMark(BaseModel):
    hour_of_week: int
    label: str
    risk_score: float


class WeeklyPoint(BaseModel):
    model_version: str
    cell_id: str
    lat: float
    lon: float
    month: int
    frame_labels: list[str]
    risk_by_hour_of_week: list[float]
    safest: HourMark
    riskiest: HourMark


class RouteRequest(BaseModel):
    waypoints: list[list[float]] | None = None
    geojson: dict | None = None
    mode: str = "climatology"
    day_of_week: int = Field(1, ge=1, le=7)
    hour: int = Field(0, ge=0, le=23)
    month: int = Field(1, ge=1, le=12)
    forecast_hours: int = Field(0, ge=0, le=48)
    provider: str = "auto"
    sample_spacing_km: float = 2.0


class RouteStep(BaseModel):
    lat: float
    lon: float
    distance_km: float
    risk_score: float
    risk_level: str
    cell_id: str


class RouteRisk(BaseModel):
    mode: str
    model_version: str
    distance_km: float
    sample_count: int
    sample_spacing_km: float
    route_risk_score_mean: float
    route_risk_score_max: float
    route_risk_level: str
    high_risk_fraction: float
    riskiest_point: RouteStep
    steps: list[RouteStep]
    live_provider: str | None = None


@dataclass
class V1Dependencies:
    service_name: str
    model_version: str
    model_ready: bool
    coverage: dict
    frame_labels: list[str]
    risk_quantiles: object
    provider_statuses: Callable[[], Sequence]
    provider_choices: list[str]
    rate_limit_per_min: int | None
    predict_point: Callable[..., dict]
    predict_point_live: Callable[..., dict]
    h3_resolution: int


def _effective_thresholds(risk_quantiles: object) -> dict[str, float]:
    if (
        isinstance(risk_quantiles, (list, tuple))
        and len(risk_quantiles) >= 3
        and all(isinstance(value, (int, float)) for value in risk_quantiles[:3])
    ):
        low_cut, mid_cut, high_cut = (float(value) for value in risk_quantiles[:3])
        return {"moderate": low_cut, "high": mid_cut, "extreme": high_cut}
    return dict(DEFAULT_RISK_THRESHOLDS)


def _extract_route_points(body: RouteRequest) -> list[tuple[float, float]]:
    """Return the route as ``[(lon, lat), ...]`` from waypoints or a GeoJSON LineString."""
    raw: object = None
    if body.geojson is not None:
        geometry = body.geojson
        if geometry.get("type") == "Feature":
            geometry = geometry.get("geometry") or {}
        if geometry.get("type") != "LineString":
            raise HTTPException(
                status_code=422,
                detail="geojson must be a LineString (optionally wrapped in a Feature)",
            )
        raw = geometry.get("coordinates")
    elif body.waypoints is not None:
        raw = body.waypoints

    if not isinstance(raw, list) or len(raw) < 2:
        raise HTTPException(
            status_code=422,
            detail="provide at least 2 waypoints as [lon, lat] pairs (or a GeoJSON LineString)",
        )
    if len(raw) > MAX_ROUTE_WAYPOINTS:
        raise HTTPException(status_code=422, detail=f"too many waypoints (> {MAX_ROUTE_WAYPOINTS})")

    points: list[tuple[float, float]] = []
    for pair in raw:
        if not isinstance(pair, (list, tuple)) or len(pair) < 2:
            raise HTTPException(status_code=422, detail="each waypoint must be a [lon, lat] pair")
        lon, lat = float(pair[0]), float(pair[1])
        if not (-180.0 <= lon <= 180.0 and -90.0 <= lat <= 90.0):
            raise HTTPException(status_code=422, detail=f"waypoint out of range: [{lon}, {lat}]")
        points.append((lon, lat))
    return points


def _area_geojson(result: dict) -> dict:
    features = []
    for segment in result.get("segments", []):
        coords = coords_from_json(segment.get("coords_json", "[]"))
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[float(lon), float(lat)] for lon, lat in coords],
                },
                "properties": {
                    "segment_id": segment.get("segment_id"),
                    "name": segment.get("fullname"),
                    "segment_idx": segment.get("segment_idx"),
                    "risk_score": segment.get("risk_score"),
                    "weather_provider": segment.get("weather_provider"),
                    "target_timestamp_local": segment.get("target_timestamp_local"),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "count": result.get("count", len(features)),
        "features": features,
    }


def _route_geojson(result: dict) -> dict:
    steps = result["steps"]
    return {
        "type": "FeatureCollection",
        "features": [
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[step["lon"], step["lat"]] for step in steps],
                },
                "properties": {
                    "mode": result["mode"],
                    "model_version": result["model_version"],
                    "distance_km": result["distance_km"],
                    "route_risk_score_mean": result["route_risk_score_mean"],
                    "route_risk_score_max": result["route_risk_score_max"],
                    "route_risk_level": result["route_risk_level"],
                    "high_risk_fraction": result["high_risk_fraction"],
                    "risk_scores": [step["risk_score"] for step in steps],
                    "risk_levels": [step["risk_level"] for step in steps],
                    "distances_km": [step["distance_km"] for step in steps],
                },
            }
        ],
    }


def build_v1_router(deps: V1Dependencies) -> APIRouter:
    router = APIRouter(prefix="/v1", tags=["v1"])

    def _validate_provider(provider: str) -> None:
        if provider.strip().lower() not in deps.provider_choices:
            raise HTTPException(
                status_code=400,
                detail=(
                    f"unknown live weather provider '{provider}'; "
                    f"choose from {deps.provider_choices}"
                ),
            )

    @router.get("/health", response_model=HealthResponse, summary="API liveness check")
    def health(response: Response) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return {
            "service": deps.service_name,
            "status": "ok",
            "api_version": API_VERSION,
            "model_version": deps.model_version,
            "model_ready": deps.model_ready,
        }

    @router.get("/meta", summary="Discovery: coverage, model, providers, limits")
    def meta(response: Response) -> dict:
        response.headers["Cache-Control"] = "public, max-age=300"
        providers = [
            {
                "name": status.name,
                "label": status.label,
                "paid": status.paid,
                "enabled": status.enabled,
                "configured": status.configured,
                "available": status.available,
            }
            for status in deps.provider_statuses()
        ]
        return {
            "service": deps.service_name,
            "api_version": API_VERSION,
            "model_version": deps.model_version,
            "model_ready": deps.model_ready,
            "coverage": deps.coverage,
            "timeline": {
                "type": "weekly_cycle",
                "frame_count": len(deps.frame_labels),
                "frame_labels": deps.frame_labels,
            },
            "risk_levels": RISK_LEVELS,
            "risk_thresholds": _effective_thresholds(deps.risk_quantiles),
            "live_providers": providers,
            "providers_accepted": deps.provider_choices,
            "units": UNITS,
            "rate_limit": {
                "scope": "per_client_ip",
                "per_minute": deps.rate_limit_per_min,
                "enabled": deps.rate_limit_per_min is not None,
            },
            "docs_url": "/v1/docs",
        }

    @router.get(
        "/risk/point",
        response_model=PointRisk,
        summary="Crash risk at a single point (climatological or live)",
    )
    def risk_point(
        response: Response,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        mode: str = Query("climatology", description="'climatology' or 'live'"),
        day_of_week: int = Query(1, ge=1, le=7, description="Monday=1..Sunday=7 (climatology)"),
        hour: int = Query(0, ge=0, le=23, description="local hour 0-23 (climatology)"),
        month: int = Query(1, ge=1, le=12, description="month 1-12 (climatology)"),
        forecast_hours: int = Query(0, ge=0, le=48, description="hours ahead (live)"),
        provider: str = Query("auto", description="live weather provider (live)"),
    ) -> dict:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"climatology", "live"}:
            raise HTTPException(status_code=422, detail="mode must be 'climatology' or 'live'")

        if mode_norm == "live":
            _validate_provider(provider)
            response.headers["Cache-Control"] = "no-store"
            try:
                return deps.predict_point_live(
                    lat=lat,
                    lon=lon,
                    forecast_hours=forecast_hours,
                    provider=provider,
                )
            except LiveWeatherProviderError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except requests.RequestException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"weather provider request failed: {exc}",
                ) from exc

        response.headers["Cache-Control"] = "public, max-age=3600"
        return deps.predict_point(
            lat=lat,
            lon=lon,
            day_of_week=day_of_week,
            hour=hour,
            month=month,
        )

    @router.get(
        "/risk/point/weekly",
        response_model=WeeklyPoint,
        summary="Weekly 168-hour climatological risk profile for a point",
    )
    def risk_point_weekly(
        response: Response,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        month: int = Query(1, ge=1, le=12),
    ) -> dict:
        response.headers["Cache-Control"] = "public, max-age=3600"
        frame_count = len(deps.frame_labels) or 168
        risks: list[float] = []
        cell_id = ""
        for hour_of_week in range(frame_count):
            prediction = deps.predict_point(
                lat=lat,
                lon=lon,
                day_of_week=hour_of_week // 24 + 1,
                hour=hour_of_week % 24,
                month=month,
            )
            risks.append(float(prediction["risk_score"]))
            cell_id = prediction["cell_id"]

        labels = deps.frame_labels

        def _mark(index: int) -> dict:
            label = labels[index] if index < len(labels) else str(index)
            return {"hour_of_week": index, "label": label, "risk_score": risks[index]}

        return {
            "model_version": deps.model_version,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "month": month,
            "frame_labels": labels,
            "risk_by_hour_of_week": risks,
            "safest": _mark(min(range(len(risks)), key=lambda i: risks[i])),
            "riskiest": _mark(max(range(len(risks)), key=lambda i: risks[i])),
        }

    @router.post(
        "/risk/route",
        response_model=RouteRisk,
        summary="Score crash risk along a driving route (JSON or GeoJSON)",
    )
    def risk_route(
        response: Response,
        body: RouteRequest,
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        points = _extract_route_points(body)
        spacing = float(body.sample_spacing_km)
        if not (MIN_SPACING_KM <= spacing <= MAX_SPACING_KM):
            raise HTTPException(
                status_code=422,
                detail=f"sample_spacing_km must be between {MIN_SPACING_KM} and {MAX_SPACING_KM} km",
            )
        mode_norm = body.mode.strip().lower()
        if mode_norm not in {"climatology", "live"}:
            raise HTTPException(status_code=422, detail="mode must be 'climatology' or 'live'")

        estimated = int(polyline_length_km(points) / spacing) + len(points)
        if estimated > MAX_ROUTE_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=(
                    f"route needs ~{estimated} samples at {spacing} km spacing "
                    f"(max {MAX_ROUTE_SAMPLES}); increase sample_spacing_km"
                ),
            )
        if mode_norm == "live":
            _validate_provider(body.provider)

        dense = densify_polyline(points, max_edge_km=spacing)
        if len(dense) > MAX_ROUTE_SAMPLES:
            raise HTTPException(
                status_code=422,
                detail=f"route produced {len(dense)} samples (max {MAX_ROUTE_SAMPLES})",
            )

        cell_cache: dict[str, dict] = {}
        steps: list[dict] = []
        cumulative = 0.0
        try:
            for index, (lon, lat) in enumerate(dense):
                if index > 0:
                    prev_lon, prev_lat = dense[index - 1]
                    cumulative += haversine_km(prev_lat, prev_lon, lat, lon)
                cell_id = h3.latlng_to_cell(float(lat), float(lon), deps.h3_resolution)
                cached = cell_cache.get(cell_id)
                if cached is None:
                    if mode_norm == "live":
                        cached = deps.predict_point_live(
                            lat=lat,
                            lon=lon,
                            forecast_hours=body.forecast_hours,
                            provider=body.provider,
                        )
                    else:
                        cached = deps.predict_point(
                            lat=lat,
                            lon=lon,
                            day_of_week=body.day_of_week,
                            hour=body.hour,
                            month=body.month,
                        )
                    cell_cache[cell_id] = cached
                steps.append(
                    {
                        "lat": float(lat),
                        "lon": float(lon),
                        "distance_km": round(float(cumulative), 4),
                        "risk_score": float(cached["risk_score"]),
                        "risk_level": cached["risk_level"],
                        "cell_id": cell_id,
                    }
                )
        except LiveWeatherProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"weather provider request failed: {exc}",
            ) from exc

        risk_scores = [step["risk_score"] for step in steps]
        riskiest = max(steps, key=lambda step: step["risk_score"])
        high_risk = sum(1 for step in steps if step["risk_level"] in HIGH_RISK_LEVELS)
        live_provider = None
        if mode_norm == "live" and cell_cache:
            live_provider = next(iter(cell_cache.values())).get("live_provider")

        result = {
            "mode": mode_norm,
            "model_version": deps.model_version,
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

        cache_control = "no-store" if mode_norm == "live" else "public, max-age=3600"
        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_route_geojson(result),
                media_type="application/geo+json",
                headers={"Cache-Control": cache_control},
            )
        response.headers["Cache-Control"] = cache_control
        return result

    @router.get(
        "/risk/area",
        response_model=None,
        summary="Scored road segments within a bounding box (JSON or GeoJSON)",
    )
    def risk_area(
        response: Response,
        min_lat: float = Query(..., ge=-90.0, le=90.0),
        max_lat: float = Query(..., ge=-90.0, le=90.0),
        min_lon: float = Query(..., ge=-180.0, le=180.0),
        max_lon: float = Query(..., ge=-180.0, le=180.0),
        forecast_hours: int = Query(0, ge=0, le=48),
        provider: str = Query("auto"),
        limit: int = Query(1500, ge=1, le=5000),
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        _validate_provider(provider)
        try:
            result = segment_runtime.score_segments_in_bbox(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                forecast_hours=forecast_hours,
                provider=provider,
                limit=limit,
            )
        except LiveWeatherProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"weather provider request failed: {exc}",
            ) from exc

        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_area_geojson(result),
                media_type="application/geo+json",
                headers={"Cache-Control": "no-store"},
            )
        response.headers["Cache-Control"] = "no-store"
        return result

    return router
