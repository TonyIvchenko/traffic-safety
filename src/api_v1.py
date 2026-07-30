"""Public, versioned ``/v1`` road-risk API.

The router is built from an injected :class:`V1Dependencies` bundle rather than
importing ``main`` directly, so this module has no dependency cycle with the web
app and stays straightforward to test. ``main`` constructs the dependencies from
its already-loaded model/overlay/provider objects and includes the router.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import math
from typing import Callable, Sequence

from fastapi import APIRouter, HTTPException, Query, Response
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel, Field
import requests

import grant_html
from live_weather import LiveWeatherProviderError
from segment_support import coords_from_json
import risk_eval
import segment_runtime
import sun_glare
from watch_store import public_view

API_VERSION = "1.0"
RISK_LEVELS = ["low", "moderate", "high", "extreme"]
DEFAULT_RISK_THRESHOLDS = {"moderate": 0.10, "high": 0.25, "extreme": 0.45}
UNITS = {
    "temperature": "celsius",
    "wind_speed": "meters_per_second",
    "distance": "kilometers",
    "risk_score": "probability_0_1",
}

MAX_ROUTE_WAYPOINTS = 1000
MAX_COMPARE_ROUTES = 5


class WeatherBlock(BaseModel):
    temp_c: float
    dewpoint_c: float
    relative_humidity_pct: float
    wind_speed_mps: float
    wet_hour: float
    summary: str


class ExplanationFactor(BaseModel):
    factor: str
    contribution: float
    direction: str


class Explanation(BaseModel):
    baseline_risk: float
    risk_score: float
    factors: list[ExplanationFactor]


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
    hazards: dict | None = None
    live_provider: str | None = None
    live_provider_label: str | None = None
    target_timestamp_local: str | None = None
    forecast_hours: int | None = None
    explanation: Explanation | None = None


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
    glare_datetime: str | None = None


class RouteStep(BaseModel):
    lat: float
    lon: float
    distance_km: float
    risk_score: float
    risk_level: str
    cell_id: str
    hazards: dict | None = None
    sun_glare: dict | None = None


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
    glare_segments: int | None = None


class RouteCandidate(BaseModel):
    waypoints: list[list[float]] | None = None
    geojson: dict | None = None
    label: str | None = None


class RouteCompareRequest(BaseModel):
    routes: list[RouteCandidate] = Field(min_length=2, max_length=MAX_COMPARE_ROUTES)
    mode: str = "climatology"
    day_of_week: int = Field(1, ge=1, le=7)
    hour: int = Field(0, ge=0, le=23)
    month: int = Field(1, ge=1, le=12)
    forecast_hours: int = Field(0, ge=0, le=48)
    provider: str = "auto"
    sample_spacing_km: float = 2.0
    objective: str = "mean"


class WatchCreateRequest(BaseModel):
    kind: str
    params: dict = Field(default_factory=dict)
    threshold_level: str = "high"
    channel: str = "poll"
    webhook_url: str | None = None
    cooldown_minutes: int = Field(60, ge=0, le=1440)


class WatchUpdateRequest(BaseModel):
    active: bool


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
    explain_point: Callable[[dict], dict]
    h3_resolution: int
    model_metrics: dict
    model_report_loader: Callable[[], dict]
    risk_cube: object
    watch_store_provider: Callable[[], object]
    grant_provider: Callable[[], object]
    equity_provider: Callable[..., dict]
    equity_overlay_provider: Callable[[], object]


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


def _validate_watch_params(kind: str, params: dict) -> dict:
    """Normalize/validate watch geometry params; raises 422 on bad shapes."""
    params = dict(params or {})
    try:
        if kind == "point":
            lat, lon = float(params["lat"]), float(params["lon"])
            if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
                raise ValueError("lat/lon out of range")
            normalized = {"lat": lat, "lon": lon}
            normalized["forecast_hours"] = int(params.get("forecast_hours", 0))
            if not (0 <= normalized["forecast_hours"] <= 48):
                raise ValueError("forecast_hours must be 0-48")
        elif kind == "route":
            candidate = RouteCandidate(
                waypoints=params.get("waypoints"), geojson=params.get("geojson")
            )
            points = _extract_route_points(candidate)
            spacing = float(params.get("sample_spacing_km", 2.0))
            if not (risk_eval.MIN_SPACING_KM <= spacing <= risk_eval.MAX_SPACING_KM):
                raise ValueError("sample_spacing_km out of range")
            normalized = {
                "waypoints": [[lon, lat] for lon, lat in points],
                "sample_spacing_km": spacing,
                "forecast_hours": int(params.get("forecast_hours", 0)),
            }
        elif kind == "area":
            normalized = {
                key: float(params[key]) for key in ("min_lat", "max_lat", "min_lon", "max_lon")
            }
            if not (
                -90.0 <= normalized["min_lat"] <= normalized["max_lat"] <= 90.0
                and -180.0 <= normalized["min_lon"] <= normalized["max_lon"] <= 180.0
            ):
                raise ValueError("bounding box out of range")
            normalized["limit"] = max(1, min(1000, int(params.get("limit", 100))))
            normalized["forecast_hours"] = int(params.get("forecast_hours", 0))
        else:
            raise ValueError("kind must be 'point', 'route', or 'area'")
    except HTTPException:
        raise
    except (KeyError, TypeError, ValueError) as exc:
        raise HTTPException(status_code=422, detail=f"invalid {kind} watch params: {exc}") from exc

    normalized["provider"] = str(params.get("provider", "auto")).strip().lower()
    return normalized


def _annotate_route_glare(steps: list[dict], glare_datetime: str | None) -> int | None:
    """Attach per-step sun-glare when a UTC datetime is supplied; return the count."""
    if not glare_datetime:
        return None
    try:
        when = sun_glare.parse_utc(glare_datetime)
    except ValueError as exc:
        raise HTTPException(status_code=422, detail=f"invalid glare_datetime: {exc}") from exc

    glare_count = 0
    last_bearing = 0.0
    for index, step in enumerate(steps):
        if index + 1 < len(steps):
            nxt = steps[index + 1]
            last_bearing = sun_glare.bearing_deg(step["lat"], step["lon"], nxt["lat"], nxt["lon"])
        assessment = sun_glare.glare_assessment(step["lat"], step["lon"], when, last_bearing)
        step["sun_glare"] = assessment
        if assessment["glare"]:
            glare_count += 1
    return glare_count


def _sample_risk_grid(cube, coverage: dict, frame_idx: int, bbox, max_cells: int, min_risk: float) -> list[dict]:
    """Sample the climatological risk grid for a frame within a bbox (bounded)."""
    frames = len(cube)
    if frames == 0:
        return []
    frame_idx = max(0, min(frames - 1, int(frame_idx)))
    frame = cube[frame_idx]
    height = len(frame)
    width = len(frame[0]) if height else 0
    if height == 0 or width == 0:
        return []

    lat_min = float(coverage.get("lat_min", -90.0))
    lat_max = float(coverage.get("lat_max", 90.0))
    lon_min = float(coverage.get("lon_min", -180.0))
    lon_max = float(coverage.get("lon_max", 180.0))
    lat_span = (lat_max - lat_min) or 1.0
    lon_span = (lon_max - lon_min) or 1.0
    row_denom = (height - 1) or 1
    col_denom = (width - 1) or 1
    qmin_lat, qmax_lat, qmin_lon, qmax_lon = bbox

    def _clip(value, hi):
        return max(0, min(hi, int(value)))

    # ceil on the min-lat/min-lon (north/left) edge and floor on the other so
    # every sampled cell centre stays within the requested bbox.
    row_top = _clip(math.ceil((lat_max - qmax_lat) / lat_span * row_denom), height - 1)
    row_bottom = _clip(math.floor((lat_max - qmin_lat) / lat_span * row_denom), height - 1)
    col_left = _clip(math.ceil((qmin_lon - lon_min) / lon_span * col_denom), width - 1)
    col_right = _clip(math.floor((qmax_lon - lon_min) / lon_span * col_denom), width - 1)
    if row_top > row_bottom or col_left > col_right:
        return []

    total = (row_bottom - row_top + 1) * (col_right - col_left + 1)
    stride = max(1, int(math.ceil(math.sqrt(total / max_cells)))) if total > max_cells else 1

    cells: list[dict] = []
    for row in range(row_top, row_bottom + 1, stride):
        lat = lat_max - (row / row_denom) * lat_span
        frame_row = frame[row]
        for col in range(col_left, col_right + 1, stride):
            risk = float(frame_row[col])
            if risk < min_risk:
                continue
            lon = lon_min + (col / col_denom) * lon_span
            cells.append({"lat": round(lat, 5), "lon": round(lon, 5), "risk": round(risk, 4)})
    return cells


def _heatmap_geojson(result: dict) -> dict:
    return {
        "type": "FeatureCollection",
        "frame_idx": result["frame_idx"],
        "frame_label": result["frame_label"],
        "features": [
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [cell["lon"], cell["lat"]]},
                "properties": {"risk": cell["risk"]},
            }
            for cell in result["cells"]
        ],
    }


def _compare_geojson(result: dict) -> dict:
    features = []
    for candidate in result["candidates"]:
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "LineString",
                    "coordinates": [[step["lon"], step["lat"]] for step in candidate["steps"]],
                },
                "properties": {
                    "index": candidate["index"],
                    "label": candidate["label"],
                    "rank": candidate["rank"],
                    "recommended": candidate["recommended"],
                    "distance_km": candidate["distance_km"],
                    "route_risk_score_mean": candidate["route_risk_score_mean"],
                    "route_risk_score_max": candidate["route_risk_score_max"],
                    "route_risk_level": candidate["route_risk_level"],
                    "high_risk_fraction": candidate["high_risk_fraction"],
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "objective": result["objective"],
        "recommended_index": result["recommended_index"],
        "features": features,
    }


def _hotspots_geojson(result: dict) -> dict:
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
                    "risk_score": segment.get("risk_score"),
                    "baseline_score": segment.get("baseline_score"),
                    "delta": segment.get("delta"),
                    "weather_provider": segment.get("weather_provider"),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "count": result.get("count", len(features)),
        "rank_by": result.get("rank_by"),
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


def _safe_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _grants_hin_geojson(corridors: list[dict]) -> dict:
    """Point features at HIN corridor centroids (the grant report stores no line
    geometry); all non-coordinate fields become feature properties. Corridors
    that are malformed (non-dict, or missing/non-numeric centroid) are skipped so
    a bad dropped-in file degrades rather than 500s."""
    features = []
    for corridor in corridors:
        if not isinstance(corridor, dict):
            continue
        lat = _safe_float(corridor.get("center_lat"))
        lon = _safe_float(corridor.get("center_lon"))
        if lat is None or lon is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    key: value
                    for key, value in corridor.items()
                    if key not in ("center_lat", "center_lon")
                },
            }
        )
    return {"type": "FeatureCollection", "count": len(features), "features": features}


def _equity_hotspots_geojson(records: list[dict]) -> dict:
    """Point features at equity-hotspot segment centroids. A record with a
    missing/non-numeric centroid is skipped so a bad overlay degrades, not 500s."""
    features = []
    for record in records:
        lat = _safe_float(record.get("center_lat"))
        lon = _safe_float(record.get("center_lon"))
        if lat is None or lon is None:
            continue
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": {
                    key: value
                    for key, value in record.items()
                    if key not in ("center_lat", "center_lon")
                },
            }
        )
    return {"type": "FeatureCollection", "count": len(features), "features": features}


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
            "model_metrics": deps.model_metrics,
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
            "watches": {
                "enabled": True,
                "kinds": ["point", "route", "area"],
                "channels": ["poll", "webhook"],
                "note": (
                    "Watches store the submitted geometry and webhook URL until "
                    "deleted; they are authorized only by the per-watch token "
                    "returned at creation."
                ),
            },
            "grants": {
                "enabled": True,
                "jurisdictions": deps.grant_provider().count(),
                "endpoints": [
                    "/v1/grants/summary",
                    "/v1/grants/hin",
                    "/v1/grants/report",
                ],
                "formats": ["json", "geojson", "html"],
                "note": (
                    "SS4A / HSIP safety-analysis datasets per jurisdiction (High "
                    "Injury Network, systemic risk, crash summary, benefit-cost). "
                    "Query by GEOID (state / county / tract) or bounding box."
                ),
            },
            "docs_url": "/v1/docs",
        }

    @router.get("/model/report", summary="Model evaluation / calibration backtest report")
    def model_report(response: Response) -> dict:
        response.headers["Cache-Control"] = "public, max-age=300"
        return deps.model_report_loader()

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
        explain: bool = Query(False, description="include a factor breakdown of the score"),
    ) -> dict:
        mode_norm = mode.strip().lower()
        if mode_norm not in {"climatology", "live"}:
            raise HTTPException(status_code=422, detail="mode must be 'climatology' or 'live'")

        if mode_norm == "live":
            _validate_provider(provider)
            response.headers["Cache-Control"] = "no-store"
            try:
                result = deps.predict_point_live(
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
        else:
            response.headers["Cache-Control"] = "public, max-age=3600"
            result = deps.predict_point(
                lat=lat,
                lon=lon,
                day_of_week=day_of_week,
                hour=hour,
                month=month,
            )

        if explain:
            result = {**result, "explanation": deps.explain_point(result)}
        return result

    @router.get(
        "/hazards/sun-glare",
        summary="Sun-glare assessment for a heading at a location/time",
    )
    def sun_glare_hazard(
        response: Response,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
        bearing: float = Query(..., ge=0.0, le=360.0, description="travel heading, degrees from North"),
        at: str | None = Query(None, alias="datetime", description="ISO UTC time; defaults to now"),
    ) -> dict:
        response.headers["Cache-Control"] = "no-store"
        try:
            when = sun_glare.parse_utc(at) if at else datetime.now(timezone.utc)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=f"invalid datetime: {exc}") from exc
        assessment = sun_glare.glare_assessment(lat, lon, when, bearing)
        return {
            "lat": lat,
            "lon": lon,
            "bearing": bearing,
            "datetime_utc": when.isoformat(),
            **assessment,
        }

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
        if body.mode.strip().lower() == "live":
            _validate_provider(body.provider)

        try:
            result = risk_eval.score_route(
                points,
                mode=body.mode,
                predict_point=deps.predict_point,
                predict_point_live=deps.predict_point_live,
                h3_resolution=deps.h3_resolution,
                sample_spacing_km=body.sample_spacing_km,
                day_of_week=body.day_of_week,
                hour=body.hour,
                month=body.month,
                forecast_hours=body.forecast_hours,
                provider=body.provider,
            )
        except (risk_eval.RouteConfigError, risk_eval.RouteTooLongError) as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except LiveWeatherProviderError as exc:
            raise HTTPException(status_code=503, detail=str(exc)) from exc
        except requests.RequestException as exc:
            raise HTTPException(
                status_code=502,
                detail=f"weather provider request failed: {exc}",
            ) from exc

        result["model_version"] = deps.model_version
        result["glare_segments"] = _annotate_route_glare(result["steps"], body.glare_datetime)

        cache_control = "no-store" if result["mode"] == "live" else "public, max-age=3600"
        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_route_geojson(result),
                media_type="application/geo+json",
                headers={"Cache-Control": cache_control},
            )
        response.headers["Cache-Control"] = cache_control
        return result

    @router.post(
        "/risk/route/compare",
        response_model=None,
        summary="Score candidate routes for the same trip and recommend the safest",
    )
    def risk_route_compare(
        response: Response,
        body: RouteCompareRequest,
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        objective = body.objective.strip().lower()
        if objective not in {"mean", "max"}:
            raise HTTPException(status_code=422, detail="objective must be 'mean' or 'max'")
        if body.mode.strip().lower() == "live":
            _validate_provider(body.provider)

        # Candidate routes for one trip usually overlap, so share the per-cell
        # prediction cache across all of them (one weather fetch per cell).
        shared_cache: dict[str, dict] = {}
        candidates: list[dict] = []
        for index, candidate in enumerate(body.routes):
            try:
                points = _extract_route_points(candidate)
                scored = risk_eval.score_route(
                    points,
                    mode=body.mode,
                    predict_point=deps.predict_point,
                    predict_point_live=deps.predict_point_live,
                    h3_resolution=deps.h3_resolution,
                    sample_spacing_km=body.sample_spacing_km,
                    day_of_week=body.day_of_week,
                    hour=body.hour,
                    month=body.month,
                    forecast_hours=body.forecast_hours,
                    provider=body.provider,
                    cell_cache=shared_cache,
                )
            except HTTPException as exc:
                raise HTTPException(
                    status_code=exc.status_code, detail=f"route[{index}]: {exc.detail}"
                ) from exc
            except (risk_eval.RouteConfigError, risk_eval.RouteTooLongError) as exc:
                raise HTTPException(status_code=422, detail=f"route[{index}]: {exc}") from exc
            except LiveWeatherProviderError as exc:
                raise HTTPException(status_code=503, detail=str(exc)) from exc
            except requests.RequestException as exc:
                raise HTTPException(
                    status_code=502,
                    detail=f"weather provider request failed: {exc}",
                ) from exc
            candidates.append({"index": index, "label": candidate.label, **scored})

        metric = "route_risk_score_mean" if objective == "mean" else "route_risk_score_max"
        order = sorted(range(len(candidates)), key=lambda i: candidates[i][metric])
        for position, candidate_index in enumerate(order):
            candidates[candidate_index]["rank"] = position + 1
            candidates[candidate_index]["recommended"] = position == 0
        recommended_index = order[0]

        result = {
            "objective": objective,
            "mode": candidates[0]["mode"],
            "model_version": deps.model_version,
            "recommended_index": recommended_index,
            "recommended_label": candidates[recommended_index]["label"],
            "candidates": candidates,
        }
        cache_control = "no-store" if result["mode"] == "live" else "public, max-age=3600"
        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_compare_geojson(result),
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

    @router.get(
        "/hotspots",
        response_model=None,
        summary="Ranked crash-risk hotspots in a bounding box (by risk or risk-vs-typical delta)",
    )
    def hotspots(
        response: Response,
        min_lat: float = Query(..., ge=-90.0, le=90.0),
        max_lat: float = Query(..., ge=-90.0, le=90.0),
        min_lon: float = Query(..., ge=-180.0, le=180.0),
        max_lon: float = Query(..., ge=-180.0, le=180.0),
        forecast_hours: int = Query(0, ge=0, le=48),
        provider: str = Query("auto"),
        top_n: int = Query(50, ge=1, le=500),
        rank_by: str = Query("risk", description="'risk' or 'delta' (risk vs typical)"),
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        _validate_provider(provider)
        rank_norm = rank_by.strip().lower()
        if rank_norm not in {"risk", "delta"}:
            raise HTTPException(status_code=422, detail="rank_by must be 'risk' or 'delta'")
        try:
            result = segment_runtime.rank_hotspots_in_bbox(
                min_lat=min_lat,
                max_lat=max_lat,
                min_lon=min_lon,
                max_lon=max_lon,
                forecast_hours=forecast_hours,
                provider=provider,
                top_n=top_n,
                rank_by=rank_norm,
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
                content=_hotspots_geojson(result),
                media_type="application/geo+json",
                headers={"Cache-Control": "no-store"},
            )
        response.headers["Cache-Control"] = "no-store"
        return result

    @router.get(
        "/heatmap",
        response_model=None,
        summary="Climatological risk heatmap grid for a time window (staging/planning)",
    )
    def heatmap(
        response: Response,
        min_lat: float = Query(..., ge=-90.0, le=90.0),
        max_lat: float = Query(..., ge=-90.0, le=90.0),
        min_lon: float = Query(..., ge=-180.0, le=180.0),
        max_lon: float = Query(..., ge=-180.0, le=180.0),
        day_of_week: int = Query(1, ge=1, le=7, description="Monday=1..Sunday=7"),
        hour: int = Query(0, ge=0, le=23),
        min_risk: float = Query(0.0, ge=0.0, le=1.0, description="only return cells at/above this risk"),
        max_cells: int = Query(2000, ge=1, le=20000),
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        response.headers["Cache-Control"] = "public, max-age=3600"
        frame_idx = (day_of_week - 1) * 24 + hour
        cells = _sample_risk_grid(
            deps.risk_cube,
            deps.coverage,
            frame_idx,
            (min_lat, max_lat, min_lon, max_lon),
            max_cells,
            min_risk,
        )
        frame_label = (
            deps.frame_labels[frame_idx] if frame_idx < len(deps.frame_labels) else str(frame_idx)
        )
        result = {
            "frame_idx": frame_idx,
            "frame_label": frame_label,
            "cell_count": len(cells),
            "cells": cells,
        }
        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_heatmap_geojson(result),
                media_type="application/geo+json",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        return result

    @router.get(
        "/grants/summary",
        response_model=None,
        summary="Headline safety-analysis summary for a jurisdiction (SS4A / HSIP)",
    )
    def grants_summary(
        response: Response,
        geoid: str = Query(
            ..., description="jurisdiction GEOID: state (2-digit), county (5), or tract (11)"
        ),
    ) -> dict:
        response.headers["Cache-Control"] = "public, max-age=3600"
        summary = deps.grant_provider().summary(geoid.strip())
        if summary is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no grant dataset for GEOID '{geoid}'; "
                    "run scripts/build_grant_dataset.py to generate it"
                ),
            )
        return summary

    @router.get(
        "/grants/hin",
        response_model=None,
        summary="High Injury Network corridors for a jurisdiction or bbox (JSON or GeoJSON)",
    )
    def grants_hin(
        response: Response,
        geoid: str | None = Query(None, description="jurisdiction GEOID (takes precedence over bbox)"),
        min_lat: float | None = Query(None, ge=-90.0, le=90.0),
        max_lat: float | None = Query(None, ge=-90.0, le=90.0),
        min_lon: float | None = Query(None, ge=-180.0, le=180.0),
        max_lon: float | None = Query(None, ge=-180.0, le=180.0),
        top_n: int = Query(100, ge=1, le=1000),
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        response.headers["Cache-Control"] = "public, max-age=3600"
        store = deps.grant_provider()
        bbox_values = (min_lat, max_lat, min_lon, max_lon)

        if geoid:
            corridors = store.hin_corridors(geoid.strip())
            if corridors is None:
                raise HTTPException(
                    status_code=404,
                    detail=(
                        f"no grant dataset for GEOID '{geoid}'; "
                        "run scripts/build_grant_dataset.py to generate it"
                    ),
                )
            corridors = corridors[:top_n]
            scope = {"geoid": geoid.strip()}
        elif any(value is not None for value in bbox_values):
            if any(value is None for value in bbox_values):
                raise HTTPException(
                    status_code=422,
                    detail="bbox requires all of min_lat, max_lat, min_lon, max_lon",
                )
            if min_lat > max_lat or min_lon > max_lon:
                raise HTTPException(status_code=422, detail="bbox min must be <= max")
            corridors = store.hin_corridors_in_bbox(bbox_values, top_n=top_n)
            scope = {"bbox": [min_lat, max_lat, min_lon, max_lon]}
        else:
            raise HTTPException(
                status_code=422,
                detail="provide a geoid or a bounding box (min_lat, max_lat, min_lon, max_lon)",
            )

        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_grants_hin_geojson(corridors),
                media_type="application/geo+json",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        return {**scope, "count": len(corridors), "corridors": corridors}

    @router.get(
        "/grants/report",
        response_model=None,
        summary="Full downloadable safety-analysis report for a jurisdiction (JSON or HTML)",
    )
    def grants_report(
        response: Response,
        geoid: str = Query(..., description="jurisdiction GEOID: state (2), county (5), or tract (11)"),
        output_format: str = Query("json", alias="format", description="'json' or 'html'"),
    ):
        report = deps.grant_provider().get_report(geoid.strip())
        if report is None:
            raise HTTPException(
                status_code=404,
                detail=(
                    f"no grant dataset for GEOID '{geoid}'; "
                    "run scripts/build_grant_dataset.py to generate it"
                ),
            )
        if output_format.strip().lower() == "html":
            # geoid reached here only via valid_geoid (digits) -> safe filename.
            filename = f"safety-analysis-{geoid.strip()}.html"
            return HTMLResponse(
                content=grant_html.render_report(report),
                headers={
                    "Cache-Control": "public, max-age=3600",
                    "Content-Disposition": f'attachment; filename="{filename}"',
                },
            )
        response.headers["Cache-Control"] = "public, max-age=3600"
        return report

    @router.get(
        "/equity/point",
        response_model=None,
        summary="Tract equity (SVI + Justice40 disadvantaged) at a location",
    )
    def equity_point(
        response: Response,
        lat: float = Query(..., ge=-90.0, le=90.0),
        lon: float = Query(..., ge=-180.0, le=180.0),
    ) -> dict:
        response.headers["Cache-Control"] = "public, max-age=3600"
        return {"lat": lat, "lon": lon, **deps.equity_provider(lat, lon)}

    @router.get(
        "/equity/hotspots",
        response_model=None,
        summary="High-risk segments in disadvantaged / high-SVI tracts (JSON or GeoJSON)",
    )
    def equity_hotspots(
        response: Response,
        min_lat: float | None = Query(None, ge=-90.0, le=90.0),
        max_lat: float | None = Query(None, ge=-90.0, le=90.0),
        min_lon: float | None = Query(None, ge=-180.0, le=180.0),
        max_lon: float | None = Query(None, ge=-180.0, le=180.0),
        top_n: int = Query(50, ge=1, le=500),
        min_risk: float = Query(0.0, ge=0.0, le=1.0),
        only_disadvantaged: bool = Query(False, description="restrict to Justice40 tracts"),
        min_svi: float | None = Query(None, ge=0.0, le=1.0, description="minimum SVI percentile"),
        rank_by: str = Query("priority", description="'priority' (equity-weighted) or 'risk'"),
        output_format: str = Query("json", alias="format", description="'json' or 'geojson'"),
    ):
        response.headers["Cache-Control"] = "public, max-age=3600"
        bbox_values = (min_lat, max_lat, min_lon, max_lon)
        bbox = None
        if any(value is not None for value in bbox_values):
            if any(value is None for value in bbox_values):
                raise HTTPException(
                    status_code=422,
                    detail="bbox requires all of min_lat, max_lat, min_lon, max_lon",
                )
            if min_lat > max_lat or min_lon > max_lon:
                raise HTTPException(status_code=422, detail="bbox min must be <= max")
            bbox = bbox_values

        records = deps.equity_overlay_provider().hotspots(
            bbox=bbox,
            top_n=top_n,
            min_risk=min_risk,
            only_disadvantaged=only_disadvantaged,
            min_svi=min_svi,
            rank_by=rank_by,
        )
        if output_format.strip().lower() == "geojson":
            return JSONResponse(
                content=_equity_hotspots_geojson(records),
                media_type="application/geo+json",
                headers={"Cache-Control": "public, max-age=3600"},
            )
        return {"count": len(records), "hotspots": records}

    def _authorized_watch(watch_id: str, token: str) -> dict:
        store = deps.watch_store_provider()
        record = store.get_watch(watch_id)
        if record is None:
            raise HTTPException(status_code=404, detail="watch not found")
        if store.get_watch_authorized(watch_id, token) is None:
            raise HTTPException(status_code=403, detail="invalid watch token")
        return record

    @router.post(
        "/watches",
        status_code=201,
        summary="Create a risk watch (point, route, or area); token returned once",
    )
    def create_watch(response: Response, body: WatchCreateRequest) -> dict:
        response.headers["Cache-Control"] = "no-store"
        kind = body.kind.strip().lower()
        params = _validate_watch_params(kind, body.params)
        _validate_provider(params["provider"])
        try:
            record = deps.watch_store_provider().create_watch(
                kind=kind,
                params=params,
                threshold_level=body.threshold_level,
                channel=body.channel,
                webhook_url=body.webhook_url,
                cooldown_minutes=body.cooldown_minutes,
            )
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        return public_view(record, include_secrets=True)

    @router.get("/watches/{watch_id}", summary="Watch status (the polling channel)")
    def get_watch(response: Response, watch_id: str, token: str = Query(...)) -> dict:
        response.headers["Cache-Control"] = "no-store"
        return public_view(_authorized_watch(watch_id, token))

    @router.patch("/watches/{watch_id}", summary="Pause or resume a watch")
    def update_watch(
        response: Response, watch_id: str, body: WatchUpdateRequest, token: str = Query(...)
    ) -> dict:
        response.headers["Cache-Control"] = "no-store"
        _authorized_watch(watch_id, token)
        record = deps.watch_store_provider().set_active(watch_id, token, body.active)
        return public_view(record)

    @router.delete("/watches/{watch_id}", summary="Delete a watch")
    def delete_watch(response: Response, watch_id: str, token: str = Query(...)) -> dict:
        response.headers["Cache-Control"] = "no-store"
        _authorized_watch(watch_id, token)
        deps.watch_store_provider().delete_watch(watch_id, token)
        return {"deleted": True, "id": watch_id}

    return router
