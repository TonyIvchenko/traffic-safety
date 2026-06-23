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
from pydantic import BaseModel
import requests

from live_weather import LiveWeatherProviderError

API_VERSION = "1.0"
RISK_LEVELS = ["low", "moderate", "high", "extreme"]
DEFAULT_RISK_THRESHOLDS = {"moderate": 0.10, "high": 0.25, "extreme": 0.45}
UNITS = {
    "temperature": "celsius",
    "wind_speed": "meters_per_second",
    "distance": "kilometers",
    "risk_score": "probability_0_1",
}


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


def _effective_thresholds(risk_quantiles: object) -> dict[str, float]:
    if (
        isinstance(risk_quantiles, (list, tuple))
        and len(risk_quantiles) >= 3
        and all(isinstance(value, (int, float)) for value in risk_quantiles[:3])
    ):
        low_cut, mid_cut, high_cut = (float(value) for value in risk_quantiles[:3])
        return {"moderate": low_cut, "high": mid_cut, "extreme": high_cut}
    return dict(DEFAULT_RISK_THRESHOLDS)


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

    return router
