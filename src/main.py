"""Standalone traffic-safety app with nationwide weekly risk overlay and predictor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
import html
import io
import json
import joblib
import os
from pathlib import Path
import smtplib
import sqlite3
import sys

from fastapi import Body, FastAPI, HTTPException, Request, Response
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
import h3
import numpy as np
import pandas as pd
from PIL import Image
import requests
import uvicorn


HOST = os.getenv("API_HOST", "0.0.0.0")
PORT = int(os.getenv("PORT", "8080"))
GMAPS_API_KEY = os.getenv("GMAPS_API_KEY", "")
GA_MEASUREMENT_ID = os.getenv("GA_MEASUREMENT_ID", "").strip()
SERVICE_NAME = os.getenv("SERVICE_NAME", "Traffic Safety")
CONTACT_EMAIL = os.getenv("CONTACT_EMAIL", "").strip()
CONTACT_DISPLAY_EMAIL = os.getenv("CONTACT_DISPLAY_EMAIL", CONTACT_EMAIL or "Set CONTACT_EMAIL to publish your inbox")
SMTP_HOST = os.getenv("SMTP_HOST", "").strip()
SMTP_PORT = int(os.getenv("SMTP_PORT", "587"))
SMTP_USERNAME = os.getenv("SMTP_USERNAME", "").strip()
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "").strip()
SMTP_FROM = os.getenv("SMTP_FROM", SMTP_USERNAME or CONTACT_EMAIL or "no-reply@roadriskmonitor.local").strip()
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "1").strip().lower() not in {"0", "false", "no", "off"}

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from live_weather import (
    LiveWeatherProviderError,
    fetch_live_weather,
    provider_statuses,
)
from model_support import build_feature_matrix, lookup_weather_climatology
from road_tiles import (
    ROAD_TILE_DB_PATH,
    ROAD_RASTER_TILE_DB_PATH,
    ROAD_TILE_BASELINE_PATH,
    ROAD_TILE_FORECAST_PATH,
    load_road_tile_meta,
    load_raster_tile_png,
    load_tile_payload,
    raster_tile_assets_ready,
    road_tile_assets_ready,
)
from segment_runtime import load_segment_runtime, score_segments_in_bbox
from scripts.common import ROAD_SEGMENTS_PATH

STATIC_DIR = SRC_DIR / "static"
STATIC_URL = "/traffic-safety-static"
MODEL_PATH = REPO_DIR / "models" / "traffic_safety.joblib"
TILES_DIR = REPO_DIR / "tiles"
CONTACT_SUBMISSIONS_PATH = REPO_DIR / "data" / "contact_submissions.jsonl"
STATIC_ASSET_VERSION = max(
    (STATIC_DIR / "map.css").stat().st_mtime_ns,
    (STATIC_DIR / "map.js").stat().st_mtime_ns,
)

MAP_ASSET_HREFS = {
    "css": f"/traffic-safety-static/map.css?v={STATIC_ASSET_VERSION}",
    "js": f"/traffic-safety-static/map.js?v={STATIC_ASSET_VERSION}",
}

TILE_SIZE = 256
SAMPLE_SIZE = 64


def print_http_startup(service_name: str, host: str, port: int) -> None:
    from scripts.service_startup import print_http_service_startup

    print_http_service_startup(service_name, host, port)


def _weekly_frame_labels() -> list[str]:
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return [f"{weekdays[idx // 24]} {idx % 24:02d}:00" for idx in range(24 * 7)]


def _load_joblib_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = joblib.load(path)
    return loaded if isinstance(loaded, dict) else {}


def _load_overlay(
    cube_path: Path,
    config_path: Path,
    default_config: dict[str, float | int | str],
    default_shape: tuple[int, int],
) -> dict[str, object]:
    config = dict(default_config)
    if config_path.exists():
        config.update(json.loads(config_path.read_text(encoding="utf-8")))

    if cube_path.exists():
        cube = np.load(cube_path)
        risk = cube["risk"].astype(np.float32)
        if "confidence" in cube:
            confidence = cube["confidence"].astype(np.float32)
        elif "activity" in cube:
            confidence = cube["activity"].astype(np.float32)
        else:
            confidence = np.zeros_like(risk, dtype=np.float32)
        frames = [str(value) for value in cube["frames"].tolist()]
    else:
        frames = _weekly_frame_labels()
        risk = np.zeros((len(frames), default_shape[0], default_shape[1]), dtype=np.float32)
        confidence = np.zeros_like(risk)

    return {
        "risk": risk,
        "confidence": confidence,
        "frames": frames,
        "config": config,
    }


OVERLAY = _load_overlay(
    cube_path=TILES_DIR / "overlay.npz",
    config_path=TILES_DIR / "overlay.json",
    default_config={
        "timeline_type": "weekly_cycle",
        "month": 1,
        "zoom_min": 3,
        "zoom_max": 9,
        "center_lat": 39.5,
        "center_lon": -98.35,
        "lat_min": 18.0,
        "lat_max": 72.0,
        "lon_min": -179.0,
        "lon_max": -66.0,
        "model_version": "missing",
    },
    default_shape=(360, 760),
)
MODEL_BUNDLE = _load_joblib_bundle(MODEL_PATH)
MODEL_VERSION = str(
    MODEL_BUNDLE.get("model_version", OVERLAY["config"].get("model_version", "missing"))
)
SEGMENT_MODEL_PATH = REPO_DIR / "models" / "traffic_safety_segments.joblib"
CELL_INDEX = {
    str(cell): idx for idx, cell in enumerate(MODEL_BUNDLE.get("candidate_cells", []))
}
LIVE_PROVIDER_CHOICES = ["auto", *[status.name for status in provider_statuses()]]


def _first_metric(bundle: dict[str, object], keys: list[str]) -> float | None:
    for key in keys:
        value = bundle.get(key)
        if isinstance(value, (int, float)):
            return float(value)
    nested_metrics = bundle.get("metrics")
    if isinstance(nested_metrics, dict):
        for key in keys:
            value = nested_metrics.get(key)
            if isinstance(value, (int, float)):
                return float(value)
    return None


def _timeline() -> dict[str, object]:
    frame_count = len(OVERLAY["frames"])
    month_value = int(OVERLAY["config"].get("month", 1))
    month_labels = [
        "",
        "January",
        "February",
        "March",
        "April",
        "May",
        "June",
        "July",
        "August",
        "September",
        "October",
        "November",
        "December",
    ]
    month_label = month_labels[month_value] if 1 <= month_value <= 12 else "Current"
    weekdays = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
    return {
        "type": "weekly_cycle",
        "step_pct": 100.0 / max(1, frame_count - 1),
        "ticks": [
            {"label": weekday, "frame_idx": idx * 24}
            for idx, weekday in enumerate(weekdays)
        ],
        "phases": [
            {
                "kind": "live",
                "label": f"{month_label} weekly pattern",
                "count": max(1, frame_count),
            }
        ],
    }


def _road_frame_labels(frame_count: int = 24) -> list[str]:
    return [f"+{offset}h" for offset in range(max(1, int(frame_count)))]


def _road_timeline(frame_count: int = 24) -> dict[str, object]:
    frame_count = max(1, int(frame_count))
    return {
        "type": "forecast_24h",
        "step_pct": 100.0 / max(1.0, float(frame_count - 1)),
        "ticks": [
            {"label": f"+{idx}h", "frame_idx": idx}
            for idx in sorted(set([0, 6, 12, 18, frame_count - 1]))
        ],
        "phases": [{"kind": "live", "label": "Forecast next 24 hours", "count": frame_count}],
    }


def _risk_level(probability: float) -> str:
    quantiles = MODEL_BUNDLE.get("risk_quantiles")
    if (
        isinstance(quantiles, list)
        and len(quantiles) >= 3
        and all(isinstance(value, (int, float)) for value in quantiles[:3])
    ):
        low_cut, mid_cut, high_cut = (float(value) for value in quantiles[:3])
        if probability < low_cut:
            return "low"
        if probability < mid_cut:
            return "moderate"
        if probability < high_cut:
            return "high"
        return "extreme"
    if probability < 0.10:
        return "low"
    if probability < 0.25:
        return "moderate"
    if probability < 0.45:
        return "high"
    return "extreme"


def _tile_bounds(z: int, x: int, y: int) -> tuple[float, float, float, float]:
    n = 2**z
    lon_left = x / n * 360.0 - 180.0
    lon_right = (x + 1) / n * 360.0 - 180.0
    lat_top = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * y / n))))
    lat_bottom = np.degrees(np.arctan(np.sinh(np.pi * (1.0 - 2.0 * (y + 1) / n))))
    return float(lat_top), float(lat_bottom), float(lon_left), float(lon_right)


def _sample_layer(
    layer_grid: np.ndarray,
    config: dict[str, float | int | str],
    z: int,
    x: int,
    y: int,
) -> tuple[np.ndarray, np.ndarray]:
    lat_top, lat_bottom, lon_left, lon_right = _tile_bounds(z, x, y)
    lat_min = float(config["lat_min"])
    lat_max = float(config["lat_max"])
    lon_min = float(config["lon_min"])
    lon_max = float(config["lon_max"])

    if (
        lat_top < lat_min
        or lat_bottom > lat_max
        or lon_right < lon_min
        or lon_left > lon_max
    ):
        return np.zeros((SAMPLE_SIZE, SAMPLE_SIZE), dtype=np.float32), np.zeros(
            (SAMPLE_SIZE, SAMPLE_SIZE), dtype=bool
        )

    row_lats = np.linspace(lat_top, lat_bottom, SAMPLE_SIZE, endpoint=False) + (
        lat_bottom - lat_top
    ) / (2.0 * SAMPLE_SIZE)
    col_lons = np.linspace(lon_left, lon_right, SAMPLE_SIZE, endpoint=False) + (
        lon_right - lon_left
    ) / (2.0 * SAMPLE_SIZE)

    valid_rows = (row_lats >= lat_min) & (row_lats <= lat_max)
    valid_cols = (col_lons >= lon_min) & (col_lons <= lon_max)
    valid_mask = np.outer(valid_rows, valid_cols)

    row_lats_clamped = np.clip(row_lats, lat_min, lat_max)
    col_lons_clamped = np.clip(col_lons, lon_min, lon_max)

    height, width = layer_grid.shape
    iy = ((lat_max - row_lats_clamped) / (lat_max - lat_min) * (height - 1)).astype(
        np.int32
    )
    ix = ((col_lons_clamped - lon_min) / (lon_max - lon_min) * (width - 1)).astype(
        np.int32
    )

    sampled = layer_grid[iy[:, None], ix[None, :]].astype(np.float32)
    sampled[~valid_mask] = 0.0
    return sampled, valid_mask


def _colorize(
    sampled_risk: np.ndarray, sampled_conf: np.ndarray, valid_mask: np.ndarray
) -> np.ndarray:
    rgba = np.zeros((SAMPLE_SIZE, SAMPLE_SIZE, 4), dtype=np.uint8)
    low = sampled_risk < 0.33
    mid = (sampled_risk >= 0.33) & (sampled_risk < 0.66)
    high = sampled_risk >= 0.66

    rgba[low, 0:3] = np.array([46, 204, 113], dtype=np.uint8)
    rgba[mid, 0:3] = np.array([241, 196, 15], dtype=np.uint8)
    rgba[high, 0:3] = np.array([231, 76, 60], dtype=np.uint8)

    conf = np.clip(sampled_conf, 0.0, 1.0)
    impact = np.clip((sampled_risk - 0.08) / 0.92, 0.0, 1.0)
    rgba[..., 3] = np.clip(conf * impact * 255.0, 0, 255).astype(np.uint8)
    rgba[~valid_mask, 3] = 0
    return rgba


@lru_cache(maxsize=40000)
def _render_tile_png(frame_idx: int, z: int, x: int, y: int) -> bytes:
    frames = OVERLAY["frames"]
    if frame_idx < 0 or frame_idx >= len(frames):
        raise ValueError("frame index out of range")

    risk_grid = OVERLAY["risk"][frame_idx]
    conf_grid = np.clip(OVERLAY["confidence"][frame_idx], 0.0, 1.0)
    sampled_risk, valid_mask = _sample_layer(
        risk_grid,
        config=OVERLAY["config"],
        z=z,
        x=x,
        y=y,
    )
    sampled_conf, _ = _sample_layer(
        conf_grid,
        config=OVERLAY["config"],
        z=z,
        x=x,
        y=y,
    )
    rgba_small = _colorize(sampled_risk, sampled_conf, valid_mask)
    image = Image.fromarray(rgba_small, mode="RGBA").resize(
        (TILE_SIZE, TILE_SIZE),
        resample=Image.Resampling.BILINEAR,
    )
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=True)
    return buffer.getvalue()


def _bundle_array(key: str, dtype: np.dtype | type, default: object | None = None) -> np.ndarray:
    value = MODEL_BUNDLE.get(key, default)
    if value is None:
        raise RuntimeError(f"traffic safety bundle is missing '{key}'")
    return np.asarray(value, dtype=dtype)


def _default_weather_for_index(idx: int, hour_of_week: int, month: int) -> np.ndarray:
    if "weather_climatology" not in MODEL_BUNDLE or "candidate_station_indices" not in MODEL_BUNDLE:
        return np.zeros(5, dtype=np.float32)

    weather = lookup_weather_climatology(
        weather_cube=_bundle_array("weather_climatology", np.float32),
        station_indices=int(_bundle_array("candidate_station_indices", np.int16)[idx]),
        months=int(month),
        hour_of_week=int(hour_of_week),
        weather_defaults=_bundle_array("weather_defaults", np.float32, default=np.zeros(5)),
    )
    return weather[0]


def _prediction_feature_row(
    *,
    model: object,
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
    prior_total: float,
    prior_same_hour: float,
    temp_c: float,
    dewpoint_c: float,
    relative_humidity_pct: float,
    wind_speed_mps: float,
    wet_hour: float,
) -> np.ndarray:
    feature_count = int(getattr(model, "n_features_in_", 16))
    if feature_count <= 11:
        hour_angle = 2.0 * np.pi * float(hour) / 24.0
        dow_angle = 2.0 * np.pi * float(day_of_week - 1) / 7.0
        month_angle = 2.0 * np.pi * float(month) / 12.0
        return np.array(
            [
                [
                    float(lat),
                    float(lon),
                    float(np.sin(hour_angle)),
                    float(np.cos(hour_angle)),
                    float(np.sin(dow_angle)),
                    float(np.cos(dow_angle)),
                    float(np.sin(month_angle)),
                    float(np.cos(month_angle)),
                    float(np.log1p(prior_total)),
                    float(np.log1p(prior_same_hour)),
                    float(prior_same_hour / max(prior_total, 1.0)),
                ]
            ],
            dtype=np.float32,
        )

    hour_of_week = (day_of_week - 1) * 24 + hour
    return build_feature_matrix(
        latitudes=np.array([lat], dtype=np.float32),
        longitudes=np.array([lon], dtype=np.float32),
        hour_of_week=np.array([hour_of_week], dtype=np.int16),
        months=np.array([month], dtype=np.int8),
        totals=np.array([prior_total], dtype=np.float32),
        same_hour=np.array([prior_same_hour], dtype=np.float32),
        temp_c=np.array([temp_c], dtype=np.float32),
        dewpoint_c=np.array([dewpoint_c], dtype=np.float32),
        relative_humidity_pct=np.array([relative_humidity_pct], dtype=np.float32),
        wind_speed_mps=np.array([wind_speed_mps], dtype=np.float32),
        wet_hour=np.array([wet_hour], dtype=np.float32),
    )


def _predict_with_weather(
    *,
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
    temp_c: float,
    dewpoint_c: float,
    relative_humidity_pct: float,
    wind_speed_mps: float,
    wet_hour: float,
    weather_source: str,
    weather_summary: str = "",
    provider: str | None = None,
    provider_label: str | None = None,
    timestamp_local: str | None = None,
    forecast_hours: int | None = None,
) -> dict[str, object]:
    if not MODEL_BUNDLE:
        raise RuntimeError(f"traffic safety model is unavailable; expected {MODEL_PATH}")

    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))

    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    cell_id = h3.latlng_to_cell(float(lat), float(lon), resolution)
    idx = CELL_INDEX.get(cell_id)
    if idx is None:
        return {
            "model_version": MODEL_VERSION,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "local_day_of_week": day_of_week,
            "local_hour": hour,
            "month": month,
            "historical_cell_events": 0,
            "historical_same_hour_events": 0,
            "risk_score": 0.0,
            "risk_level": "low",
            "weather_source": weather_source,
            "weather": {
                "temp_c": float(temp_c),
                "dewpoint_c": float(dewpoint_c),
                "relative_humidity_pct": float(relative_humidity_pct),
                "wind_speed_mps": float(wind_speed_mps),
                "wet_hour": float(wet_hour),
                "summary": weather_summary,
            },
            "live_provider": provider,
            "live_provider_label": provider_label,
            "target_timestamp_local": timestamp_local,
            "forecast_hours": forecast_hours,
        }

    model = MODEL_BUNDLE["model"]
    candidate_lats = _bundle_array("candidate_lats", np.float32)
    candidate_lons = _bundle_array("candidate_lons", np.float32)
    cell_total_counts = _bundle_array("cell_total_counts", np.float32)
    cell_hour_counts = _bundle_array("cell_hour_counts", np.float32)

    hour_of_week = (day_of_week - 1) * 24 + hour
    prior_total = float(cell_total_counts[idx])
    prior_same_hour = float(cell_hour_counts[idx, hour_of_week])
    features = _prediction_feature_row(
        model=model,
        lat=float(candidate_lats[idx]),
        lon=float(candidate_lons[idx]),
        day_of_week=day_of_week,
        hour=hour,
        month=month,
        prior_total=prior_total,
        prior_same_hour=prior_same_hour,
        temp_c=float(temp_c),
        dewpoint_c=float(dewpoint_c),
        relative_humidity_pct=float(relative_humidity_pct),
        wind_speed_mps=float(wind_speed_mps),
        wet_hour=float(wet_hour),
    )
    probability = float(model.predict_proba(features)[0, 1])
    probability = max(0.0, min(1.0, probability))
    return {
        "model_version": MODEL_VERSION,
        "cell_id": cell_id,
        "lat": float(lat),
        "lon": float(lon),
        "local_day_of_week": day_of_week,
        "local_hour": hour,
        "month": month,
        "historical_cell_events": int(prior_total),
        "historical_same_hour_events": int(prior_same_hour),
        "risk_score": probability,
        "risk_level": _risk_level(probability),
        "weather_source": weather_source,
        "weather": {
            "temp_c": float(temp_c),
            "dewpoint_c": float(dewpoint_c),
            "relative_humidity_pct": float(relative_humidity_pct),
            "wind_speed_mps": float(wind_speed_mps),
            "wet_hour": float(wet_hour),
            "summary": weather_summary,
        },
        "live_provider": provider,
        "live_provider_label": provider_label,
        "target_timestamp_local": timestamp_local,
        "forecast_hours": forecast_hours,
    }


def predict_traffic_safety(
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
) -> dict[str, object]:
    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))
    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    cell_id = h3.latlng_to_cell(float(lat), float(lon), resolution)
    idx = CELL_INDEX.get(cell_id)

    if idx is None:
        return {
            "model_version": MODEL_VERSION,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "local_day_of_week": day_of_week,
            "local_hour": hour,
            "month": month,
            "historical_cell_events": 0,
            "historical_same_hour_events": 0,
            "risk_score": 0.0,
            "risk_level": "low",
            "weather_source": "climatology",
        }

    hour_of_week = (day_of_week - 1) * 24 + hour
    default_weather = _default_weather_for_index(idx=idx, hour_of_week=hour_of_week, month=month)
    return _predict_with_weather(
        lat=float(lat),
        lon=float(lon),
        day_of_week=day_of_week,
        hour=hour,
        month=month,
        temp_c=float(default_weather[0]),
        dewpoint_c=float(default_weather[1]),
        relative_humidity_pct=float(default_weather[2]),
        wind_speed_mps=float(default_weather[3]),
        wet_hour=float(default_weather[4]),
        weather_source="climatology",
        weather_summary="station climatology",
    )


def predict_traffic_safety_live(
    lat: float,
    lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
) -> dict[str, object]:
    snapshot = fetch_live_weather(
        lat=float(lat),
        lon=float(lon),
        forecast_hours=int(forecast_hours),
        provider=provider,
    )
    timestamp_local = snapshot.timestamp_local
    return _predict_with_weather(
        lat=float(lat),
        lon=float(lon),
        day_of_week=timestamp_local.weekday() + 1,
        hour=timestamp_local.hour,
        month=timestamp_local.month,
        temp_c=snapshot.temp_c,
        dewpoint_c=snapshot.dewpoint_c,
        relative_humidity_pct=snapshot.relative_humidity_pct,
        wind_speed_mps=snapshot.wind_speed_mps,
        wet_hour=snapshot.wet_hour,
        weather_source=f"live_{snapshot.observed_or_forecast}",
        weather_summary=snapshot.summary,
        provider=snapshot.provider,
        provider_label=snapshot.provider_label,
        timestamp_local=timestamp_local.isoformat(),
        forecast_hours=int(snapshot.forecast_hours),
    )


def _page_config() -> dict[str, object]:
    config = OVERLAY["config"]
    road_meta = load_road_tile_meta() if road_tile_assets_ready() else {}
    road_mode = bool(SEGMENT_MODEL_PATH.exists())
    frames = (
        [str(frame) for frame in road_meta.get("frame_labels", _road_frame_labels())]
        if road_mode
        else [str(frame) for frame in OVERLAY["frames"]]
    )
    return {
        "api_key": GMAPS_API_KEY,
        "service_id": "traffic_safety",
        "frames": frames,
        "center_lat": float(config["center_lat"]),
        "center_lon": float(config["center_lon"]),
        "default_zoom": 5 if road_mode else int(config.get("zoom_min", 4)),
        "zoom_min": (
            int(road_meta.get("tile_zoom_min", 4))
            if road_mode
            else int(config.get("zoom_min", 2))
        ),
        "zoom_max": int(config.get("zoom_max", 10)),
        "tile_zoom_min": int(road_meta.get("tile_zoom_min", 4)),
        "tile_zoom_max": int(road_meta.get("tile_zoom_max", 11)),
        "raster_zoom_min": int(road_meta.get("raster_zoom_min", 4)),
        "raster_zoom_max": int(road_meta.get("raster_zoom_max", 8)),
        "vector_zoom_min": int(road_meta.get("vector_zoom_min", 9)),
        "road_tile_revision": str(road_meta.get("run_id", "")),
        "generated_at_utc": str(road_meta.get("generated_at_utc", "")),
        "generated_at_label": str(road_meta.get("generated_at_utc", "")),
        "forecast_start_utc": str(road_meta.get("forecast_start_utc", "")),
        "forecast_end_utc": str(road_meta.get("forecast_end_utc", "")),
        "provider_label": str(road_meta.get("provider", "nws")).upper(),
        "timeline": _road_timeline(len(frames)) if road_mode else _timeline(),
        "road_mode": road_mode,
        "default_frame_idx": 0,
    }

def _site_nav(active_page: str) -> str:
    links = [
        ("map", "/map", "Map"),
        ("about", "/about", "About"),
        ("contact", "/contact", "Contact"),
    ]
    items = []
    for page_key, href, label in links:
        active_class = " active" if active_page == page_key else ""
        items.append(
            f'<a class="site-nav-link{active_class}" href="{href}">{html.escape(label)}</a>'
        )
    return f"""
    <div class="site-chrome">
      <a class="site-brandmark" href="/map">
        <div class="ops-brand">
          <div class="ops-kicker">US ROAD RISK MONITOR</div>
          <div class="ops-title-row">
            <h1>Road Risk Outlook</h1>
            <span class="ops-badge">24h forecast</span>
          </div>
        </div>
      </a>
      <nav class="site-nav" aria-label="Primary">
        {''.join(items)}
      </nav>
    </div>
    """


def _document_html(
    *,
    title: str,
    content: str,
    include_map_assets: bool = False,
    extra_script: str = "",
) -> str:
    map_script_tag = (
        f'<script src="{MAP_ASSET_HREFS["js"]}" defer></script>' if include_map_assets else ""
    )
    analytics_script_tag = ""
    if GA_MEASUREMENT_ID:
        analytics_id = html.escape(GA_MEASUREMENT_ID, quote=True)
        analytics_script_tag = f"""
    <script async src="https://www.googletagmanager.com/gtag/js?id={analytics_id}"></script>
    <script>
      window.dataLayer = window.dataLayer || [];
      function gtag(){{dataLayer.push(arguments);}}
      gtag('js', new Date());
      gtag('config', '{analytics_id}');
    </script>"""
    return f"""<!doctype html>
<html lang="en">
  <head>
    <meta charset="utf-8">
    <meta name="viewport" content="width=device-width, initial-scale=1">
    <title>{html.escape(title)}</title>
    <meta name="description" content="Road Risk Monitor turns incident history, road geometry, and live weather into a nationwide operational road-safety forecast.">
    <link rel="stylesheet" href="{MAP_ASSET_HREFS['css']}">
    {analytics_script_tag}
    {map_script_tag}
  </head>
  <body>
    {content}
    {extra_script}
  </body>
</html>
"""


def _map_page_content() -> str:
    config_blob = html.escape(json.dumps(_page_config()), quote=True)
    return f"""
    <div id="risk-map-shell" class="risk-map-shell" data-config="{config_blob}">
      {_site_nav("map")}
      <header class="ops-topbar">
        <div class="ops-top-pills">
          <div class="ops-pill">
            <span>Frame</span>
            <strong id="ops-frame-chip">+0h</strong>
          </div>
          <div class="ops-pill">
            <span>Updated</span>
            <strong id="ops-updated-chip">-</strong>
          </div>
        </div>
      </header>
      <header class="ops-timeline-dock">
        <div class="timeline-row">
          <div class="timeline-wrap">
            <div id="risk-timeline-ticks" class="timeline-years"></div>
            <div id="risk-timeline-track" class="timeline-track" style="--frame-step:1%;">
              <div id="risk-timeline-phases" class="timeline-phases"></div>
              <div id="risk-time-progress" class="timeline-progress"></div>
              <div id="risk-now-marker" class="timeline-marker"></div>
              <input id="risk-time-slider" type="range" min="0" max="0" value="0" step="1">
            </div>
            <div id="risk-frame-label" class="timeline-current-label"></div>
          </div>
          <button id="risk-play" type="button" aria-label="Play timeline">
            <span class="play-icon" aria-hidden="true">&#9658;</span>
            <span class="pause-icon" aria-hidden="true">&#10074;&#10074;</span>
          </button>
        </div>
      </header>

      <section class="ops-stage">
        <aside class="ops-sidecard">
          <div class="side-section">
            <div class="side-title">Layers</div>
            <label class="layer-toggle">
              <input id="layer-risk" type="checkbox" checked>
              <span>Risk overlay</span>
            </label>
            <label class="layer-toggle">
              <input id="layer-roads" type="checkbox" checked>
              <span>Google roads</span>
            </label>
          </div>

          <div class="side-section">
            <div class="side-title">Legend</div>
            <div class="legend-row"><span class="legend-swatch low"></span><span>Low</span></div>
            <div class="legend-row"><span class="legend-swatch moderate"></span><span>Elevated</span></div>
            <div class="legend-row"><span class="legend-swatch high"></span><span>High</span></div>
            <div class="legend-row"><span class="legend-swatch severe"></span><span>Severe</span></div>
          </div>

          <div class="side-section">
            <div class="side-title">Last updated</div>
            <div id="ops-freshness-text" class="meta-line">-</div>
            <div id="ops-provider-text" class="meta-subline">-</div>
          </div>

          <div class="side-section">
            <div class="side-title">Road details</div>
            <div class="help-copy">Click a road to see the current index and how far it is from normal.</div>
          </div>
        </aside>

        <div class="ops-mapwrap">
          <div id="risk-map" class="risk-map"></div>
          <div id="risk-zoom-hint" class="zoom-hint">Zoom in for road-level details.</div>
        </div>
      </section>

      
      <div id="risk-map-status" class="risk-map-status"></div>
    </div>
    """


def _page_html() -> str:
    return _document_html(
        title="Road Risk Monitor | Map",
        content=_map_page_content(),
        include_map_assets=True,
    )


def _about_page_html() -> str:
    return _document_html(
        title="Road Risk Monitor | About",
        content=f"""
        <div class="site-shell">
          {_site_nav("about")}
          <main class="site-main about-main">
            <section class="page-hero about-hero">
              <div class="about-hero-copy">
                <span class="page-eyebrow">Traffic safety intelligence</span>
                <h1>From crash archives to live risk assessment at national level</h1>
                <p>
                  Road Risk Monitor transforms historical incidents, road geometry and live weather
                  into a continuously refreshed 24-hour risk outlook for major U.S. road segments. It does not wait for incidents to accumulate. It learns where risk normally lives, watches the
                atmosphere change and turns that signal into a road-level forecast that is fast enough for real
                operational use.
                </p>
                <div class="about-actions">
                  <a class="button-primary about-button" href="/map">Explore</a>
                  <a class="button-secondary about-button" href="/contact">Ask question</a>
                </div>
                <div class="metric-grid about-metric-grid">
                  <article class="metric-card">
                    <strong>Segment-level</strong>
                    <span>Risk scores mapped to roads, not county blobs.</span>
                  </article>
                  <article class="metric-card">
                    <strong>24-hour</strong>
                    <span>Precomputed nationwide forecast frames.</span>
                  </article>
                  <article class="metric-card">
                    <strong>Weather-aware</strong>
                    <span>Live conditions fused with learned baselines.</span>
                  </article>
                </div>
              </div>
              <div class="page-card about-hero-visual">
                <div class="about-visual-head">
                  <span>Major-road risk overlay</span>
                </div>
                <svg class="about-map-svg" viewBox="0 0 620 430" role="img" aria-label="Stylized U.S. road risk map">
                  <defs>
                    <filter id="aboutGlow"><feGaussianBlur stdDeviation="4" result="blur"/><feMerge><feMergeNode in="blur"/><feMergeNode in="SourceGraphic"/></feMerge></filter>
                    <linearGradient id="aboutRiskLine" x1="0" x2="1">
                      <stop offset="0%" stop-color="#48bb78"/>
                      <stop offset="38%" stop-color="#f2c94c"/>
                      <stop offset="72%" stop-color="#f28a31"/>
                      <stop offset="100%" stop-color="#cb2b27"/>
                    </linearGradient>
                  </defs>
                  <path d="M60 112 C160 62 253 94 330 75 C433 49 508 82 560 136 C516 196 557 260 487 311 C416 363 328 342 260 368 C183 398 123 360 88 302 C52 244 20 177 60 112Z" fill="rgba(72, 187, 120, 0.06)" stroke="rgba(20,32,43,0.14)"/>
                  <g filter="url(#aboutGlow)" fill="none" stroke-linecap="round" stroke-width="8">
                    <path d="M84 166 C154 194 226 170 304 206 C390 246 456 236 538 286" stroke="url(#aboutRiskLine)"/>
                    <path d="M132 286 C220 230 286 284 374 240 C448 202 496 148 552 168" stroke="#f28a31"/>
                    <path d="M184 92 C206 168 198 256 226 352" stroke="#f2c94c"/>
                    <path d="M342 88 C316 154 354 218 332 346" stroke="#48bb78"/>
                    <path d="M472 104 C438 158 464 238 432 322" stroke="#cb2b27"/>
                  </g>
                </svg>
                <aside class="about-risk-panel">
                  <strong>Segment detail</strong>
                  <div class="about-risk-row"><span>Current index</span><strong>74 / 100</strong></div>
                  <div class="about-risk-row"><span>Compared with normal</span><strong>+31%</strong></div>
                  <div class="about-risk-row"><span>Risk factor</span><strong>Rain</strong></div>
                  <div class="about-risk-meter"><div></div></div>
                </aside>
              </div>
            </section>

            <section class="page-card page-card--wide about-intro">
              <div class="page-section-kicker">What it is</div>
              <div class="about-section-head">
                <div>
                  <h2>Road Risk Monitor is the missing intelligence layer between weather maps and crash reports</h2>
                </div>
                <p>
                  Most transportation analytics stop at summaries of what already happened. Road Risk
                  Monitor does the harder work - it converts static historical records into a forecast
                  surface that operators can browse, audit, and use.
                </p>
              </div>
              <div class="grid-3 about-feature-grid">
                <article class="feature-card">
                  <h3>Road-segment intelligence</h3>
                  <p>Major corridors are represented as scored roadway segments, preserving the geography that county-level summaries erase.</p>
                </article>
                <article class="feature-card">
                  <h3>Weather-aware forecasting</h3>
                  <p>Historical crash patterns are adjusted with current and forecast atmospheric signals to detect when conditions drift away from normal.</p>
                </article>
                <article class="feature-card">
                  <h3>Operational delivery</h3>
                  <p>Forecast frames are precomputed and published as map-ready tiles, so the interface feels like a live instrument instead of a research notebook.</p>
                </article>
              </div>
            </section>

            <section class="page-card page-card--wide">
              <div class="page-section-kicker">System architecture</div>
              <div class="about-section-head">
                <div>
                  <h2>How raw records become a national risk layer</h2>
                </div>
                <p>Road Risk Monitor combines geospatial processing, time-series alignment, predictive modeling, and tile publishing into one repeatable production pipeline.</p>
              </div>
              <div class="about-pipeline">
                <article class="about-pipe-node"><small>01</small><h3>Road graph</h3><p>National roadway geometry is cleaned, simplified, and split into major-road segments that can be scored consistently.</p></article>
                <article class="about-pipe-node"><small>02</small><h3>Incident labeling</h3><p>Historical roadway incidents are projected onto nearby segments and assigned temporal context.</p></article>
                <article class="about-pipe-node"><small>03</small><h3>Weather alignment</h3><p>Hourly weather observations and forecasts are spatially matched to representative road locations.</p></article>
                <article class="about-pipe-node"><small>04</small><h3>Model training</h3><p>The model learns baseline risk, weather sensitivity, seasonality, and same-hour patterns by segment.</p></article>
                <article class="about-pipe-node"><small>05</small><h3>Tile publishing</h3><p>Forecast scores are converted into vector and raster tiles for fast national map rendering.</p></article>
              </div>
            </section>

            

            <section class="page-card page-card--wide">
              <div class="page-section-kicker">Inference</div>
              <div class="about-section-head">
                <div>
                  <h2>End-to-end pipeline</h2>
                </div>
                <p>Operational loop keeps the forecast current and presents it in a format people can use.</p>
              </div>
              <div class="about-live-loop">
                <article class="about-pipe-node about-loop-node"><small>01</small><h3>Forecast ingestion</h3><p>Fresh weather forecasts are pulled into the scoring pipeline and aligned to relevant road locations.</p></article>
                <article class="about-pipe-node about-loop-node"><small>02</small><h3>Segment-hour scoring</h3><p>Every forecast frame is evaluated against pretrained priors and live atmospheric context.</p></article>
                <article class="about-pipe-node about-loop-node"><small>03</small><h3>Tile generation</h3><p>Scores are packaged into tiled layers so nationwide browsing stays fast and smooth.</p></article>
                <article class="about-pipe-node about-loop-node"><small>04</small><h3>Map delivery</h3><p>Operators can inspect risk overlays, click roads, and compare present conditions against normal.</p></article>
              </div>
            </section>

            <section class="page-card page-card--wide about-proof">
              <div class="about-proof-band">
                <div>
                  <div class="page-section-kicker">Why it matters</div>
                  <h2>Earlier awareness for agencies, planners and operators</h2>
                  <p>
                    Road Risk Monitor helps transportation organizations move from reactive reporting to proactive
                    attention. It gives teams a common risk layer for planning, weather response, staffing,
                    maintenance coordination and executive communication.
                  </p>
                </div>
                <ul class="feature-list about-proof-list">
                  <li>Turns passive crash archives into forward-looking awareness</li>
                  <li>Connects atmospheric change to roadway-specific exposure</li>
                  <li>Supports planning and monitoring from the same forecast base</li>
                  <li>Packages scientific modeling as a product people can actually use</li>
                </ul>
              </div>

              <div class="about-risk-scale">
                <div class="diagram-card"><div class="scale-bar low"></div><h3>Low</h3><p>Conditions are near or below the learned normal for this segment and hour</p></div>
                <div class="diagram-card"><div class="scale-bar elevated"></div><h3>Elevated</h3><p>Risk is drifting upward and deserves routine awareness</p></div>
                <div class="diagram-card"><div class="scale-bar high"></div><h3>High</h3><p>Weather, timing, and segment history combine into a stronger warning signal</p></div>
                <div class="diagram-card"><div class="scale-bar severe"></div><h3>Severe</h3><p>The system sees a meaningful departure from normal operating conditions</p></div>
              </div>
            </section>

            <section class="page-card page-card--wide about-cta">
              <div class="page-section-kicker">About</div>
              <h2>Road Risk Monitor is the missing intelligence layer between weather maps and crash reports.</h2>
              <p>
                It does not wait for incidents to accumulate. It learns where risk normally lives, watches the
                atmosphere change, and turns that signal into a road-level forecast that is fast enough for real
                operational use.
              </p>
              <div class="about-actions about-actions-center">
                <a class="button-primary about-button" href="/map">View the national outlook</a>
                <a class="button-secondary about-button" href="/contact">Request a briefing</a>
              </div>
            </section>
          </main>
        </div>
        """,
    )


def _contact_delivery_configured() -> bool:
    return bool(CONTACT_EMAIL and SMTP_HOST and SMTP_FROM)


def _contact_email_link() -> str:
    if not CONTACT_EMAIL:
        return f'<span class="contact-email-placeholder">{html.escape(CONTACT_DISPLAY_EMAIL)}</span>'
    return f'<a class="contact-email-link" href="mailto:{html.escape(CONTACT_EMAIL)}">{html.escape(CONTACT_EMAIL)}</a>'


def _contact_page_html() -> str:
    delivery_state = (
        "Email forwarding is live. Form submissions will be delivered to the configured inbox and logged locally."
        if _contact_delivery_configured()
        else "Email forwarding is not configured yet. The form will still capture and store submissions on this server."
    )
    return _document_html(
        title="Road Risk Monitor | Contact",
        content=f"""
        <div class="site-shell">
          {_site_nav("contact")}
          <main class="site-main">

            <section class="contact-grid">
              <aside class="page-card contact-card">
                <div class="page-section-kicker">Contact</div>
                <h2>Anton Ivchenko</h2>
                <p class="contact-primary">{_contact_email_link()}</p>
                <p>Please feel free to send me a message through the form or to my email if you have any questions or want to contribute to the project. I'll be happy to answer your questions!</p>
                <div class="contact-info-list">
                  <div>
                    <strong>Routing</strong>
                    <span>{html.escape(delivery_state)}</span>
                  </div>
                  <div>
                    <strong>Stored submissions</strong>
                    <span>{html.escape(str(CONTACT_SUBMISSIONS_PATH))}</span>
                  </div>
                  <div>
                    <strong>Best for</strong>
                    <span>Deployments, pilots, research collaboration, and product inquiries.</span>
                  </div>
                </div>
              </aside>

              <section class="page-card contact-form-card">
                <div class="page-section-kicker">Send a message</div>
                <div id="contact-form-status" class="status-banner" role="status" aria-live="polite"></div>
                <form id="contact-form" class="contact-form" data-endpoint="/api/contact">
                  <div class="field-grid">
                    <label class="field">
                      <span>Name</span>
                      <input type="text" name="name" placeholder="Your name" required>
                    </label>
                    <label class="field">
                      <span>Email</span>
                      <input type="email" name="email" placeholder="your@email.com" required>
                    </label>
                  </div>
                  <div class="field-grid">
                    <label class="field">
                      <span>Organization</span>
                      <input type="text" name="organization" placeholder="Company">
                    </label>
                    <label class="field">
                      <span>Subject</span>
                      <input type="text" name="subject" placeholder="What this is about">
                    </label>
                  </div>
                  <label class="field">
                    <span>Message</span>
                    <textarea name="message" rows="8" placeholder="How I can help" required></textarea>
                  </label>
                  <div class="contact-actions">
                    <button id="contact-submit" class="button-primary" type="submit">Send</button>
                  </div>
                </form>
              </section>
            </section>
          </main>
        </div>
        """,
        extra_script="""
<script>
(() => {
  const form = document.getElementById("contact-form");
  const status = document.getElementById("contact-form-status");
  const submitButton = document.getElementById("contact-submit");
  if (!form || !status || !submitButton) {
    return;
  }
  const defaultLabel = submitButton.textContent;
  const setStatus = (kind, message) => {
    status.className = `status-banner show ${kind || "info"}`;
    status.textContent = message || "";
  };
  form.addEventListener("submit", async (event) => {
    event.preventDefault();
    submitButton.disabled = true;
    submitButton.textContent = "Sending...";
    const payload = Object.fromEntries(new FormData(form).entries());
    try {
      const response = await fetch(form.dataset.endpoint, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(payload),
      });
      const data = await response.json().catch(() => ({}));
      if (!response.ok) {
        setStatus("error", data.detail || data.message || "Unable to send the message right now.");
        return;
      }
      setStatus(data.kind || "success", data.message || "Message received.");
      if ((data.kind || "success") !== "error") {
        form.reset();
      }
    } catch (error) {
      setStatus("error", "Unable to reach the contact service right now. Please try again.");
    } finally {
      submitButton.disabled = false;
      submitButton.textContent = defaultLabel;
    }
  });
})();
</script>
""",
    )


def _map_html() -> str:
    return _page_html() + "\n<!-- Traffic Safety -->\n"


def _coerce_contact_value(payload: dict[str, object], key: str, limit: int) -> str:
    value = str(payload.get(key, "") or "").strip()
    return value[:limit]


def _append_contact_submission(record: dict[str, object]) -> None:
    CONTACT_SUBMISSIONS_PATH.parent.mkdir(parents=True, exist_ok=True)
    with CONTACT_SUBMISSIONS_PATH.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=True) + "\n")


def _send_contact_email(record: dict[str, object]) -> tuple[str, str]:
    if not CONTACT_EMAIL:
        return ("warning", "Set CONTACT_EMAIL to route submissions to your inbox.")
    if not SMTP_HOST:
        return ("warning", "Set SMTP_HOST and SMTP_* credentials to enable email forwarding.")

    message = EmailMessage()
    sender_name = str(record.get("name") or "Website contact")
    sender_email = str(record.get("email") or "")
    reply_to = sender_email or SMTP_FROM
    subject = str(record.get("subject") or "Website contact")
    message["To"] = CONTACT_EMAIL
    message["From"] = SMTP_FROM
    message["Reply-To"] = reply_to
    message["Subject"] = f"[Road Risk Monitor] {subject}"
    message.set_content(
        "\n".join(
            [
                f"Submitted at: {record.get('submitted_at_utc', '')}",
                f"Name: {sender_name}",
                f"Email: {sender_email}",
                f"Organization: {record.get('organization', '')}",
                "",
                "Message:",
                str(record.get("message") or ""),
            ]
        )
    )

    try:
        with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=20) as smtp:
            if SMTP_USE_TLS:
                smtp.starttls()
            if SMTP_USERNAME:
                smtp.login(SMTP_USERNAME, SMTP_PASSWORD)
            smtp.send_message(message)
    except Exception as exc:  # noqa: BLE001
        return ("warning", f"Email forwarding failed, but the message was stored locally: {exc}")

    return ("success", f"Message sent to {CONTACT_EMAIL} and logged by the service.")


def _map_bootstrap_js() -> str:
    script = (STATIC_DIR / "map.js").read_text(encoding="utf-8")
    # Compatibility shim for older tests that expected a named bootstrap helper.
    return "function bootstrapTrafficSafetyMap() {}\n" + script


api = FastAPI(title=SERVICE_NAME)
api.add_middleware(GZipMiddleware, minimum_size=1024)
api.mount(STATIC_URL, StaticFiles(directory=str(STATIC_DIR)), name="traffic-safety-static")


def _utc_label(value: str | None) -> str:
    if not value:
        return "n/a"
    try:
        timestamp = datetime.fromisoformat(str(value).replace("Z", "+00:00"))
    except ValueError:
        return str(value)
    if timestamp.tzinfo is None:
        timestamp = timestamp.replace(tzinfo=timezone.utc)
    return timestamp.astimezone(timezone.utc).strftime("%Y-%m-%d %H:%M UTC")


def _road_class_label(rttyp: str | None, mtfcc: str | None) -> str:
    rttyp_value = (rttyp or "").strip().upper()
    mtfcc_value = (mtfcc or "").strip().upper()
    if rttyp_value == "I" or mtfcc_value == "S1100":
        return "Interstate"
    if rttyp_value in {"U", "S"} or mtfcc_value == "S1200":
        return "Highway"
    return "Major road"


def _risk_band(risk_index: float) -> str:
    if risk_index < 18.0:
        return "Low"
    if risk_index < 38.0:
        return "Elevated"
    if risk_index < 62.0:
        return "High"
    return "Severe"


def _timeline_kind_factors(zoom: int) -> tuple[float, float, float]:
    if int(zoom) <= 4:
        return (0.12, 0.34, 1.0)
    if int(zoom) <= 5:
        return (0.18, 0.48, 1.0)
    if int(zoom) <= 6:
        return (0.28, 0.62, 1.0)
    if int(zoom) <= 8:
        return (0.4, 0.76, 1.0)
    return (0.55, 0.86, 1.0)


def _path_blob_weight(path_blob: bytes) -> float:
    coords = np.frombuffer(path_blob, dtype=np.int16)
    if coords.size < 4:
        return 1.0
    points = coords.reshape(-1, 2).astype(np.float32)
    diffs = points[1:] - points[:-1]
    length = float(np.hypot(diffs[:, 0], diffs[:, 1]).sum())
    return max(1.0, length / 24.0)


RASTER_RISK_STOPS = np.asarray(
    [
        (17.0, 122.0, 101.0, 0.0),
        (72.0, 187.0, 120.0, 0.28),
        (242.0, 201.0, 76.0, 0.52),
        (242.0, 130.0, 49.0, 0.74),
        (203.0, 43.0, 39.0, 1.0),
    ],
    dtype=np.float32,
)


@lru_cache(maxsize=4)
def _load_segment_score_cube(path_str: str, mtime_ns: int) -> np.ndarray:
    del mtime_ns
    return np.load(path_str, mmap_mode="r")


def _current_segment_scores() -> np.ndarray:
    return _load_segment_score_cube(
        str(ROAD_TILE_FORECAST_PATH),
        ROAD_TILE_FORECAST_PATH.stat().st_mtime_ns,
    )


def _baseline_segment_scores() -> np.ndarray | None:
    if not ROAD_TILE_BASELINE_PATH.exists():
        return None
    return _load_segment_score_cube(
        str(ROAD_TILE_BASELINE_PATH),
        ROAD_TILE_BASELINE_PATH.stat().st_mtime_ns,
    )


@lru_cache(maxsize=1)
def _segment_detail_runtime() -> dict[str, object]:
    runtime = load_segment_runtime()
    roads = pd.read_parquet(
        ROAD_SEGMENTS_PATH,
        columns=[
            "segment_id",
            "fullname",
            "rttyp",
            "mtfcc",
            "length_km",
            "center_lat",
            "center_lon",
        ],
    ).reset_index(drop=True)
    if "segment_idx" not in roads.columns:
        roads = roads.copy()
        roads["segment_idx"] = np.arange(len(roads), dtype=np.int32)
    segment_idx = roads["segment_idx"].to_numpy(dtype=np.int32)
    order = np.argsort(segment_idx, kind="mergesort")
    roads_sorted = roads.iloc[order].reset_index(drop=True)
    return {
        "bundle": runtime["bundle"],
        "rep_by_index": runtime["rep_by_index"],
        "roads": roads_sorted,
        "segment_idx": roads_sorted["segment_idx"].to_numpy(dtype=np.int32),
    }


@api.get("/", response_class=HTMLResponse)
def root() -> HTMLResponse:
    return HTMLResponse(_map_html())


@api.get("/map", response_class=HTMLResponse)
def map_page() -> HTMLResponse:
    return HTMLResponse(_map_html())


@api.get("/about", response_class=HTMLResponse)
def about_page() -> HTMLResponse:
    return HTMLResponse(_about_page_html())


@api.get("/contact", response_class=HTMLResponse)
def contact_page() -> HTMLResponse:
    return HTMLResponse(_contact_page_html())


@api.post("/api/contact")
def submit_contact(request: Request, payload: dict[str, object] = Body(...)) -> JSONResponse:
    name = _coerce_contact_value(payload, "name", 160)
    email_value = _coerce_contact_value(payload, "email", 240)
    organization = _coerce_contact_value(payload, "organization", 240)
    subject = _coerce_contact_value(payload, "subject", 240) or "Website contact"
    message = _coerce_contact_value(payload, "message", 6000)

    if not name or not email_value or not message:
        raise HTTPException(status_code=400, detail="Name, email, and message are required.")

    if email_value.count("@") != 1 or "." not in email_value.split("@", 1)[1]:
        raise HTTPException(status_code=400, detail="Enter a valid email address.")

    submitted_at_utc = datetime.now(timezone.utc).isoformat()
    record = {
        "submitted_at_utc": submitted_at_utc,
        "name": name,
        "email": email_value,
        "organization": organization,
        "subject": subject,
        "message": message,
        "client_host": getattr(request.client, "host", ""),
        "user_agent": request.headers.get("user-agent", ""),
    }
    delivery_kind, delivery_message = _send_contact_email(record)
    record["delivery_kind"] = delivery_kind
    record["delivery_message"] = delivery_message

    try:
        _append_contact_submission(record)
    except OSError as exc:
        raise HTTPException(status_code=500, detail=f"Unable to store the message locally: {exc}") from exc

    return JSONResponse(
        {
            "kind": delivery_kind,
            "message": delivery_message,
            "submitted_at_utc": submitted_at_utc,
        }
    )


@api.get("/health")
def health() -> dict[str, object]:
    segment_model_path = REPO_DIR / "models" / "traffic_safety_segments.joblib"
    road_tile_meta = load_road_tile_meta() if road_tile_assets_ready() else {}
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "frames": len(OVERLAY["frames"]),
        "model_version": MODEL_VERSION,
        "model_ready": bool(MODEL_BUNDLE),
        "segment_model_ready": bool(segment_model_path.exists()),
        "overlay_ready": bool((TILES_DIR / "overlay.npz").exists()),
        "road_tiles_ready": road_tile_assets_ready(),
        "road_raster_tiles_ready": raster_tile_assets_ready(),
        "road_tiles_generated_at_utc": road_tile_meta.get("generated_at_utc"),
        "live_providers": [
            {
                "name": status.name,
                "label": status.label,
                "paid": status.paid,
                "enabled": status.enabled,
                "configured": status.configured,
                "available": status.available,
            }
            for status in provider_statuses()
        ],
    }


@api.get("/tiles/{frame_idx}/{z}/{x}/{y}.png")
def tile(frame_idx: int, z: int, x: int, y: int) -> Response:
    if z < 0 or z > 12:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        return Response(content=buffer.getvalue(), media_type="image/png")

    try:
        png = _render_tile_png(frame_idx=frame_idx, z=z, x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


@api.get("/api/live-risk")
def live_risk(
    lat: float,
    lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
) -> dict[str, object]:
    try:
        return predict_traffic_safety_live(
            lat=lat,
            lon=lon,
            forecast_hours=forecast_hours,
            provider=provider,
        )
    except LiveWeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"weather provider request failed: {exc}") from exc


@api.get("/api/segment-risk")
def segment_risk(
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
    limit: int = 1500,
) -> dict[str, object]:
    try:
        return score_segments_in_bbox(
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
    except requests.HTTPError as exc:
        raise HTTPException(status_code=502, detail=f"weather provider request failed: {exc}") from exc


@api.get("/api/segment-detail")
def segment_detail(segment_idx: int, frame_idx: int = 0) -> dict[str, object]:
    if not road_tile_assets_ready():
        raise HTTPException(status_code=404, detail="road tiles are not ready")

    meta = load_road_tile_meta()
    frames = [str(frame) for frame in meta.get("frame_labels", _road_frame_labels())]
    if frame_idx < 0 or frame_idx >= len(frames):
        raise HTTPException(status_code=404, detail="frame is out of range")

    runtime = _segment_detail_runtime()
    segment_index = runtime["segment_idx"]
    pos = int(np.searchsorted(segment_index, int(segment_idx)))
    if pos >= len(segment_index) or int(segment_index[pos]) != int(segment_idx):
        raise HTTPException(status_code=404, detail="segment was not found")

    roads = runtime["roads"]
    road = roads.iloc[pos]
    bundle = runtime["bundle"]
    rep_by_index = runtime["rep_by_index"]
    current_scores = _current_segment_scores()
    baseline_scores = _baseline_segment_scores()

    current_byte = int(current_scores[int(segment_idx), int(frame_idx)])
    if baseline_scores is not None:
        baseline_row = np.asarray(baseline_scores[int(segment_idx)], dtype=np.uint8)
        baseline_byte = int(baseline_row[int(frame_idx)])
        if baseline_byte == 0 and baseline_row.size:
            baseline_byte = int(np.rint(float(np.mean(baseline_row))))
    else:
        baseline_byte = int(np.rint(np.mean(current_scores[int(segment_idx)])))

    risk_index = float(current_byte) / 255.0 * 100.0
    normal_risk_index = float(baseline_byte) / 255.0 * 100.0
    safety_index = max(0.0, 100.0 - risk_index)
    delta_points = risk_index - normal_risk_index
    if abs(delta_points) < 2.0:
        relative_label = "Near normal"
    elif delta_points > 0:
        relative_label = "Higher than normal"
    else:
        relative_label = "Lower than normal"

    station_index = int(np.asarray(bundle["segment_station_indices"], dtype=np.int32)[int(segment_idx)])
    utc_offset_hours = 0
    if station_index in rep_by_index.index:
        utc_offset_hours = int(rep_by_index.loc[station_index].get("utc_offset_hours", 0))
    forecast_start_value = str(meta.get("forecast_start_utc") or "")
    if forecast_start_value:
        forecast_start_utc = datetime.fromisoformat(forecast_start_value.replace("Z", "+00:00"))
    else:
        forecast_start_utc = datetime.fromisoformat(
            str(meta.get("generated_at_utc", "")).replace("Z", "+00:00")
        ).replace(minute=0, second=0, microsecond=0)
    if forecast_start_utc.tzinfo is None:
        forecast_start_utc = forecast_start_utc.replace(tzinfo=timezone.utc)
    local_tz = timezone(timedelta(hours=int(utc_offset_hours)))
    target_local = (forecast_start_utc + timedelta(hours=int(frame_idx))).astimezone(local_tz)
    hour_of_week = target_local.weekday() * 24 + target_local.hour

    total_counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    hour_counts = np.asarray(bundle["segment_hour_counts"], dtype=np.float32)
    historical_total = int(total_counts[int(segment_idx)])
    historical_same_hour = int(hour_counts[int(segment_idx), int(hour_of_week)])

    road_name = str(road.get("fullname") or "").strip() or _road_class_label(
        road.get("rttyp"),
        road.get("mtfcc"),
    )
    road_class = _road_class_label(road.get("rttyp"), road.get("mtfcc"))

    return {
        "segment_idx": int(segment_idx),
        "segment_id": str(road.get("segment_id") or ""),
        "road_name": road_name,
        "road_class": road_class,
        "length_km": round(float(road.get("length_km") or 0.0), 3),
        "center_lat": float(road.get("center_lat") or 0.0),
        "center_lon": float(road.get("center_lon") or 0.0),
        "frame_idx": int(frame_idx),
        "frame_label": frames[int(frame_idx)],
        "target_local": target_local.isoformat(),
        "target_local_label": target_local.strftime("%a %H:%M"),
        "generated_at_utc": str(meta.get("generated_at_utc") or ""),
        "generated_at_label": _utc_label(meta.get("generated_at_utc")),
        "risk_index": round(risk_index, 1),
        "safety_index": round(safety_index, 1),
        "normal_risk_index": round(normal_risk_index, 1),
        "delta_points": round(delta_points, 1),
        "relative_label": relative_label,
        "risk_band": _risk_band(risk_index),
        "historical_total": historical_total,
        "historical_same_hour": historical_same_hour,
    }


@lru_cache(maxsize=131072)
def _vector_tile_risk_summary(
    db_mtime_ns: int,
    forecast_mtime_ns: int,
    revision: str,
    z: int,
    x: int,
    y: int,
    zoom: int,
) -> tuple[list[float], float]:
    del db_mtime_ns, forecast_mtime_ns, revision
    factors = _timeline_kind_factors(int(zoom))
    scores = _current_segment_scores()
    frame_count = int(scores.shape[1])
    frame_sums = np.zeros(frame_count, dtype=np.float64)
    total_weight = 0.0
    db_uri = f"file:{ROAD_TILE_DB_PATH}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as connection:
        rows = connection.execute(
            """
            SELECT segment_idx, road_kind, path
            FROM tile_entries
            WHERE z = ? AND x = ? AND y = ?
            """,
            (int(z), int(x), int(y)),
        ).fetchall()
    for segment_idx, road_kind, path_blob in rows:
        kind_idx = max(0, min(2, int(road_kind)))
        weight = _path_blob_weight(bytes(path_blob)) * float(factors[kind_idx])
        if weight <= 0.0:
            continue
        frame_sums += np.asarray(scores[int(segment_idx)], dtype=np.float64) * weight
        total_weight += weight
    return (frame_sums.tolist(), float(total_weight))


@lru_cache(maxsize=131072)
def _raster_tile_risk_summary(
    db_mtime_ns: int,
    revision: str,
    frame_idx: int,
    z: int,
    x: int,
    y: int,
) -> tuple[float, float]:
    del db_mtime_ns, revision
    png = load_raster_tile_png(frame_idx=int(frame_idx), z=int(z), x=int(x), y=int(y))
    if png is None:
        return (0.0, 0.0)
    with Image.open(io.BytesIO(png)) as image:
        sampled = image.convert("RGBA").resize((64, 64), resample=Image.Resampling.BILINEAR)
        rgba = np.asarray(sampled, dtype=np.float32)
    alpha = rgba[:, :, 3] / 255.0
    mask = alpha > 0.08
    if not np.any(mask):
        return (0.0, 0.0)
    rgb = rgba[:, :, :3][mask]
    weights = alpha[mask].astype(np.float32)
    deltas = rgb[:, None, :] - RASTER_RISK_STOPS[None, :, :3]
    nearest_idx = np.argmin(np.sum(deltas * deltas, axis=2), axis=1)
    stop_scores = RASTER_RISK_STOPS[nearest_idx, 3]
    total_weight = float(weights.sum())
    if total_weight <= 1e-6:
        return (0.0, 0.0)
    risk_score = float(np.sum(stop_scores * weights) / total_weight) * 255.0
    return (risk_score, total_weight)


@api.post("/api/raster-timeline-summary")
def raster_timeline_summary(payload: dict[str, object] = Body(...)) -> dict[str, object]:
    if not raster_tile_assets_ready():
        raise HTTPException(status_code=404, detail="road raster tiles are not ready")

    meta = load_road_tile_meta()
    frame_labels = [str(frame) for frame in meta.get("frame_labels", _road_frame_labels())]
    empty_risks = [0.0 for _ in frame_labels]
    zoom = int(payload.get("z", 0) or 0)
    raw_tiles = payload.get("tiles", [])
    if not isinstance(raw_tiles, list) or not raw_tiles:
        return {
            "risks": empty_risks,
            "frame_labels": frame_labels,
            "zoom": int(zoom),
            "tile_count": 0,
        }

    unique_tiles: list[tuple[int, int]] = []
    seen: set[tuple[int, int]] = set()
    for item in raw_tiles[:64]:
        if not isinstance(item, dict):
            continue
        try:
            tile = (int(item.get("x", 0) or 0), int(item.get("y", 0) or 0))
        except (TypeError, ValueError):
            continue
        if tile in seen:
            continue
        seen.add(tile)
        unique_tiles.append(tile)

    if not unique_tiles:
        return {
            "risks": empty_risks,
            "frame_labels": frame_labels,
            "zoom": int(zoom),
            "tile_count": 0,
        }

    db_mtime_ns = ROAD_RASTER_TILE_DB_PATH.stat().st_mtime_ns
    forecast_mtime_ns = ROAD_TILE_FORECAST_PATH.stat().st_mtime_ns
    revision = str(meta.get("run_id") or meta.get("generated_at_utc") or "")
    if int(zoom) <= int(meta.get("raster_zoom_max", 8)):
        frame_sums = np.zeros(len(frame_labels), dtype=np.float64)
        frame_weights = np.zeros(len(frame_labels), dtype=np.float64)
        for tile_x, tile_y in unique_tiles:
            for frame_idx in range(len(frame_labels)):
                tile_risk, tile_weight = _raster_tile_risk_summary(
                    db_mtime_ns,
                    revision,
                    int(frame_idx),
                    int(zoom),
                    int(tile_x),
                    int(tile_y),
                )
                if tile_weight <= 0.0:
                    continue
                frame_sums[int(frame_idx)] += float(tile_risk) * float(tile_weight)
                frame_weights[int(frame_idx)] += float(tile_weight)
        risks = (
            np.divide(
                frame_sums,
                np.maximum(frame_weights, 1e-6),
                out=np.zeros_like(frame_sums),
                where=frame_weights > 0.0,
            )
            .astype(np.float64)
            .tolist()
        )
        return {
            "risks": risks,
            "frame_labels": frame_labels,
            "zoom": int(zoom),
            "tile_count": len(unique_tiles),
        }

    frame_sums = np.zeros(len(frame_labels), dtype=np.float64)
    total_weight = 0.0
    for tile_x, tile_y in unique_tiles:
        tile_sums, tile_weight = _vector_tile_risk_summary(
            ROAD_TILE_DB_PATH.stat().st_mtime_ns,
            forecast_mtime_ns,
            revision,
            int(zoom),
            int(tile_x),
            int(tile_y),
            int(zoom),
        )
        if tile_weight <= 0.0:
            continue
        frame_sums += np.asarray(tile_sums, dtype=np.float64)
        total_weight += float(tile_weight)
    risks = (
        (frame_sums / max(total_weight, 1e-6)).astype(np.float64).tolist()
        if total_weight > 0.0
        else [0.0 for _ in frame_labels]
    )
    return {
        "risks": risks,
        "frame_labels": frame_labels,
        "zoom": int(zoom),
        "tile_count": len(unique_tiles),
    }


@api.get("/segment-tiles/meta")
def segment_tiles_meta() -> JSONResponse:
    if not road_tile_assets_ready():
        raise HTTPException(status_code=404, detail="road tiles are not ready")
    return JSONResponse(
        content=load_road_tile_meta(),
        headers={"Cache-Control": "no-store"},
    )


@api.get("/segment-tiles/{z}/{x}/{y}.json")
def segment_tile(z: int, x: int, y: int) -> JSONResponse:
    if not road_tile_assets_ready():
        raise HTTPException(status_code=404, detail="road tiles are not ready")
    payload = load_tile_payload(z=z, x=x, y=y)
    if payload is None:
        raise HTTPException(status_code=404, detail="road tiles are not ready")
    return JSONResponse(
        content=payload,
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@api.get("/segment-raster-tiles/{frame_idx}/{z}/{x}/{y}.png")
def segment_raster_tile(frame_idx: int, z: int, x: int, y: int) -> Response:
    if not raster_tile_assets_ready():
        raise HTTPException(status_code=404, detail="road raster tiles are not ready")
    png = load_raster_tile_png(frame_idx=frame_idx, z=z, x=x, y=y)
    if png is None:
        blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
        buffer = io.BytesIO()
        blank.save(buffer, format="PNG")
        png = buffer.getvalue()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )

app = api


def main() -> None:
    print_http_startup(SERVICE_NAME, HOST, PORT)
    uvicorn.run(app, host=HOST, port=PORT)


if __name__ == "__main__":
    main()
