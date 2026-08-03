"""Standalone traffic-safety app with nationwide weekly risk overlay and predictor."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from email.message import EmailMessage
from functools import lru_cache
import html
import io
import json
import os
from pathlib import Path
import smtplib
import sqlite3
import sys

from fastapi import Body, FastAPI, HTTPException, Query, Request, Response
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles
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
    provider_statuses,
)
from road_tiles import (
    ROAD_TILE_DB_PATH,
    ROAD_RASTER_TILE_DB_PATH,
    ROAD_TILE_BASELINE_PATH,
    ROAD_TILE_FORECAST_PATH,
    load_road_tile_meta,
    load_weather_overlay,
    load_raster_tile_png,
    load_weather_raster_tile_png,
    load_tile_payload,
    raster_tile_assets_ready,
    road_tile_assets_ready,
    weather_overlay_assets_ready,
    weather_raster_assets_ready,
)
from segment_runtime import load_segment_runtime, score_segments_in_bbox
from scripts.common import CDC_SVI_YEAR, CEJST_VERSION, ROAD_SEGMENTS_PATH
from predict import (
    MODEL_BUNDLE,
    MODEL_VERSION,
    explain_for_result,
    predict_traffic_safety,
    predict_traffic_safety_live,
)
from api_ratelimit import install_rate_limit_middleware, rate_limiter_from_env
from api_v1 import V1Dependencies, build_v1_router
from equity import equity_for_tract as _equity_for_tract
from equity import load_equity_overlay as _get_equity_overlay
from geo_lookup import tract_of as _tract_of
from grant_store import get_default_store as _get_grant_store
from watch_store import get_default_store as _get_watch_store

STATIC_DIR = SRC_DIR / "static"
STATIC_URL = "/traffic-safety-static"
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
SEGMENT_MODEL_PATH = REPO_DIR / "models" / "traffic_safety_segments.joblib"
LIVE_PROVIDER_CHOICES = ["auto", *[status.name for status in provider_statuses()]]


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


@lru_cache(maxsize=1)
def _blank_tile_png() -> bytes:
    blank = Image.new("RGBA", (TILE_SIZE, TILE_SIZE), (0, 0, 0, 0))
    buffer = io.BytesIO()
    blank.save(buffer, format="PNG")
    return buffer.getvalue()


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
        "weather_overlay_ready": bool(weather_overlay_assets_ready()),
        "weather_raster_ready": bool(weather_raster_assets_ready()),
        "weather_tile_revision": str(road_meta.get("run_id", "")),
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
          <div class="ops-kicker">US ROAD RISK ASSESSMENT</div>
          <div class="ops-title-row">
            <h1>Road Risk Monitor</h1>
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
              <input id="layer-weather" type="checkbox">
              <span>Weather overlay</span>
            </label>
            <label class="layer-toggle layer-toggle-subtle">
              <input id="layer-weather-precip" type="checkbox">
              <span>Precipitation</span>
            </label>
            <label class="layer-toggle layer-toggle-subtle">
              <input id="layer-weather-wind" type="checkbox">
              <span>Wind</span>
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

          <div id="weather-legend" class="side-section weather-legend" hidden>
            <div class="side-title">Weather legend</div>
            <div id="weather-legend-precip" class="weather-legend-group" hidden>
              <div class="weather-legend-title">Precipitation</div>
              <div class="weather-legend-bar weather-legend-bar-precipitation"></div>
              <div class="weather-legend-labels">
                <span>Low</span>
                <span>High</span>
              </div>
              <div class="help-copy">Raster forecast tiles where darker blue means stronger wetness probability.</div>
            </div>
            <div id="weather-legend-wind" class="weather-legend-group" hidden>
              <div class="weather-legend-title">Wind</div>
              <div class="weather-legend-bar weather-legend-bar-wind"></div>
              <div class="weather-legend-labels">
                <span>Light</span>
                <span>Strong</span>
              </div>
              <div class="help-copy">Live station arrows from the hourly forecast feed. Direction is heading; amber intensity shows speed.</div>
            </div>
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
              <div class="page-section-kicker">Data sources</div>
              <div class="about-section-head">
                <div>
                  <h2>Public national datasets feed the forecast stack</h2>
                </div>
                <p>
                  This system integrates multiple public datasets so roadway geometry, crash history, weather archives,
                  and live forecast signals can be scored together in one national pipeline.
                </p>
              </div>
              <div class="grid-3 about-feature-grid">
                <article class="feature-card">
                  <h3>FARS</h3>
                  <p>Fatal crash records published by NHTSA provide high-severity historical incident labels.</p>
                </article>
                <article class="feature-card">
                  <h3>US-Accidents</h3>
                  <p>Large-scale roadway incident records expand the historical sample beyond fatal crashes alone.</p>
                </article>
                <article class="feature-card">
                  <h3>NOAA ISD-Lite</h3>
                  <p>Hourly historical weather observations anchor the training data with temperature, wind, and wet-hour context.</p>
                </article>
                <article class="feature-card">
                  <h3>TIGER/Line roads</h3>
                  <p>National roadway geometry is simplified into machine-readable segments that can be forecast consistently.</p>
                </article>
                <article class="feature-card">
                  <h3>National Weather Service</h3>
                  <p>Live forecasts update the current 24-hour outlook so the map reflects changing atmospheric conditions.</p>
                </article>
              </div>
            </section>

            <section class="page-card page-card--wide">
              <div class="page-section-kicker">Methodology</div>
              <div class="about-section-head">
                <div>
                  <h2>Dual-layer modeling balances national coverage with road-level detail</h2>
                </div>
                <p>
                  The system uses a dual-layer approach: a nationwide H3-based baseline model and a road-segment level
                  forecasting model. Predictions incorporate historical incident frequency, temporal features such as
                  hour, day, and seasonality, and weather conditions including temperature, precipitation, wind, and
                  related atmospheric context. Live predictions are updated using real-time weather data.
                </p>
              </div>
              <div class="grid-3 about-feature-grid">
                <article class="feature-card">
                  <h3>Nationwide baseline</h3>
                  <p>The H3 layer provides a coarse national prior so broad spatial patterns can be learned consistently across the country.</p>
                </article>
                <article class="feature-card">
                  <h3>Segment forecasting</h3>
                  <p>The road model refines that prior at segment scale, preserving corridor-level variation that county or state summaries lose.</p>
                </article>
                <article class="feature-card">
                  <h3>Live adjustment</h3>
                  <p>Current and forecast weather conditions shift the learned baseline so each frame responds to changing operating conditions.</p>
                </article>
              </div>
            </section>

            <section class="page-card page-card--wide about-limitations">
              <div class="page-section-kicker">Limitations</div>
              <div class="about-section-head">
                <div>
                  <h2>This is a research and engineering prototype</h2>
                </div>
                <p>
                  The system is useful for exploration and hypothesis testing, but it should not be treated as production-grade operational guidance without deeper validation.
                </p>
              </div>
              <div class="about-warning-box">
                <ul class="feature-list">
                  <li>Historical datasets may be incomplete or uneven across regions and time periods.</li>
                  <li>Exposure is simplified because the current system does not incorporate traffic volume data.</li>
                  <li>Incident reporting datasets can contain structural bias in what gets recorded and how it is classified.</li>
                  <li>Evaluation is difficult because traffic incidents are highly imbalanced and spatially persistent.</li>
                </ul>
                <p class="about-warning-note">
                  Predictions should not be used for operational decision-making without further validation.
                </p>
              </div>
            </section>

            <section class="page-card page-card--wide">
              <div class="page-section-kicker">Purpose</div>
              <div class="about-section-head">
                <div>
                  <h2>Why build this system at all</h2>
                </div>
                <p>
                  The goal of this project is to explore how machine learning systems can support transportation safety,
                  disaster response, and infrastructure resilience at national scale. It is intended to help people think
                  more clearly about where predictive risk layers can be useful, where they remain fragile, and what
                  better public safety tooling could look like.
                </p>
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


api = FastAPI(
    title=SERVICE_NAME,
    version="1.0.0",
    description=(
        "Public road-risk API exposing climatological and live weather-adjusted "
        "crash risk for points, routes, and map areas across the United States. "
        "See /v1/docs for the interactive reference."
    ),
    openapi_tags=[{"name": "v1", "description": "Public, versioned road-risk endpoints."}],
)
api.add_middleware(GZipMiddleware, minimum_size=1024)
api.mount(STATIC_URL, StaticFiles(directory=str(STATIC_DIR)), name="traffic-safety-static")

SECURITY_HEADERS = {
    "X-Content-Type-Options": "nosniff",
    "Referrer-Policy": "strict-origin-when-cross-origin",
}


@api.middleware("http")
async def add_security_headers(request: Request, call_next):
    response = await call_next(request)
    for header, value in SECURITY_HEADERS.items():
        response.headers.setdefault(header, value)
    return response


# --- Public /v1 API: CORS, rate limiting, and router wiring ---
_CORS_ORIGINS = [
    origin.strip()
    for origin in os.getenv("TRAFFIC_SAFETY_CORS_ORIGINS", "*").split(",")
    if origin.strip()
] or ["*"]
api.add_middleware(
    CORSMiddleware,
    allow_origins=_CORS_ORIGINS,
    allow_methods=["GET", "POST", "OPTIONS"],
    allow_headers=["*"],
)

_RATE_LIMITER = rate_limiter_from_env()
install_rate_limit_middleware(api, _RATE_LIMITER, path_prefix="/v1")

_MODEL_EVAL_REPORT_PATH = REPO_DIR / "data" / "model_eval_report.json"


def _load_model_report() -> dict:
    if _MODEL_EVAL_REPORT_PATH.exists():
        try:
            return json.loads(_MODEL_EVAL_REPORT_PATH.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            pass
    return {"available": False, "metrics": dict(MODEL_BUNDLE.get("metrics", {}) or {})}


def _equity_at_point(lat: float, lon: float) -> dict:
    """Resolve a point to its census tract, then that tract's equity record."""
    geoid = _tract_of(lat, lon)
    record = _equity_for_tract(geoid or "")
    record["tract_geoid"] = geoid
    return record


_v1_overlay_config = OVERLAY["config"]
api.include_router(
    build_v1_router(
        V1Dependencies(
            service_name=SERVICE_NAME,
            model_version=MODEL_VERSION,
            model_ready=bool(MODEL_BUNDLE),
            coverage={
                key: float(_v1_overlay_config[key])
                for key in ("lat_min", "lat_max", "lon_min", "lon_max")
                if key in _v1_overlay_config
            },
            frame_labels=[str(frame) for frame in OVERLAY["frames"]],
            risk_quantiles=MODEL_BUNDLE.get("risk_quantiles"),
            provider_statuses=provider_statuses,
            provider_choices=LIVE_PROVIDER_CHOICES,
            rate_limit_per_min=(_RATE_LIMITER.rate_per_min if _RATE_LIMITER else None),
            predict_point=predict_traffic_safety,
            predict_point_live=predict_traffic_safety_live,
            explain_point=explain_for_result,
            h3_resolution=int(MODEL_BUNDLE.get("resolution", 5)),
            model_metrics=dict(MODEL_BUNDLE.get("metrics", {}) or {}),
            model_report_loader=_load_model_report,
            risk_cube=OVERLAY["risk"],
            watch_store_provider=_get_watch_store,
            grant_provider=_get_grant_store,
            equity_provider=_equity_at_point,
            equity_overlay_provider=_get_equity_overlay,
            equity_vintage={
                "svi": f"CDC/ATSDR SVI {CDC_SVI_YEAR}",
                "cejst": f"CEJST {CEJST_VERSION}",
                "tract_boundaries": "2010 census tracts",
            },
        )
    )
)


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
def health(response: Response) -> dict[str, object]:
    response.headers["Cache-Control"] = "no-store"
    road_tile_meta = load_road_tile_meta() if road_tile_assets_ready() else {}
    return {
        "service": SERVICE_NAME,
        "status": "ok",
        "frames": len(OVERLAY["frames"]),
        "model_version": MODEL_VERSION,
        "model_ready": bool(MODEL_BUNDLE),
        "segment_model_ready": bool(SEGMENT_MODEL_PATH.exists()),
        "overlay_ready": bool((TILES_DIR / "overlay.npz").exists()),
        "road_tiles_ready": road_tile_assets_ready(),
        "road_raster_tiles_ready": raster_tile_assets_ready(),
        "weather_overlay_ready": weather_overlay_assets_ready(),
        "weather_raster_tiles_ready": weather_raster_assets_ready(),
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
        return Response(content=_blank_tile_png(), media_type="image/png")

    try:
        png = _render_tile_png(frame_idx=frame_idx, z=z, x=x, y=y)
    except ValueError as exc:
        raise HTTPException(status_code=404, detail=str(exc)) from exc

    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=86400"},
    )


def _validate_provider(provider: str) -> None:
    if provider.strip().lower() not in LIVE_PROVIDER_CHOICES:
        raise HTTPException(
            status_code=400,
            detail=f"unknown live weather provider '{provider}'; choose from {LIVE_PROVIDER_CHOICES}",
        )


@api.get("/api/live-risk")
def live_risk(
    response: Response,
    lat: float = Query(..., ge=-90.0, le=90.0),
    lon: float = Query(..., ge=-180.0, le=180.0),
    forecast_hours: int = Query(0, ge=0, le=48),
    provider: str = "auto",
) -> dict[str, object]:
    response.headers["Cache-Control"] = "no-store"
    _validate_provider(provider)
    try:
        return predict_traffic_safety_live(
            lat=lat,
            lon=lon,
            forecast_hours=forecast_hours,
            provider=provider,
        )
    except LiveWeatherProviderError as exc:
        raise HTTPException(status_code=503, detail=str(exc)) from exc
    except requests.RequestException as exc:
        raise HTTPException(status_code=502, detail=f"weather provider request failed: {exc}") from exc


@api.get("/api/segment-risk")
def segment_risk(
    response: Response,
    min_lat: float = Query(..., ge=-90.0, le=90.0),
    max_lat: float = Query(..., ge=-90.0, le=90.0),
    min_lon: float = Query(..., ge=-180.0, le=180.0),
    max_lon: float = Query(..., ge=-180.0, le=180.0),
    forecast_hours: int = Query(0, ge=0, le=48),
    provider: str = "auto",
    limit: int = Query(1500, ge=1, le=5000),
) -> dict[str, object]:
    response.headers["Cache-Control"] = "no-store"
    _validate_provider(provider)
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
    except requests.RequestException as exc:
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


@api.get("/weather-overlay/meta")
def weather_overlay_meta() -> JSONResponse:
    if not weather_overlay_assets_ready():
        raise HTTPException(status_code=404, detail="weather overlay is not ready")
    return JSONResponse(
        content=load_weather_overlay(),
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
        png = _blank_tile_png()
    return Response(
        content=png,
        media_type="image/png",
        headers={"Cache-Control": "public, max-age=31536000, immutable"},
    )


@api.get("/weather-raster-tiles/{mode}/{frame_idx}/{z}/{x}/{y}.png")
def weather_raster_tile(mode: str, frame_idx: int, z: int, x: int, y: int) -> Response:
    if mode not in {"temperature", "precipitation"}:
        raise HTTPException(status_code=404, detail="unsupported weather raster mode")
    if not weather_raster_assets_ready():
        raise HTTPException(status_code=404, detail="weather raster tiles are not ready")
    png = load_weather_raster_tile_png(mode=mode, frame_idx=frame_idx, z=z, x=x, y=y)
    if png is None:
        png = _blank_tile_png()
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
