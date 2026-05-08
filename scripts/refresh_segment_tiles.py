from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
import io
import json
import math
from pathlib import Path
import sqlite3
import sys
from typing import Any

import joblib
import numpy as np
import pandas as pd
from PIL import Image, ImageDraw, ImageFilter
import requests
from sklearn.neighbors import BallTree

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from live_weather import DEFAULT_USER_AGENT
from model_support import (
    fahrenheit_to_celsius,
    lookup_weather_climatology,
    parse_wind_speed_string_mps,
    relative_humidity_from_temp_dewpoint,
)
from road_tiles import (
    ROAD_TILE_BASELINE_PATH,
    DEFAULT_TILE_ZOOM_MAX,
    DEFAULT_TILE_ZOOM_MIN,
    ROAD_TILE_DB_PATH,
    ROAD_TILE_FORECAST_PATH,
    ROAD_TILE_META_PATH,
    ROAD_RASTER_TILE_DB_PATH,
    WEATHER_OVERLAY_PATH,
    WEATHER_RASTER_TILE_DB_PATH,
)
from segment_model_support import (
    DEFAULT_MTFCC_VALUES,
    DEFAULT_RTTYP_VALUES,
    build_segment_feature_matrix,
    build_static_segment_frame,
)
from scripts.common import (
    REPRESENTATIVE_STATIONS_PATH,
    ROAD_SEGMENTS_PATH,
    SEGMENT_MODEL_BUNDLE_PATH,
    ensure_dirs,
)


NWS_BASE_URL = "https://api.weather.gov"
WET_KEYWORDS = (
    "rain",
    "shower",
    "storm",
    "snow",
    "sleet",
    "hail",
    "drizzle",
    "ice",
    "freezing",
    "thunder",
)
ROAD_TILE_COORD_SCALE = 8.0
RASTER_ZOOM_MIN = 4
RASTER_LEAF_ZOOM = 8
WEATHER_BLEND_K = 8
WEATHER_BLEND_DECAY_KM = 420.0
SPATIAL_SMOOTH_K = 10
SPATIAL_SMOOTH_DECAY_KM = 18.0
SPATIAL_SMOOTH_MAX_KM = 42.0
SPATIAL_SMOOTH_BLEND = 0.55
TILE_NORMALIZE_HIGH_QUANTILE = 0.93
EARTH_RADIUS_KM = 6371.0088
WEATHER_HEATMAP_FIELD_ZOOM = 6
WEATHER_HEATMAP_RADIUS_PX = 220.0
WEATHER_TEMPERATURE_ALPHA_MAX = 168.0
WEATHER_TEMPERATURE_EDGE_FEATHER_SCALE = 1.75
WEATHER_RASTER_WEIGHT_DECAY_KM = 420.0
WEATHER_RASTER_BLEND_K = 24
WEATHER_PRECIP_MAX_DISTANCE_KM = 900.0
WEATHER_WIND_SAMPLE_STRIDE = 10
WEATHER_RASTER_HALO_TILES = 1
WEATHER_FIELD_SAMPLES_PER_TILE = 64


@dataclass(frozen=True)
class StationForecastHour:
    timestamp_local: datetime
    month: int
    hour_of_week: int
    temp_c: float
    relative_humidity_pct: float
    wind_speed_mps: float
    wind_dir_deg: float | None
    wet_hour: float
    precip_probability_pct: float
    source: str


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    parser.add_argument("--provider", default="nws")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--zoom-min", type=int, default=DEFAULT_TILE_ZOOM_MIN)
    parser.add_argument("--zoom-max", type=int, default=DEFAULT_TILE_ZOOM_MAX)
    parser.add_argument("--raster-zoom-min", type=int, default=RASTER_ZOOM_MIN)
    parser.add_argument("--raster-zoom-max", type=int, default=RASTER_LEAF_ZOOM)
    return parser.parse_args()


def request_json(url: str) -> dict[str, Any]:
    response = requests.get(
        url,
        headers={
            "User-Agent": DEFAULT_USER_AGENT,
            "Accept": "application/geo+json, application/json",
        },
        timeout=20,
    )
    response.raise_for_status()
    return response.json()


def wet_hour_from_summary(summary: str | None, probability: float | None) -> float:
    if probability is not None and float(probability) >= 30.0:
        return 1.0
    summary_value = (summary or "").lower()
    return 1.0 if any(keyword in summary_value for keyword in WET_KEYWORDS) else 0.0


def quantitative_value(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        value = payload.get("value")
        if value is None:
            return None
        return float(value)
    return None


def humidity_value(humidity_pct: float | None, temp_c: float, dewpoint_c: float) -> float:
    if humidity_pct is not None:
        return max(0.0, min(100.0, float(humidity_pct)))
    return float(relative_humidity_from_temp_dewpoint(temp_c, dewpoint_c).item())


def parse_wind_direction_degrees(value: Any) -> float | None:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        return float(value) % 360.0
    text = str(value).strip().upper()
    if not text or text in {"CALM", "VRB", "VARIABLE"}:
        return None
    try:
        return float(text) % 360.0
    except ValueError:
        pass

    cardinal_map = {
        "N": 0.0,
        "NNE": 22.5,
        "NE": 45.0,
        "ENE": 67.5,
        "E": 90.0,
        "ESE": 112.5,
        "SE": 135.0,
        "SSE": 157.5,
        "S": 180.0,
        "SSW": 202.5,
        "SW": 225.0,
        "WSW": 247.5,
        "W": 270.0,
        "WNW": 292.5,
        "NW": 315.0,
        "NNW": 337.5,
    }
    return cardinal_map.get(text)


def fallback_station_series(
    *,
    station_index: int,
    utc_offset_hours: int,
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
    hours: int,
) -> list[StationForecastHour]:
    base_utc = datetime.now(timezone.utc).replace(minute=0, second=0, microsecond=0)
    local_tz = timezone(timedelta(hours=int(utc_offset_hours)))
    output: list[StationForecastHour] = []
    for offset in range(hours):
        timestamp_local = (base_utc + timedelta(hours=offset)).astimezone(local_tz)
        hour_of_week = timestamp_local.weekday() * 24 + timestamp_local.hour
        weather = lookup_weather_climatology(
            weather_cube=weather_cube,
            station_indices=np.array([station_index], dtype=np.int16),
            months=np.array([timestamp_local.month], dtype=np.int8),
            hour_of_week=np.array([hour_of_week], dtype=np.int16),
            weather_defaults=weather_defaults,
        )[0]
        output.append(
            StationForecastHour(
                timestamp_local=timestamp_local,
                month=int(timestamp_local.month),
                hour_of_week=int(hour_of_week),
                temp_c=float(weather[0]),
                relative_humidity_pct=float(weather[1]),
                wind_speed_mps=float(weather[2]),
                wind_dir_deg=None,
                wet_hour=float(weather[3]),
                precip_probability_pct=float(weather[3]) * 100.0,
                source="climatology",
            )
        )
    return output


def fetch_nws_station_series(
    *,
    lat: float,
    lon: float,
    hours: int,
) -> list[StationForecastHour]:
    points_payload = request_json(f"{NWS_BASE_URL}/points/{lat},{lon}")
    hourly_url = points_payload.get("properties", {}).get("forecastHourly")
    if not hourly_url:
        raise RuntimeError("NWS points lookup returned no hourly forecast URL")

    payload = request_json(str(hourly_url))
    periods = payload.get("properties", {}).get("periods") or []
    if not periods:
        raise RuntimeError("NWS hourly forecast returned no periods")

    output: list[StationForecastHour] = []
    for offset in range(hours):
        period = periods[min(offset, len(periods) - 1)]
        timestamp_local = datetime.fromisoformat(
            str(period.get("startTime")).replace("Z", "+00:00")
        )
        temp_c = float(
            fahrenheit_to_celsius(period.get("temperature"))
            if str(period.get("temperatureUnit", "")).upper() == "F"
            else float(period.get("temperature") or 0.0)
        )
        dewpoint_c = float(quantitative_value(period.get("dewpoint")) or 0.0)
        humidity_pct = quantitative_value(period.get("relativeHumidity"))
        wind_speed_mps = float(parse_wind_speed_string_mps(period.get("windSpeed")))
        wind_dir_deg = parse_wind_direction_degrees(period.get("windDirection"))
        precipitation_probability = quantitative_value(
            period.get("probabilityOfPrecipitation")
        )
        summary = " ".join(
            part.strip()
            for part in (
                str(period.get("shortForecast", "")).strip(),
                str(period.get("detailedForecast", "")).strip(),
            )
            if part
        )
        hour_of_week = timestamp_local.weekday() * 24 + timestamp_local.hour
        output.append(
            StationForecastHour(
                timestamp_local=timestamp_local,
                month=int(timestamp_local.month),
                hour_of_week=int(hour_of_week),
                temp_c=temp_c,
                relative_humidity_pct=humidity_value(humidity_pct, temp_c, dewpoint_c),
                wind_speed_mps=max(0.0, wind_speed_mps),
                wind_dir_deg=wind_dir_deg,
                wet_hour=wet_hour_from_summary(summary, precipitation_probability),
                precip_probability_pct=float(precipitation_probability or 0.0),
                source="nws",
            )
        )
    return output


def build_station_series(
    representative: pd.DataFrame,
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
    hours: int,
    workers: int,
) -> tuple[dict[int, list[StationForecastHour]], dict[str, int]]:
    series_by_station: dict[int, list[StationForecastHour]] = {}
    counts = {"nws": 0, "climatology": 0}

    with ThreadPoolExecutor(max_workers=max(1, int(workers))) as executor:
        future_map = {
            executor.submit(
                fetch_nws_station_series,
                lat=float(row["LAT"]),
                lon=float(row["LON"]),
                hours=hours,
            ): row
            for _, row in representative.iterrows()
        }
        for future in as_completed(future_map):
            row = future_map[future]
            station_index = int(row["station_index"])
            try:
                station_series = future.result()
            except Exception:
                station_series = fallback_station_series(
                    station_index=station_index,
                    utc_offset_hours=int(row["utc_offset_hours"]),
                    weather_cube=weather_cube,
                    weather_defaults=weather_defaults,
                    hours=hours,
                )
                counts["climatology"] += 1
            else:
                counts["nws"] += 1
            series_by_station[station_index] = station_series
            print(
                f"station_index={station_index} source={station_series[0].source}",
                flush=True,
            )

    return series_by_station, counts


def build_station_weather_matrices(
    series_by_station: dict[int, list[StationForecastHour]],
    station_count: int,
    hours: int,
) -> dict[str, np.ndarray]:
    matrices = {
        "temp_c": np.zeros((station_count, hours), dtype=np.float32),
        "relative_humidity_pct": np.zeros((station_count, hours), dtype=np.float32),
        "wind_speed_mps": np.zeros((station_count, hours), dtype=np.float32),
        "wind_dir_deg": np.full((station_count, hours), np.nan, dtype=np.float32),
        "wet_hour": np.zeros((station_count, hours), dtype=np.float32),
        "precip_probability_pct": np.zeros((station_count, hours), dtype=np.float32),
    }
    for station_index, rows in series_by_station.items():
        for hour_idx, row in enumerate(rows[:hours]):
            matrices["temp_c"][station_index, hour_idx] = float(row.temp_c)
            matrices["relative_humidity_pct"][station_index, hour_idx] = float(
                row.relative_humidity_pct
            )
            matrices["wind_speed_mps"][station_index, hour_idx] = float(row.wind_speed_mps)
            if row.wind_dir_deg is not None:
                matrices["wind_dir_deg"][station_index, hour_idx] = float(row.wind_dir_deg)
            matrices["wet_hour"][station_index, hour_idx] = float(row.wet_hour)
            matrices["precip_probability_pct"][station_index, hour_idx] = float(
                row.precip_probability_pct
            )
    return matrices


def write_weather_overlay(
    *,
    representative: pd.DataFrame,
    series_by_station: dict[int, list[StationForecastHour]],
    station_weather: dict[str, np.ndarray],
    source_counts: dict[str, int],
    generated_at: datetime,
    forecast_start_utc: datetime,
    forecast_end_utc: datetime,
    provider: str,
    hours: int,
) -> None:
    stations_payload = []
    precip_matrix = np.asarray(
        station_weather["precip_probability_pct"],
        dtype=np.float32,
    )
    temp_matrix = np.asarray(station_weather["temp_c"], dtype=np.float32)
    wind_matrix = np.asarray(station_weather["wind_speed_mps"], dtype=np.float32)
    wind_dir_matrix = np.asarray(station_weather["wind_dir_deg"], dtype=np.float32)
    wet_matrix = np.asarray(station_weather["wet_hour"], dtype=np.float32)
    for _, row in representative.iterrows():
        station_index = int(row["station_index"])
        stations_payload.append(
            {
                "station_index": station_index,
                "lat": round(float(row["LAT"]), 6),
                "lon": round(float(row["LON"]), 6),
                "source": str(series_by_station[station_index][0].source),
                "precip_probability_pct": [
                    round(float(value), 2) for value in precip_matrix[station_index, :hours]
                ],
                "temp_c": [round(float(value), 2) for value in temp_matrix[station_index, :hours]],
                "wind_speed_mps": [
                    round(float(value), 2) for value in wind_matrix[station_index, :hours]
                ],
                "wind_dir_deg": [
                    None if math.isnan(float(value)) else round(float(value), 1)
                    for value in wind_dir_matrix[station_index, :hours]
                ],
                "wet_hour": [round(float(value), 3) for value in wet_matrix[station_index, :hours]],
            }
        )

    payload = {
        "run_id": generated_at.strftime("%Y%m%dT%H%M%SZ"),
        "generated_at_utc": generated_at.isoformat(),
        "forecast_start_utc": forecast_start_utc.isoformat(),
        "forecast_end_utc": forecast_end_utc.isoformat(),
        "provider": str(provider),
        "hours": int(hours),
        "frame_labels": [f"+{hour}h" for hour in range(int(hours))],
        "layer_kind": "precip_probability_pct",
        "layer_label": "Precipitation probability",
        "scale_min": 0.0,
        "scale_max": 100.0,
        "available_layers": [
            {
                "id": "precipitation",
                "label": "Precipitation",
                "renderer": "raster",
                "data_key": "precip_probability_pct",
            },
            {
                "id": "temperature",
                "label": "Temperature",
                "renderer": "raster",
                "data_key": "temp_c",
            },
            {
                "id": "wind",
                "label": "Wind",
                "renderer": "arrows",
                "speed_key": "wind_speed_mps",
                "direction_key": "wind_dir_deg",
            },
        ],
        "station_count": int(len(stations_payload)),
        "station_source_counts": dict(source_counts),
        "stations": stations_payload,
    }
    WEATHER_OVERLAY_PATH.write_text(json.dumps(payload), encoding="utf-8")


def build_station_timing_matrices(
    series_by_station: dict[int, list[StationForecastHour]],
    station_count: int,
    hours: int,
) -> dict[str, np.ndarray]:
    matrices = {
        "month": np.zeros((station_count, hours), dtype=np.int8),
        "hour_of_week": np.zeros((station_count, hours), dtype=np.int16),
    }
    for station_index, rows in series_by_station.items():
        for hour_idx, row in enumerate(rows[:hours]):
            matrices["month"][station_index, hour_idx] = int(row.month)
            matrices["hour_of_week"][station_index, hour_idx] = int(row.hour_of_week)
    return matrices


def build_station_climatology_matrices(
    *,
    timing: dict[str, np.ndarray],
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
) -> dict[str, np.ndarray]:
    station_count, hours = timing["month"].shape
    station_idx = np.arange(station_count, dtype=np.int16)[:, None]
    station_idx = np.repeat(station_idx, hours, axis=1).reshape(-1)
    months = timing["month"].reshape(-1).astype(np.int8)
    hour_of_week = timing["hour_of_week"].reshape(-1).astype(np.int16)
    values = lookup_weather_climatology(
        weather_cube=weather_cube,
        station_indices=station_idx,
        months=months,
        hour_of_week=hour_of_week,
        weather_defaults=weather_defaults,
    ).reshape(station_count, hours, 4)
    return {
        "temp_c": values[:, :, 0].astype(np.float32),
        "relative_humidity_pct": values[:, :, 1].astype(np.float32),
        "wind_speed_mps": values[:, :, 2].astype(np.float32),
        "wet_hour": values[:, :, 3].astype(np.float32),
    }


def build_segment_weather_blend(
    roads: pd.DataFrame,
    representative: pd.DataFrame,
    k: int = WEATHER_BLEND_K,
) -> tuple[np.ndarray, np.ndarray]:
    rep_coords = np.radians(representative[["LAT", "LON"]].to_numpy(dtype=np.float64))
    road_coords = np.radians(roads[["center_lat", "center_lon"]].to_numpy(dtype=np.float64))
    tree = BallTree(rep_coords, metric="haversine")
    k = max(1, min(int(k), len(representative)))
    dist_rad, idx = tree.query(road_coords, k=k)
    dist_km = dist_rad * EARTH_RADIUS_KM
    weights = np.exp(-dist_km / WEATHER_BLEND_DECAY_KM).astype(np.float32)
    weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)
    return idx.astype(np.int16), weights.astype(np.float32)


def interpolate_hour_counts(
    hour_counts: np.ndarray,
    local_hour_of_week: np.ndarray,
) -> np.ndarray:
    local_hour_of_week = np.asarray(local_hour_of_week, dtype=np.float32)
    lower_idx = np.floor(local_hour_of_week).astype(np.int16)
    upper_idx = ((lower_idx.astype(np.int32) + 1) % 168).astype(np.int16)
    upper_weight = (local_hour_of_week - lower_idx.astype(np.float32)).astype(np.float32)
    lower_weight = 1.0 - upper_weight
    rows = np.arange(hour_counts.shape[0], dtype=np.int32)
    lower_values = hour_counts[rows, lower_idx].astype(np.float32)
    upper_values = hour_counts[rows, upper_idx].astype(np.float32)
    return lower_weight * lower_values + upper_weight * upper_values


def risk_rgb(risk_byte: int) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(risk_byte) / 255.0))
    t = 1.0 / (1.0 + math.exp(-6.0 * (t - 0.46)))
    stops = [
        (0.0, (17, 122, 101)),
        (0.28, (72, 187, 120)),
        (0.52, (242, 201, 76)),
        (0.74, (242, 130, 49)),
        (1.0, (203, 43, 39)),
    ]
    for idx in range(1, len(stops)):
        if t <= stops[idx][0]:
            start_t, start_color = stops[idx - 1]
            end_t, end_color = stops[idx]
            local_t = (t - start_t) / max(1e-6, end_t - start_t)
            return (
                int(round(start_color[0] + (end_color[0] - start_color[0]) * local_t)),
                int(round(start_color[1] + (end_color[1] - start_color[1]) * local_t)),
                int(round(start_color[2] + (end_color[2] - start_color[2]) * local_t)),
            )
    return (203, 43, 39)


def raster_rgba(
    *,
    risk_byte: int,
    road_kind: int,
    zoom: int,
) -> tuple[int, int, int, int] | None:
    risk_value = int(risk_byte)
    if zoom <= 4:
        if road_kind < 2 and risk_value < 176:
            return None
        alpha = 96 if risk_value < 92 else 132 if risk_value < 148 else 188 if risk_value < 205 else 232
    elif zoom <= 5:
        if road_kind < 2 and risk_value < 144:
            return None
        alpha = 110 if risk_value < 88 else 146 if risk_value < 144 else 196 if risk_value < 205 else 234
    elif zoom <= 6:
        alpha = 120 if risk_value < 82 else 156 if risk_value < 136 else 204 if risk_value < 205 else 236
    elif zoom <= 8:
        alpha = 134 if risk_value < 74 else 172 if risk_value < 128 else 214 if risk_value < 205 else 236
    else:
        alpha = 236
    red, green, blue = risk_rgb(risk_value)
    return (red, green, blue, alpha)


def raster_stroke_width(road_kind: int, zoom: int) -> int:
    if zoom <= 4:
        return 3 if road_kind >= 2 else 2
    if zoom <= 6:
        return 4 if road_kind >= 2 else 3 if road_kind >= 1 else 2
    return 5 if road_kind >= 2 else 4 if road_kind >= 1 else 3


def tile_normalize_radius(zoom: int) -> float:
    return 2.0 if int(zoom) <= 5 else 1.5 if int(zoom) == 6 else 1.0


def tile_normalize_decay(zoom: int) -> float:
    return 1.2 if int(zoom) <= 5 else 0.9 if int(zoom) == 6 else 0.7


def tile_normalize_blend(zoom: int) -> float:
    if int(zoom) <= 4:
        return 0.68
    if int(zoom) <= 5:
        return 0.58
    if int(zoom) == 6:
        return 0.46
    if int(zoom) == 7:
        return 0.34
    return 0.24


def tile_normalize_min_span(zoom: int) -> float:
    if int(zoom) <= 4:
        return 56.0
    if int(zoom) <= 5:
        return 48.0
    if int(zoom) == 6:
        return 40.0
    if int(zoom) == 7:
        return 32.0
    return 26.0


def tile_normalize_max_shift(zoom: int) -> float:
    if int(zoom) <= 4:
        return 34.0
    if int(zoom) <= 5:
        return 28.0
    if int(zoom) == 6:
        return 22.0
    if int(zoom) == 7:
        return 18.0
    return 14.0


def build_tile_normalizers(
    *,
    source: sqlite3.Connection,
    render_scores: np.ndarray,
    zoom: int,
    zoom_tiles: list[tuple[int, int]],
) -> dict[tuple[int, int], tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    tile_count = len(zoom_tiles)
    hours = int(render_scores.shape[1])
    tile_center = np.zeros((tile_count, hours), dtype=np.float32)
    tile_high = np.full((tile_count, hours), 255.0, dtype=np.float32)

    for tile_idx, (tile_x, tile_y) in enumerate(zoom_tiles, start=1):
        rows = source.execute(
            """
            SELECT segment_idx
            FROM tile_entries
            WHERE z = ? AND x = ? AND y = ?
            """,
            (int(zoom), int(tile_x), int(tile_y)),
        ).fetchall()
        segment_idx = np.fromiter((int(row[0]) for row in rows), dtype=np.int32)
        if segment_idx.size:
            tile_scores = np.asarray(render_scores[segment_idx], dtype=np.float32)
            tile_center[tile_idx - 1] = tile_scores.mean(axis=0).astype(np.float32)
            tile_high[tile_idx - 1] = np.quantile(
                tile_scores,
                TILE_NORMALIZE_HIGH_QUANTILE,
                axis=0,
            ).astype(np.float32)
        if tile_idx % 100 == 0 or tile_idx == tile_count:
            print(
                f"raster_stats_zoom={zoom} tiles={tile_idx}/{tile_count}",
                flush=True,
            )

    coords = np.asarray(zoom_tiles, dtype=np.float32)
    radius = float(tile_normalize_radius(int(zoom)))
    decay = float(tile_normalize_decay(int(zoom)))
    neighborhood_center = np.empty_like(tile_center)
    neighborhood_high = np.empty_like(tile_high)

    for idx, (tile_x, tile_y) in enumerate(coords):
        dist = np.hypot(coords[:, 0] - tile_x, coords[:, 1] - tile_y)
        mask = dist <= radius + 1e-6
        masked_dist = dist[mask]
        weights = np.exp(-masked_dist / max(decay, 1e-6)).astype(np.float32)
        weights[masked_dist < 1e-6] *= 2.0
        weights /= np.maximum(weights.sum(), 1e-6)
        neighborhood_center[idx] = (tile_center[mask] * weights[:, None]).sum(axis=0)
        neighborhood_high[idx] = (tile_high[mask] * weights[:, None]).sum(axis=0)

    return {
        (int(tile_x), int(tile_y)): (
            tile_center[idx].astype(np.float32),
            tile_high[idx].astype(np.float32),
            neighborhood_center[idx].astype(np.float32),
            neighborhood_high[idx].astype(np.float32),
        )
        for idx, (tile_x, tile_y) in enumerate(zoom_tiles)
    }


def normalize_tile_frame_scores(
    *,
    raw_scores: np.ndarray,
    local_center: float,
    local_high: float,
    neighborhood_center: float,
    neighborhood_high: float,
    zoom: int,
) -> np.ndarray:
    raw = np.asarray(raw_scores, dtype=np.float32)
    local_span = max(
        float(local_high) - float(local_center),
        tile_normalize_min_span(int(zoom)),
    )
    neighborhood_span = max(
        float(neighborhood_high) - float(neighborhood_center),
        tile_normalize_min_span(int(zoom)),
    )
    scale = float(np.clip(neighborhood_span / max(local_span, 1e-6), 0.82, 1.18))
    shifted = float(neighborhood_center) + (raw - float(local_center)) * scale
    max_shift = tile_normalize_max_shift(int(zoom))
    shifted = np.clip(shifted, raw - max_shift, raw + max_shift)
    blend = float(tile_normalize_blend(int(zoom)))
    adjusted = np.clip(raw * (1.0 - blend) + shifted * blend, 0.0, 255.0)
    return np.rint(adjusted).astype(np.uint8)


def build_display_scores(
    *,
    forecast_scores: np.ndarray,
    total_counts: np.ndarray,
    road_kind: np.ndarray,
    strength: float = 0.75,
    floor: float = 0.35,
) -> np.ndarray:
    count_bins = np.linspace(0.0, np.log1p(float(np.max(total_counts)) + 1.0), 13)[1:-1]
    count_bucket = np.digitize(np.log1p(total_counts.astype(np.float32)), count_bins).astype(np.int16)
    bucket_id = (road_kind.astype(np.int16) * 12 + count_bucket).astype(np.int16)
    bucket_count = int(bucket_id.max()) + 1
    display_scores = np.empty_like(forecast_scores, dtype=np.uint8)

    for frame_idx in range(forecast_scores.shape[1]):
        raw = forecast_scores[:, frame_idx].astype(np.float32) / 255.0
        bucket_sum = np.bincount(bucket_id, weights=raw, minlength=bucket_count).astype(np.float32)
        bucket_n = np.bincount(bucket_id, minlength=bucket_count).astype(np.float32)
        bucket_mean = bucket_sum / np.maximum(bucket_n, 1.0)
        mean = bucket_mean[bucket_id]
        log_relative = np.log((raw + 0.02) / (mean + 0.02))
        uplift = 1.0 / (1.0 + np.exp(-2.5 * log_relative))
        factor = float(floor) + float(strength) * uplift
        display = np.clip(raw * factor, 0.0, 1.0)
        display_scores[:, frame_idx] = np.rint(display * 255.0).astype(np.uint8)
        print(f"display_frame={frame_idx} recalibrated", flush=True)

    return display_scores


def apply_spatial_smoothing(
    *,
    score_map: np.ndarray,
    road_coords_rad: np.ndarray,
    neighbor_tree: BallTree | None = None,
    chunk_size: int = 10000,
    neighbor_k: int = SPATIAL_SMOOTH_K,
    decay_km: float = SPATIAL_SMOOTH_DECAY_KM,
    max_distance_km: float = SPATIAL_SMOOTH_MAX_KM,
    blend: float = SPATIAL_SMOOTH_BLEND,
) -> np.ndarray:
    tree = neighbor_tree or BallTree(road_coords_rad, metric="haversine")
    total_rows = score_map.shape[0]
    hours = score_map.shape[1]
    smoothed = np.empty_like(score_map, dtype=np.uint8)

    for start in range(0, total_rows, int(chunk_size)):
        stop = min(total_rows, start + int(chunk_size))
        dist_rad, idx = tree.query(
            road_coords_rad[start:stop],
            k=max(1, min(int(neighbor_k), total_rows)),
        )
        dist_km = dist_rad * EARTH_RADIUS_KM
        weights = np.exp(-dist_km / float(decay_km)).astype(np.float32)
        weights[dist_km > float(max_distance_km)] = 0.0
        weights[:, 0] += 1.0
        weights /= np.maximum(weights.sum(axis=1, keepdims=True), 1e-6)

        neighbor_scores = np.asarray(score_map[idx], dtype=np.float32)
        neighborhood = (neighbor_scores * weights[:, :, None]).sum(axis=1)
        original = np.asarray(score_map[start:stop], dtype=np.float32).reshape(-1, hours)
        blended = np.clip(
            original * (1.0 - float(blend)) + neighborhood * float(blend),
            0.0,
            255.0,
        )
        smoothed[start:stop] = np.rint(blended).astype(np.uint8)
        print(f"spatial_smoothing={stop}/{total_rows}", flush=True)

    return smoothed


def decode_path_blob(path_blob: bytes) -> list[tuple[float, float]]:
    coords = np.frombuffer(path_blob, dtype=np.int16).reshape(-1, 2).astype(np.float32)
    coords /= ROAD_TILE_COORD_SCALE
    return [(float(x_coord), float(y_coord)) for x_coord, y_coord in coords]


def png_bytes(image: Image.Image) -> bytes:
    buffer = io.BytesIO()
    image.save(buffer, format="PNG", optimize=False, compress_level=1)
    return buffer.getvalue()


def mercator_pixel_to_latlon(world_x: float, world_y: float, zoom: int) -> tuple[float, float]:
    scale = 256.0 * (2 ** int(zoom))
    lon = float(world_x) / scale * 360.0 - 180.0
    n = math.pi - (2.0 * math.pi * float(world_y)) / scale
    lat = math.degrees(math.atan(math.sinh(n)))
    return lat, lon


def mercator_latlon_to_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    scale = 256.0 * (2 ** int(zoom))
    x = (float(lon) + 180.0) / 360.0 * scale
    lat_rad = math.radians(float(lat))
    sin_lat = min(max(math.sin(lat_rad), -0.9999), 0.9999)
    y = (
        0.5
        - math.log((1.0 + sin_lat) / (1.0 - sin_lat)) / (4.0 * math.pi)
    ) * scale
    return x, y


def weather_field_radius_for_zoom(zoom: int) -> float:
    value = int(zoom)
    if value <= 4:
        return 54.0
    if value <= 5:
        return 62.0
    if value <= 6:
        return 72.0
    if value <= 7:
        return 84.0
    if value <= 8:
        return 96.0
    if value <= 10:
        return 112.0
    return 124.0


def build_temperature_frame_normalizers(temp_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    temp_values = np.asarray(temp_matrix, dtype=np.float32)
    lower = np.quantile(temp_values, 0.08, axis=0).astype(np.float32)
    upper = np.quantile(temp_values, 0.92, axis=0).astype(np.float32)
    return lower, upper


def weather_precip_alpha(precip_probability_pct: np.ndarray, wet_hour: np.ndarray) -> np.ndarray:
    certainty = np.clip(
        np.maximum(
            np.asarray(precip_probability_pct, dtype=np.float32) / 100.0,
            np.asarray(wet_hour, dtype=np.float32) * 0.45,
        ),
        0.0,
        1.0,
    )
    certainty = np.where(certainty > 0.02, certainty, 0.0)
    visible = np.clip(np.power(certainty, 0.72) * 1.12, 0.0, 0.98)
    return np.where(certainty > 0.02, np.maximum(visible, 0.06), 0.0).astype(np.float32)


def weather_temperature_weight(temp_c: np.ndarray) -> np.ndarray:
    values = np.clip((np.asarray(temp_c, dtype=np.float32) + 20.0) / 60.0, 0.0, 1.0)
    return np.clip(0.05 + values * 0.95, 0.0, 1.0).astype(np.float32)


def weather_heatmap_gradient(mode: str) -> np.ndarray:
    if str(mode) == "temperature":
        return np.asarray(
            [
                (0.0, 0.0, 0.0, 0.0),
                (28.0, 91.0, 215.0, 46.0),
                (65.0, 149.0, 241.0, 87.0),
                (102.0, 205.0, 255.0, 128.0),
                (242.0, 211.0, 82.0, 163.0),
                (242.0, 139.0, 55.0, 204.0),
                (203.0, 43.0, 39.0, 235.0),
            ],
            dtype=np.float32,
        )
    return np.asarray(
        [
            (0.0, 0.0, 0.0, 0.0),
            (112.0, 168.0, 255.0, 20.0),
            (95.0, 153.0, 248.0, 51.0),
            (78.0, 133.0, 231.0, 92.0),
            (58.0, 110.0, 210.0, 143.0),
            (38.0, 81.0, 178.0, 194.0),
            (20.0, 52.0, 132.0, 235.0),
        ],
        dtype=np.float32,
    )


def apply_weather_heatmap_gradient(values: np.ndarray, mode: str) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    stops = np.linspace(0.0, 1.0, 7, dtype=np.float32)
    gradient = weather_heatmap_gradient(mode)
    channels = [
        np.interp(clipped, stops, gradient[:, channel_idx]).astype(np.float32)
        for channel_idx in range(4)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def apply_weather_heatmap_gradient_rgb(values: np.ndarray, mode: str) -> np.ndarray:
    clipped = np.clip(np.asarray(values, dtype=np.float32), 0.0, 1.0)
    gradient = weather_heatmap_gradient(mode)
    if str(mode) == "temperature":
        # The first stop is transparent black for RGBA compositing. Temperature
        # RGB should start at blue, not black.
        gradient = gradient[1:, :]
    stops = np.linspace(0.0, 1.0, gradient.shape[0], dtype=np.float32)
    channels = [
        np.interp(clipped, stops, gradient[:, channel_idx]).astype(np.float32)
        for channel_idx in range(3)
    ]
    return np.stack(channels, axis=-1).astype(np.uint8)


def weather_kernel(radius_px: float) -> tuple[np.ndarray, int]:
    influence_radius = max(4, int(math.ceil(float(radius_px) * 2.25)))
    sigma = max(1.0, float(radius_px) / 2.0)
    axis = np.arange(-influence_radius, influence_radius + 1, dtype=np.float32)
    xx, yy = np.meshgrid(axis, axis)
    kernel = np.exp(-((xx * xx) + (yy * yy)) / (2.0 * sigma * sigma)).astype(np.float32)
    kernel /= max(float(kernel.max()), 1e-6)
    return kernel, influence_radius


def build_weather_field_sampler(
    representative: pd.DataFrame,
    *,
    blend_k: int,
    weight_decay_km: float,
):
    station_lat = representative["LAT"].to_numpy(dtype=np.float64)
    station_lon = representative["LON"].to_numpy(dtype=np.float64)
    station_coords_rad = np.radians(np.column_stack((station_lat, station_lon)))
    station_tree = BallTree(station_coords_rad, metric="haversine")
    blend_k = max(1, min(int(blend_k), len(representative)))
    decay_km = max(1.0, float(weight_decay_km))

    def sample_field(
        station_values: np.ndarray,
        *,
        field_zoom: int,
        bbox_min_x: int,
        bbox_min_y: int,
        bbox_max_x: int,
        bbox_max_y: int,
        samples_per_tile: int,
        max_distance_km: float | None = None,
    ) -> np.ndarray:
        width_samples = max(1, (int(bbox_max_x) - int(bbox_min_x) + 1) * int(samples_per_tile))
        height_samples = max(1, (int(bbox_max_y) - int(bbox_min_y) + 1) * int(samples_per_tile))
        tile_span = 256.0 / float(samples_per_tile)
        origin_x = float(int(bbox_min_x) * 256)
        origin_y = float(int(bbox_min_y) * 256)
        xs = origin_x + (np.arange(width_samples, dtype=np.float64) + 0.5) * tile_span
        ys = origin_y + (np.arange(height_samples, dtype=np.float64) + 0.5) * tile_span

        sampled = np.zeros((height_samples, width_samples), dtype=np.float32)
        for row_start in range(0, height_samples, 128):
            row_stop = min(height_samples, row_start + 128)
            yy = ys[row_start:row_stop]
            grid_x, grid_y = np.meshgrid(xs, yy)
            flat_x = grid_x.reshape(-1)
            flat_y = grid_y.reshape(-1)
            latlon = np.asarray(
                [
                    mercator_pixel_to_latlon(float(world_x), float(world_y), int(field_zoom))
                    for world_x, world_y in zip(flat_x, flat_y, strict=False)
                ],
                dtype=np.float64,
            )
            query_rad = np.radians(latlon)
            distances, indices = station_tree.query(query_rad, k=blend_k)
            distances_km = np.asarray(distances, dtype=np.float32) * EARTH_RADIUS_KM
            local_weights = np.exp(-distances_km / decay_km).astype(np.float32)
            values = np.asarray(station_values, dtype=np.float32)[indices]
            if max_distance_km is not None:
                local_weights *= (distances_km <= float(max_distance_km)).astype(np.float32)
            local_weights_sum = np.maximum(local_weights.sum(axis=1, keepdims=True), 1e-6)
            local_weights /= local_weights_sum
            flat_values = np.sum(values * local_weights, axis=1, dtype=np.float32)
            sampled[row_start:row_stop, :] = flat_values.reshape(row_stop - row_start, width_samples)
        return sampled

    return sample_field


def render_weather_heatmap_field_frame(
    *,
    representative: pd.DataFrame,
    mode: str,
    temp_values: np.ndarray,
    precip_probability_pct: np.ndarray,
    wet_hour: np.ndarray,
    temp_lower: float,
    temp_upper: float,
    field_zoom: int,
    bbox_min_x: int,
    bbox_min_y: int,
    width_px: int,
    height_px: int,
    radius_px: float | None = None,
    edge_feather: np.ndarray | None = None,
) -> Image.Image:
    if width_px <= 0 or height_px <= 0:
        return Image.new("RGBA", (1, 1), (0, 0, 0, 0))

    if mode == "temperature":
        span = max(2.0, float(temp_upper) - float(temp_lower))
        station_temps_normalized = np.clip(
            (np.asarray(temp_values, dtype=np.float32) - float(temp_lower)) / span,
            0.0,
            1.0,
        )
    else:
        station_weights = weather_precip_alpha(
            np.asarray(precip_probability_pct, dtype=np.float32),
            np.asarray(wet_hour, dtype=np.float32),
        )

    field = np.zeros((int(height_px), int(width_px)), dtype=np.float32)
    support = np.zeros((int(height_px), int(width_px)), dtype=np.float32) if mode == "temperature" else None
    radius_px = float(radius_px) if radius_px is not None else weather_field_radius_for_zoom(int(field_zoom))
    kernel, influence_radius = weather_kernel(radius_px)
    kernel_size = kernel.shape[0]
    origin_x = float(int(bbox_min_x) * 256)
    origin_y = float(int(bbox_min_y) * 256)

    if mode == "temperature":
        for temp_value, (lat, lon) in zip(
            station_temps_normalized,
            representative[["LAT", "LON"]].to_numpy(dtype=np.float64),
            strict=False,
        ):
            world_x, world_y = mercator_latlon_to_pixel(float(lat), float(lon), int(field_zoom))
            center_x = int(round(world_x - origin_x))
            center_y = int(round(world_y - origin_y))
            left = center_x - influence_radius
            top = center_y - influence_radius
            right = left + kernel_size
            bottom = top + kernel_size
            clip_left = max(0, left)
            clip_top = max(0, top)
            clip_right = min(int(width_px), right)
            clip_bottom = min(int(height_px), bottom)
            if clip_left >= clip_right or clip_top >= clip_bottom:
                continue
            kernel_left = clip_left - left
            kernel_top = clip_top - top
            kernel_right = kernel_left + (clip_right - clip_left)
            kernel_bottom = kernel_top + (clip_bottom - clip_top)
            kernel_slice = kernel[kernel_top:kernel_bottom, kernel_left:kernel_right]
            contribution = kernel_slice
            support[clip_top:clip_bottom, clip_left:clip_right] += contribution
            field[clip_top:clip_bottom, clip_left:clip_right] += contribution * float(temp_value)

        valid = support > 1e-6
        temp_field = np.zeros_like(field, dtype=np.float32)
        temp_field[valid] = field[valid] / support[valid]
        max_support = float(support.max())
        alpha_field = np.zeros_like(field, dtype=np.float32)
        if max_support > 1e-6:
            support_norm = np.clip(support / max_support, 0.0, 1.0)
            # Cut off the weakest fringe, then fade up quickly enough that the
            # outer contour dissolves instead of reading as a colored ring.
            alpha_field = np.clip((support_norm - 0.14) / 0.22, 0.0, 1.0)
            alpha_field = np.power(alpha_field, 1.15)
        rgb = apply_weather_heatmap_gradient_rgb(temp_field, mode)
        rgba = np.zeros((int(height_px), int(width_px), 4), dtype=np.uint8)
        rgba[..., :3] = rgb
        rgba[..., 3] = np.clip(
            alpha_field * float(WEATHER_TEMPERATURE_ALPHA_MAX),
            0.0,
            float(WEATHER_TEMPERATURE_ALPHA_MAX),
        ).astype(np.uint8)
    else:
        for weight, (lat, lon) in zip(
            station_weights,
            representative[["LAT", "LON"]].to_numpy(dtype=np.float64),
            strict=False,
        ):
            if not (float(weight) > 1e-4):
                continue
            world_x, world_y = mercator_latlon_to_pixel(float(lat), float(lon), int(field_zoom))
            center_x = int(round(world_x - origin_x))
            center_y = int(round(world_y - origin_y))
            left = center_x - influence_radius
            top = center_y - influence_radius
            right = left + kernel_size
            bottom = top + kernel_size

            clip_left = max(0, left)
            clip_top = max(0, top)
            clip_right = min(int(width_px), right)
            clip_bottom = min(int(height_px), bottom)
            if clip_left >= clip_right or clip_top >= clip_bottom:
                continue

            kernel_left = clip_left - left
            kernel_top = clip_top - top
            kernel_right = kernel_left + (clip_right - clip_left)
            kernel_bottom = kernel_top + (clip_bottom - clip_top)
            kernel_slice = kernel[kernel_top:kernel_bottom, kernel_left:kernel_right]
            field[clip_top:clip_bottom, clip_left:clip_right] += (
                kernel_slice * float(weight)
            )

        max_value = float(field.max())
        if max_value > 1e-6:
            field = np.clip(field / max_value, 0.0, 1.0)
        rgba = apply_weather_heatmap_gradient(field, mode)

    if edge_feather is not None:
        rgba[..., 3] = np.clip(
            rgba[..., 3].astype(np.float32) * np.asarray(edge_feather, dtype=np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)
    return Image.fromarray(rgba, mode="RGBA").filter(ImageFilter.GaussianBlur(radius=1.1))


def render_weather_interpolated_field_frame(
    *,
    sample_field,
    mode: str,
    temp_values: np.ndarray,
    precip_probability_pct: np.ndarray,
    wet_hour: np.ndarray,
    temp_lower: float,
    temp_upper: float,
    field_zoom: int,
    bbox_min_x: int,
    bbox_min_y: int,
    bbox_max_x: int,
    bbox_max_y: int,
    samples_per_tile: int,
    edge_feather: np.ndarray | None = None,
) -> Image.Image:
    if mode == "temperature":
        span = max(2.0, float(temp_upper) - float(temp_lower))
        station_values = np.clip(
            (np.asarray(temp_values, dtype=np.float32) - float(temp_lower)) / span,
            0.0,
            1.0,
        )
    else:
        station_values = weather_precip_alpha(
            np.asarray(precip_probability_pct, dtype=np.float32),
            np.asarray(wet_hour, dtype=np.float32),
        )

    sampled = sample_field(
        station_values,
        field_zoom=int(field_zoom),
        bbox_min_x=int(bbox_min_x),
        bbox_min_y=int(bbox_min_y),
        bbox_max_x=int(bbox_max_x),
        bbox_max_y=int(bbox_max_y),
        samples_per_tile=int(samples_per_tile),
        max_distance_km=float(WEATHER_PRECIP_MAX_DISTANCE_KM if mode != "temperature" else 0.0) or None,
    )

    if mode == "temperature":
        rgba = weather_temperature_rgba(sampled.reshape(-1)).reshape((*sampled.shape, 4))
    else:
        active = sampled[sampled > 0.01]
        if active.size:
            lower = float(np.quantile(active, 0.45))
            upper = float(np.quantile(active, 0.92))
            span = max(0.04, upper - lower)
            sampled = np.clip((sampled - lower) / span, 0.0, 1.0).astype(np.float32)
        else:
            sampled = np.zeros_like(sampled, dtype=np.float32)
        rgba = weather_precip_rgba(sampled.reshape(-1)).reshape((*sampled.shape, 4))

    if edge_feather is not None:
        rgba[..., 3] = np.clip(
            rgba[..., 3].astype(np.float32) * np.asarray(edge_feather, dtype=np.float32),
            0.0,
            255.0,
        ).astype(np.uint8)

    base = Image.fromarray(rgba, mode="RGBA")
    width_px = (int(bbox_max_x) - int(bbox_min_x) + 1) * 256
    height_px = (int(bbox_max_y) - int(bbox_min_y) + 1) * 256
    return base.resize((width_px, height_px), resample=Image.Resampling.BICUBIC).filter(
        ImageFilter.GaussianBlur(radius=1.35)
    )


def weather_temperature_rgba(normalized: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(normalized, dtype=np.float32), 0.0, 1.0)
    stops = np.array([0.0, 0.18, 0.38, 0.58, 0.78, 1.0], dtype=np.float32)
    red = np.interp(values, stops, np.array([28, 65, 102, 242, 242, 203], dtype=np.float32))
    green = np.interp(values, stops, np.array([91, 149, 205, 211, 139, 43], dtype=np.float32))
    blue = np.interp(values, stops, np.array([215, 241, 255, 82, 55, 39], dtype=np.float32))
    alpha = np.full(values.shape, 184.0, dtype=np.float32)
    return np.stack([red, green, blue, alpha], axis=1).astype(np.uint8)


def weather_precip_rgba(intensity: np.ndarray) -> np.ndarray:
    values = np.clip(np.asarray(intensity, dtype=np.float32), 0.0, 1.0)
    stops = np.array([0.0, 0.18, 0.36, 0.56, 0.78, 1.0], dtype=np.float32)
    red = np.interp(values, stops, np.array([112, 95, 78, 58, 38, 20], dtype=np.float32))
    green = np.interp(values, stops, np.array([168, 153, 133, 110, 81, 52], dtype=np.float32))
    blue = np.interp(values, stops, np.array([255, 248, 231, 210, 178, 132], dtype=np.float32))
    alpha = np.clip(values * 255.0, 0.0, 246.0)
    return np.stack([red, green, blue, alpha], axis=1).astype(np.uint8)


def weather_edge_feather(
    *,
    sample_size: int,
    fade_north: bool,
    fade_south: bool,
    fade_west: bool,
    fade_east: bool,
) -> np.ndarray:
    fade = np.ones((int(sample_size), int(sample_size)), dtype=np.float32)
    width = max(8, int(round(float(sample_size) * 0.24)))
    ramp = np.linspace(0.0, 1.0, width, dtype=np.float32)
    if fade_north:
        fade[:width, :] *= ramp[:, None]
    if fade_south:
        fade[-width:, :] *= ramp[::-1, None]
    if fade_west:
        fade[:, :width] *= ramp[None, :]
    if fade_east:
        fade[:, -width:] *= ramp[None, ::-1]
    return fade


def weather_blank_png() -> bytes:
    return png_bytes(Image.new("RGBA", (256, 256), (0, 0, 0, 0)))


def weather_wind_rgb(speed_mps: float) -> tuple[int, int, int]:
    t = max(0.0, min(1.0, float(speed_mps) / 18.0))
    if t <= 0.55:
        local = t / 0.55
        start = (38.0, 132.0, 255.0)
        mid = (82.0, 196.0, 139.0)
        return (
            int(round(start[0] + (mid[0] - start[0]) * local)),
            int(round(start[1] + (mid[1] - start[1]) * local)),
            int(round(start[2] + (mid[2] - start[2]) * local)),
        )
    local = (t - 0.55) / 0.45
    mid = (82.0, 196.0, 139.0)
    end = (242.0, 139.0, 55.0)
    return (
        int(round(mid[0] + (end[0] - mid[0]) * local)),
        int(round(mid[1] + (end[1] - mid[1]) * local)),
        int(round(mid[2] + (end[2] - mid[2]) * local)),
    )


def weather_wind_alpha(speed_mps: float) -> int:
    value = max(0.42, min(0.94, 0.42 + float(speed_mps) / 20.0))
    return int(round(value * 255.0))


def weather_wind_length(speed_mps: float) -> float:
    return max(6.5, min(14.0, 6.5 + float(speed_mps) * 0.45))


def weighted_direction_deg(direction_deg: np.ndarray, weights: np.ndarray) -> float | None:
    directions = np.asarray(direction_deg, dtype=np.float32)
    valid = np.isfinite(directions)
    if not np.any(valid):
        return None
    radians = np.radians(directions[valid].astype(np.float64))
    local_weights = np.asarray(weights, dtype=np.float64)[valid]
    sin_sum = float(np.sum(np.sin(radians) * local_weights))
    cos_sum = float(np.sum(np.cos(radians) * local_weights))
    if abs(sin_sum) < 1e-9 and abs(cos_sum) < 1e-9:
        return None
    return (math.degrees(math.atan2(sin_sum, cos_sum)) + 360.0) % 360.0


def draw_weather_wind_arrow(
    draw: ImageDraw.ImageDraw,
    *,
    x: float,
    y: float,
    heading_deg: float,
    length_px: float,
    rgba: tuple[int, int, int, int],
) -> None:
    angle = math.radians(float(heading_deg))
    dx = math.sin(angle)
    dy = -math.cos(angle)
    half = float(length_px) * 0.5
    tail_x = x - dx * half
    tail_y = y - dy * half
    head_x = x + dx * half
    head_y = y + dy * half
    draw.line((tail_x, tail_y, head_x, head_y), fill=rgba, width=2)
    head_len = max(3.5, float(length_px) * 0.38)
    head_angle = math.radians(26.0)
    left_dx = math.sin(angle + math.pi - head_angle)
    left_dy = -math.cos(angle + math.pi - head_angle)
    right_dx = math.sin(angle + math.pi + head_angle)
    right_dy = -math.cos(angle + math.pi + head_angle)
    left_x = head_x + left_dx * head_len
    left_y = head_y + left_dy * head_len
    right_x = head_x + right_dx * head_len
    right_y = head_y + right_dy * head_len
    draw.polygon([(head_x, head_y), (left_x, left_y), (right_x, right_y)], fill=rgba)


def build_weather_field_edge_feather(
    *,
    width: int,
    height: int,
    samples_per_tile: int,
    halo_tiles: int,
    feather_scale: float = 1.0,
) -> np.ndarray:
    fade = np.ones((int(height), int(width)), dtype=np.float32)
    ramp_width = max(
        24,
        int(round(int(samples_per_tile) * max(1, int(halo_tiles)) * float(feather_scale))),
    )
    ramp_width = min(ramp_width, max(1, width // 3), max(1, height // 3))
    if ramp_width <= 1:
        return fade
    ramp = np.linspace(0.0, 1.0, ramp_width, dtype=np.float32)
    fade[:ramp_width, :] *= ramp[:, None]
    fade[-ramp_width:, :] *= ramp[::-1, None]
    fade[:, :ramp_width] *= ramp[None, :]
    fade[:, -ramp_width:] *= ramp[None, ::-1]
    return fade


def render_weather_tile_from_field(
    *,
    field_image: Image.Image,
    field_zoom: int,
    bbox_min_x: int,
    bbox_min_y: int,
    tile_x: int,
    tile_y: int,
    zoom: int,
) -> Image.Image:
    samples_per_tile = 256.0
    bbox_min_x = float(bbox_min_x)
    bbox_min_y = float(bbox_min_y)
    field_tiles_per_target_tile = 2.0 ** (field_zoom - int(zoom))
    left = (float(tile_x) * field_tiles_per_target_tile - bbox_min_x) * samples_per_tile
    top = (float(tile_y) * field_tiles_per_target_tile - bbox_min_y) * samples_per_tile
    right = (float(tile_x + 1) * field_tiles_per_target_tile - bbox_min_x) * samples_per_tile
    bottom = (float(tile_y + 1) * field_tiles_per_target_tile - bbox_min_y) * samples_per_tile
    return field_image.transform(
        (256, 256),
        Image.Transform.EXTENT,
        data=(left, top, right, bottom),
        resample=Image.Resampling.BICUBIC,
    )


def build_weather_raster_tiles(
    *,
    representative: pd.DataFrame,
    station_weather: dict[str, np.ndarray],
    hours: int,
    raster_zoom_min: int,
    raster_zoom_max: int,
) -> None:
    tmp_path = WEATHER_RASTER_TILE_DB_PATH.with_suffix(".tmp.sqlite3")
    if tmp_path.exists():
        tmp_path.unlink()
    temp_lower, temp_upper = build_temperature_frame_normalizers(station_weather["temp_c"])
    modes = ("temperature", "precipitation")
    source = sqlite3.connect(ROAD_TILE_DB_PATH)
    target = sqlite3.connect(tmp_path)
    target.execute("PRAGMA journal_mode=OFF")
    target.execute("PRAGMA synchronous=OFF")
    target.execute("PRAGMA temp_store=MEMORY")
    target.execute(
        """
        CREATE TABLE raster_tiles (
            mode TEXT NOT NULL,
            frame_idx INTEGER NOT NULL,
            z INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            png BLOB NOT NULL,
            PRIMARY KEY (mode, frame_idx, z, x, y)
        )
        """
    )

    try:
        target.execute(
            "CREATE INDEX idx_weather_raster_tiles_lookup ON raster_tiles(mode, frame_idx, z, x, y)"
        )
        leaf_bounds = source.execute(
            """
            SELECT MIN(x), MAX(x), MIN(y), MAX(y)
            FROM tile_entries
            WHERE z = ?
            """,
            (int(raster_zoom_max),),
        ).fetchone()
        if leaf_bounds is None or leaf_bounds[0] is None:
            raise RuntimeError(f"no road tile entries found at z={int(raster_zoom_max)}")
        min_x, max_x, min_y, max_y = (int(value) for value in leaf_bounds)
        halo = int(WEATHER_RASTER_HALO_TILES)
        temp_matrix = np.asarray(station_weather["temp_c"], dtype=np.float32)
        precip_matrix = np.asarray(station_weather["precip_probability_pct"], dtype=np.float32)
        wet_matrix = np.asarray(station_weather["wet_hour"], dtype=np.float32)
        mode_layouts: dict[str, dict[str, Any]] = {}
        for mode in modes:
            field_zoom = min(int(raster_zoom_max), int(WEATHER_HEATMAP_FIELD_ZOOM))
            max_field_tile = 2 ** int(field_zoom) - 1
            zoom_scale = 2 ** max(0, int(raster_zoom_max) - int(field_zoom))
            core_min_x = max(0, int(min_x) // zoom_scale)
            core_max_x = min(max_field_tile, int(max_x) // zoom_scale)
            core_min_y = max(0, int(min_y) // zoom_scale)
            core_max_y = min(max_field_tile, int(max_y) // zoom_scale)
            bbox_min_x = max(0, core_min_x - halo)
            bbox_max_x = min(max_field_tile, core_max_x + halo)
            bbox_min_y = max(0, core_min_y - halo)
            bbox_max_y = min(max_field_tile, core_max_y + halo)
            edge_feather = (
                build_weather_field_edge_feather(
                    width=(bbox_max_x - bbox_min_x + 1) * 256,
                    height=(bbox_max_y - bbox_min_y + 1) * 256,
                    samples_per_tile=256,
                    halo_tiles=halo,
                    feather_scale=(
                        WEATHER_TEMPERATURE_EDGE_FEATHER_SCALE
                        if str(mode) == "temperature"
                        else 1.0
                    ),
                )
                if halo > 0
                else None
            )
            zoom_tile_ranges: dict[int, tuple[int, int, int, int]] = {}
            for zoom in range(int(raster_zoom_min), int(raster_zoom_max) + 1):
                scale = 2 ** (int(zoom) - int(field_zoom))
                min_tile_x = max(0, int(math.floor(core_min_x * scale)))
                max_tile_x = min(2 ** int(zoom) - 1, int(math.ceil((core_max_x + 1) * scale) - 1))
                min_tile_y = max(0, int(math.floor(core_min_y * scale)))
                max_tile_y = min(2 ** int(zoom) - 1, int(math.ceil((core_max_y + 1) * scale) - 1))
                zoom_tile_ranges[int(zoom)] = (min_tile_x, max_tile_x, min_tile_y, max_tile_y)
            mode_layouts[str(mode)] = {
                "field_zoom": int(field_zoom),
                "bbox_min_x": int(bbox_min_x),
                "bbox_min_y": int(bbox_min_y),
                "bbox_max_x": int(bbox_max_x),
                "bbox_max_y": int(bbox_max_y),
                "edge_feather": edge_feather,
                "zoom_tile_ranges": zoom_tile_ranges,
            }

        for frame_idx in range(int(hours)):
            temp_values = temp_matrix[:, frame_idx]
            precip_values = precip_matrix[:, frame_idx]
            wet_values = wet_matrix[:, frame_idx]
            rendered_fields: dict[str, Image.Image] = {}
            for mode in modes:
                layout = mode_layouts[str(mode)]
                rendered_fields[str(mode)] = render_weather_heatmap_field_frame(
                    representative=representative,
                    mode=str(mode),
                    temp_values=temp_values,
                    precip_probability_pct=precip_values,
                    wet_hour=wet_values,
                    temp_lower=float(temp_lower[frame_idx]),
                    temp_upper=float(temp_upper[frame_idx]),
                    field_zoom=int(layout["field_zoom"]),
                    bbox_min_x=int(layout["bbox_min_x"]),
                    bbox_min_y=int(layout["bbox_min_y"]),
                    width_px=(int(layout["bbox_max_x"]) - int(layout["bbox_min_x"]) + 1) * 256,
                    height_px=(int(layout["bbox_max_y"]) - int(layout["bbox_min_y"]) + 1) * 256,
                    radius_px=float(WEATHER_HEATMAP_RADIUS_PX),
                    edge_feather=np.asarray(layout["edge_feather"], dtype=np.float32)
                    if layout["edge_feather"] is not None
                    else None,
                )
            for zoom in range(int(raster_zoom_min), int(raster_zoom_max) + 1):
                insert_rows: list[tuple[str, int, int, int, int, sqlite3.Binary]] = []
                total_tiles = max(
                    1,
                    sum(
                        (
                            int(layout["zoom_tile_ranges"][int(zoom)][1])
                            - int(layout["zoom_tile_ranges"][int(zoom)][0])
                            + 1
                        )
                        * (
                            int(layout["zoom_tile_ranges"][int(zoom)][3])
                            - int(layout["zoom_tile_ranges"][int(zoom)][2])
                            + 1
                        )
                        for layout in mode_layouts.values()
                    ),
                )
                tile_counter = 0
                for mode in modes:
                    layout = mode_layouts[str(mode)]
                    min_tile_x, max_tile_x, min_tile_y, max_tile_y = layout["zoom_tile_ranges"][int(zoom)]
                    for tile_y in range(min_tile_y, max_tile_y + 1):
                        for tile_x in range(min_tile_x, max_tile_x + 1):
                            tile_counter += 1
                            image = render_weather_tile_from_field(
                                field_image=rendered_fields[str(mode)],
                                field_zoom=int(layout["field_zoom"]),
                                bbox_min_x=int(layout["bbox_min_x"]),
                                bbox_min_y=int(layout["bbox_min_y"]),
                                tile_x=int(tile_x),
                                tile_y=int(tile_y),
                                zoom=int(zoom),
                            )
                            insert_rows.append(
                                (
                                    str(mode),
                                    int(frame_idx),
                                    int(zoom),
                                    int(tile_x),
                                    int(tile_y),
                                    sqlite3.Binary(png_bytes(image)),
                                )
                            )
                            if len(insert_rows) >= 200:
                                target.executemany(
                                    """
                                    INSERT INTO raster_tiles(mode, frame_idx, z, x, y, png)
                                    VALUES (?, ?, ?, ?, ?, ?)
                                    """,
                                    insert_rows,
                                )
                                insert_rows = []
                            if tile_counter % 100 == 0 or tile_counter == total_tiles:
                                target.commit()
                                print(
                                    f"weather_zoom={zoom} frame={frame_idx + 1}/{hours} tiles={tile_counter}/{total_tiles}",
                                    flush=True,
                                )
                if insert_rows:
                    target.executemany(
                        """
                        INSERT INTO raster_tiles(mode, frame_idx, z, x, y, png)
                        VALUES (?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                    target.commit()

        target.commit()
        if WEATHER_RASTER_TILE_DB_PATH.exists():
            WEATHER_RASTER_TILE_DB_PATH.unlink()
        tmp_path.replace(WEATHER_RASTER_TILE_DB_PATH)
    finally:
        source.close()
        target.close()


def build_raster_tiles(
    *,
    render_scores: np.ndarray,
    hours: int,
    raster_zoom_min: int,
    raster_zoom_max: int,
) -> None:
    tmp_path = ROAD_RASTER_TILE_DB_PATH.with_suffix(".tmp.sqlite3")
    if tmp_path.exists():
        tmp_path.unlink()

    source = sqlite3.connect(ROAD_TILE_DB_PATH)
    target = sqlite3.connect(tmp_path)
    target.execute("PRAGMA journal_mode=OFF")
    target.execute("PRAGMA synchronous=OFF")
    target.execute("PRAGMA temp_store=MEMORY")
    target.execute(
        """
        CREATE TABLE raster_tiles (
            frame_idx INTEGER NOT NULL,
            z INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            png BLOB NOT NULL,
            PRIMARY KEY (frame_idx, z, x, y)
        )
        """
    )

    try:
        target.execute(
            "CREATE INDEX idx_raster_tiles_lookup ON raster_tiles(frame_idx, z, x, y)"
        )
        for zoom in range(int(raster_zoom_max), int(raster_zoom_min) - 1, -1):
            zoom_tiles = source.execute(
                "SELECT x, y FROM tile_entries WHERE z = ? GROUP BY x, y ORDER BY x, y",
                (int(zoom),),
            ).fetchall()
            tile_normalizers = build_tile_normalizers(
                source=source,
                render_scores=render_scores,
                zoom=int(zoom),
                zoom_tiles=zoom_tiles,
            )
            for tile_idx, (tile_x, tile_y) in enumerate(zoom_tiles, start=1):
                rows = source.execute(
                    """
                    SELECT segment_idx, road_kind, path
                    FROM tile_entries
                    WHERE z = ? AND x = ? AND y = ?
                    """,
                    (int(zoom), int(tile_x), int(tile_y)),
                ).fetchall()
                decoded = sorted(
                    [
                        (int(segment_idx), int(road_kind), decode_path_blob(path_blob))
                        for segment_idx, road_kind, path_blob in rows
                    ],
                    key=lambda value: value[1],
                )
                local_center, local_high, neighborhood_center, neighborhood_high = (
                    tile_normalizers[(int(tile_x), int(tile_y))]
                )
                insert_rows: list[tuple[int, int, int, int, sqlite3.Binary]] = []
                for frame_idx in range(int(hours)):
                    raw_frame_scores = np.fromiter(
                        (
                            int(render_scores[segment_idx, frame_idx])
                            for segment_idx, _, _ in decoded
                        ),
                        dtype=np.uint8,
                        count=len(decoded),
                    )
                    normalized_scores = normalize_tile_frame_scores(
                        raw_scores=raw_frame_scores,
                        local_center=float(local_center[frame_idx]),
                        local_high=float(local_high[frame_idx]),
                        neighborhood_center=float(neighborhood_center[frame_idx]),
                        neighborhood_high=float(neighborhood_high[frame_idx]),
                        zoom=int(zoom),
                    )
                    image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
                    draw = ImageDraw.Draw(image, "RGBA")
                    for score_byte, (_, road_kind, path) in zip(
                        normalized_scores,
                        decoded,
                        strict=False,
                    ):
                        color = raster_rgba(
                            risk_byte=int(score_byte),
                            road_kind=int(road_kind),
                            zoom=int(zoom),
                        )
                        if color is None:
                            continue
                        base_width = raster_stroke_width(road_kind, int(zoom))
                        draw.line(
                            path,
                            fill=color,
                            width=base_width,
                            joint="curve",
                        )
                    insert_rows.append(
                        (
                            int(frame_idx),
                            int(zoom),
                            int(tile_x),
                            int(tile_y),
                            sqlite3.Binary(png_bytes(image)),
                        )
                    )

                target.executemany(
                    """
                    INSERT INTO raster_tiles(frame_idx, z, x, y, png)
                    VALUES (?, ?, ?, ?, ?)
                    """,
                    insert_rows,
                )
                commit_every = 25 if int(zoom) == int(raster_zoom_max) else 50
                if tile_idx % commit_every == 0 or tile_idx == len(zoom_tiles):
                    target.commit()
                    print(
                        f"raster_zoom={zoom} tiles={tile_idx}/{len(zoom_tiles)}",
                        flush=True,
                    )

        target.commit()
        if ROAD_RASTER_TILE_DB_PATH.exists():
            ROAD_RASTER_TILE_DB_PATH.unlink()
        tmp_path.replace(ROAD_RASTER_TILE_DB_PATH)
    finally:
        source.close()
        target.close()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if not ROAD_TILE_DB_PATH.exists():
        raise FileNotFoundError(
            f"missing {ROAD_TILE_DB_PATH}; run build_road_tile_index.py first"
        )

    bundle = joblib.load(SEGMENT_MODEL_BUNDLE_PATH, mmap_mode="r")
    representative = pd.read_csv(REPRESENTATIVE_STATIONS_PATH)
    roads = pd.read_parquet(
        ROAD_SEGMENTS_PATH,
        columns=["center_lat", "center_lon", "length_km", "fullname", "rttyp", "mtfcc"],
    )
    road_coords_rad = np.radians(
        roads[["center_lat", "center_lon"]].to_numpy(dtype=np.float64)
    )
    road_kind = np.where(
        roads["rttyp"].fillna("").eq("I").to_numpy()
        | roads["mtfcc"].fillna("").eq("S1100").to_numpy(),
        2,
        np.where(
            roads["rttyp"].fillna("").isin(["U", "S"]).to_numpy()
            | roads["mtfcc"].fillna("").eq("S1200").to_numpy(),
            1,
            0,
        ),
    ).astype(np.int8)
    static_frame, _ = build_static_segment_frame(
        roads,
        rttyp_values=DEFAULT_RTTYP_VALUES,
        mtfcc_values=DEFAULT_MTFCC_VALUES,
    )
    static_features = static_frame.to_numpy(dtype=np.float32)

    model = bundle["model"]
    station_indices = np.asarray(bundle["segment_station_indices"], dtype=np.int32)
    total_counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    hour_counts = np.asarray(bundle["segment_hour_counts"], dtype=np.float32)
    weather_cube = np.asarray(bundle["weather_climatology"], dtype=np.float32)
    weather_defaults = np.asarray(bundle["weather_defaults"], dtype=np.float32)

    station_series, source_counts = build_station_series(
        representative=representative,
        weather_cube=weather_cube,
        weather_defaults=weather_defaults,
        hours=max(1, int(args.hours)),
        workers=int(args.workers),
    )

    station_weather = build_station_weather_matrices(
        series_by_station=station_series,
        station_count=len(representative),
        hours=max(1, int(args.hours)),
    )
    station_timing = build_station_timing_matrices(
        series_by_station=station_series,
        station_count=len(representative),
        hours=max(1, int(args.hours)),
    )
    station_climatology = build_station_climatology_matrices(
        timing=station_timing,
        weather_cube=weather_cube,
        weather_defaults=weather_defaults,
    )
    blend_idx, blend_weights = build_segment_weather_blend(
        roads=roads,
        representative=representative,
        k=WEATHER_BLEND_K,
    )
    representative_utc_offsets = representative["utc_offset_hours"].to_numpy(dtype=np.float32)
    del roads
    del static_frame

    tmp_forecast_path = ROAD_TILE_FORECAST_PATH.with_suffix(".tmp.npy")
    if tmp_forecast_path.exists():
        tmp_forecast_path.unlink()
    risk_map = np.lib.format.open_memmap(
        tmp_forecast_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(station_indices), max(1, int(args.hours))),
    )
    tmp_baseline_path = ROAD_TILE_BASELINE_PATH.with_suffix(".tmp.npy")
    if tmp_baseline_path.exists():
        tmp_baseline_path.unlink()
    baseline_map = np.lib.format.open_memmap(
        tmp_baseline_path,
        mode="w+",
        dtype=np.uint8,
        shape=(len(station_indices), max(1, int(args.hours))),
    )

    base_times = next(iter(station_series.values()))
    forecast_start_utc = (
        base_times[0].timestamp_local.astimezone(timezone.utc).replace(microsecond=0)
    )
    forecast_end_utc = (
        base_times[-1].timestamp_local.astimezone(timezone.utc).replace(microsecond=0)
    )
    chunk_size = 100000
    for start in range(0, len(station_indices), chunk_size):
        stop = min(len(station_indices), start + chunk_size)
        static_group = static_features[start:stop]
        totals_group = total_counts[start:stop]
        hour_counts_group = hour_counts[start:stop]
        blend_idx_group = blend_idx[start:stop]
        blend_weights_group = blend_weights[start:stop]
        blended_utc_offset_group = (
            representative_utc_offsets[blend_idx_group] * blend_weights_group
        ).sum(axis=1).astype(np.float32)
        size = stop - start

        for hour_offset in range(len(base_times)):
            temp_c = (
                station_weather["temp_c"][blend_idx_group, hour_offset] * blend_weights_group
            ).sum(axis=1)
            relative_humidity_pct = (
                station_weather["relative_humidity_pct"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)
            wind_speed_mps = (
                station_weather["wind_speed_mps"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)
            wet_hour = (
                station_weather["wet_hour"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)
            climatology_temp_c = (
                station_climatology["temp_c"][blend_idx_group, hour_offset] * blend_weights_group
            ).sum(axis=1)
            climatology_relative_humidity_pct = (
                station_climatology["relative_humidity_pct"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)
            climatology_wind_speed_mps = (
                station_climatology["wind_speed_mps"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)
            climatology_wet_hour = (
                station_climatology["wet_hour"][blend_idx_group, hour_offset]
                * blend_weights_group
            ).sum(axis=1)

            utc_target = forecast_start_utc + timedelta(hours=int(hour_offset))
            utc_hour_of_week = float(utc_target.weekday() * 24 + utc_target.hour)
            local_hour_of_week = np.mod(
                utc_hour_of_week + blended_utc_offset_group,
                168.0,
            ).astype(np.float32)
            month = np.full(size, int(utc_target.month), dtype=np.int8)
            same_hour = interpolate_hour_counts(
                hour_counts=hour_counts_group,
                local_hour_of_week=local_hour_of_week,
            )
            features = build_segment_feature_matrix(
                static_features=static_group,
                hour_of_week=local_hour_of_week,
                months=month,
                totals=totals_group,
                same_hour=same_hour,
                temp_c=temp_c.astype(np.float32),
                relative_humidity_pct=relative_humidity_pct.astype(np.float32),
                wind_speed_mps=wind_speed_mps.astype(np.float32),
                wet_hour=wet_hour.astype(np.float32),
            )
            probabilities = model.predict_proba(features)[:, 1]
            climatology_features = build_segment_feature_matrix(
                static_features=static_group,
                hour_of_week=local_hour_of_week,
                months=month,
                totals=totals_group,
                same_hour=same_hour,
                temp_c=climatology_temp_c.astype(np.float32),
                relative_humidity_pct=climatology_relative_humidity_pct.astype(np.float32),
                wind_speed_mps=climatology_wind_speed_mps.astype(np.float32),
                wet_hour=climatology_wet_hour.astype(np.float32),
            )
            climatology_probabilities = model.predict_proba(climatology_features)[:, 1]
            risk_map[start:stop, hour_offset] = np.clip(
                np.rint(probabilities * 255.0),
                0,
                255,
            ).astype(np.uint8)
            baseline_map[start:stop, hour_offset] = np.clip(
                np.rint(climatology_probabilities * 255.0),
                0,
                255,
            ).astype(np.uint8)

        print(f"scored segments={stop}/{len(station_indices)}", flush=True)

    risk_map.flush()
    baseline_map.flush()

    display_scores = build_display_scores(
        forecast_scores=np.asarray(risk_map),
        total_counts=total_counts,
        road_kind=road_kind,
        strength=0.55,
        floor=0.25,
    )
    smoothing_tree = BallTree(road_coords_rad, metric="haversine")
    display_scores = apply_spatial_smoothing(
        score_map=display_scores,
        road_coords_rad=road_coords_rad,
        neighbor_tree=smoothing_tree,
    )
    risk_map[:] = display_scores
    risk_map.flush()
    baseline_display_scores = build_display_scores(
        forecast_scores=np.asarray(baseline_map),
        total_counts=total_counts,
        road_kind=road_kind,
        strength=0.55,
        floor=0.25,
    )
    baseline_display_scores = apply_spatial_smoothing(
        score_map=baseline_display_scores,
        road_coords_rad=road_coords_rad,
        neighbor_tree=smoothing_tree,
    )
    baseline_map[:] = baseline_display_scores
    baseline_map.flush()

    if ROAD_TILE_FORECAST_PATH.exists():
        ROAD_TILE_FORECAST_PATH.unlink()
    tmp_forecast_path.replace(ROAD_TILE_FORECAST_PATH)
    if ROAD_TILE_BASELINE_PATH.exists():
        ROAD_TILE_BASELINE_PATH.unlink()
    tmp_baseline_path.replace(ROAD_TILE_BASELINE_PATH)

    build_raster_tiles(
        render_scores=np.asarray(risk_map),
        hours=int(args.hours),
        raster_zoom_min=int(args.raster_zoom_min),
        raster_zoom_max=int(args.raster_zoom_max),
    )
    build_weather_raster_tiles(
        representative=representative,
        station_weather=station_weather,
        hours=int(args.hours),
        raster_zoom_min=int(args.raster_zoom_min),
        raster_zoom_max=int(args.raster_zoom_max),
    )

    generated_at = datetime.now(timezone.utc).replace(microsecond=0)
    write_weather_overlay(
        representative=representative,
        series_by_station=station_series,
        station_weather=station_weather,
        source_counts=source_counts,
        generated_at=generated_at,
        forecast_start_utc=forecast_start_utc,
        forecast_end_utc=forecast_end_utc,
        provider=str(args.provider),
        hours=int(args.hours),
    )
    meta = {
        "run_id": generated_at.strftime("%Y%m%dT%H%M%SZ"),
        "generated_at_utc": generated_at.isoformat(),
        "forecast_start_utc": forecast_start_utc.isoformat(),
        "forecast_end_utc": forecast_end_utc.isoformat(),
        "provider": str(args.provider),
        "hours": int(args.hours),
        "frame_labels": [f"+{hour}h" for hour in range(int(args.hours))],
        "tile_zoom_min": int(args.zoom_min),
        "tile_zoom_max": int(args.zoom_max),
        "raster_zoom_min": int(args.raster_zoom_min),
        "raster_zoom_max": int(args.raster_zoom_max),
        "vector_zoom_min": int(args.raster_zoom_max) + 1,
        "weather_blend_k": WEATHER_BLEND_K,
        "spatial_smooth_k": SPATIAL_SMOOTH_K,
        "spatial_smooth_decay_km": SPATIAL_SMOOTH_DECAY_KM,
        "spatial_smooth_max_km": SPATIAL_SMOOTH_MAX_KM,
        "spatial_smooth_blend": SPATIAL_SMOOTH_BLEND,
        "forecast_path": str(ROAD_TILE_FORECAST_PATH),
        "baseline_path": str(ROAD_TILE_BASELINE_PATH),
        "tile_db_path": str(ROAD_TILE_DB_PATH),
        "raster_tile_db_path": str(ROAD_RASTER_TILE_DB_PATH),
        "weather_overlay_path": str(WEATHER_OVERLAY_PATH),
        "weather_raster_tile_db_path": str(WEATHER_RASTER_TILE_DB_PATH),
        "segment_count": int(len(station_indices)),
        "station_count": int(len(representative)),
        "station_source_counts": source_counts,
    }
    ROAD_TILE_META_PATH.write_text(json.dumps(meta, indent=2), encoding="utf-8")
    print(f"wrote {ROAD_TILE_FORECAST_PATH}")
    print(f"wrote {WEATHER_OVERLAY_PATH}")
    print(f"wrote {WEATHER_RASTER_TILE_DB_PATH}")
    print(f"wrote {ROAD_TILE_META_PATH}")


if __name__ == "__main__":
    main()
