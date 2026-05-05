from __future__ import annotations

import math
import re

import numpy as np


WEATHER_FEATURE_NAMES = [
    "temp_c",
    "dewpoint_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wet_hour",
]


def approximate_utc_offset_hours(lat: float, lon: float) -> int:
    if lat < 26.0 and lon < -150.0:
        return -10
    if lat >= 50.0 and lon < -130.0:
        return -9
    if lon < -114.0:
        return -8
    if lon < -101.0:
        return -7
    if lon < -86.0:
        return -6
    return -5


def relative_humidity_from_temp_dewpoint(
    temp_c: np.ndarray | float,
    dewpoint_c: np.ndarray | float,
) -> np.ndarray:
    temp = np.asarray(temp_c, dtype=np.float32)
    dewpoint = np.asarray(dewpoint_c, dtype=np.float32)
    exponent = np.exp((17.625 * dewpoint) / (243.04 + dewpoint)) / np.exp(
        (17.625 * temp) / (243.04 + temp)
    )
    return np.clip(exponent * 100.0, 0.0, 100.0).astype(np.float32)


def build_feature_matrix(
    latitudes: np.ndarray,
    longitudes: np.ndarray,
    hour_of_week: np.ndarray,
    months: np.ndarray,
    totals: np.ndarray,
    same_hour: np.ndarray,
    temp_c: np.ndarray,
    dewpoint_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
    wind_speed_mps: np.ndarray,
    wet_hour: np.ndarray,
) -> np.ndarray:
    latitudes = np.asarray(latitudes, dtype=np.float32)
    longitudes = np.asarray(longitudes, dtype=np.float32)
    hour_of_week = np.asarray(hour_of_week, dtype=np.int16)
    months = np.asarray(months, dtype=np.int8)
    totals = np.asarray(totals, dtype=np.float32)
    same_hour = np.asarray(same_hour, dtype=np.float32)
    temp_c = np.asarray(temp_c, dtype=np.float32)
    dewpoint_c = np.asarray(dewpoint_c, dtype=np.float32)
    relative_humidity_pct = np.asarray(relative_humidity_pct, dtype=np.float32)
    wind_speed_mps = np.asarray(wind_speed_mps, dtype=np.float32)
    wet_hour = np.asarray(wet_hour, dtype=np.float32)

    dow = hour_of_week // 24 + 1
    hour = hour_of_week % 24

    hour_angle = 2.0 * math.pi * hour.astype(np.float32) / 24.0
    dow_angle = 2.0 * math.pi * (dow.astype(np.float32) - 1.0) / 7.0
    month_angle = 2.0 * math.pi * months.astype(np.float32) / 12.0

    return np.column_stack(
        [
            latitudes,
            longitudes,
            np.sin(hour_angle),
            np.cos(hour_angle),
            np.sin(dow_angle),
            np.cos(dow_angle),
            np.sin(month_angle),
            np.cos(month_angle),
            np.log1p(totals),
            np.log1p(same_hour),
            np.divide(same_hour, np.maximum(totals, 1.0), dtype=np.float32),
            temp_c,
            dewpoint_c,
            relative_humidity_pct / 100.0,
            wind_speed_mps,
            wet_hour,
        ]
    ).astype(np.float32)


def lookup_weather_climatology(
    weather_cube: np.ndarray,
    station_indices: np.ndarray | int,
    months: np.ndarray | int,
    hour_of_week: np.ndarray | int,
    weather_defaults: np.ndarray | list[float],
) -> np.ndarray:
    weather_cube = np.asarray(weather_cube, dtype=np.float32)
    if weather_cube.ndim != 4:
        raise ValueError("weather_cube must have shape [station, month, hour_of_week, feature]")

    station_indices = np.asarray(station_indices, dtype=np.int32)
    months = np.asarray(months, dtype=np.int16)
    hour_of_week = np.asarray(hour_of_week, dtype=np.int16)
    defaults = np.asarray(weather_defaults, dtype=np.float32)

    station_indices = np.clip(station_indices, 0, max(0, weather_cube.shape[0] - 1))
    month_indices = np.clip(months - 1, 0, weather_cube.shape[1] - 1)
    hour_indices = np.clip(hour_of_week, 0, weather_cube.shape[2] - 1)

    weather = weather_cube[station_indices, month_indices, hour_indices].astype(
        np.float32,
        copy=True,
    )
    if weather.ndim == 1:
        weather = weather.reshape(1, -1)

    if np.isnan(weather).any():
        missing_rows, missing_cols = np.where(np.isnan(weather))
        weather[missing_rows, missing_cols] = defaults[missing_cols]
    return weather


def parse_wind_speed_string_mps(value: str | None) -> float | None:
    if not value:
        return None
    matches = [float(match) for match in re.findall(r"\d+(?:\.\d+)?", value)]
    if not matches:
        return None
    speed = max(matches)
    if "km" in value.lower():
        return speed / 3.6
    return speed * 0.44704


def fahrenheit_to_celsius(value_f: float | None) -> float | None:
    if value_f is None:
        return None
    return (float(value_f) - 32.0) * (5.0 / 9.0)


def pascal_to_hpa(value_pa: float | None) -> float | None:
    if value_pa is None:
        return None
    return float(value_pa) / 100.0
