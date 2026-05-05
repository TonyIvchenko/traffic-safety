from __future__ import annotations

import math

import numpy as np
import pandas as pd


DEFAULT_RTTYP_VALUES = ["I", "U", "S", "C", "M"]
DEFAULT_MTFCC_VALUES = ["S1100", "S1200", "S1630", "S1640"]


def build_static_segment_frame(
    roads: pd.DataFrame,
    *,
    rttyp_values: list[str] | None = None,
    mtfcc_values: list[str] | None = None,
) -> tuple[pd.DataFrame, list[str]]:
    rttyp_values = rttyp_values or DEFAULT_RTTYP_VALUES
    mtfcc_values = mtfcc_values or DEFAULT_MTFCC_VALUES

    frame = pd.DataFrame(
        {
            "center_lat": roads["center_lat"].astype(np.float32),
            "center_lon": roads["center_lon"].astype(np.float32),
            "length_km": roads["length_km"].astype(np.float32),
            "named_road": roads["fullname"].fillna("").astype(str).str.len().gt(0).astype(np.float32),
        }
    )
    for value in rttyp_values:
        frame[f"rttyp_{value}"] = roads["rttyp"].fillna("").eq(value).astype(np.float32)
    for value in mtfcc_values:
        frame[f"mtfcc_{value}"] = roads["mtfcc"].fillna("").eq(value).astype(np.float32)
    return frame, frame.columns.tolist()


def build_segment_feature_matrix(
    static_features: np.ndarray,
    hour_of_week: np.ndarray,
    months: np.ndarray,
    totals: np.ndarray,
    same_hour: np.ndarray,
    temp_c: np.ndarray,
    relative_humidity_pct: np.ndarray,
    wind_speed_mps: np.ndarray,
    wet_hour: np.ndarray,
) -> np.ndarray:
    static_features = np.asarray(static_features, dtype=np.float32)
    hour_of_week = np.asarray(hour_of_week, dtype=np.float32)
    months = np.asarray(months, dtype=np.int8)
    totals = np.asarray(totals, dtype=np.float32)
    same_hour = np.asarray(same_hour, dtype=np.float32)
    temp_c = np.asarray(temp_c, dtype=np.float32)
    relative_humidity_pct = np.asarray(relative_humidity_pct, dtype=np.float32)
    wind_speed_mps = np.asarray(wind_speed_mps, dtype=np.float32)
    wet_hour = np.asarray(wet_hour, dtype=np.float32)

    hour = np.mod(hour_of_week, 24.0).astype(np.float32)
    dow = np.floor(hour_of_week / 24.0).astype(np.float32) + 1.0

    hour_angle = 2.0 * math.pi * hour / 24.0
    dow_angle = 2.0 * math.pi * (dow - 1.0) / 7.0
    month_angle = 2.0 * math.pi * months.astype(np.float32) / 12.0

    temporal = np.column_stack(
        [
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
            relative_humidity_pct / 100.0,
            wind_speed_mps,
            wet_hour,
        ]
    ).astype(np.float32)
    return np.hstack([static_features, temporal]).astype(np.float32)
