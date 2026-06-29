from __future__ import annotations

from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from live_weather import fetch_live_weather
from model_support import lookup_weather_climatology
from segment_model_support import (
    DEFAULT_MTFCC_VALUES,
    DEFAULT_RTTYP_VALUES,
    build_segment_feature_matrix,
    build_static_segment_frame,
)

from scripts.common import (
    ACTIVE_ROAD_SEGMENTS_PATH,
    REPRESENTATIVE_STATIONS_PATH,
    SEGMENT_MODEL_BUNDLE_PATH,
    SEGMENT_SERVE_LIMIT,
)


def _local_target_time(utc_offset_hours: int, forecast_hours: int) -> datetime:
    base = datetime.now(timezone.utc) + timedelta(hours=int(forecast_hours))
    return base + timedelta(hours=int(utc_offset_hours))


@lru_cache(maxsize=1)
def load_segment_runtime() -> dict[str, object]:
    bundle = joblib.load(SEGMENT_MODEL_BUNDLE_PATH)
    roads = pd.read_parquet(ACTIVE_ROAD_SEGMENTS_PATH).reset_index(drop=True)
    representative = pd.read_csv(REPRESENTATIVE_STATIONS_PATH)
    static_frame, _ = build_static_segment_frame(
        roads,
        rttyp_values=DEFAULT_RTTYP_VALUES,
        mtfcc_values=DEFAULT_MTFCC_VALUES,
    )
    rep_by_index = representative.set_index("station_index")
    return {
        "bundle": bundle,
        "roads": roads,
        "static_features": static_frame.to_numpy(dtype=np.float32),
        "rep_by_index": rep_by_index,
    }


def _climatology_context(target: datetime) -> dict[str, object]:
    return {
        "month": target.month,
        "hour_of_week": target.weekday() * 24 + target.hour,
        "temp_c": None,
        "relative_humidity_pct": None,
        "wind_speed_mps": None,
        "wet_hour": None,
        "provider": "climatology",
        "timestamp_local": target.isoformat(),
    }


def _station_live_contexts(
    station_indices: np.ndarray,
    rep_by_index: pd.DataFrame,
    forecast_hours: int,
    provider: str,
) -> dict[int, dict[str, object]]:
    contexts: dict[int, dict[str, object]] = {}
    for station_index in sorted(set(int(v) for v in station_indices.tolist())):
        if station_index not in rep_by_index.index:
            # No representative weather station for this segment; fall back to a
            # neutral climatology context instead of raising a KeyError that
            # would fail the whole bbox request.
            contexts[station_index] = _climatology_context(
                _local_target_time(0, forecast_hours)
            )
            continue
        rep = rep_by_index.loc[station_index]
        try:
            snapshot = fetch_live_weather(
                lat=float(rep["LAT"]),
                lon=float(rep["LON"]),
                forecast_hours=int(forecast_hours),
                provider=provider,
            )
            contexts[station_index] = {
                "month": snapshot.timestamp_local.month,
                "hour_of_week": snapshot.timestamp_local.weekday() * 24 + snapshot.timestamp_local.hour,
                "temp_c": snapshot.temp_c,
                "relative_humidity_pct": snapshot.relative_humidity_pct,
                "wind_speed_mps": snapshot.wind_speed_mps,
                "wet_hour": snapshot.wet_hour,
                "provider": snapshot.provider,
                "timestamp_local": snapshot.timestamp_local.isoformat(),
            }
        except Exception:
            target = _local_target_time(int(rep.get("utc_offset_hours", 0)), forecast_hours)
            contexts[station_index] = _climatology_context(target)
    return contexts


def _prepare_bbox_candidates(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    limit: int,
):
    runtime = load_segment_runtime()
    bundle = runtime["bundle"]
    roads = runtime["roads"]

    filtered = roads.loc[
        roads["max_lat"].ge(min_lat)
        & roads["min_lat"].le(max_lat)
        & roads["max_lon"].ge(min_lon)
        & roads["min_lon"].le(max_lon)
    ].copy()
    if filtered.empty:
        return None

    counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    filtered["historical_events"] = counts[filtered["segment_idx"].to_numpy(dtype=np.int32)]
    filtered = filtered.sort_values("historical_events", ascending=False)
    candidate_limit = max(int(limit), 3000)
    if len(filtered) > candidate_limit:
        filtered = filtered.head(candidate_limit).copy()

    station_indices_all = np.asarray(bundle["segment_station_indices"], dtype=np.int16)
    filtered_station_indices = station_indices_all[filtered["segment_idx"].to_numpy(dtype=np.int32)]
    return runtime, filtered, filtered_station_indices


def _score_rows(filtered, filtered_station_indices, station_contexts, runtime, forecast_hours):
    bundle = runtime["bundle"]
    static_features = runtime["static_features"]
    counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    weather_cube = np.asarray(bundle["weather_climatology"], dtype=np.float32)
    weather_defaults = np.asarray(bundle["weather_defaults"], dtype=np.float32)
    hour_counts = np.asarray(bundle["segment_hour_counts"], dtype=np.float32)
    model = bundle["model"]

    rows = []
    for station_index, station_frame in filtered.groupby(filtered_station_indices, sort=False):
        ctx = station_contexts[int(station_index)]
        segment_idx = station_frame["segment_idx"].to_numpy(dtype=np.int32)
        static = static_features[station_frame.index.to_numpy(dtype=np.int32)]
        month = int(ctx["month"])
        hour_of_week = int(ctx["hour_of_week"])

        climatology_weather = lookup_weather_climatology(
            weather_cube=weather_cube,
            station_indices=np.full(len(station_frame), int(station_index), dtype=np.int16),
            months=np.full(len(station_frame), month, dtype=np.int8),
            hour_of_week=np.full(len(station_frame), hour_of_week, dtype=np.int16),
            weather_defaults=weather_defaults,
        )
        temp_c = climatology_weather[:, 0] if ctx["temp_c"] is None else np.full(len(station_frame), float(ctx["temp_c"]), dtype=np.float32)
        rh = climatology_weather[:, 1] if ctx["relative_humidity_pct"] is None else np.full(len(station_frame), float(ctx["relative_humidity_pct"]), dtype=np.float32)
        wind = climatology_weather[:, 2] if ctx["wind_speed_mps"] is None else np.full(len(station_frame), float(ctx["wind_speed_mps"]), dtype=np.float32)
        wet = climatology_weather[:, 3] if ctx["wet_hour"] is None else np.full(len(station_frame), float(ctx["wet_hour"]), dtype=np.float32)

        features = build_segment_feature_matrix(
            static_features=static,
            hour_of_week=np.full(len(station_frame), hour_of_week, dtype=np.int16),
            months=np.full(len(station_frame), month, dtype=np.int8),
            totals=counts[segment_idx],
            same_hour=hour_counts[segment_idx, hour_of_week],
            temp_c=temp_c,
            relative_humidity_pct=rh,
            wind_speed_mps=wind,
            wet_hour=wet,
        )
        risk = model.predict_proba(features)[:, 1].astype(np.float32)
        part = station_frame[
            ["segment_id", "fullname", "coords_json", "center_lat", "center_lon", "segment_idx"]
        ].copy()
        part["risk_score"] = risk
        part["forecast_hours"] = int(forecast_hours)
        part["target_timestamp_local"] = ctx["timestamp_local"]
        part["weather_provider"] = ctx["provider"]
        rows.append(part)
    return pd.concat(rows, ignore_index=True)


def _baseline_contexts(live_contexts: dict[int, dict]) -> dict[int, dict]:
    # Same time-of-week as the live contexts but climatology weather, so a
    # live-vs-baseline delta isolates the current weather anomaly.
    return {
        index: {
            **context,
            "temp_c": None,
            "relative_humidity_pct": None,
            "wind_speed_mps": None,
            "wet_hour": None,
            "provider": "climatology",
        }
        for index, context in live_contexts.items()
    }


def score_segments_in_bbox(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
    limit: int = SEGMENT_SERVE_LIMIT,
) -> dict[str, object]:
    prepared = _prepare_bbox_candidates(
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, limit=limit
    )
    if prepared is None:
        return {"segments": [], "count": 0}
    runtime, filtered, filtered_station_indices = prepared

    station_contexts = _station_live_contexts(
        filtered_station_indices,
        rep_by_index=runtime["rep_by_index"],
        forecast_hours=forecast_hours,
        provider=provider,
    )
    scored = _score_rows(
        filtered, filtered_station_indices, station_contexts, runtime, forecast_hours
    ).sort_values("risk_score", ascending=False)
    if len(scored) > limit:
        scored = scored.head(limit).copy()
    return {"count": int(len(scored)), "segments": scored.to_dict(orient="records")}


def rank_hotspots_in_bbox(
    *,
    min_lat: float,
    max_lat: float,
    min_lon: float,
    max_lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
    top_n: int = 50,
    rank_by: str = "risk",
) -> dict[str, object]:
    rank_by = "delta" if str(rank_by).lower() == "delta" else "risk"
    prepared = _prepare_bbox_candidates(
        min_lat=min_lat, max_lat=max_lat, min_lon=min_lon, max_lon=max_lon, limit=top_n
    )
    if prepared is None:
        return {"segments": [], "count": 0, "rank_by": rank_by}
    runtime, filtered, filtered_station_indices = prepared

    live_contexts = _station_live_contexts(
        filtered_station_indices,
        rep_by_index=runtime["rep_by_index"],
        forecast_hours=forecast_hours,
        provider=provider,
    )
    live = _score_rows(filtered, filtered_station_indices, live_contexts, runtime, forecast_hours)
    baseline = _score_rows(
        filtered, filtered_station_indices, _baseline_contexts(live_contexts), runtime, forecast_hours
    )[["segment_idx", "risk_score"]].rename(columns={"risk_score": "baseline_score"})

    merged = live.merge(baseline, on="segment_idx", how="left")
    merged["baseline_score"] = merged["baseline_score"].fillna(merged["risk_score"])
    merged["delta"] = merged["risk_score"] - merged["baseline_score"]

    sort_column = "delta" if rank_by == "delta" else "risk_score"
    merged = merged.sort_values(sort_column, ascending=False).head(int(top_n)).copy()
    return {
        "count": int(len(merged)),
        "rank_by": rank_by,
        "segments": merged.to_dict(orient="records"),
    }
