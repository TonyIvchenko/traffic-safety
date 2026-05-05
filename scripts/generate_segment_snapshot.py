from __future__ import annotations

import argparse
from datetime import datetime, timedelta
from pathlib import Path
import sys

import joblib
import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_support import lookup_weather_climatology
from segment_model_support import (
    DEFAULT_MTFCC_VALUES,
    DEFAULT_RTTYP_VALUES,
    build_segment_feature_matrix,
    build_static_segment_frame,
)

from common import (
    ROAD_SEGMENTS_PATH,
    SEGMENT_MODEL_BUNDLE_PATH,
    SEGMENT_RISK_SNAPSHOT_PATH,
    ensure_dirs,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--hours", type=int, default=24)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    roads = pd.read_parquet(ROAD_SEGMENTS_PATH).sort_values("segment_id").reset_index(drop=True)
    bundle = joblib.load(SEGMENT_MODEL_BUNDLE_PATH)
    static_frame, _ = build_static_segment_frame(
        roads,
        rttyp_values=DEFAULT_RTTYP_VALUES,
        mtfcc_values=DEFAULT_MTFCC_VALUES,
    )
    static_features = static_frame.to_numpy(dtype=np.float32)
    model = bundle["model"]
    station_indices = np.asarray(bundle["segment_station_indices"], dtype=np.int16)
    total_counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    hour_counts = np.asarray(bundle["segment_hour_counts"], dtype=np.float32)
    weather_cube = np.asarray(bundle["weather_climatology"], dtype=np.float32)
    weather_defaults = np.asarray(bundle["weather_defaults"], dtype=np.float32)

    now = datetime.now().replace(minute=0, second=0, microsecond=0)
    rows = []
    for offset in range(max(1, int(args.hours))):
        target = now + timedelta(hours=offset)
        month = target.month
        hour_of_week = target.weekday() * 24 + target.hour
        weather = lookup_weather_climatology(
            weather_cube=weather_cube,
            station_indices=station_indices,
            months=np.full(len(roads), month, dtype=np.int8),
            hour_of_week=np.full(len(roads), hour_of_week, dtype=np.int16),
            weather_defaults=weather_defaults,
        )
        features = build_segment_feature_matrix(
            static_features=static_features,
            hour_of_week=np.full(len(roads), hour_of_week, dtype=np.int16),
            months=np.full(len(roads), month, dtype=np.int8),
            totals=total_counts,
            same_hour=hour_counts[:, hour_of_week],
            temp_c=weather[:, 0],
            relative_humidity_pct=weather[:, 1],
            wind_speed_mps=weather[:, 2],
            wet_hour=weather[:, 3],
        )
        risk = model.predict_proba(features)[:, 1].astype(np.float32)
        frame = roads[
            ["segment_id", "fullname", "center_lat", "center_lon", "coords_json", "state_fips"]
        ].copy()
        frame["target_time"] = pd.Timestamp(target)
        frame["hour_offset"] = offset
        frame["risk_score"] = risk
        rows.append(frame)
        print(f"hour_offset={offset} generated={len(frame)}")

    snapshot = pd.concat(rows, ignore_index=True)
    snapshot.to_parquet(SEGMENT_RISK_SNAPSHOT_PATH, index=False)
    print(f"wrote {SEGMENT_RISK_SNAPSHOT_PATH}")
    print(f"rows={len(snapshot)}")


if __name__ == "__main__":
    main()
