"""Build a persistent high-risk corridor report for engineering review.

Groups active road segments into corridors (state + TIGER linear feature),
aggregates crash history and the current 24h forecast, and writes durable
artifacts under data/reports/:

- hotspots.csv       ranked corridor table
- hotspots.geojson   the same corridors as MultiLineString features

    python scripts/build_hotspot_report.py --top-n 100 --rank-by events
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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

from road_tiles import ROAD_TILE_FORECAST_PATH
from segment_support import coords_from_json
from scripts.common import (
    ACTIVE_ROAD_SEGMENTS_PATH,
    SEGMENT_MODEL_BUNDLE_PATH,
    weekly_frame_labels,
)

REPORTS_DIR = REPO_DIR / "data" / "reports"
MIN_CORRIDOR_KM = 0.1  # floor so tiny corridors do not dominate events_per_km

CORRIDOR_KEYS = ["state_fips", "linearid"]
RANK_COLUMNS = {"events": "historical_events", "events_per_km": "events_per_km"}


def build_corridor_table(
    roads: pd.DataFrame,
    total_counts: np.ndarray,
    hour_counts: np.ndarray,
    forecast_max: np.ndarray | None = None,
    *,
    top_n: int = 100,
    min_events: int = 3,
    rank_by: str = "events",
) -> pd.DataFrame:
    """Aggregate per-segment history into ranked corridors.

    ``total_counts``/``hour_counts``/``forecast_max`` are indexed by
    ``segment_idx``; ``forecast_max`` is already scaled to [0, 1].
    """
    rank_column = RANK_COLUMNS.get(str(rank_by).lower())
    if rank_column is None:
        raise ValueError(f"rank_by must be one of {sorted(RANK_COLUMNS)}")

    frame = roads[
        CORRIDOR_KEYS + ["segment_idx", "fullname", "length_km", "center_lat", "center_lon"]
    ].copy()
    segment_idx = frame["segment_idx"].to_numpy(dtype=np.int64)
    frame["events"] = total_counts[segment_idx].astype(np.float64)
    frame["name"] = frame["fullname"].astype(str).str.strip().replace("", np.nan)

    # Vectorized corridor aggregation; the expensive per-corridor work (peak
    # hour, forecast max) runs only for the ranked top-N survivors below.
    grouped = frame.groupby(CORRIDOR_KEYS, sort=False).agg(
        name=("name", "first"),
        segments=("segment_idx", "size"),
        length_km=("length_km", "sum"),
        historical_events=("events", "sum"),
        center_lat=("center_lat", "mean"),
        center_lon=("center_lon", "mean"),
    )
    grouped = grouped.loc[grouped["historical_events"] >= min_events].copy()
    if grouped.empty:
        return pd.DataFrame()
    grouped["name"] = grouped["name"].fillna("Unnamed road")
    grouped["length_km"] = grouped["length_km"].round(3)
    grouped["events_per_km"] = (
        grouped["historical_events"] / grouped["length_km"].clip(lower=MIN_CORRIDOR_KM)
    ).round(4)
    grouped["center_lat"] = grouped["center_lat"].round(6)
    grouped["center_lon"] = grouped["center_lon"].round(6)

    table = (
        grouped.sort_values(rank_column, ascending=False)
        .head(int(top_n))
        .reset_index()
    )
    table.insert(0, "rank", np.arange(1, len(table) + 1))

    labels = weekly_frame_labels()
    members = frame.loc[
        pd.MultiIndex.from_frame(frame[CORRIDOR_KEYS]).isin(
            pd.MultiIndex.from_frame(table[CORRIDOR_KEYS])
        )
    ].groupby(CORRIDOR_KEYS)["segment_idx"]
    peak_hours, peak_labels, peak_events, forecast_risks = [], [], [], []
    for row in table.itertuples():
        idx = members.get_group((row.state_fips, row.linearid)).to_numpy(dtype=np.int64)
        corridor_hours = hour_counts[idx].sum(axis=0)
        peak_idx = int(np.argmax(corridor_hours))
        peak_hours.append(peak_idx)
        peak_labels.append(labels[peak_idx])
        peak_events.append(float(corridor_hours[peak_idx]))
        forecast_risks.append(
            round(float(forecast_max[idx].max()), 4) if forecast_max is not None else None
        )
    table["peak_hour_of_week"] = peak_hours
    table["peak_hour_label"] = peak_labels
    table["peak_hour_events"] = peak_events
    table["forecast_risk_max"] = forecast_risks
    return table


def corridors_geojson(table: pd.DataFrame, roads: pd.DataFrame) -> dict:
    """MultiLineString feature per corridor, geometry from member segments."""
    features = []
    if not table.empty:
        member_coords = roads.groupby(CORRIDOR_KEYS)["coords_json"]
        for row in table.to_dict(orient="records"):
            key = (row["state_fips"], row["linearid"])
            try:
                payloads = member_coords.get_group(key)
            except KeyError:
                payloads = []
            lines = [
                [[float(lon), float(lat)] for lon, lat in coords_from_json(payload)]
                for payload in payloads
            ]
            lines = [line for line in lines if len(line) >= 2]
            features.append(
                {
                    "type": "Feature",
                    "geometry": {"type": "MultiLineString", "coordinates": lines},
                    "properties": {
                        key: value for key, value in row.items() if key != "coords_json"
                    },
                }
            )
    return {"type": "FeatureCollection", "count": len(features), "features": features}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=100)
    parser.add_argument("--min-events", type=int, default=3)
    parser.add_argument("--rank-by", choices=sorted(RANK_COLUMNS), default="events")
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    roads = pd.read_parquet(ACTIVE_ROAD_SEGMENTS_PATH)
    bundle = joblib.load(SEGMENT_MODEL_BUNDLE_PATH)
    total_counts = np.asarray(bundle["segment_total_counts"], dtype=np.float32)
    hour_counts = np.asarray(bundle["segment_hour_counts"], dtype=np.float32)

    forecast_max = None
    if ROAD_TILE_FORECAST_PATH.exists():
        scores = np.load(ROAD_TILE_FORECAST_PATH, mmap_mode="r")
        forecast_max = np.zeros(scores.shape[0], dtype=np.float32)
        active_idx = roads["segment_idx"].to_numpy(dtype=np.int64)
        forecast_max[active_idx] = scores[active_idx].max(axis=1).astype(np.float32) / 255.0

    table = build_corridor_table(
        roads,
        total_counts,
        hour_counts,
        forecast_max,
        top_n=args.top_n,
        min_events=args.min_events,
        rank_by=args.rank_by,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "hotspots.csv"
    geojson_path = args.output_dir / "hotspots.geojson"
    table.to_csv(csv_path, index=False)
    payload = corridors_geojson(table, roads)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    payload["rank_by"] = args.rank_by
    geojson_path.write_text(json.dumps(payload), encoding="utf-8")

    print(f"corridors={len(table)} rank_by={args.rank_by} min_events={args.min_events}")
    if not table.empty:
        top = table.iloc[0]
        print(f"top: {top['name']} ({top['state_fips']}) events={top['historical_events']}")
    print(f"wrote {csv_path}")
    print(f"wrote {geojson_path}")


if __name__ == "__main__":
    main()
