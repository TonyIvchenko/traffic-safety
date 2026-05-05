from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.parquet as pq
from sklearn.neighbors import BallTree

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from common import (
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    ROAD_SEGMENTS_PATH,
    SEGMENT_EVENTS_PATH,
    SEGMENT_MATCH_CANDIDATES,
    US_ACCIDENTS_PATH,
    ensure_dirs,
)
from segment_support import coords_from_json, point_to_polyline_distance_km


USECOLS = [
    "ID",
    "Severity",
    "Start_Time",
    "Start_Lat",
    "Start_Lng",
    "State",
    "Timezone",
    "Temperature(F)",
    "Humidity(%)",
    "Wind_Speed(mph)",
    "Precipitation(in)",
    "Weather_Condition",
]

STATE_ABBR_TO_FIPS = {
    "AL": "01",
    "AK": "02",
    "AZ": "04",
    "AR": "05",
    "CA": "06",
    "CO": "08",
    "CT": "09",
    "DE": "10",
    "DC": "11",
    "FL": "12",
    "GA": "13",
    "HI": "15",
    "ID": "16",
    "IL": "17",
    "IN": "18",
    "IA": "19",
    "KS": "20",
    "KY": "21",
    "LA": "22",
    "ME": "23",
    "MD": "24",
    "MA": "25",
    "MI": "26",
    "MN": "27",
    "MS": "28",
    "MO": "29",
    "MT": "30",
    "NE": "31",
    "NV": "32",
    "NH": "33",
    "NJ": "34",
    "NM": "35",
    "NY": "36",
    "NC": "37",
    "ND": "38",
    "OH": "39",
    "OK": "40",
    "OR": "41",
    "PA": "42",
    "RI": "44",
    "SC": "45",
    "SD": "46",
    "TN": "47",
    "TX": "48",
    "UT": "49",
    "VT": "50",
    "VA": "51",
    "WA": "53",
    "WV": "54",
    "WI": "55",
    "WY": "56",
}


def fahrenheit_to_celsius(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return (float(value) - 32.0) * (5.0 / 9.0)


def mph_to_mps(value: float | None) -> float | None:
    if value is None or pd.isna(value):
        return None
    return float(value) * 0.44704


def wet_hour_from_row(precip_in: float | None, weather_condition: str | None) -> float:
    if precip_in is not None and not pd.isna(precip_in) and float(precip_in) > 0.0:
        return 1.0
    summary = str(weather_condition or "").lower()
    if any(token in summary for token in ("rain", "storm", "snow", "drizzle", "sleet", "hail")):
        return 1.0
    return 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--chunksize", type=int, default=50000)
    parser.add_argument("--max-match-km", type=float, default=1.0)
    parser.add_argument("--max-chunks", type=int, default=0)
    return parser.parse_args()


def load_segments() -> tuple[
    pd.DataFrame,
    list[list[tuple[float, float]]],
    dict[str, np.ndarray],
    dict[str, BallTree],
]:
    segments = pd.read_parquet(ROAD_SEGMENTS_PATH)
    segments = segments.reset_index(drop=True)
    coords = [coords_from_json(payload) for payload in segments["coords_json"].tolist()]
    state_indices: dict[str, np.ndarray] = {}
    state_trees: dict[str, BallTree] = {}
    for state_fips, state_frame in segments.groupby("state_fips", sort=False):
        index_values = state_frame.index.to_numpy(dtype=np.int32)
        center_radians = np.radians(
            state_frame[["center_lat", "center_lon"]].to_numpy(dtype=np.float64)
        )
        state_indices[str(state_fips)] = index_values
        state_trees[str(state_fips)] = BallTree(center_radians, metric="haversine")
    return segments, coords, state_indices, state_trees


def normalize_chunk(chunk: pd.DataFrame) -> pd.DataFrame:
    chunk["Start_Lat"] = pd.to_numeric(chunk["Start_Lat"], errors="coerce")
    chunk["Start_Lng"] = pd.to_numeric(chunk["Start_Lng"], errors="coerce")
    chunk["Severity"] = pd.to_numeric(chunk["Severity"], errors="coerce")
    chunk["Temperature(F)"] = pd.to_numeric(chunk["Temperature(F)"], errors="coerce")
    chunk["Humidity(%)"] = pd.to_numeric(chunk["Humidity(%)"], errors="coerce")
    chunk["Wind_Speed(mph)"] = pd.to_numeric(chunk["Wind_Speed(mph)"], errors="coerce")
    chunk["Precipitation(in)"] = pd.to_numeric(chunk["Precipitation(in)"], errors="coerce")
    chunk["Start_Time"] = pd.to_datetime(chunk["Start_Time"], errors="coerce")
    chunk = chunk.dropna(subset=["Start_Time", "Start_Lat", "Start_Lng", "Severity", "State"]).copy()
    chunk = chunk.loc[
        chunk["Start_Lat"].between(LAT_MIN, LAT_MAX)
        & chunk["Start_Lng"].between(LON_MIN, LON_MAX)
    ].copy()
    chunk["state_fips"] = chunk["State"].map(STATE_ABBR_TO_FIPS)
    chunk = chunk.dropna(subset=["state_fips"]).copy()
    return chunk


def matched_rows_for_chunk(
    chunk: pd.DataFrame,
    segments: pd.DataFrame,
    coords: list[list[tuple[float, float]]],
    state_indices: dict[str, np.ndarray],
    state_trees: dict[str, BallTree],
    max_match_km: float,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for state_fips, state_chunk in chunk.groupby("state_fips", sort=False):
        tree = state_trees.get(str(state_fips))
        state_segment_indices = state_indices.get(str(state_fips))
        if tree is None or state_segment_indices is None:
            continue

        query_points = np.radians(
            state_chunk[["Start_Lat", "Start_Lng"]].to_numpy(dtype=np.float64)
        )
        _, candidate_pos = tree.query(query_points, k=SEGMENT_MATCH_CANDIDATES)

        for row_idx, (record, candidate_row) in enumerate(
            zip(state_chunk.to_dict(orient="records"), candidate_pos, strict=False)
        ):
            lat = float(record["Start_Lat"])
            lon = float(record["Start_Lng"])
            best_idx = None
            best_dist = float("inf")
            for pos in candidate_row:
                segment_idx = int(state_segment_indices[int(pos)])
                dist = point_to_polyline_distance_km(lat, lon, coords[segment_idx])
                if dist < best_dist:
                    best_dist = dist
                    best_idx = segment_idx
            if best_idx is None or best_dist > max_match_km:
                continue

            start_time = pd.Timestamp(record["Start_Time"])
            rows.append(
                {
                    "segment_id": segments.at[best_idx, "segment_id"],
                    "state_fips": segments.at[best_idx, "state_fips"],
                    "severity": int(record["Severity"]),
                    "event_id": str(record["ID"]),
                    "year": int(start_time.year),
                    "month": int(start_time.month),
                    "day": int(start_time.day),
                    "hour": int(start_time.hour),
                    "hour_of_week": int(start_time.dayofweek * 24 + start_time.hour),
                    "lat": lat,
                    "lon": lon,
                    "temp_c": fahrenheit_to_celsius(record.get("Temperature(F)")),
                    "relative_humidity_pct": (
                        float(record["Humidity(%)"])
                        if not pd.isna(record.get("Humidity(%)"))
                        else None
                    ),
                    "wind_speed_mps": mph_to_mps(record.get("Wind_Speed(mph)")),
                    "wet_hour": wet_hour_from_row(
                        record.get("Precipitation(in)"),
                        record.get("Weather_Condition"),
                    ),
                    "weather_condition": str(record.get("Weather_Condition") or ""),
                    "match_distance_km": float(best_dist),
                    "timezone": str(record.get("Timezone") or ""),
                }
            )
    return rows


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not ROAD_SEGMENTS_PATH.exists():
        raise FileNotFoundError(
            f"missing {ROAD_SEGMENTS_PATH}; run build_segments.py first"
        )
    if not US_ACCIDENTS_PATH.exists():
        raise FileNotFoundError(
            f"missing {US_ACCIDENTS_PATH}; run download_us_accidents.py first"
        )

    segments, coords, state_indices, state_trees = load_segments()
    writer = None
    matched_total = 0
    source_total = 0

    for chunk_idx, chunk in enumerate(
        pd.read_csv(
            US_ACCIDENTS_PATH,
            usecols=USECOLS,
            low_memory=False,
            chunksize=args.chunksize,
        )
    ):
        if args.max_chunks and chunk_idx >= args.max_chunks:
            break
        normalized = normalize_chunk(chunk)
        source_total += len(normalized)
        rows = matched_rows_for_chunk(
            normalized,
            segments=segments,
            coords=coords,
            state_indices=state_indices,
            state_trees=state_trees,
            max_match_km=args.max_match_km,
        )
        matched_total += len(rows)
        if rows:
            table = pa.Table.from_pandas(pd.DataFrame(rows), preserve_index=False)
            if writer is None:
                writer = pq.ParquetWriter(SEGMENT_EVENTS_PATH, table.schema, compression="zstd")
            writer.write_table(table)
        print(
            f"chunk={chunk_idx} normalized={len(normalized)} matched={len(rows)} total_matched={matched_total}"
        )

    if writer is not None:
        writer.close()
    print(f"wrote {SEGMENT_EVENTS_PATH}")
    print(f"normalized_rows={source_total} matched_rows={matched_total}")


if __name__ == "__main__":
    main()
