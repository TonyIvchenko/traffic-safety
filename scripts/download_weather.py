from __future__ import annotations

import argparse
from pathlib import Path
import sys

import h3
import numpy as np
import pandas as pd
import requests
from sklearn.neighbors import BallTree

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_support import approximate_utc_offset_hours, relative_humidity_from_temp_dewpoint

from common import (
    CANDIDATE_CELLS_PATH,
    CELL_WEATHER_STATIONS_PATH,
    DEFAULT_YEARS,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    REPRESENTATIVE_STATIONS_PATH,
    STATION_HISTORY_PATH,
    WEATHER_CLIMATOLOGY_PATH,
    WEATHER_HOURLY_DIR,
    WEATHER_REPRESENTATION_H3_RES,
    ensure_dirs,
    noaa_isd_lite_path,
    noaa_isd_lite_url,
)


STATION_HISTORY_URL = "https://www.ncei.noaa.gov/pub/data/noaa/isd-history.csv"
WEATHER_RAW_COLUMNS = [
    "year",
    "month",
    "day",
    "hour",
    "temp_tenths_c",
    "dewpoint_tenths_c",
    "slp_tenths_hpa",
    "wind_dir_deg",
    "wind_speed_tenths_mps",
    "sky_cover_oktas",
    "precip1_tenths_mm",
    "precip6_tenths_mm",
]
HOURLY_OUTPUT_COLUMNS = [
    "station_index",
    "year",
    "month",
    "day",
    "hour",
    "hour_of_week",
    "temp_c",
    "dewpoint_c",
    "relative_humidity_pct",
    "wind_speed_mps",
    "wet_hour",
]


def download_file(url: str, path: Path, force: bool = False) -> None:
    if path.exists() and not force:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    response = requests.get(url, timeout=60)
    response.raise_for_status()
    path.write_bytes(response.content)


def load_station_history(force: bool = False) -> pd.DataFrame:
    download_file(STATION_HISTORY_URL, STATION_HISTORY_PATH, force=force)
    stations = pd.read_csv(STATION_HISTORY_PATH, dtype={"USAF": str, "WBAN": str})
    stations = stations.loc[
        (stations["CTRY"] == "US")
        & stations["LAT"].notna()
        & stations["LON"].notna()
        & stations["LAT"].between(LAT_MIN, LAT_MAX)
        & stations["LON"].between(LON_MIN, LON_MAX)
    ].copy()
    stations["USAF"] = stations["USAF"].fillna("").str.zfill(6)
    stations["WBAN"] = stations["WBAN"].fillna("").str.zfill(5)
    stations["station_id"] = stations["USAF"] + "-" + stations["WBAN"]
    stations["begin_int"] = pd.to_numeric(stations["BEGIN"], errors="coerce").fillna(0).astype(int)
    stations["end_int"] = pd.to_numeric(stations["END"], errors="coerce").fillna(0).astype(int)
    return stations


def coverage_fraction(begin_int: int, end_int: int, years: list[int]) -> float:
    target_start = years[0] * 10000 + 101
    target_end = years[-1] * 10000 + 1231
    overlap_start = max(begin_int, target_start)
    overlap_end = min(end_int, target_end)
    if overlap_end < overlap_start:
        return 0.0
    return (overlap_end - overlap_start) / max(1, target_end - target_start)


def haversine_km(lat1: float, lon1: float, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    earth_radius_km = 6371.0088
    lat1_rad = np.radians(lat1)
    lon1_rad = np.radians(lon1)
    lat2_rad = np.radians(lat2)
    lon2_rad = np.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    return 2.0 * earth_radius_km * np.arcsin(np.sqrt(a))


def select_representative_stations(
    candidate_cells: pd.DataFrame,
    stations: pd.DataFrame,
    years: list[int],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    coarse_cells = candidate_cells.copy()
    coarse_cells["weather_coarse_cell"] = [
        h3.latlng_to_cell(lat, lon, WEATHER_REPRESENTATION_H3_RES)
        for lat, lon in zip(coarse_cells["center_lat"], coarse_cells["center_lon"], strict=False)
    ]
    centroids = (
        coarse_cells.groupby("weather_coarse_cell")[["center_lat", "center_lon"]]
        .mean()
        .reset_index()
    )

    station_coords = np.radians(stations[["LAT", "LON"]].to_numpy(dtype=np.float64))
    tree = BallTree(station_coords, metric="haversine")
    centroid_coords = np.radians(centroids[["center_lat", "center_lon"]].to_numpy(dtype=np.float64))
    query_k = min(12, len(stations))
    distances, indices = tree.query(centroid_coords, k=query_k)

    chosen_rows: list[pd.Series] = []
    for row_idx, centroid in centroids.iterrows():
        candidate_idx = indices[row_idx]
        nearby = stations.iloc[candidate_idx].copy()
        nearby["distance_km"] = haversine_km(
            float(centroid["center_lat"]),
            float(centroid["center_lon"]),
            nearby["LAT"].to_numpy(dtype=np.float64),
            nearby["LON"].to_numpy(dtype=np.float64),
        )
        nearby["coverage_fraction"] = nearby.apply(
            lambda station: coverage_fraction(
                int(station["begin_int"]),
                int(station["end_int"]),
                years,
            ),
            axis=1,
        )
        nearby["score"] = nearby["distance_km"] + (1.0 - nearby["coverage_fraction"]) * 500.0
        best = nearby.sort_values(["score", "distance_km"]).iloc[0]
        chosen_rows.append(best)

    representative = pd.DataFrame(chosen_rows).drop_duplicates(subset=["station_id"]).copy()
    representative = representative.sort_values("station_id").reset_index(drop=True)
    representative["station_index"] = np.arange(len(representative), dtype=np.int16)
    representative["utc_offset_hours"] = [
        approximate_utc_offset_hours(float(lat), float(lon))
        for lat, lon in zip(representative["LAT"], representative["LON"], strict=False)
    ]

    rep_coords = np.radians(representative[["LAT", "LON"]].to_numpy(dtype=np.float64))
    rep_tree = BallTree(rep_coords, metric="haversine")
    cell_coords = np.radians(candidate_cells[["center_lat", "center_lon"]].to_numpy(dtype=np.float64))
    _, rep_indices = rep_tree.query(cell_coords, k=1)
    mapped = candidate_cells[["cell_id"]].copy()
    matched = representative.iloc[rep_indices[:, 0]].reset_index(drop=True)
    mapped["station_index"] = matched["station_index"].to_numpy(dtype=np.int16)
    mapped["station_id"] = matched["station_id"].to_numpy(dtype=object)

    return representative, mapped


def parse_isd_lite_file(path: Path) -> pd.DataFrame:
    return pd.read_csv(
        path,
        sep=r"\s+",
        header=None,
        names=WEATHER_RAW_COLUMNS,
        compression="gzip",
        low_memory=False,
    )


def process_station_year(
    station: pd.Series,
    year: int,
    force: bool = False,
) -> pd.DataFrame:
    raw_path = noaa_isd_lite_path(str(station["station_id"]), year)
    processed_path = WEATHER_HOURLY_DIR / str(year) / f"{int(station['station_index']):04d}.csv.gz"
    processed_path.parent.mkdir(parents=True, exist_ok=True)
    if processed_path.exists() and not force:
        return pd.read_csv(processed_path)

    frame = parse_isd_lite_file(raw_path)
    timestamp_utc = pd.to_datetime(
        {
            "year": frame["year"],
            "month": frame["month"],
            "day": frame["day"],
            "hour": frame["hour"],
        },
        errors="coerce",
        utc=True,
    )
    timestamp_local = timestamp_utc + pd.to_timedelta(int(station["utc_offset_hours"]), unit="h")

    temp_c = frame["temp_tenths_c"].replace(-9999, np.nan).astype(np.float32) / 10.0
    dewpoint_c = frame["dewpoint_tenths_c"].replace(-9999, np.nan).astype(np.float32) / 10.0
    wind_speed_mps = frame["wind_speed_tenths_mps"].replace(-9999, np.nan).astype(np.float32) / 10.0
    precip_raw = frame["precip1_tenths_mm"].astype(np.float32)
    wet_hour = np.where(precip_raw == -9999, np.nan, np.where(precip_raw == 0, 0.0, 1.0)).astype(
        np.float32
    )
    humidity = relative_humidity_from_temp_dewpoint(temp_c, dewpoint_c)
    humidity[np.isnan(temp_c) | np.isnan(dewpoint_c)] = np.nan

    hourly = pd.DataFrame(
        {
            "station_index": np.full(len(frame), int(station["station_index"]), dtype=np.int16),
            "year": timestamp_local.dt.year.astype(np.int16),
            "month": timestamp_local.dt.month.astype(np.int8),
            "day": timestamp_local.dt.day.astype(np.int8),
            "hour": timestamp_local.dt.hour.astype(np.int8),
            "hour_of_week": (
                timestamp_local.dt.dayofweek.astype(np.int16) * 24
                + timestamp_local.dt.hour.astype(np.int16)
            ),
            "temp_c": temp_c,
            "dewpoint_c": dewpoint_c,
            "relative_humidity_pct": humidity,
            "wind_speed_mps": wind_speed_mps,
            "wet_hour": wet_hour,
        }
    )
    hourly = hourly.loc[hourly["year"] == year].reset_index(drop=True)
    hourly.to_csv(processed_path, index=False, compression="gzip")
    return hourly


def aggregate_climatology(hourly: pd.DataFrame) -> pd.DataFrame:
    return (
        hourly.groupby(["station_index", "month", "hour_of_week"], as_index=False)
        .agg(
            temp_c=("temp_c", "mean"),
            dewpoint_c=("dewpoint_c", "mean"),
            relative_humidity_pct=("relative_humidity_pct", "mean"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            wet_hour=("wet_hour", "mean"),
            obs_count=("hour_of_week", "size"),
        )
        .astype(
            {
                "station_index": np.int16,
                "month": np.int8,
                "hour_of_week": np.int16,
                "obs_count": np.int32,
            }
        )
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not CANDIDATE_CELLS_PATH.exists():
        raise FileNotFoundError(
            f"missing {CANDIDATE_CELLS_PATH}; run build_dataset.py first"
        )

    candidate_cells = pd.read_csv(CANDIDATE_CELLS_PATH)
    stations = load_station_history(force=args.force)
    representative, cell_station_map = select_representative_stations(
        candidate_cells=candidate_cells,
        stations=stations,
        years=args.years,
    )
    representative.to_csv(REPRESENTATIVE_STATIONS_PATH, index=False, compression="gzip")
    cell_station_map.to_csv(CELL_WEATHER_STATIONS_PATH, index=False, compression="gzip")
    print(f"representative_stations={len(representative)}")

    partial_climatology: list[pd.DataFrame] = []
    for station in representative.to_dict(orient="records"):
        station_row = pd.Series(station)
        for year in args.years:
            begin_year = int(str(int(station_row["begin_int"]))[:4] or 0)
            end_year = int(str(int(station_row["end_int"]))[:4] or 0)
            if begin_year and year < begin_year:
                continue
            if end_year and year > end_year:
                continue
            url = noaa_isd_lite_url(str(station_row["station_id"]), year)
            path = noaa_isd_lite_path(str(station_row["station_id"]), year)
            try:
                download_file(url, path, force=args.force)
                hourly = process_station_year(station_row, year, force=args.force)
                partial_climatology.append(aggregate_climatology(hourly))
            except requests.HTTPError as exc:
                print(f"skip {station_row['station_id']} {year}: {exc}")

    climatology = (
        pd.concat(partial_climatology, ignore_index=True)
        .groupby(["station_index", "month", "hour_of_week"], as_index=False)
        .agg(
            temp_c=("temp_c", "mean"),
            dewpoint_c=("dewpoint_c", "mean"),
            relative_humidity_pct=("relative_humidity_pct", "mean"),
            wind_speed_mps=("wind_speed_mps", "mean"),
            wet_hour=("wet_hour", "mean"),
            obs_count=("obs_count", "sum"),
        )
        .sort_values(["station_index", "month", "hour_of_week"])
        .reset_index(drop=True)
    )
    climatology.to_csv(WEATHER_CLIMATOLOGY_PATH, index=False, compression="gzip")
    print(f"wrote {REPRESENTATIVE_STATIONS_PATH}")
    print(f"wrote {CELL_WEATHER_STATIONS_PATH}")
    print(f"wrote {WEATHER_CLIMATOLOGY_PATH}")


if __name__ == "__main__":
    main()
