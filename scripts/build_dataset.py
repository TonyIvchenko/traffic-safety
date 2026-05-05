from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

import h3
import pandas as pd

from common import (
    ACCIDENTS_CLEAN_PATH,
    CANDIDATE_CELLS_PATH,
    DEFAULT_YEARS,
    H3_RESOLUTION,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    WEEKLY_COUNTS_PATH,
    ensure_dirs,
    fars_zip_path,
)


USECOLS = [
    "STATE",
    "STATENAME",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_WEEK",
    "HOUR",
    "LATITUDE",
    "LONGITUD",
    "FATALS",
    "RUR_URB",
    "FUNC_SYS",
    "WEATHER",
    "LGT_COND",
    "WRK_ZONE",
]


def normalize_day_of_week(day_week: int) -> int:
    # FARS uses Sunday=1..Saturday=7. Convert to Monday=1..Sunday=7.
    return ((int(day_week) + 5) % 7) + 1


def load_year_accidents(year: int) -> pd.DataFrame:
    zip_path = fars_zip_path(year)
    if not zip_path.exists():
        raise FileNotFoundError(
            f"missing {zip_path}; run download_data.py for year {year} first"
        )

    with zipfile.ZipFile(zip_path) as archive:
        accident_member = next(
            name for name in archive.namelist() if name.lower().endswith("accident.csv")
        )
        frame = None
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                with archive.open(accident_member) as handle:
                    frame = pd.read_csv(
                        handle,
                        usecols=USECOLS,
                        low_memory=False,
                        encoding=encoding,
                    )
                break
            except UnicodeDecodeError:
                continue
        if frame is None:
            raise UnicodeDecodeError(
                "utf-8",
                b"",
                0,
                1,
                f"unable to decode {accident_member} in {zip_path.name}",
            )

    frame["LATITUDE"] = pd.to_numeric(frame["LATITUDE"], errors="coerce")
    frame["LONGITUD"] = pd.to_numeric(frame["LONGITUD"], errors="coerce")
    frame["MONTH"] = pd.to_numeric(frame["MONTH"], errors="coerce")
    frame["DAY"] = pd.to_numeric(frame["DAY"], errors="coerce")
    frame["DAY_WEEK"] = pd.to_numeric(frame["DAY_WEEK"], errors="coerce")
    frame["HOUR"] = pd.to_numeric(frame["HOUR"], errors="coerce")
    frame["FATALS"] = pd.to_numeric(frame["FATALS"], errors="coerce")
    frame["RUR_URB"] = pd.to_numeric(frame["RUR_URB"], errors="coerce")
    frame["FUNC_SYS"] = pd.to_numeric(frame["FUNC_SYS"], errors="coerce")
    frame["WEATHER"] = pd.to_numeric(frame["WEATHER"], errors="coerce")
    frame["LGT_COND"] = pd.to_numeric(frame["LGT_COND"], errors="coerce")
    frame["WRK_ZONE"] = pd.to_numeric(frame["WRK_ZONE"], errors="coerce")

    frame = frame.dropna(
        subset=["LATITUDE", "LONGITUD", "MONTH", "DAY", "DAY_WEEK", "HOUR"]
    )
    frame = frame.loc[
        frame["LATITUDE"].between(LAT_MIN, LAT_MAX)
        & frame["LONGITUD"].between(LON_MIN, LON_MAX)
        & frame["MONTH"].between(1, 12)
        & frame["DAY"].between(1, 31)
        & frame["DAY_WEEK"].between(1, 7)
        & frame["HOUR"].between(0, 23)
    ].copy()

    frame["day_of_week"] = frame["DAY_WEEK"].astype(int).map(normalize_day_of_week)
    frame["day"] = frame["DAY"].astype(int)
    frame["hour"] = frame["HOUR"].astype(int)
    frame["month"] = frame["MONTH"].astype(int)
    frame["fatals"] = frame["FATALS"].fillna(1).astype(int).clip(lower=1)
    frame["lat"] = frame["LATITUDE"].astype(float)
    frame["lon"] = frame["LONGITUD"].astype(float)
    frame["hour_of_week"] = (frame["day_of_week"] - 1) * 24 + frame["hour"]
    frame["cell_id"] = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=False)
    ]
    return frame[
        [
            "YEAR",
            "STATE",
            "STATENAME",
            "month",
            "day",
            "day_of_week",
            "hour",
            "hour_of_week",
            "lat",
            "lon",
            "fatals",
            "RUR_URB",
            "FUNC_SYS",
            "WEATHER",
            "LGT_COND",
            "WRK_ZONE",
            "cell_id",
        ]
    ].rename(
        columns={
            "YEAR": "year",
            "STATE": "state_code",
            "STATENAME": "state_name",
            "RUR_URB": "rur_urb",
            "FUNC_SYS": "func_sys",
            "WEATHER": "weather_code",
            "LGT_COND": "light_code",
            "WRK_ZONE": "work_zone_code",
        }
    )


def build_candidate_cells(events: pd.DataFrame) -> pd.DataFrame:
    active_cells = set(events["cell_id"].unique())
    candidate_cells: set[str] = set(active_cells)
    for cell_id in active_cells:
        candidate_cells.update(h3.grid_disk(cell_id, 1))

    weekly_counts = (
        events.groupby(["cell_id", "hour_of_week"])
        .size()
        .rename("event_count")
        .reset_index()
    )
    total_counts = weekly_counts.groupby("cell_id")["event_count"].sum().rename("event_count")

    rows = []
    for cell_id in sorted(candidate_cells):
        lat, lon = h3.cell_to_latlng(cell_id)
        rows.append(
            {
                "cell_id": cell_id,
                "center_lat": float(lat),
                "center_lon": float(lon),
                "event_count": int(total_counts.get(cell_id, 0)),
            }
        )
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    yearly_frames = []
    for year in args.years:
        print(f"load {year}")
        yearly_frames.append(load_year_accidents(year))

    events = pd.concat(yearly_frames, ignore_index=True)
    events = events.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    candidate_cells = build_candidate_cells(events)
    weekly_counts = (
        events.groupby(["cell_id", "hour_of_week"]).size().rename("event_count").reset_index()
    )

    events.to_csv(ACCIDENTS_CLEAN_PATH, index=False, compression="gzip")
    candidate_cells.to_csv(CANDIDATE_CELLS_PATH, index=False, compression="gzip")
    weekly_counts.to_csv(WEEKLY_COUNTS_PATH, index=False, compression="gzip")

    print(f"wrote {ACCIDENTS_CLEAN_PATH}")
    print(f"wrote {CANDIDATE_CELLS_PATH}")
    print(f"wrote {WEEKLY_COUNTS_PATH}")
    print(f"events={len(events)} candidate_cells={len(candidate_cells)}")


if __name__ == "__main__":
    main()
