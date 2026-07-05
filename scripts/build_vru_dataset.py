"""Build a vulnerable-road-user (pedestrian/cyclist) crash dataset from FARS.

FARS accident.csv is crash-level, so it has no person type. This reads the
companion person.csv, keeps crashes that involved a pedestrian or cyclist
(PER_TYP 5/8 = pedestrian & personal conveyance, 6/7 = bicyclist & other
pedalcyclist), joins them back to the accident geometry/time, and emits the same
cleaned-events / candidate-cells / weekly-counts artifacts as build_dataset.py so
train_vru_model.py can reuse the training machinery.

    python scripts/build_vru_dataset.py --years 2018 2019 2020 2021 2022 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys
import zipfile

import h3
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from build_dataset import build_candidate_cells, normalize_day_of_week
from common import (
    DEFAULT_YEARS,
    H3_RESOLUTION,
    LAT_MAX,
    LAT_MIN,
    LON_MAX,
    LON_MIN,
    VRU_ACCIDENTS_CLEAN_PATH,
    VRU_CANDIDATE_CELLS_PATH,
    VRU_WEEKLY_COUNTS_PATH,
    ensure_dirs,
    fars_zip_path,
)

PEDESTRIAN_TYPES = (5, 8)  # Pedestrian, Person on a Personal Conveyance
CYCLIST_TYPES = (6, 7)     # Bicyclist, Other Pedalcyclist
VRU_PER_TYPES = PEDESTRIAN_TYPES + CYCLIST_TYPES

ACCIDENT_USECOLS = [
    "ST_CASE",
    "STATE",
    "STATENAME",
    "YEAR",
    "MONTH",
    "DAY",
    "DAY_WEEK",
    "HOUR",
    "LATITUDE",
    "LONGITUD",
    "RUR_URB",
    "FUNC_SYS",
    "LGT_COND",
]
PERSON_USECOLS = ["ST_CASE", "PER_TYP"]


def _read_member(zip_path: Path, suffix: str, usecols: list[str]) -> pd.DataFrame:
    with zipfile.ZipFile(zip_path) as archive:
        member = next(name for name in archive.namelist() if name.lower().endswith(suffix))
        for encoding in ("utf-8", "cp1252", "latin-1"):
            try:
                with archive.open(member) as handle:
                    return pd.read_csv(
                        handle, usecols=usecols, low_memory=False, encoding=encoding
                    )
            except UnicodeDecodeError:
                continue
    raise UnicodeDecodeError("utf-8", b"", 0, 1, f"unable to decode {suffix} in {zip_path.name}")


def vru_case_stats(person: pd.DataFrame) -> pd.DataFrame:
    """Per-crash pedestrian/cyclist/VRU counts, indexed by ST_CASE.

    Only crashes that involved at least one VRU appear in the result.
    """
    per_typ = pd.to_numeric(person["PER_TYP"], errors="coerce")
    vru = person.loc[per_typ.isin(VRU_PER_TYPES)].copy()
    vru["PER_TYP"] = per_typ.loc[vru.index]
    if vru.empty:
        return pd.DataFrame(columns=["ped_count", "cyc_count", "vru_count"]).rename_axis("ST_CASE")

    is_ped = vru["PER_TYP"].isin(PEDESTRIAN_TYPES)
    stats = pd.DataFrame(
        {
            "ped_count": vru.loc[is_ped].groupby("ST_CASE").size(),
            "cyc_count": vru.loc[~is_ped].groupby("ST_CASE").size(),
        }
    )
    stats = stats.reindex(vru["ST_CASE"].unique()).fillna(0).astype(int)
    stats["vru_count"] = stats["ped_count"] + stats["cyc_count"]
    stats.index.name = "ST_CASE"
    return stats.sort_index()


def clean_accident_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Numeric-coerce, bound-check, and add day-of-week/hour-of-week/cell_id."""
    frame = raw.copy()
    for column in ("LATITUDE", "LONGITUD", "MONTH", "DAY", "DAY_WEEK", "HOUR",
                   "RUR_URB", "FUNC_SYS", "LGT_COND"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")

    frame = frame.dropna(subset=["LATITUDE", "LONGITUD", "MONTH", "DAY", "DAY_WEEK", "HOUR"])
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
    frame["lat"] = frame["LATITUDE"].astype(float)
    frame["lon"] = frame["LONGITUD"].astype(float)
    frame["hour_of_week"] = (frame["day_of_week"] - 1) * 24 + frame["hour"]
    frame["cell_id"] = [
        h3.latlng_to_cell(lat, lon, H3_RESOLUTION)
        for lat, lon in zip(frame["lat"], frame["lon"], strict=False)
    ]
    return frame


def build_vru_events(accidents: pd.DataFrame, stats: pd.DataFrame) -> pd.DataFrame:
    """Keep only VRU-involved crashes and attach ped/cyc counts."""
    merged = accidents.merge(stats, left_on="ST_CASE", right_index=True, how="inner")
    return merged[
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
            "ped_count",
            "cyc_count",
            "vru_count",
            "RUR_URB",
            "FUNC_SYS",
            "LGT_COND",
            "cell_id",
        ]
    ].rename(
        columns={
            "YEAR": "year",
            "STATE": "state_code",
            "STATENAME": "state_name",
            "RUR_URB": "rur_urb",
            "FUNC_SYS": "func_sys",
            "LGT_COND": "light_code",
        }
    )


def load_year_vru(year: int) -> pd.DataFrame:
    zip_path = fars_zip_path(year)
    if not zip_path.exists():
        raise FileNotFoundError(f"missing {zip_path}; run download_data.py for year {year} first")
    stats = vru_case_stats(_read_member(zip_path, "person.csv", PERSON_USECOLS))
    accidents = clean_accident_frame(_read_member(zip_path, "accident.csv", ACCIDENT_USECOLS))
    return build_vru_events(accidents, stats)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    yearly_frames = []
    for year in args.years:
        frame = load_year_vru(year)
        print(f"load {year}: vru_crashes={len(frame)}")
        yearly_frames.append(frame)

    events = pd.concat(yearly_frames, ignore_index=True)
    events = events.sort_values(["year", "month", "day", "hour"]).reset_index(drop=True)
    candidate_cells = build_candidate_cells(events)
    weekly_counts = (
        events.groupby(["cell_id", "hour_of_week"]).size().rename("event_count").reset_index()
    )

    events.to_csv(VRU_ACCIDENTS_CLEAN_PATH, index=False, compression="gzip")
    candidate_cells.to_csv(VRU_CANDIDATE_CELLS_PATH, index=False, compression="gzip")
    weekly_counts.to_csv(VRU_WEEKLY_COUNTS_PATH, index=False, compression="gzip")

    print(f"wrote {VRU_ACCIDENTS_CLEAN_PATH}")
    print(f"wrote {VRU_CANDIDATE_CELLS_PATH}")
    print(f"wrote {VRU_WEEKLY_COUNTS_PATH}")
    print(
        f"vru_events={len(events)} pedestrians={int(events['ped_count'].sum())} "
        f"cyclists={int(events['cyc_count'].sum())} candidate_cells={len(candidate_cells)}"
    )


if __name__ == "__main__":
    main()
