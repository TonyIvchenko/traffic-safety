"""Join each road segment to its tract equity, current risk, and crash history.

Per-segment overlay powering the equity endpoints and hotspot ranking: for every
segment attach the containing tract's SVI percentile + Justice40 disadvantaged
flag (via the equity index), the current model forecast risk, and the recorded
fatal-crash count.

Tract vintage: segments are tagged from TIGER 2024 (2020 census tracts) while the
equity index is keyed on 2010 tracts (CEJST 2.0). Tracts unchanged between 2010
and 2020 join; re-tracted segments fall through to ``in_equity_index=False`` (null
equity) rather than a wrong value. A 2010->2020 crosswalk would close that gap.

    python scripts/build_equity_overlay.py

Output: data/processed/equity/segment_equity.parquet
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from equity import load_equity_index
from road_tiles import ROAD_TILE_FORECAST_PATH

from common import (
    ACTIVE_ROAD_SEGMENTS_PATH,
    EQUITY_OVERLAY_PATH,
    HIGH_INJURY_NETWORK_PATH,
    SEGMENT_GEOID_PATH,
    ensure_dirs,
)

OVERLAY_COLUMNS = [
    "segment_id",
    "tract_geoid",
    "svi_percentile",
    "svi_category",
    "disadvantaged",
    "in_equity_index",
    "risk",
    "crashes",
    "center_lat",
    "center_lon",
    "fullname",
]
_STRING_COLUMNS = ["segment_id", "tract_geoid", "svi_category", "fullname"]
_FLOAT_COLUMNS = ["svi_percentile", "risk", "crashes", "center_lat", "center_lon"]
_BOOL_COLUMNS = ["disadvantaged", "in_equity_index"]


def segment_forecast_risk(segment_idx, forecast_scores) -> np.ndarray:
    """Max forecast risk per segment (0-1) from the uint8 road-tile forecast."""
    idx = np.asarray(segment_idx, dtype=np.int64)
    if forecast_scores is None or len(idx) == 0:
        return np.zeros(len(idx), dtype=np.float64)
    rows = np.asarray(forecast_scores)[idx]
    return rows.max(axis=1).astype(np.float64) / 255.0


def _coerce_schema(frame: pd.DataFrame) -> pd.DataFrame:
    """Pin the overlay's Parquet dtypes so empty/all-null columns stay stable."""
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    for column in _BOOL_COLUMNS:
        frame[column] = frame[column].astype(bool)
    return frame


def build_equity_overlay(segments: pd.DataFrame, equity_index, *, tract_col="tract_geoid") -> pd.DataFrame:
    """Attach tract equity to each segment; unknown/re-tracted tracts read as unknown."""
    frame = segments.copy()
    if tract_col not in frame.columns:
        frame[tract_col] = ""
    tracts = [("" if pd.isna(value) else str(value).strip()) for value in frame[tract_col]]
    records = [equity_index.equity_for_tract(tract) for tract in tracts]

    frame[tract_col] = tracts
    frame["svi_percentile"] = [record["svi_percentile"] for record in records]
    frame["svi_category"] = [record["svi_category"] for record in records]
    frame["disadvantaged"] = [record["disadvantaged"] for record in records]
    frame["in_equity_index"] = [record["in_index"] for record in records]

    for column in OVERLAY_COLUMNS:
        if column not in frame.columns:
            frame[column] = pd.NA
    return _coerce_schema(frame[OVERLAY_COLUMNS]).reset_index(drop=True)


def _attach_tract(network: pd.DataFrame) -> pd.DataFrame:
    if "tract_geoid" in network.columns:
        return network
    if SEGMENT_GEOID_PATH.exists():
        geoids = pd.read_csv(SEGMENT_GEOID_PATH, dtype=str)
        return network.merge(geoids[["segment_id", "tract_geoid"]], on="segment_id", how="left")
    print(f"note: {SEGMENT_GEOID_PATH.name} missing; run build_geo_lookup.py for tract GEOIDs")
    network["tract_geoid"] = ""
    return network


def _attach_risk(network: pd.DataFrame) -> pd.Series:
    risk = pd.Series(np.nan, index=network.index, dtype="float64")
    if not (ROAD_TILE_FORECAST_PATH.exists() and ACTIVE_ROAD_SEGMENTS_PATH.exists()):
        print("note: no segment forecast; risk left null")
        return risk
    active = pd.read_parquet(ACTIVE_ROAD_SEGMENTS_PATH, columns=["segment_id", "segment_idx"])
    merged = network.merge(active, on="segment_id", how="left")
    scores = np.load(ROAD_TILE_FORECAST_PATH, mmap_mode="r")
    has_idx = merged["segment_idx"].notna().to_numpy()
    idx = merged.loc[has_idx, "segment_idx"].to_numpy(dtype=np.int64)
    risk.iloc[np.flatnonzero(has_idx)] = segment_forecast_risk(idx, scores)
    return risk


def main() -> None:
    ensure_dirs()
    network = pd.read_parquet(HIGH_INJURY_NETWORK_PATH)
    network = _attach_tract(network)
    network["crashes"] = network["fatal_crashes"] if "fatal_crashes" in network.columns else 0.0
    network["risk"] = _attach_risk(network).to_numpy()

    overlay = build_equity_overlay(network, load_equity_index())
    EQUITY_OVERLAY_PATH.parent.mkdir(parents=True, exist_ok=True)
    overlay.to_parquet(EQUITY_OVERLAY_PATH, index=False)

    matched = int(overlay["in_equity_index"].sum())
    disadvantaged = int(overlay["disadvantaged"].sum())
    print(
        f"segments={len(overlay)} matched_to_equity={matched} "
        f"disadvantaged={disadvantaged} -> {EQUITY_OVERLAY_PATH}"
    )


if __name__ == "__main__":
    main()
