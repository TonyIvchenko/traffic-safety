"""Join road-segment and H3-cell centroids to their county / census-tract GEOID.

Uses the TIGER boundaries (download_geographies.py) via the shapely point-in-
polygon index in src/geo_lookup.py, and writes lookup tables reused by the grant,
equity, and countermeasure features.

    python scripts/build_geo_lookup.py --year 2024

Outputs:
  data/processed/geo/segment_geoid.csv.gz   (segment_id, county_geoid, tract_geoid)
  data/processed/geo/cell_geoid.csv.gz      (cell_id,    county_geoid, tract_geoid)
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import geo_lookup
from common import (
    ACTIVE_ROAD_SEGMENTS_PATH,
    CANDIDATE_CELLS_PATH,
    CELL_GEOID_PATH,
    SEGMENT_GEOID_PATH,
    ensure_dirs,
)


def enrich_with_geoids(frame, *, lat_col, lon_col, county_index, tract_index):
    """Return a copy of ``frame`` with ``county_geoid``/``tract_geoid`` columns.

    Unmatched points (outside all polygons) get ``None``.
    """
    county = []
    tract = []
    for lat, lon in zip(frame[lat_col], frame[lon_col], strict=False):
        county.append(county_index.lookup(float(lat), float(lon)))
        tract.append(tract_index.lookup(float(lat), float(lon)))
    result = frame.copy()
    result["county_geoid"] = county
    result["tract_geoid"] = tract
    return result


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    county_index = geo_lookup.load_county_index(args.year)
    tract_index = geo_lookup.load_tract_index(args.year)
    print(f"county polygons={len(county_index)} tract polygons={len(tract_index)}")

    segments = pd.read_parquet(ACTIVE_ROAD_SEGMENTS_PATH, columns=["segment_id", "center_lat", "center_lon"])
    segment_geoids = enrich_with_geoids(
        segments, lat_col="center_lat", lon_col="center_lon",
        county_index=county_index, tract_index=tract_index,
    )[["segment_id", "county_geoid", "tract_geoid"]]
    segment_geoids.to_csv(SEGMENT_GEOID_PATH, index=False, compression="gzip")

    cells = pd.read_csv(CANDIDATE_CELLS_PATH, usecols=["cell_id", "center_lat", "center_lon"])
    cell_geoids = enrich_with_geoids(
        cells, lat_col="center_lat", lon_col="center_lon",
        county_index=county_index, tract_index=tract_index,
    )[["cell_id", "county_geoid", "tract_geoid"]]
    cell_geoids.to_csv(CELL_GEOID_PATH, index=False, compression="gzip")

    seg_matched = int(segment_geoids["county_geoid"].notna().sum())
    cell_matched = int(cell_geoids["county_geoid"].notna().sum())
    print(f"segments matched to county: {seg_matched}/{len(segment_geoids)}")
    print(f"cells matched to county: {cell_matched}/{len(cell_geoids)}")
    print(f"wrote {SEGMENT_GEOID_PATH}")
    print(f"wrote {CELL_GEOID_PATH}")


if __name__ == "__main__":
    main()
