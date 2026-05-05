from __future__ import annotations

import argparse
from pathlib import Path
import sys

import pandas as pd
import shapefile

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
    SEGMENT_MAX_LENGTH_KM,
    STATE_FIPS,
    ensure_dirs,
    tiger_prisecroads_path,
)
from segment_support import (
    coords_to_json,
    polyline_bounds,
    polyline_centroid,
    polyline_length_km,
    segmentize_polyline,
)


GRID_SIZE_DEG = 0.25


def grid_bucket(value: float) -> int:
    return int(value // GRID_SIZE_DEG)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--state-fips", nargs="*", default=STATE_FIPS)
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--max-length-km", type=float, default=SEGMENT_MAX_LENGTH_KM)
    return parser.parse_args()


def iter_shape_parts(shape) -> list[list[tuple[float, float]]]:
    points = shape.points
    parts = list(shape.parts) + [len(points)]
    polylines = []
    for idx in range(len(parts) - 1):
        part_points = points[parts[idx] : parts[idx + 1]]
        if len(part_points) >= 2:
            polylines.append([(float(lon), float(lat)) for lon, lat in part_points])
    return polylines


def segment_rows_for_zip(
    zip_path: Path,
    state_fips: str,
    max_length_km: float,
) -> list[dict[str, object]]:
    reader = shapefile.Reader(str(zip_path))
    rows: list[dict[str, object]] = []
    for shape_record in reader.iterShapeRecords():
        record = shape_record.record.as_dict()
        linearid = str(record.get("LINEARID", "")).strip()
        fullname = str(record.get("FULLNAME", "")).strip()
        rttyp = str(record.get("RTTYP", "")).strip()
        mtfcc = str(record.get("MTFCC", "")).strip()

        for part_idx, polyline in enumerate(iter_shape_parts(shape_record.shape)):
            if not polyline:
                continue
            segment_polylines = segmentize_polyline(polyline, max_length_km=max_length_km)
            for piece_idx, segment_points in enumerate(segment_polylines):
                min_lat, max_lat, min_lon, max_lon = polyline_bounds(segment_points)
                if max_lat < LAT_MIN or min_lat > LAT_MAX or max_lon < LON_MIN or min_lon > LON_MAX:
                    continue
                center_lat, center_lon = polyline_centroid(segment_points)
                rows.append(
                    {
                        "segment_id": f"{state_fips}:{linearid}:{part_idx}:{piece_idx}",
                        "state_fips": state_fips,
                        "linearid": linearid,
                        "fullname": fullname,
                        "rttyp": rttyp,
                        "mtfcc": mtfcc,
                        "length_km": polyline_length_km(segment_points),
                        "center_lat": center_lat,
                        "center_lon": center_lon,
                        "min_lat": min_lat,
                        "max_lat": max_lat,
                        "min_lon": min_lon,
                        "max_lon": max_lon,
                        "grid_lat": grid_bucket(center_lat),
                        "grid_lon": grid_bucket(center_lon),
                        "coords_json": coords_to_json(segment_points),
                    }
                )
    return rows


def main() -> None:
    args = parse_args()
    ensure_dirs()
    all_rows: list[dict[str, object]] = []
    for state_fips in args.state_fips:
        zip_path = tiger_prisecroads_path(state_fips, year=args.year)
        if not zip_path.exists():
            raise FileNotFoundError(
                f"missing {zip_path}; run download_roads.py first"
            )
        print(f"load {zip_path.name}")
        rows = segment_rows_for_zip(
            zip_path=zip_path,
            state_fips=state_fips,
            max_length_km=args.max_length_km,
        )
        all_rows.extend(rows)
        print(f"state={state_fips} segments={len(rows)}")

    frame = pd.DataFrame(all_rows)
    frame = frame.sort_values(["state_fips", "linearid", "segment_id"]).reset_index(drop=True)
    frame.to_parquet(ROAD_SEGMENTS_PATH, index=False)
    print(f"wrote {ROAD_SEGMENTS_PATH}")
    print(f"segments={len(frame)}")


if __name__ == "__main__":
    main()
