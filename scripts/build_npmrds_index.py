"""Build the runtime NPMRDS speed index from an aggregated speed export.

NPMRDS (FHWA/RITIS) publishes travel-time readings; aggregate them (e.g. with the
RITIS Massive Data Downloader) into per-segment reference + peak/offpeak average
speeds, then run this to write the compact index the runtime reads
(``data/processed/npmrds_segments.parquet``), which ``src/traffic_speed.py``
prefers over the curated reference sample.

    python scripts/build_npmrds_index.py --input npmrds_agg.csv
    python scripts/build_npmrds_index.py --input npmrds_agg.csv --json out.json

Accepted columns (case-insensitive; first match wins):
    id/tmc, lat/latitude, lon/longitude, reference_speed[_mph],
    peak_speed[_mph], offpeak_speed[_mph], road_class, year, name.

NPMRDS: https://ops.fhwa.dot.gov/perf_measurement/npmrds/
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import traffic_speed  # noqa: E402  (after sys.path bootstrap)

DEFAULT_OUTPUT = REPO_DIR / "data" / "processed" / "npmrds_segments.parquet"


def _first_column(frame: pd.DataFrame, *names: str):
    lower = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def build_records(frame: pd.DataFrame) -> list[dict]:
    lat_col = _first_column(frame, "lat", "latitude")
    lon_col = _first_column(frame, "lon", "longitude")
    ref_col = _first_column(frame, "reference_speed_mph", "reference_speed", "ref_speed")
    if lat_col is None or lon_col is None or ref_col is None:
        raise SystemExit("input must have lat, lon, and reference_speed columns")

    peak_col = _first_column(frame, "peak_speed_mph", "peak_speed")
    offpeak_col = _first_column(frame, "offpeak_speed_mph", "offpeak_speed", "off_peak_speed")
    id_col = _first_column(frame, "id", "tmc", "tmc_code")
    name_col = _first_column(frame, "name", "road", "roadname")
    class_col = _first_column(frame, "road_class", "f_system", "functional_class")
    year_col = _first_column(frame, "year", "data_year")

    records: list[dict] = []
    for _, row in frame.iterrows():
        lat = traffic_speed._finite_float(row.get(lat_col))
        lon = traffic_speed._finite_float(row.get(lon_col))
        reference = traffic_speed._finite_float(row.get(ref_col))
        if lat is None or lon is None or reference is None or reference <= 0:
            continue
        records.append(
            {
                "id": (str(row.get(id_col)) if id_col else None) or None,
                "name": (str(row.get(name_col)) if name_col else None) or None,
                "lat": lat,
                "lon": lon,
                "reference_speed_mph": reference,
                "peak_speed_mph": traffic_speed._finite_float(row.get(peak_col)) if peak_col else None,
                "offpeak_speed_mph": (
                    traffic_speed._finite_float(row.get(offpeak_col)) if offpeak_col else None
                ),
                "road_class": (str(row.get(class_col)) if class_col else None) or None,
                "year": (row.get(year_col) if year_col else None),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="aggregated NPMRDS CSV")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="parquet index path")
    parser.add_argument("--json", type=Path, default=None, help="also write a JSON index here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    records = build_records(frame)
    if not records:
        raise SystemExit("no valid speed rows found in the input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(f"wrote {len(records)} NPMRDS speed segments to {args.output}")

    if args.json is not None:
        payload = {"schema_version": "1.0", "source": "NPMRDS", "segments": records}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote JSON index to {args.json}")


if __name__ == "__main__":
    main()
