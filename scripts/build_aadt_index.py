"""Build the runtime AADT exposure index from an FHWA HPMS-style extract.

HPMS publishes road segments with a functional system and AADT. This converts a
flat CSV export of those fields into the compact index the runtime reads
(``data/processed/aadt_segments.parquet``), which ``src/exposure.py`` prefers
over the curated reference sample.

    python scripts/build_aadt_index.py --input hpms_export.csv
    python scripts/build_aadt_index.py --input hpms_export.csv --json out.json

Accepted columns (case-insensitive; first match wins):
    lat/latitude, lon/longitude, aadt/aadt_vn, f_system (HPMS 1-7) or
    functional_class, length_km or length_miles, name/route_id, state, year.

HPMS source: https://www.fhwa.dot.gov/policyinformation/hpms/
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

import exposure  # noqa: E402  (after sys.path bootstrap)

DEFAULT_OUTPUT = REPO_DIR / "data" / "processed" / "aadt_segments.parquet"
MILES_PER_KM = exposure.MILES_PER_KM


def _first_column(frame: pd.DataFrame, *names: str):
    lower = {str(col).strip().lower(): col for col in frame.columns}
    for name in names:
        if name in lower:
            return lower[name]
    return None


def _resolve_class(row, class_col, fsystem_col) -> str | None:
    if class_col is not None:
        resolved = exposure.normalize_class(row.get(class_col))
        if resolved:
            return resolved
    if fsystem_col is not None:
        try:
            code = int(float(row.get(fsystem_col)))
        except (TypeError, ValueError):
            code = None
        if code is not None:
            return exposure.HPMS_F_SYSTEM_TO_CLASS.get(code)
    return None


def build_records(frame: pd.DataFrame) -> list[dict]:
    lat_col = _first_column(frame, "lat", "latitude", "begin_lat")
    lon_col = _first_column(frame, "lon", "longitude", "begin_lon")
    aadt_col = _first_column(frame, "aadt", "aadt_vn", "aadt_value")
    if lat_col is None or lon_col is None or aadt_col is None:
        raise SystemExit("input must have lat, lon, and aadt columns")

    class_col = _first_column(frame, "functional_class", "f_class")
    fsystem_col = _first_column(frame, "f_system", "fsystem", "functional_system")
    length_km_col = _first_column(frame, "length_km")
    length_mi_col = _first_column(frame, "length_miles", "length_mi", "section_length")
    name_col = _first_column(frame, "name", "route_id", "route_number")
    state_col = _first_column(frame, "state", "state_code")
    year_col = _first_column(frame, "year", "data_year")

    records: list[dict] = []
    for _, row in frame.iterrows():
        lat = exposure._finite_float(row.get(lat_col))
        lon = exposure._finite_float(row.get(lon_col))
        aadt = exposure._finite_float(row.get(aadt_col))
        if lat is None or lon is None or aadt is None or aadt < 0:
            continue
        length_km = None
        if length_km_col is not None:
            length_km = exposure._finite_float(row.get(length_km_col))
        if length_km is None and length_mi_col is not None:
            miles = exposure._finite_float(row.get(length_mi_col))
            length_km = None if miles is None else miles / MILES_PER_KM
        records.append(
            {
                "id": (str(row.get(name_col)) if name_col else None) or None,
                "name": (str(row.get(name_col)) if name_col else None) or None,
                "lat": lat,
                "lon": lon,
                "aadt": aadt,
                "functional_class": _resolve_class(row, class_col, fsystem_col),
                "length_km": length_km,
                "state": (str(row.get(state_col)) if state_col else None) or None,
                "year": (row.get(year_col) if year_col else None),
            }
        )
    return records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--input", type=Path, required=True, help="HPMS-style CSV export")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="parquet index path")
    parser.add_argument("--json", type=Path, default=None, help="also write a JSON index here")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.input)
    records = build_records(frame)
    if not records:
        raise SystemExit("no valid AADT rows found in the input")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_parquet(args.output, index=False)
    print(f"wrote {len(records)} AADT segments to {args.output}")

    if args.json is not None:
        payload = {"schema_version": "1.0", "source": "FHWA HPMS", "segments": records}
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(payload, indent=2), encoding="utf-8")
        print(f"wrote JSON index to {args.json}")


if __name__ == "__main__":
    main()
