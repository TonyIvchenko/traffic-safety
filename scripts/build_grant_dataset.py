"""Build per-county federal safety-grant datasets (SS4A / HSIP analysis).

For every county with fatal-crash history this aggregates the High Injury
Network, the systemic risk ranking, a crash summary and the data vintage into a
single structured JSON — the data-driven safety analysis that SS4A and HSIP
applications require. The per-county JSON is what the HTML renderer (F1.7) and
the /v1/grants endpoints consume.

Systemic rates are computed once over the whole national network (roadway-class
rates need exposure to be reliable); counties then inherit their segments'
scores. Crashes and segments are assigned to a county by point-in-polygon on the
TIGER county boundaries, unless the High Injury Network already carries a
``county_geoid`` from build_geo_lookup.py.

Inputs:
- data/processed/safety/high_injury_network.parquet   (build_hin.py)
- data/processed/accidents_clean.csv.gz               (FARS fatal crashes)
- TIGER county boundaries                             (download_geographies.py)

    python scripts/build_grant_dataset.py --counties 06037 --top-n 15

Output: data/reports/grant/<GEOID>.json (one per county)
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
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

import systemic
from geo_lookup import load_county_index
from grant_report import assemble_grant_report

from common import (
    ACCIDENTS_CLEAN_PATH,
    HIGH_INJURY_NETWORK_PATH,
    tiger_county_path,
)

REPORTS_DIR = REPO_DIR / "data" / "reports"
GRANT_DIR = REPORTS_DIR / "grant"

COUNTY_NAME_FIELD = "NAMELSAD"
GEOID_COL = "county_geoid"
# accidents_clean.csv.gz carries no ped/cyc columns (those live in the VRU
# file); they are read opportunistically so mode breakdowns light up for free
# if a richer crash file is ever supplied.
CRASH_COLUMNS = ("year", "lat", "lon", "fatals", "ped_count", "cyc_count")


def assign_geoids(frame: pd.DataFrame, index, *, lat_col="center_lat", lon_col="center_lon") -> pd.Series:
    """GEOID per row from a point-in-polygon index (``None`` where uncontained)."""
    lats = frame[lat_col].to_numpy(dtype=float)
    lons = frame[lon_col].to_numpy(dtype=float)
    geoids = [index.lookup(float(lat), float(lon)) for lat, lon in zip(lats, lons)]
    return pd.Series(geoids, index=frame.index, dtype=object)


def record_geoid_names(
    records, field_names, *, geoid_field="GEOID", name_field=COUNTY_NAME_FIELD
) -> dict[str, str]:
    """Map GEOID -> human name from parallel shapefile record/field lists."""
    if geoid_field not in field_names:
        raise ValueError(f"missing '{geoid_field}' field (has {field_names})")
    geoid_col = field_names.index(geoid_field)
    name_col = field_names.index(name_field) if name_field in field_names else geoid_col
    return {str(record[geoid_col]): str(record[name_col]) for record in records}


def county_names(path=None, *, geoid_field="GEOID", name_field=COUNTY_NAME_FIELD) -> dict[str, str]:
    """GEOID -> county name read from the TIGER county shapefile."""
    reader = shapefile.Reader(str(path or tiger_county_path()))
    field_names = [field[0] for field in reader.fields[1:]]
    return record_geoid_names(
        list(reader.iterRecords()), field_names, geoid_field=geoid_field, name_field=name_field
    )


def crash_years(crashes: pd.DataFrame) -> list[int]:
    if "year" not in crashes.columns or len(crashes) == 0:
        return []
    return sorted(int(year) for year in crashes["year"].dropna().unique())


def data_vintage(crashes: pd.DataFrame) -> dict:
    years = crash_years(crashes)
    return {
        "fars_years": [years[0], years[-1]] if years else [],
        "crash_source": "FARS fatal crashes",
        "network_source": "TIGER/Line primary & secondary roads",
    }


def build_county_reports(
    segments: pd.DataFrame,
    crashes: pd.DataFrame,
    *,
    names: dict[str, str] | None = None,
    geoid_col: str = GEOID_COL,
    top_n: int = 10,
    min_fatal_crashes: int = 1,
    generated_at: str | None = None,
    data_vintage: dict | None = None,
) -> dict[str, dict]:
    """One assembled grant report per county present in the tagged frames.

    ``segments`` and ``crashes`` must both carry ``geoid_col``; rows with no
    county (point outside every boundary) are dropped by the groupby. A county
    is emitted only when its fatal-crash count reaches ``min_fatal_crashes``.
    """
    names = names or {}
    seg_groups = (
        {str(key): frame for key, frame in segments.groupby(geoid_col)}
        if geoid_col in segments.columns and len(segments)
        else {}
    )
    crash_groups = (
        {str(key): frame for key, frame in crashes.groupby(geoid_col)}
        if geoid_col in crashes.columns and len(crashes)
        else {}
    )
    empty_segments = segments.iloc[0:0]
    empty_crashes = crashes.iloc[0:0]

    reports: dict[str, dict] = {}
    for geoid in sorted(set(seg_groups) | set(crash_groups)):
        county_crashes = crash_groups.get(geoid, empty_crashes)
        if len(county_crashes) < int(min_fatal_crashes):
            continue
        reports[geoid] = assemble_grant_report(
            geoid=geoid,
            name=names.get(geoid, geoid),
            crashes=county_crashes,
            segments=seg_groups.get(geoid, empty_segments),
            top_n=top_n,
            generated_at=generated_at,
            data_vintage=data_vintage,
        )
    return reports


def write_county_reports(reports: dict[str, dict], out_dir) -> list[Path]:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for geoid, report in reports.items():
        path = out_dir / f"{geoid}.json"
        path.write_text(json.dumps(report, indent=2), encoding="utf-8")
        written.append(path)
    return written


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=10, help="corridors/locations per report")
    parser.add_argument("--min-fatal-crashes", type=int, default=1)
    parser.add_argument("--counties", nargs="*", default=None, help="limit to these county GEOIDs")
    parser.add_argument("--output-dir", type=Path, default=GRANT_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not tiger_county_path().exists():
        print(f"missing {tiger_county_path()}; run scripts/download_geographies.py first")
        return

    network = pd.read_parquet(HIGH_INJURY_NETWORK_PATH)
    rates = systemic.systemic_rates(network)
    network = systemic.apply_systemic_scores(network, rates)

    county_index = load_county_index()
    if GEOID_COL not in network.columns:
        print(f"tagging {len(network)} segments to counties via TIGER boundaries ...")
        network[GEOID_COL] = assign_geoids(network, county_index)
    network[GEOID_COL] = network[GEOID_COL].astype("string")

    crashes = pd.read_csv(
        ACCIDENTS_CLEAN_PATH, usecols=lambda column: column in CRASH_COLUMNS
    )
    print(f"tagging {len(crashes)} fatal crashes to counties ...")
    crashes[GEOID_COL] = assign_geoids(crashes, county_index, lat_col="lat", lon_col="lon").astype(
        "string"
    )

    # Match coverage mirrors build_hin.py / build_geo_lookup.py: uncontained
    # points (None GEOID) are dropped downstream, so a low ratio warns the
    # operator of a boundary-vintage or coordinate problem before it silently
    # yields near-empty datasets.
    print(f"segments matched to a county: {int(network[GEOID_COL].notna().sum())}/{len(network)}")
    print(f"crashes matched to a county: {int(crashes[GEOID_COL].notna().sum())}/{len(crashes)}")

    if args.counties:
        wanted = {str(geoid) for geoid in args.counties}
        network = network[network[GEOID_COL].isin(wanted)]
        crashes = crashes[crashes[GEOID_COL].isin(wanted)]

    reports = build_county_reports(
        network,
        crashes,
        names=county_names(),
        top_n=args.top_n,
        min_fatal_crashes=args.min_fatal_crashes,
        generated_at=datetime.now(timezone.utc).isoformat(),
        data_vintage=data_vintage(crashes),
    )
    written = write_county_reports(reports, args.output_dir)
    print(f"wrote {len(written)} county grant datasets to {args.output_dir}")
    if written:
        first = json.loads(written[0].read_text(encoding="utf-8"))
        summary = first["crash_summary"]
        print(
            f"e.g. {first['jurisdiction']['name']} ({first['jurisdiction']['geoid']}): "
            f"{summary['total_fatal_crashes']} fatal crashes, "
            f"{first['high_injury_network'].get('hin_segments', 0)} HIN segments"
        )


if __name__ == "__main__":
    main()
