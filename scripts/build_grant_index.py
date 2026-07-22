"""Precompute a one-row-per-jurisdiction index of the per-county grant datasets.

build_grant_dataset.py writes a full ``<GEOID>.json`` per county. Serving an
all-county rollup (a national jurisdictions list, a leaderboard, /v1/meta's
jurisdiction count) by reading thousands of those files per request is slow;
this job flattens each report's headline fields into a single Parquet table.

    python scripts/build_grant_index.py

Output: data/reports/grant/index.parquet (one row per jurisdiction)
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

import grant_store

_STRING_COLUMNS = ["geoid", "name", "level", "generated_at_utc"]
_INT_COLUMNS = [
    "fars_year_start",
    "fars_year_end",
    "total_fatal_crashes",
    "total_fatalities",
    "hin_segments",
    "hin_corridor_count",
    "systemic_location_count",
]
_FLOAT_COLUMNS = [
    "hin_length_km",
    "length_share",
    "weighted_crash_share",
    "benefit_cost_ratio",
    "net_benefit",
]
INDEX_COLUMNS = [
    "geoid",
    "name",
    "level",
    "fars_year_start",
    "fars_year_end",
    "total_fatal_crashes",
    "total_fatalities",
    "hin_segments",
    "hin_length_km",
    "length_share",
    "weighted_crash_share",
    "hin_corridor_count",
    "systemic_location_count",
    "benefit_cost_ratio",
    "net_benefit",
    "generated_at_utc",
]


def _dict(value) -> dict:
    return value if isinstance(value, dict) else {}


def _list(value) -> list:
    return value if isinstance(value, list) else []


def _scalar(value):
    """A leaf where a scalar is expected — nested object/array degrades to None
    (mirrors the container guards) so one bad leaf can't poison the Parquet writer."""
    return None if isinstance(value, (dict, list)) else value


def index_row(geoid: str, report) -> dict:
    """Flatten one grant report into a single index row (tolerant of malformed reports)."""
    report = _dict(report)
    jurisdiction = _dict(report.get("jurisdiction"))
    vintage = _dict(report.get("data_vintage"))
    crash = _dict(report.get("crash_summary"))
    hin = _dict(report.get("high_injury_network"))
    benefit_cost = _dict(report.get("benefit_cost"))
    years = _list(vintage.get("fars_years"))
    return {
        "geoid": str(jurisdiction.get("geoid", geoid)),
        "name": _scalar(jurisdiction.get("name")),
        "level": _scalar(jurisdiction.get("level")),
        "fars_year_start": _scalar(years[0]) if years else None,
        "fars_year_end": _scalar(years[-1]) if years else None,
        "total_fatal_crashes": _scalar(crash.get("total_fatal_crashes")),
        "total_fatalities": _scalar(crash.get("total_fatalities")),
        "hin_segments": _scalar(hin.get("hin_segments")),
        "hin_length_km": _scalar(hin.get("hin_length_km")),
        "length_share": _scalar(hin.get("length_share")),
        "weighted_crash_share": _scalar(hin.get("weighted_crash_share")),
        "hin_corridor_count": len(_list(report.get("hin_corridors"))),
        "systemic_location_count": len(_list(report.get("systemic_locations"))),
        "benefit_cost_ratio": _scalar(benefit_cost.get("benefit_cost_ratio")),
        "net_benefit": _scalar(benefit_cost.get("net_benefit")),
        "generated_at_utc": _scalar(report.get("generated_at_utc")),
    }


def build_index(reports) -> pd.DataFrame:
    """Index DataFrame from an iterable of ``(geoid, report)`` pairs, sorted by GEOID.

    Columns are coerced to a fixed, nullable schema (string / Int64 / float64) so
    the Parquet types are identical whether the index is empty, fully populated,
    or carries an all-null optional column — and so a malformed scalar leaf
    (e.g. a stray string in a numeric field) degrades to null instead of aborting
    the whole build.
    """
    rows = [index_row(geoid, report) for geoid, report in reports]
    frame = pd.DataFrame(rows, columns=INDEX_COLUMNS)
    for column in _STRING_COLUMNS:
        frame[column] = frame[column].astype("string")
    for column in _INT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").round().astype("Int64")
    for column in _FLOAT_COLUMNS:
        frame[column] = pd.to_numeric(frame[column], errors="coerce").astype("float64")
    if len(frame):
        frame = frame.sort_values("geoid").reset_index(drop=True)
    return frame


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--grant-dir", type=Path, default=None,
        help="directory of <GEOID>.json reports (default: the grant store directory)",
    )
    parser.add_argument(
        "--output", type=Path, default=None,
        help="index parquet path (default: <grant-dir>/index.parquet)",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = grant_store.GrantStore(args.grant_dir) if args.grant_dir else grant_store.get_default_store()
    output = args.output or (store.directory / "index.parquet")

    frame = build_index(store.iter_reports())
    output.parent.mkdir(parents=True, exist_ok=True)
    frame.to_parquet(output, index=False)

    print(f"indexed {len(frame)} jurisdictions -> {output}")
    if len(frame):
        preview = frame[["geoid", "name", "total_fatal_crashes", "benefit_cost_ratio"]].head()
        print(preview.to_string(index=False))


if __name__ == "__main__":
    main()
