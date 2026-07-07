"""Download Census TIGER/Line county and tract boundary shapefiles.

These boundaries power the geographic enrichment (segment/cell -> county/tract
GEOID) that the grant, equity, and countermeasure features rely on. Counties are
a single national file; tracts are published per state.

    python scripts/download_geographies.py --year 2024
"""

from __future__ import annotations

import argparse
import os

import requests

from common import (
    STATE_FIPS,
    ensure_dirs,
    tiger_county_path,
    tiger_county_url,
    tiger_tract_path,
    tiger_tract_url,
)


def download_file(url: str, output_path, force: bool = False) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    # Write to a temporary file and atomically replace so an interrupted write
    # never leaves a truncated shapefile archive behind.
    tmp_path = output_path.with_name(output_path.name + ".part")
    try:
        tmp_path.write_bytes(response.content)
        os.replace(tmp_path, output_path)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()
    print(f"wrote {output_path}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--year", type=int, default=2024)
    parser.add_argument("--state-fips", nargs="*", default=STATE_FIPS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    download_file(tiger_county_url(args.year), tiger_county_path(args.year), force=args.force)
    for state_fips in args.state_fips:
        download_file(
            tiger_tract_url(state_fips, year=args.year),
            tiger_tract_path(state_fips, year=args.year),
            force=args.force,
        )


if __name__ == "__main__":
    main()
