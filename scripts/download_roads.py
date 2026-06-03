from __future__ import annotations

import argparse
import os

import requests

from common import STATE_FIPS, ensure_dirs, tiger_prisecroads_path, tiger_prisecroads_url


def download_file(url: str, output_path, force: bool) -> None:
    if output_path.exists() and not force:
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    response = requests.get(url, timeout=120)
    response.raise_for_status()
    # Write to a temporary file and atomically replace the destination so an
    # interrupted write never leaves a truncated shapefile archive behind.
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
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--state-fips", nargs="*", default=STATE_FIPS)
    parser.add_argument("--year", type=int, default=2024)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    for state_fips in args.state_fips:
        url = tiger_prisecroads_url(state_fips, year=args.year)
        path = tiger_prisecroads_path(state_fips, year=args.year)
        download_file(url, path, force=args.force)


if __name__ == "__main__":
    main()
