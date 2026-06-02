from __future__ import annotations

import argparse
import os
from pathlib import Path

import requests

from common import DEFAULT_YEARS, ensure_dirs, fars_zip_path, fars_zip_url


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"skip {destination.name}")
        return

    print(f"download {url}")
    destination.parent.mkdir(parents=True, exist_ok=True)
    # Stream into a temporary file and atomically move it into place so an
    # interrupted download never leaves a truncated file that later runs would
    # mistake for a complete cache.
    tmp_path = destination.with_name(destination.name + ".part")
    try:
        with requests.get(url, stream=True, timeout=120) as response:
            response.raise_for_status()
            with tmp_path.open("wb") as handle:
                for chunk in response.iter_content(chunk_size=1 << 20):
                    if chunk:
                        handle.write(chunk)
        os.replace(tmp_path, destination)
    finally:
        if tmp_path.exists():
            tmp_path.unlink()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--years", nargs="+", type=int, default=DEFAULT_YEARS)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    for year in args.years:
        download_file(fars_zip_url(year), fars_zip_path(year), force=args.force)


if __name__ == "__main__":
    main()
