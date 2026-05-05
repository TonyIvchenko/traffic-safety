from __future__ import annotations

import argparse
from pathlib import Path

import requests

from common import DEFAULT_YEARS, ensure_dirs, fars_zip_path, fars_zip_url


def download_file(url: str, destination: Path, force: bool = False) -> None:
    if destination.exists() and not force:
        print(f"skip {destination.name}")
        return

    print(f"download {url}")
    with requests.get(url, stream=True, timeout=120) as response:
        response.raise_for_status()
        with destination.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=1 << 20):
                if chunk:
                    handle.write(chunk)


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
