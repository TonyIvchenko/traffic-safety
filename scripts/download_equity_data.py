"""Download the equity source datasets for the Justice40 / SVI overlay.

Two tract-level inputs:

- CDC/ATSDR Social Vulnerability Index (SVI) — a percentile of social
  vulnerability per census tract.
- CEJST (Climate and Economic Justice Screening Tool) — the federal Justice40
  "disadvantaged community" designation per census tract.

Both are large national CSVs. The hosting for these federal datasets has moved
before, so the URLs are overridable on the command line; the defaults come from
scripts/common.py.

    python scripts/download_equity_data.py
    python scripts/download_equity_data.py --svi-url https://.../SVI.csv --force

Output: data/raw/equity/
"""

from __future__ import annotations

import argparse
import os

import requests

from common import (
    cdc_svi_path,
    cdc_svi_url,
    cejst_path,
    cejst_url,
    ensure_dirs,
)


def download_file(url: str, output_path, force: bool = False) -> None:
    if output_path.exists() and not force:
        print(f"skip {output_path} (exists)")
        return
    output_path.parent.mkdir(parents=True, exist_ok=True)
    print(f"download {url}")
    response = requests.get(url, timeout=300)
    response.raise_for_status()
    # Atomic replace so an interrupted download never leaves a truncated CSV.
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
    parser.add_argument("--svi-url", default=None, help="override the CDC SVI CSV URL")
    parser.add_argument("--cejst-url", default=None, help="override the CEJST communities CSV URL")
    parser.add_argument("--force", action="store_true", help="re-download even if present")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    download_file(args.svi_url or cdc_svi_url(), cdc_svi_path(), force=args.force)
    download_file(args.cejst_url or cejst_url(), cejst_path(), force=args.force)


if __name__ == "__main__":
    main()
