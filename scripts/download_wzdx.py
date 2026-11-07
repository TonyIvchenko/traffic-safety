"""Fetch a WZDx work-zone feed and save it for the runtime to serve.

WZDx feeds are published per agency as GeoJSON RoadEvent feature collections;
there is no single national feed. Point this at one feed URL (see
https://www.transportation.gov/wzdx), then set TRAFFIC_SAFETY_WZDX_PATH to the
saved file so /v1/work-zones reads it.

    python scripts/download_wzdx.py --url https://<agency>/wzdx/feed.json
    python scripts/download_wzdx.py --url <url> --output data/processed/work_zones.json

Only validates that the response is a GeoJSON-ish FeatureCollection; the runtime
parser (src/work_zones.py) is defensive about the rest.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import requests

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_OUTPUT = REPO_DIR / "data" / "processed" / "work_zones.json"
TIMEOUT_SECONDS = 30
USER_AGENT = "road-risk-monitor-wzdx/1.0"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--url", required=True, help="WZDx feed URL (GeoJSON FeatureCollection)")
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT, help="where to save the feed")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    response = requests.get(args.url, headers={"User-Agent": USER_AGENT}, timeout=TIMEOUT_SECONDS)
    response.raise_for_status()
    payload = response.json()

    features = payload.get("features") if isinstance(payload, dict) else None
    if not isinstance(features, list):
        raise SystemExit("response is not a GeoJSON FeatureCollection with a 'features' list")

    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(payload), encoding="utf-8")
    print(f"saved {len(features)} work-zone features to {args.output}")
    print(f"set TRAFFIC_SAFETY_WZDX_PATH={args.output} to serve it")


if __name__ == "__main__":
    main()
