"""Precompute the Road Risk Advisory reference profiles and a national snapshot.

For every metro region, score its 168-hour weekly climatology from the raw model
(one batched call per representative grid point, reduced per frame) and store the
per-region profiles. Then build a national snapshot at a chosen hour-of-week: each
region's relative advisory (its percentile within its own weekly climatology),
ranked by how elevated it is versus normal.

The profiles are the discriminating reference distribution the /v1/advisory
endpoints use; precomputing them offline lets the API warm its cache instead of
paying the first-request cost. The heavy step (the raw model) needs the model
bundle, so ``main`` is not exercised in CI — the pure builders below are.

    python scripts/build_advisory_snapshot.py --month 1 --day-of-week 5 --hour 17

Output: data/advisory/region_weekly_profiles.json,
        data/advisory/national_snapshot.json,
        data/advisory/national_snapshot.geojson
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import advisory
import advisory_messages
import region_index
import regions

ADVISORY_DIR = REPO_DIR / "data" / "advisory"
WEEKLY_FRAMES = 24 * 7
_WEEKDAYS = ("Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun")


def frame_label(frame_idx: int) -> str:
    idx = max(0, min(WEEKLY_FRAMES - 1, int(frame_idx)))
    return f"{_WEEKDAYS[idx // 24]} {idx % 24:02d}:00"


def build_region_profiles(profile_fn, *, month: int = 1, region_catalog=None) -> dict:
    """region_id -> 168-frame weekly climatology, from an injected profile builder.

    ``profile_fn(lat, lon, month)`` returns a length-168 profile for a point (e.g.
    ``predict.weekly_risk_profile``); each region reduces its grid per frame.
    """
    catalog = region_catalog if region_catalog is not None else regions.list_regions()
    profiles: dict[str, list[float]] = {}
    for region in catalog:
        profile = region_index.region_weekly_profile_batched(profile_fn, region, month=month)
        profiles[region["id"]] = [round(float(value), 6) for value in profile]
    return profiles


def national_snapshot(
    region_profiles: dict,
    *,
    day_of_week: int = 1,
    hour: int = 0,
    month: int = 1,
    region_catalog=None,
) -> dict:
    """A national advisory snapshot at one hour-of-week from stored profiles."""
    catalog = region_catalog if region_catalog is not None else regions.list_regions()
    by_id = {region["id"]: region for region in catalog}
    frame_idx = (day_of_week - 1) * 24 + hour

    entries = []
    for region_id, profile in region_profiles.items():
        region = by_id.get(region_id)
        if region is None:
            continue
        value = profile[frame_idx] if 0 <= frame_idx < len(profile) else 0.0
        advisory_block = advisory.relative_advisory(value, profile)
        advisory_block["message"] = advisory_messages.compose(
            advisory_block, location_name=region["name"], frame_idx=frame_idx
        )
        entries.append(
            {
                "region_id": region_id,
                "region_name": region["name"],
                "bbox": list(region["bbox"]),
                "risk_score": advisory_block["risk_score"],
                "percentile": advisory_block["percentile"],
                "advisory": advisory_block,
            }
        )
    entries.sort(
        key=lambda entry: (
            entry["percentile"] if entry["percentile"] is not None else -1.0,
            entry["risk_score"],
        ),
        reverse=True,
    )
    return {
        "mode": "climatology",
        "basis": "relative_to_local_weekly_normal",
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "frame_idx": frame_idx,
        "frame_label": frame_label(frame_idx),
        "count": len(entries),
        "regions": entries,
    }


def snapshot_geojson(snapshot: dict, *, geometry: str = "point") -> dict:
    """FeatureCollection of the snapshot's regions (centroid points or bbox polygons)."""
    features = []
    for entry in snapshot.get("regions", []):
        bbox = entry.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            min_lat, max_lat, min_lon, max_lon = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        advisory_block = entry.get("advisory") or {}
        properties = {
            "region_id": entry.get("region_id"),
            "region_name": entry.get("region_name"),
            "risk_score": entry.get("risk_score"),
            "percentile": advisory_block.get("percentile"),
            "level": advisory_block.get("level"),
            "level_index": advisory_block.get("level_index"),
            "color": advisory_block.get("color"),
            "advice": advisory_block.get("advice"),
        }
        if geometry == "bbox":
            geom = {
                "type": "Polygon",
                "coordinates": [
                    [
                        [min_lon, min_lat],
                        [max_lon, min_lat],
                        [max_lon, max_lat],
                        [min_lon, max_lat],
                        [min_lon, min_lat],
                    ]
                ],
            }
        else:
            geom = {
                "type": "Point",
                "coordinates": [
                    round((min_lon + max_lon) / 2.0, 6),
                    round((min_lat + max_lat) / 2.0, 6),
                ],
            }
        features.append({"type": "Feature", "geometry": geom, "properties": properties})
    return {
        "type": "FeatureCollection",
        "frame_idx": snapshot.get("frame_idx"),
        "frame_label": snapshot.get("frame_label"),
        "features": features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=1, help="month 1-12 for the reference")
    parser.add_argument("--day-of-week", type=int, default=1, help="Monday=1..Sunday=7")
    parser.add_argument("--hour", type=int, default=0, help="local hour 0-23")
    parser.add_argument("--output-dir", type=Path, default=ADVISORY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from predict import weekly_risk_profile

    profiles = build_region_profiles(weekly_risk_profile, month=args.month)
    snapshot = national_snapshot(
        profiles, day_of_week=args.day_of_week, hour=args.hour, month=args.month
    )
    geojson = snapshot_geojson(snapshot)

    generated_at = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    (args.output_dir / "region_weekly_profiles.json").write_text(
        json.dumps(
            {
                "month": args.month,
                "frames": WEEKLY_FRAMES,
                "generated_at_utc": generated_at,
                "profiles": profiles,
            }
        ),
        encoding="utf-8",
    )
    snapshot["generated_at_utc"] = generated_at
    (args.output_dir / "national_snapshot.json").write_text(
        json.dumps(snapshot, allow_nan=False), encoding="utf-8"
    )
    geojson["generated_at_utc"] = generated_at
    (args.output_dir / "national_snapshot.geojson").write_text(
        json.dumps(geojson, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Wrote {len(profiles)} region profiles and a national snapshot "
        f"({snapshot['frame_label']}, month {args.month}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
