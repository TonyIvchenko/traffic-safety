"""Precompute a national evacuation-readiness snapshot across metro regions.

For each region: the relative road-risk advisory at a chosen hour-of-week (from the
raw-model weekly climatology), the critical-facility coverage within its bounds,
and the combined evacuation-readiness rating. Regions are ranked least-ready
first so operators see the gaps.

Reuses build_advisory_snapshot.build_region_profiles for the weekly climatology
(needs the model bundle), so ``main`` is not exercised in CI; the pure builders
below are unit-tested.

    python scripts/build_emergency_snapshot.py --month 1 --day-of-week 5 --hour 17

Output: data/emergency/national_readiness.json, data/emergency/national_readiness.geojson
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
for _path in (REPO_DIR / "scripts", str(SRC_DIR), str(REPO_DIR)):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import advisory
import advisory_messages
import evacuation
import facilities
import regions

from build_advisory_snapshot import build_region_profiles, frame_label

EMERGENCY_DIR = REPO_DIR / "data" / "emergency"
_RATING_RANK = {"limited": 0, "moderate": 1, "good": 2}


def region_facility_coverage(store, region) -> dict:
    """Facility counts per kind within a region's bbox, plus the total."""
    bbox = region["bbox"]
    counts = {kind: len(store.within_bbox(bbox, kind=kind)) for kind in facilities.FACILITY_KINDS}
    return {"counts": counts, "total": sum(counts.values())}


def _percentile_key(entry: dict) -> float:
    percentile = entry["advisory"].get("percentile")
    return percentile if isinstance(percentile, (int, float)) else -1.0


def build_readiness_table(
    region_profiles: dict,
    store,
    *,
    day_of_week: int = 1,
    hour: int = 0,
    month: int = 1,
    region_catalog=None,
) -> dict:
    """A national readiness table (least-ready first) from profiles + facilities."""
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
        coverage = region_facility_coverage(store, region)
        readiness = evacuation.assess_readiness(advisory_block["level_index"], coverage["counts"])
        entries.append(
            {
                "region_id": region_id,
                "region_name": region["name"],
                "bbox": list(region["bbox"]),
                "advisory": advisory_block,
                "facility_coverage": coverage,
                "readiness": readiness,
            }
        )

    # Least-ready first (limited < moderate < good), then most-elevated conditions.
    entries.sort(key=lambda entry: (_RATING_RANK.get(entry["readiness"]["rating"], 3), -_percentile_key(entry)))
    return {
        "mode": "climatology",
        "day_of_week": day_of_week,
        "hour": hour,
        "month": month,
        "frame_idx": frame_idx,
        "frame_label": frame_label(frame_idx),
        "count": len(entries),
        "regions": entries,
    }


def readiness_geojson(table: dict) -> dict:
    """FeatureCollection of region-centroid points carrying the readiness summary."""
    features = []
    for entry in table.get("regions", []):
        bbox = entry.get("bbox")
        if not isinstance(bbox, (list, tuple)) or len(bbox) != 4:
            continue
        try:
            min_lat, max_lat, min_lon, max_lon = (float(value) for value in bbox)
        except (TypeError, ValueError):
            continue
        advisory_block = entry.get("advisory") or {}
        readiness = entry.get("readiness") or {}
        coverage = entry.get("facility_coverage") or {}
        features.append(
            {
                "type": "Feature",
                "geometry": {
                    "type": "Point",
                    "coordinates": [
                        round((min_lon + max_lon) / 2.0, 6),
                        round((min_lat + max_lat) / 2.0, 6),
                    ],
                },
                "properties": {
                    "region_id": entry.get("region_id"),
                    "region_name": entry.get("region_name"),
                    "readiness": readiness.get("rating"),
                    "level": advisory_block.get("level"),
                    "color": advisory_block.get("color"),
                    "facility_total": coverage.get("total"),
                },
            }
        )
    return {
        "type": "FeatureCollection",
        "frame_idx": table.get("frame_idx"),
        "frame_label": table.get("frame_label"),
        "features": features,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--month", type=int, default=1)
    parser.add_argument("--day-of-week", type=int, default=1)
    parser.add_argument("--hour", type=int, default=0)
    parser.add_argument("--output-dir", type=Path, default=EMERGENCY_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    from predict import weekly_risk_profile

    profiles = build_region_profiles(weekly_risk_profile, month=args.month)
    store = facilities.load_facility_store()
    table = build_readiness_table(
        profiles, store, day_of_week=args.day_of_week, hour=args.hour, month=args.month
    )
    geojson = readiness_geojson(table)

    generated_at = datetime.now(timezone.utc).isoformat()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    table["generated_at_utc"] = generated_at
    (args.output_dir / "national_readiness.json").write_text(
        json.dumps(table, allow_nan=False), encoding="utf-8"
    )
    geojson["generated_at_utc"] = generated_at
    (args.output_dir / "national_readiness.geojson").write_text(
        json.dumps(geojson, allow_nan=False), encoding="utf-8"
    )
    print(
        f"Wrote readiness for {table['count']} regions "
        f"({table['frame_label']}, month {args.month}) to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
