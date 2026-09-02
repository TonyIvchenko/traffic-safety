from __future__ import annotations

from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_advisory_snapshot as bas

# Fake regions inside a shared synthetic space; profile_fn ignores location.
FAKE_REGIONS = [
    {"id": "alpha", "name": "Alpha City", "bbox": [1.0, 2.0, 1.0, 2.0]},
    {"id": "beta", "name": "Beta City", "bbox": [3.0, 4.0, 3.0, 4.0]},
]


def _rising_profile_fn(lat, lon, month):
    # Rises across the week; independent of location.
    return [0.20 + how / 1000.0 for how in range(168)]


def test_frame_label():
    assert bas.frame_label(0) == "Mon 00:00"
    assert bas.frame_label((5 - 1) * 24 + 17) == "Fri 17:00"
    assert bas.frame_label(9999) == "Sun 23:00"  # clamped


def test_build_region_profiles():
    profiles = bas.build_region_profiles(
        _rising_profile_fn, month=1, region_catalog=FAKE_REGIONS
    )
    assert set(profiles) == {"alpha", "beta"}
    assert len(profiles["alpha"]) == 168
    assert profiles["alpha"][0] == pytest.approx(0.20)
    assert profiles["alpha"][167] == pytest.approx(0.20 + 167 / 1000.0)


def test_national_snapshot_shape_and_ranking():
    # Give beta a higher current-frame percentile by shifting its profile so the
    # chosen frame sits near its top while alpha sits mid-range.
    profiles = {
        "alpha": [0.5] * 168,  # flat -> absolute fallback
        "beta": [0.10 + how / 1000.0 for how in range(168)],  # rising -> discriminates
    }
    snap = bas.national_snapshot(
        profiles, day_of_week=7, hour=23, month=1, region_catalog=FAKE_REGIONS
    )
    assert snap["mode"] == "climatology"
    assert snap["basis"] == "relative_to_local_weekly_normal"
    assert snap["frame_idx"] == (7 - 1) * 24 + 23  # 167
    assert snap["frame_label"] == "Sun 23:00"
    assert snap["count"] == 2
    # beta at its weekly peak (percentile ~1) ranks above alpha (flat -> absolute).
    assert snap["regions"][0]["region_id"] == "beta"
    assert snap["regions"][0]["advisory"]["level"] == "Extreme"
    assert snap["regions"][0]["advisory"]["message"].startswith("Extreme for Beta City")


def test_national_snapshot_skips_unknown_region():
    snap = bas.national_snapshot(
        {"ghost": [0.5] * 168}, region_catalog=FAKE_REGIONS
    )
    assert snap["count"] == 0


def test_snapshot_geojson_point_and_bbox():
    profiles = bas.build_region_profiles(
        _rising_profile_fn, region_catalog=FAKE_REGIONS
    )
    snap = bas.national_snapshot(
        profiles, day_of_week=7, hour=23, region_catalog=FAKE_REGIONS
    )
    point = bas.snapshot_geojson(snap, geometry="point")
    assert point["type"] == "FeatureCollection"
    assert len(point["features"]) == 2
    feat = point["features"][0]
    assert feat["geometry"]["type"] == "Point"
    assert {"region_id", "level", "percentile", "color"} <= set(feat["properties"])

    poly = bas.snapshot_geojson(snap, geometry="bbox")["features"][0]
    assert poly["geometry"]["type"] == "Polygon"
    ring = poly["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]
