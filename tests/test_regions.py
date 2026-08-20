from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import regions


def test_list_regions():
    catalog = regions.list_regions()
    assert len(catalog) >= 15
    assert all({"id", "name", "bbox"} <= set(region) for region in catalog)


def test_ids_unique_and_bboxes_well_formed():
    ids = [region["id"] for region in regions.REGIONS]
    assert len(ids) == len(set(ids))
    for region in regions.REGIONS:
        min_lat, max_lat, min_lon, max_lon = region["bbox"]
        assert min_lat < max_lat, region["id"]
        assert min_lon < max_lon, region["id"]
        assert -90 <= min_lat <= 90 and -180 <= min_lon <= 180, region["id"]


def test_get_region():
    la = regions.get_region("los_angeles")
    assert la["name"] == "Los Angeles, CA"
    assert regions.get_region("LOS_ANGELES")["id"] == "los_angeles"  # case-insensitive
    assert regions.get_region("nope") is None


def test_region_for_point():
    la = regions.region_for_point(34.0522, -118.2437)  # downtown LA
    assert la["id"] == "los_angeles"
    ny = regions.region_for_point(40.7128, -74.0060)  # Manhattan
    assert ny["id"] == "new_york"


def test_region_for_point_none_when_uncovered():
    assert regions.region_for_point(0.0, 0.0) is None  # Gulf of Guinea
    assert regions.region_for_point(64.0, -150.0) is None  # interior Alaska


def test_region_for_point_bad_input():
    assert regions.region_for_point("x", None) is None


def test_region_bbox():
    bbox = regions.region_bbox(regions.get_region("chicago"))
    assert bbox == (41.60, 42.10, -88.05, -87.45)
