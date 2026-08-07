from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_typing as ct


def test_road_tier_from_mtfcc():
    assert ct.road_tier("S1100") == "highway"
    assert ct.road_tier("S1200") == "arterial"
    assert ct.road_tier("S1400") == "local"


def test_road_tier_falls_back_to_func_sys():
    assert ct.road_tier(None, func_sys=1) == "highway"
    assert ct.road_tier(None, func_sys=4) == "arterial"
    assert ct.road_tier(None, func_sys=7) == "local"
    assert ct.road_tier(None, None) == "arterial"  # unknown -> middle


def test_is_urban():
    assert ct.is_urban(2) is True
    assert ct.is_urban("urban") is True
    assert ct.is_urban(1) is False
    assert ct.is_urban("rural") is False
    assert ct.is_urban(None) is True  # unknown defaults urban


def test_context_class():
    assert ct.context_class(1, "S1100") == "rural_highway"
    assert ct.context_class(2, "S1200") == "urban_arterial"
    assert ct.context_class(2, "S1400") == "urban_local"


def test_rural_highway_run_off_road_dominant():
    profile = ct.crash_type_profile(rur_urb=1, mtfcc="S1100")
    assert profile["run_off_road"] == max(profile.values())
    assert ct.dominant_crash_types(profile)[0] == "run_off_road"
    assert profile["pedestrian"] == 0.0  # no peds on a rural highway base


def test_urban_arterial_profile():
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200")
    top = ct.dominant_crash_types(profile, top_n=3)
    assert "angle" in top
    assert profile["pedestrian"] > 0.0


def test_vru_share_boosts_pedestrian_and_bicycle():
    base = ct.crash_type_profile(rur_urb=2, mtfcc="S1200")
    boosted = ct.crash_type_profile(rur_urb=2, mtfcc="S1200", vru_share=0.8)
    assert boosted["pedestrian"] >= 0.8
    assert boosted["pedestrian"] > base["pedestrian"]
    assert boosted["bicycle"] == round(0.8 * 0.6, 4)


def test_condition_shares_set_weights():
    profile = ct.crash_type_profile(rur_urb=1, mtfcc="S1200", nighttime_share=0.4, wet_share=0.25)
    assert profile["nighttime"] == 0.4
    assert profile["wet_road"] == 0.25


def test_all_weights_in_unit_range():
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200", vru_share=2.0, nighttime_share=5.0)
    assert set(profile) == set(ct.CRASH_TYPES)
    assert all(0.0 <= w <= 1.0 for w in profile.values())


def test_missing_attributes_do_not_crash():
    profile = ct.crash_type_profile()  # all defaults -> urban_arterial
    assert profile["angle"] > 0.0
    assert all(0.0 <= w <= 1.0 for w in profile.values())


def test_dominant_crash_types_respects_min_weight():
    profile = {t: 0.0 for t in ct.CRASH_TYPES}
    profile["angle"] = 0.8
    profile["rear_end"] = 0.1  # below default min_weight 0.2
    assert ct.dominant_crash_types(profile) == ["angle"]
