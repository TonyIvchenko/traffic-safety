from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import countermeasures as cm
import crash_typing as ct


# A small synthetic catalog for precise assertions, independent of the real file.
_CATALOG = [
    {
        "id": "rumble_strips", "name": "Rumble Strips", "category": "roadway_departure",
        "cmf": 0.74, "cmf_basis": "ROR", "cmf_star_rating": 4,
        "applicable_crash_types": ["run_off_road"],
        "roadway": {"mtfcc": ["S1100", "S1200"], "context": ["rural"], "setting": ["segment"]},
        "vru_focused": False, "typical_cost_usd": 15000, "cost_unit": "per_mile",
    },
    {
        "id": "rrfb", "name": "RRFB", "category": "pedestrian_crossing",
        "cmf": 0.53, "cmf_basis": "ped", "cmf_star_rating": 3,
        "applicable_crash_types": ["pedestrian"],
        "roadway": {"mtfcc": ["S1200"], "context": ["urban", "suburban"], "setting": ["crossing"]},
        "vru_focused": True, "typical_cost_usd": 25000, "cost_unit": "per_location",
    },
    {
        "id": "road_diet", "name": "Road Diet", "category": "roadway_reconfiguration",
        "cmf": 0.71, "cmf_basis": "total", "cmf_star_rating": 4,
        "applicable_crash_types": ["total", "angle", "rear_end", "pedestrian"],
        "roadway": {"mtfcc": ["S1200"], "context": ["urban", "suburban"], "setting": ["segment"]},
        "vru_focused": False, "typical_cost_usd": 100000, "cost_unit": "per_mile",
    },
]


def test_load_reference_returns_entries():
    catalog = cm.load_countermeasures()
    assert isinstance(catalog, list) and len(catalog) >= 10
    assert all("id" in entry and "cmf" in entry for entry in catalog)


def test_get_countermeasure():
    assert cm.get_countermeasure("road_diet", catalog=_CATALOG)["name"] == "Road Diet"
    assert cm.get_countermeasure("nope", catalog=_CATALOG) is None


def test_rural_highway_recommends_run_off_road_treatment():
    attrs = {"mtfcc": "S1100", "rur_urb": 1}
    profile = ct.crash_type_profile(rur_urb=1, mtfcc="S1100")  # run_off_road dominant
    result = cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG)
    ids = [r["id"] for r in result]
    assert "rumble_strips" in ids
    assert "rrfb" not in ids  # urban crossing treatment, wrong roadway
    assert "road_diet" not in ids  # urban arterial, wrong roadway


def test_urban_arterial_high_vru_recommends_ped_and_road_diet():
    attrs = {"mtfcc": "S1200", "rur_urb": 2}
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200", vru_share=0.8)
    result = cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG)
    ids = [r["id"] for r in result]
    assert "rrfb" in ids and "road_diet" in ids
    assert "rumble_strips" not in ids  # rural treatment
    # RRFB best-addresses the boosted pedestrian weight (0.8).
    rrfb = next(r for r in result if r["id"] == "rrfb")
    assert rrfb["match_score"] >= 0.8


def test_results_ranked_by_match_score():
    attrs = {"mtfcc": "S1200", "rur_urb": 2}
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200", vru_share=0.9)
    result = cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG)
    scores = [r["match_score"] for r in result]
    assert scores == sorted(scores, reverse=True)


def test_crash_reduction_is_one_minus_cmf():
    attrs = {"mtfcc": "S1200", "rur_urb": 2}
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200")
    road_diet = next(
        r for r in cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG)
        if r["id"] == "road_diet"
    )
    assert road_diet["crash_reduction"] == round(1 - 0.71, 4)


def test_min_score_and_top_n():
    attrs = {"mtfcc": "S1200", "rur_urb": 2}
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200")
    all_matches = cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG, min_score=0.0)
    capped = cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG, top_n=1)
    assert len(capped) == 1
    # A very high threshold filters everything out.
    assert cm.applicable_countermeasures(attrs, profile, catalog=_CATALOG, min_score=0.99) == []
    assert len(all_matches) >= len(capped)


def test_real_catalog_urban_arterial_recommends_something():
    attrs = {"mtfcc": "S1200", "rur_urb": 2}
    profile = ct.crash_type_profile(rur_urb=2, mtfcc="S1200", vru_share=0.7)
    result = cm.applicable_countermeasures(attrs, profile)  # real committed catalog
    assert result, "expected recommendations for an urban arterial"
    assert all(r["match_score"] >= 0.2 for r in result)
    assert any(r["vru_focused"] for r in result)  # a VRU treatment surfaces
