from __future__ import annotations

from pathlib import Path
import sys

import pytest

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


def test_catalog_metadata():
    meta = cm.catalog_metadata()
    assert meta["version"]
    assert meta["source"]
    assert meta["count"] == len(cm.load_countermeasures())


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


# --- Benefit-cost -------------------------------------------------------------

_RUMBLE = {"id": "rumble_strips", "cmf": 0.74, "typical_cost_usd": 15000, "cost_unit": "per_mile"}
_HAWK = {"id": "hawk", "cmf": 0.45, "typical_cost_usd": 120000, "cost_unit": "per_location"}


def test_annualize_crashes():
    assert cm.annualize_crashes(15, 5) == pytest.approx(3.0)
    assert cm.annualize_crashes(15, 0) == pytest.approx(15.0)  # 0 years -> treat as 1
    assert cm.annualize_crashes(-4, 5) == 0.0


def test_treatment_cost_per_mile_scales_with_length():
    # 1.60934 km ~= 1 mile.
    cost = cm.treatment_cost(_RUMBLE, length_km=1.60934)
    assert cost == pytest.approx(15000, abs=5)


def test_treatment_cost_per_location_is_flat():
    assert cm.treatment_cost(_HAWK, length_km=3.0) == pytest.approx(120000)


def test_countermeasure_benefit_cost_structure():
    bc = cm.countermeasure_benefit_cost(_RUMBLE, expected_annual_crashes=2.0, length_km=1.60934)
    # CRF 0.26 * 2/yr = 0.52 crashes avoided per year.
    assert bc["annual_crashes_reduced"] == pytest.approx(0.52)
    assert bc["crash_reduction"] == pytest.approx(0.26)
    assert bc["countermeasure_id"] == "rumble_strips"
    assert bc["treatment_cost"] == pytest.approx(15000, abs=5)
    assert bc["benefit_cost_ratio"] > 0
    assert bc["net_benefit"] > 0  # fatal-crash value dwarfs a low-cost treatment


def test_lower_cmf_yields_higher_benefit():
    strong = cm.countermeasure_benefit_cost(
        {"id": "x", "cmf": 0.4, "typical_cost_usd": 15000, "cost_unit": "per_mile"},
        expected_annual_crashes=2.0, length_km=1.0,
    )
    weak = cm.countermeasure_benefit_cost(
        {"id": "y", "cmf": 0.9, "typical_cost_usd": 15000, "cost_unit": "per_mile"},
        expected_annual_crashes=2.0, length_km=1.0,
    )
    assert strong["annual_benefit"] > weak["annual_benefit"]
    assert strong["benefit_cost_ratio"] > weak["benefit_cost_ratio"]


def test_crash_cost_override():
    bc = cm.countermeasure_benefit_cost(
        _RUMBLE, expected_annual_crashes=2.0, length_km=1.0, crash_cost=100000
    )
    assert bc["crash_cost_each"] == 100000
    assert bc["annual_benefit"] == pytest.approx(0.52 * 100000)


# --- recommend_countermeasures + CountermeasureStore --------------------------

import pandas as pd  # noqa: E402


def test_recommend_countermeasures_ranked_with_benefit_cost():
    attrs = {"mtfcc": "S1200", "rur_urb": 2, "length_km": 1.5, "fatal_crashes": 6.0}
    recs = cm.recommend_countermeasures(attrs, analysis_years=5, catalog=_CATALOG, vru_share=0.7)
    assert recs, "expected recommendations for an urban arterial"
    assert "benefit_cost" in recs[0] and "benefit_cost_ratio" in recs[0]["benefit_cost"]
    ratios = [r["benefit_cost"]["benefit_cost_ratio"] or 0.0 for r in recs]
    assert ratios == sorted(ratios, reverse=True)  # ranked by BCR


def _segment_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["seg-1", "seg-2"],
            "fullname": ["Main St", "Rural Rd"],
            "mtfcc": ["S1200", "S1100"],
            "rur_urb": [2, 1],
            "length_km": [1.5, 2.0],
            "center_lat": [34.0, 36.0],
            "center_lon": [-118.2, -119.0],
            "fatal_crashes": [6.0, 10.0],
            "hin_rank": [1, 2],
        }
    )


def test_store_from_parquet_and_recommend(tmp_path):
    path = tmp_path / "hin.parquet"
    _segment_frame().to_parquet(path, index=False)
    store = cm.CountermeasureStore.from_parquet(path, analysis_years=5)
    assert len(store) == 2

    result = store.recommend("seg-1")
    assert result["segment"]["segment_id"] == "seg-1"
    assert result["analysis_years"] == 5
    assert result["count"] >= 1
    assert result["recommendations"][0]["benefit_cost"]["benefit_cost_ratio"] > 0
    import json

    json.dumps(result)  # JSON-safe (no numpy/NaN leak)


def test_store_recommend_nan_fields_no_nan_leak(tmp_path):
    import json

    import numpy as np

    frame = pd.DataFrame(
        {
            "segment_id": ["seg-nan"], "fullname": ["X"], "mtfcc": ["S1200"], "rur_urb": [2],
            "length_km": [np.nan], "center_lat": [34.0], "center_lon": [-118.2],
            "fatal_crashes": [np.nan], "hin_rank": [1],
        }
    )
    path = tmp_path / "hin.parquet"
    frame.to_parquet(path, index=False)
    result = cm.CountermeasureStore.from_parquet(path).recommend("seg-nan")
    # NaN inputs degrade to 0; nothing NaN leaks (allow_nan=False raises on NaN).
    json.dumps(result, allow_nan=False)


def test_store_unknown_segment_returns_none(tmp_path):
    path = tmp_path / "hin.parquet"
    _segment_frame().to_parquet(path, index=False)
    assert cm.CountermeasureStore.from_parquet(path).recommend("nope") is None


def test_store_missing_file_is_empty(tmp_path):
    store = cm.CountermeasureStore.from_parquet(tmp_path / "nope.parquet")
    assert len(store) == 0
    assert store.recommend("seg-1") is None


def test_load_store_honors_env(tmp_path, monkeypatch):
    path = tmp_path / "hin.parquet"
    _segment_frame().to_parquet(path, index=False)
    monkeypatch.setenv(cm.CM_SEGMENTS_PATH_ENV, str(path))
    assert len(cm.load_countermeasure_store()) == 2


def test_store_hotspots_ranked_with_recommendation(tmp_path):
    path = tmp_path / "hin.parquet"
    _segment_frame().to_parquet(path, index=False)  # seg-1 fatal 6, seg-2 fatal 10
    hotspots = cm.CountermeasureStore.from_parquet(path).hotspots()
    assert [h["segment_id"] for h in hotspots] == ["seg-2", "seg-1"]  # by fatal_crashes desc
    assert hotspots[0]["recommended"] is not None
    assert "benefit_cost_ratio" in hotspots[0]["recommended"]
    import json

    json.dumps(hotspots)  # JSON-safe


def test_store_hotspots_bbox_and_min_fatal(tmp_path):
    path = tmp_path / "hin.parquet"
    _segment_frame().to_parquet(path, index=False)
    store = cm.CountermeasureStore.from_parquet(path)
    # bbox over seg-1 only (34.0/-118.2).
    in_la = store.hotspots(bbox=(33.9, 34.1, -118.3, -118.1))
    assert {h["segment_id"] for h in in_la} == {"seg-1"}
    # min_fatal_crashes filters seg-1 (6) out, keeps seg-2 (10).
    assert {h["segment_id"] for h in store.hotspots(min_fatal_crashes=8)} == {"seg-2"}
