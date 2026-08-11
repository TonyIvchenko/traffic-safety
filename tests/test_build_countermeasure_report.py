from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_countermeasure_report as bcr


_CATALOG = [
    {
        "id": "rumble_strips", "name": "Rumble Strips", "category": "roadway_departure",
        "cmf": 0.74, "cmf_basis": "ROR", "cmf_star_rating": 4,
        "applicable_crash_types": ["run_off_road"],
        "roadway": {"mtfcc": ["S1100", "S1200"], "context": ["rural"], "setting": ["segment"]},
        "vru_focused": False, "typical_cost_usd": 15000, "cost_unit": "per_mile",
    },
    {
        "id": "road_diet", "name": "Road Diet", "category": "roadway_reconfiguration",
        "cmf": 0.71, "cmf_basis": "total", "cmf_star_rating": 4,
        "applicable_crash_types": ["total", "angle", "pedestrian"],
        "roadway": {"mtfcc": ["S1200"], "context": ["urban", "suburban"], "setting": ["segment"]},
        "vru_focused": False, "typical_cost_usd": 100000, "cost_unit": "per_mile",
    },
]


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "fullname": ["Rural Rd", "Main St"],
            "mtfcc": ["S1100", "S1200"],
            "rur_urb": [1, 2],
            "length_km": [2.0, 1.5],
            "center_lat": [36.0, 34.0],
            "center_lon": [-119.0, -118.2],
            "fatal_crashes": [10.0, 6.0],
            "hin_rank": [1, 2],
        }
    )


def test_recommend_for_segment_rural():
    segment = _segments().iloc[0].to_dict()  # rural S1100
    rec = bcr.recommend_for_segment(segment, analysis_years=5, catalog=_CATALOG)
    assert rec["segment_id"] == "a"
    assert rec["recommended_id"] == "rumble_strips"  # rural roadway-departure fit
    assert rec["expected_annual_crashes"] == pytest.approx(2.0)  # 10 / 5
    assert rec["annual_crashes_reduced"] == pytest.approx(2.0 * 0.26)
    assert rec["benefit_cost_ratio"] > 0
    assert rec["applicable_count"] >= 1


def test_recommend_for_segment_none_when_no_fit():
    # An MTFCC no catalog entry targets -> no recommendation.
    segment = {"segment_id": "x", "mtfcc": "S9999", "rur_urb": 2, "length_km": 1.0,
               "fatal_crashes": 4.0}
    assert bcr.recommend_for_segment(segment, analysis_years=5, catalog=_CATALOG) is None


def test_build_report_columns_and_sort():
    report = bcr.build_countermeasure_report(_segments(), analysis_years=5, catalog=_CATALOG)
    assert list(report.columns) == bcr.REPORT_COLUMNS
    assert len(report) == 2
    # Sorted by benefit-cost ratio descending.
    ratios = report["benefit_cost_ratio"].tolist()
    assert ratios == sorted(ratios, reverse=True)


def test_geojson_builder():
    report = bcr.build_countermeasure_report(_segments(), analysis_years=5, catalog=_CATALOG)
    geojson = bcr.countermeasures_geojson(report)
    assert geojson["type"] == "FeatureCollection"
    assert geojson["count"] == 2
    feature = geojson["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert "recommended_id" in feature["properties"]
    assert "center_lat" not in feature["properties"]
    import json

    json.dumps(geojson)  # JSON-serializable


def test_build_report_uses_real_catalog():
    # Integration: no explicit catalog -> the committed reference is used.
    report = bcr.build_countermeasure_report(_segments(), analysis_years=5)
    assert len(report) >= 1
    assert report.iloc[0]["benefit_cost_ratio"] > 0
    assert report.iloc[0]["recommended_id"]
