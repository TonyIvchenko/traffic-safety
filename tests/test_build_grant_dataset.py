from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_grant_dataset as bgd


class FakeIndex:
    """Assigns a GEOID by hemisphere so lookups are deterministic in tests."""

    def lookup(self, lat: float, lon: float):
        if lat is None:
            return None
        return "06037" if lat >= 34.0 else "06059"


def _segments() -> pd.DataFrame:
    # Two counties: 06037 = {a, b}, 06059 = {c, d}. Systemic columns are
    # pre-attached (build_grant_dataset applies them before this stage).
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "county_geoid": ["06037", "06037", "06059", "06059"],
            "fullname": ["Main St", "1st Ave", "Oak Rd", "Elm St"],
            "rttyp": ["U", "S", "I", "C"],
            "mtfcc": ["S1200", "S1200", "S1100", "S1400"],
            "length_km": [1.0, 1.0, 2.0, 1.0],
            "fatal_crashes": [5.0, 3.0, 0.0, 1.0],
            "weighted_crashes": [5.0, 3.0, 0.0, 1.0],
            "hin": [True, True, False, True],
            "hin_intensity": [5.0, 3.0, 0.0, 1.0],
            "hin_rank": [1, 2, 4, 3],
            "systemic_score": [0.9, 0.8, 0.95, 0.2],
            "systemic_rate": [0.5, 0.4, 0.6, 0.05],
            "systemic_expected_crashes": [0.5, 0.4, 1.2, 0.05],
            "center_lat": [34.0, 34.1, 33.6, 33.7],
            "center_lon": [-118.2, -118.3, -117.8, -117.9],
        }
    )


def _crashes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "county_geoid": ["06037", "06037", "06037", "06059"],
            "year": [2022, 2022, 2023, 2023],
            "fatals": [1, 2, 1, 1],
            "lat": [34.0, 34.1, 34.0, 33.6],
            "lon": [-118.2, -118.3, -118.2, -117.8],
        }
    )


def test_assign_geoids_uses_index_lookup():
    frame = pd.DataFrame({"center_lat": [34.5, 33.0], "center_lon": [-118.0, -117.0]})
    geoids = bgd.assign_geoids(frame, FakeIndex())
    assert list(geoids) == ["06037", "06059"]
    assert list(geoids.index) == list(frame.index)


def test_assign_geoids_custom_columns():
    frame = pd.DataFrame({"lat": [34.5], "lon": [-118.0]})
    geoids = bgd.assign_geoids(frame, FakeIndex(), lat_col="lat", lon_col="lon")
    assert list(geoids) == ["06037"]


def test_record_geoid_names_maps_geoid_to_name():
    records = [["06037", "Los Angeles County"], ["06059", "Orange County"]]
    names = bgd.record_geoid_names(records, ["GEOID", "NAMELSAD"])
    assert names == {"06037": "Los Angeles County", "06059": "Orange County"}


def test_record_geoid_names_missing_geoid_field_raises():
    with pytest.raises(ValueError, match="GEOID"):
        bgd.record_geoid_names([["x"]], ["NAMELSAD"])


def test_data_vintage_spans_crash_years():
    vintage = bgd.data_vintage(_crashes())
    assert vintage["fars_years"] == [2022, 2023]
    assert vintage["crash_source"] == "FARS fatal crashes"


def test_build_county_reports_splits_by_county():
    reports = bgd.build_county_reports(
        _segments(), _crashes(), names={"06037": "Los Angeles County"}
    )
    assert set(reports) == {"06037", "06059"}
    assert reports["06037"]["jurisdiction"]["name"] == "Los Angeles County"
    # Name falls back to the GEOID when not supplied.
    assert reports["06059"]["jurisdiction"]["name"] == "06059"
    assert reports["06037"]["crash_summary"]["total_fatal_crashes"] == 3
    assert reports["06059"]["crash_summary"]["total_fatal_crashes"] == 1


def test_build_county_reports_hin_scoped_to_county():
    reports = bgd.build_county_reports(_segments(), _crashes())
    la_ids = {corridor["segment_id"] for corridor in reports["06037"]["hin_corridors"]}
    assert la_ids <= {"a", "b"}
    # 06059's only HIN corridor is segment d.
    oc_ids = {corridor["segment_id"] for corridor in reports["06059"]["hin_corridors"]}
    assert oc_ids == {"d"}


def test_build_county_reports_min_fatal_crashes_filters():
    reports = bgd.build_county_reports(_segments(), _crashes(), min_fatal_crashes=2)
    # 06059 has a single fatal crash -> dropped; 06037 has three -> kept.
    assert set(reports) == {"06037"}


def test_build_county_reports_json_serializable():
    reports = bgd.build_county_reports(_segments(), _crashes())
    for report in reports.values():
        json.dumps(report)  # raises if a numpy scalar leaked through


def test_analysis_span_years():
    assert bgd.analysis_span_years(_crashes()) == 2  # 2022..2023 inclusive
    assert bgd.analysis_span_years(pd.DataFrame({"year": []})) == 0


def test_build_county_reports_includes_benefit_cost_with_analysis_years():
    reports = bgd.build_county_reports(_segments(), _crashes(), analysis_years=5)
    benefit_cost = reports["06037"]["benefit_cost"]
    assert benefit_cost is not None
    assert benefit_cost["treated_corridors"] == 2  # a, b are HIN in 06037
    assert benefit_cost["analysis_years"] == 5
    assert "benefit_cost_ratio" in benefit_cost
    json.dumps(reports["06037"])  # still serializable


def test_build_county_reports_omits_benefit_cost_without_analysis_years():
    reports = bgd.build_county_reports(_segments(), _crashes())
    assert "benefit_cost" not in reports["06037"]


def test_write_county_reports_round_trips(tmp_path):
    reports = bgd.build_county_reports(_segments(), _crashes())
    written = bgd.write_county_reports(reports, tmp_path)
    assert {path.name for path in written} == {"06037.json", "06059.json"}
    loaded = json.loads((tmp_path / "06037.json").read_text(encoding="utf-8"))
    assert loaded["jurisdiction"]["geoid"] == "06037"
    assert loaded["crash_summary"]["total_fatal_crashes"] == 3
