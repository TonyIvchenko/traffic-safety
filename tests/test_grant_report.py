from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs
import grant_report as gr


def _crashes() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "year": [2022, 2022, 2023],
            "fatals": [1, 2, 1],
            "ped_count": [1, 0, 1],
            "cyc_count": [0, 1, 0],
        }
    )


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["hin1", "hin2", "sys1", "sys2"],
            "fullname": ["Main St", "1st Ave", "New Rd", "Quiet Ln"],
            "rttyp": ["U", "S", "I", "C"],
            "mtfcc": ["S1200", "S1200", "S1100", "S1400"],
            "length_km": [1.0, 1.0, 2.0, 1.0],
            "fatal_crashes": [5.0, 3.0, 0.0, 0.0],
            "weighted_crashes": [5.0, 3.0, 0.0, 0.0],
            "hin": [True, True, False, False],
            "hin_intensity": [5.0, 3.0, 0.0, 0.0],
            "hin_rank": [1, 2, 3, 4],
            "systemic_score": [0.9, 0.8, 0.95, 0.2],
            "systemic_rate": [0.5, 0.4, 0.6, 0.05],
            "systemic_expected_crashes": [0.5, 0.4, 1.2, 0.05],
            "center_lat": [34.0, 34.1, 34.2, 34.3],
            "center_lon": [-118.0, -118.1, -118.2, -118.3],
        }
    )


def test_geoid_level_by_length():
    assert gr.geoid_level("06") == "state"
    assert gr.geoid_level("06037") == "county"
    assert gr.geoid_level("06037920100") == "tract"


def test_crash_summary_totals_and_modes():
    summary = gr.crash_summary(_crashes())
    assert summary["total_fatal_crashes"] == 3
    assert summary["total_fatalities"] == 4  # sum of fatals
    assert summary["years"] == [2022, 2023]
    assert summary["by_year"] == {2022: 2, 2023: 1}
    assert summary["by_mode"] == {"pedestrian": 2, "cyclist": 1}


def test_crash_summary_empty():
    summary = gr.crash_summary(pd.DataFrame(columns=["year", "fatals"]))
    assert summary["total_fatal_crashes"] == 0
    assert summary["by_year"] == {}


def test_top_hin_corridors_only_hin_sorted_by_rank():
    corridors = gr.top_hin_corridors(_segments(), top_n=10)
    assert [c["segment_id"] for c in corridors] == ["hin1", "hin2"]
    assert corridors[0]["fullname"] == "Main St"
    assert corridors[0]["hin_rank"] == 1


def test_top_systemic_locations_are_history_poor():
    locations = gr.top_systemic_locations(_segments(), top_n=10)
    # Only zero-history segments, ranked by systemic score (sys1=0.95 > sys2=0.2).
    assert [loc["segment_id"] for loc in locations] == ["sys1", "sys2"]
    assert locations[0]["systemic_score"] == 0.95


def test_assemble_report_structure_and_json_serializable():
    report = gr.assemble_grant_report(
        geoid="06037",
        name="Los Angeles County",
        crashes=_crashes(),
        segments=_segments(),
        benefit_cost={"benefit_cost_ratio": 2.5},
        generated_at="2026-07-14T00:00:00+00:00",
        data_vintage={"fars_years": [2018, 2024]},
    )
    assert report["jurisdiction"] == {
        "geoid": "06037", "name": "Los Angeles County", "level": "county"
    }
    assert report["crash_summary"]["total_fatal_crashes"] == 3
    assert report["high_injury_network"]["hin_segments"] == 2
    assert len(report["hin_corridors"]) == 2
    assert report["benefit_cost"]["benefit_cost_ratio"] == 2.5
    assert "high_injury_network" in report["methodology"]
    # Must be JSON-serializable (no numpy scalars leak through).
    json.dumps(report)


def test_assemble_report_omits_benefit_cost_when_absent():
    report = gr.assemble_grant_report(
        geoid="06037", name="LA", crashes=_crashes(), segments=_segments()
    )
    assert "benefit_cost" not in report


def _corridors():
    return [
        {"segment_id": "a", "fatal_crashes": 10.0, "length_km": 1.0},
        {"segment_id": "b", "fatal_crashes": 5.0, "length_km": 1.0},
    ]


def test_hin_benefit_cost_computes_ratio_and_inputs():
    result = gr.hin_benefit_cost(_corridors(), analysis_years=5)
    # 15 fatal crashes / 5 years = 3.0/yr; reduction 0.30 of the KABCO-K cost.
    fatal_cost = crash_costs.severity_cost("K")
    assert result["annual_fatal_crashes_treated"] == pytest.approx(3.0)
    assert result["annual_benefit"] == pytest.approx(3.0 * fatal_cost * 0.30)
    assert result["treated_corridors"] == 2
    assert result["treated_length_km"] == pytest.approx(2.0)
    assert result["treatment_cost"] == pytest.approx(2.0 * gr.DEFAULT_TREATMENT_COST_PER_KM)
    assert result["analysis_years"] == 5
    assert result["benefit_cost_ratio"] > 0


def test_hin_benefit_cost_empty_or_zero_years_returns_none():
    assert gr.hin_benefit_cost([], analysis_years=5) is None
    assert gr.hin_benefit_cost(_corridors(), analysis_years=0) is None


def test_hin_benefit_cost_reduction_scales_benefit_linearly():
    low = gr.hin_benefit_cost(_corridors(), analysis_years=5, crash_reduction=0.20)
    high = gr.hin_benefit_cost(_corridors(), analysis_years=5, crash_reduction=0.40)
    assert high["annual_benefit"] == pytest.approx(2.0 * low["annual_benefit"])


def test_hin_benefit_cost_skips_non_dict_and_bad_values():
    corridors = [
        {"segment_id": "a", "fatal_crashes": 10.0, "length_km": 1.0},
        "oops",  # non-dict -> ignored
        {"segment_id": "b", "fatal_crashes": None, "length_km": "bad"},  # coerced to 0
    ]
    result = gr.hin_benefit_cost(corridors, analysis_years=5)
    assert result["treated_corridors"] == 2  # only the two dicts
    assert result["annual_fatal_crashes_treated"] == pytest.approx(2.0)  # 10/5, bad->0
    assert result["treated_length_km"] == pytest.approx(1.0)


def test_hin_benefit_cost_is_json_serializable():
    json.dumps(gr.hin_benefit_cost(_corridors(), analysis_years=5))
