from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs
import impact


def test_prevention_impact_sums_and_values_crashes():
    treatments = [
        {"annual_crashes_reduced": 0.2, "treatment_cost": 100_000.0},
        {"annual_crashes_reduced": 0.3, "treatment_cost": 200_000.0},
    ]
    result = impact.prevention_impact(treatments, service_life_years=20, discount_rate=0.03)
    assert result["segments_with_treatment"] == 2
    assert result["annual_crashes_averted"] == pytest.approx(0.5)
    assert result["crashes_averted_service_life"] == pytest.approx(10.0)
    assert result["investment_required"] == pytest.approx(300_000.0)
    # PV of 0.5 fatal crashes/yr valued at the FHWA K cost, over 20yr @3%.
    annual_benefit = 0.5 * crash_costs.severity_cost("K")
    expected_pv = crash_costs.present_value(annual_benefit, 20, 0.03)
    assert result["present_value_benefit"] == pytest.approx(round(expected_pv, 2), rel=1e-6)
    assert result["benefit_cost_ratio"] == pytest.approx(round(expected_pv / 300_000.0, 3), rel=1e-6)


def test_prevention_impact_skips_invalid_treatments():
    treatments = [
        {"annual_crashes_reduced": 0.1, "treatment_cost": 50_000.0},
        {"annual_crashes_reduced": None, "treatment_cost": 50_000.0},  # no benefit -> skipped
        {"treatment_cost": 50_000.0},  # no benefit key -> skipped
        "not a dict",
    ]
    result = impact.prevention_impact(treatments)
    assert result["segments_with_treatment"] == 1
    assert result["annual_crashes_averted"] == pytest.approx(0.1)


def test_prevention_impact_zero_investment_has_no_bcr():
    result = impact.prevention_impact([{"annual_crashes_reduced": 0.1, "treatment_cost": 0.0}])
    assert result["investment_required"] == 0.0
    assert result["benefit_cost_ratio"] is None


def test_equity_impact_estimates_disadvantaged_share():
    disparity = {
        "crash_share_disadvantaged": 0.4,
        "crash_disparity_ratio": 1.8,
        "segment_share_disadvantaged": 0.25,
    }
    result = impact.equity_impact(0.5, disparity)
    assert result["disadvantaged_crash_share"] == pytest.approx(0.4)
    assert result["estimated_disadvantaged_crashes_averted_annual"] == pytest.approx(0.2)
    assert result["crash_disparity_ratio"] == 1.8


def test_equity_impact_handles_missing_share():
    result = impact.equity_impact(0.5, {"crash_share_disadvantaged": None})
    assert result["estimated_disadvantaged_crashes_averted_annual"] is None


def test_impact_summary_composes_all_parts():
    treatments = [{"annual_crashes_reduced": 0.5, "treatment_cost": 300_000.0}]
    disparity = {"crash_share_disadvantaged": 0.4, "crash_disparity_ratio": 1.8}
    coverage = {"hin_segments": 10, "grant_jurisdictions": 3}
    summary = impact.impact_summary(
        treatments, equity_disparity=disparity, coverage=coverage,
        service_life_years=20, discount_rate=0.03,
    )
    assert summary["status"] == "ok"
    assert summary["method"] == impact.METHOD
    assert summary["prevention"]["annual_crashes_averted"] == pytest.approx(0.5)
    assert summary["equity"]["estimated_disadvantaged_crashes_averted_annual"] == pytest.approx(0.2)
    assert summary["coverage"]["hin_segments"] == 10
    assert "crashes/year avertable" in summary["headline"]


def test_impact_summary_without_equity():
    summary = impact.impact_summary([{"annual_crashes_reduced": 0.1, "treatment_cost": 1000.0}])
    assert "equity" not in summary
    assert summary["prevention"]["annual_crashes_averted"] == pytest.approx(0.1)


def test_impact_summary_empty_degrades_to_zeros():
    summary = impact.impact_summary([], equity_disparity={"crash_share_disadvantaged": None})
    assert summary["status"] == "ok"
    assert summary["prevention"]["segments_with_treatment"] == 0
    assert summary["prevention"]["annual_crashes_averted"] == 0.0
    assert summary["prevention"]["benefit_cost_ratio"] is None
