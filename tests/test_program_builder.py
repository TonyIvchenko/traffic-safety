from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs
import program_builder as pb


def _candidate(**overrides) -> dict:
    base = {
        "project_id": "P1",
        "treatment_cost": 100_000.0,
        "annual_crashes_reduced": 0.1,
        "svi_percentile": 0.5,
        "disadvantaged": False,
    }
    base.update(overrides)
    return base


def test_equity_weight_formula():
    # 1 + svi_weight*svi + boost*disadvantaged
    assert pb.equity_weight(0.5, False, svi_weight=1.0, disadvantaged_boost=0.5) == pytest.approx(1.5)
    assert pb.equity_weight(0.5, True, svi_weight=1.0, disadvantaged_boost=0.5) == pytest.approx(2.0)
    # Unknown SVI adds no boost; weights are floored at 0 (never below 1).
    assert pb.equity_weight(None, False) == pytest.approx(1.0)
    assert pb.equity_weight(0.9, False, svi_weight=-5.0, disadvantaged_boost=-5.0) == pytest.approx(1.0)


def test_score_candidate_matches_crash_cost_math():
    record = pb.score_candidate(
        _candidate(),
        crash_cost=crash_costs.severity_cost("K"),
        service_life_years=20,
        discount_rate=0.03,
        svi_weight=1.0,
        disadvantaged_boost=0.5,
    )
    annual_benefit = 0.1 * crash_costs.severity_cost("K")
    expected = crash_costs.benefit_cost(annual_benefit, 100_000.0, 20, 0.03)
    assert record["annual_benefit"] == pytest.approx(annual_benefit, rel=1e-6)
    assert record["present_value_benefit"] == pytest.approx(expected["present_value_benefit"], rel=1e-6)
    assert record["benefit_cost_ratio"] == pytest.approx(expected["benefit_cost_ratio"], rel=1e-6)
    # svi 0.5, not disadvantaged -> multiplier 1.5
    assert record["equity_multiplier"] == pytest.approx(1.5)
    assert record["equity_weighted_benefit"] == pytest.approx(expected["present_value_benefit"] * 1.5, rel=1e-6)


def test_score_candidate_from_cmf():
    # expected 0.2 crashes/yr, cmf 0.5 -> 0.1 crashes/yr reduced
    record = pb.score_candidate(
        _candidate(annual_crashes_reduced=None, expected_annual_crashes=0.2, cmf=0.5),
        crash_cost=crash_costs.severity_cost("K"),
        service_life_years=20,
        discount_rate=0.03,
    )
    assert record["annual_crashes_reduced"] == pytest.approx(0.1, abs=1e-6)


def test_unscorable_candidates_return_none():
    # no benefit source
    assert pb.score_candidate(
        {"treatment_cost": 100.0}, crash_cost=1.0, service_life_years=20, discount_rate=0.03
    ) is None
    # non-positive cost
    assert pb.score_candidate(
        {"treatment_cost": 0.0, "annual_crashes_reduced": 1.0},
        crash_cost=1.0,
        service_life_years=20,
        discount_rate=0.03,
    ) is None


def test_build_program_respects_budget_and_ranks_by_value_per_dollar():
    candidates = [
        # cheap, high value-per-dollar
        _candidate(project_id="cheap", treatment_cost=50_000.0, annual_crashes_reduced=0.1, svi_percentile=0.1),
        # expensive, lower value-per-dollar, but disadvantaged (equity boost)
        _candidate(project_id="big", treatment_cost=500_000.0, annual_crashes_reduced=0.3,
                   svi_percentile=0.9, disadvantaged=True),
        # tiny benefit -> screened out by BCR floor
        _candidate(project_id="weak", treatment_cost=400_000.0, annual_crashes_reduced=0.0001),
    ]
    program = pb.build_program(candidates, budget=100_000.0)
    assert program["status"] == "ok"
    assert program["candidate_count"] == 3
    # 'weak' fails the BCR>=1 screen
    assert program["eligible_count"] == 2
    # Only the cheap one fits in a 100k budget; 'big' is deferred (exceeds budget).
    selected_ids = [row["project_id"] for row in program["selected"]]
    assert selected_ids == ["cheap"]
    assert program["programmed_cost"] == pytest.approx(50_000.0)
    assert program["budget_remaining"] == pytest.approx(50_000.0)
    deferred_reasons = {row["project_id"]: row["defer_reason"] for row in program["deferred"]}
    assert deferred_reasons["big"] == "exceeds_budget"
    assert deferred_reasons["weak"] == "below_min_benefit_cost_ratio"


def test_build_program_greedy_skips_unaffordable_then_funds_cheaper():
    candidates = [
        _candidate(project_id="unaffordable", treatment_cost=900_000.0, annual_crashes_reduced=1.0,
                   svi_percentile=1.0, disadvantaged=True),
        _candidate(project_id="affordable", treatment_cost=80_000.0, annual_crashes_reduced=0.15,
                   svi_percentile=0.2),
    ]
    program = pb.build_program(candidates, budget=100_000.0)
    # 'unaffordable' ranks first by value but does not fit; greedy continues and funds 'affordable'.
    selected_ids = [row["project_id"] for row in program["selected"]]
    assert selected_ids == ["affordable"]
    assert program["selected"][0]["rank"] == 1


def test_equity_boost_changes_the_ranking():
    candidates = [
        _candidate(project_id="rich", treatment_cost=100_000.0, annual_crashes_reduced=0.12, svi_percentile=0.1),
        _candidate(project_id="poor", treatment_cost=100_000.0, annual_crashes_reduced=0.10,
                   svi_percentile=0.95, disadvantaged=True),
    ]
    # No equity weighting: the higher raw benefit wins.
    plain = pb.build_program(candidates, budget=100_000.0, svi_weight=0.0, disadvantaged_boost=0.0)
    assert plain["selected"][0]["project_id"] == "rich"
    # Strong equity weighting: the disadvantaged high-SVI project wins despite lower raw benefit.
    weighted = pb.build_program(candidates, budget=100_000.0, svi_weight=2.0, disadvantaged_boost=1.0)
    assert weighted["selected"][0]["project_id"] == "poor"


def test_program_rollups_and_equity_shares():
    candidates = [
        _candidate(project_id="a", treatment_cost=60_000.0, annual_crashes_reduced=0.2,
                   svi_percentile=0.8, disadvantaged=True),
        _candidate(project_id="b", treatment_cost=40_000.0, annual_crashes_reduced=0.15, svi_percentile=0.3),
    ]
    program = pb.build_program(candidates, budget=200_000.0)
    assert program["selected_count"] == 2
    assert program["programmed_cost"] == pytest.approx(100_000.0)
    assert program["annual_crashes_reduced"] == pytest.approx(0.35, abs=1e-6)
    assert program["program_benefit_cost_ratio"] is not None
    equity = program["equity"]
    assert equity["disadvantaged_project_count"] == 1
    assert equity["disadvantaged_cost"] == pytest.approx(60_000.0)
    assert equity["disadvantaged_cost_share"] == pytest.approx(0.6, abs=1e-4)


def test_no_budget_funds_everything_eligible():
    candidates = [_candidate(project_id=f"p{i}", treatment_cost=100_000.0, annual_crashes_reduced=0.1)
                  for i in range(5)]
    program = pb.build_program(candidates)  # no budget
    assert program["selected_count"] == 5
    assert program["budget"] is None
    assert program["budget_remaining"] is None


def test_max_projects_caps_selection():
    candidates = [_candidate(project_id=f"p{i}", treatment_cost=10_000.0, annual_crashes_reduced=0.1)
                  for i in range(5)]
    program = pb.build_program(candidates, max_projects=2)
    assert program["selected_count"] == 2
    assert all(row["defer_reason"] == "project_cap_reached" for row in program["deferred"])


def test_hsip_export_shape():
    candidates = [
        _candidate(project_id="seg1", segment_id="S-1", countermeasure="Road diet",
                   treatment_cost=100_000.0, annual_crashes_reduced=0.2, disadvantaged=True),
    ]
    program = pb.build_program(candidates, budget=200_000.0)
    rows = pb.hsip_export(program)
    assert len(rows) == 1
    row = rows[0]
    assert row["project_id"] == "seg1"
    assert row["location"] == "S-1"
    assert row["countermeasure"] == "Road diet"
    assert row["estimated_cost"] == pytest.approx(100_000.0)
    assert row["serves_disadvantaged_community"] is True
    assert row["funding_rank"] == 1


def test_hsip_export_location_from_latlon():
    candidates = [_candidate(project_id="pt", segment_id=None, lat=40.0, lon=-75.0,
                             treatment_cost=100_000.0, annual_crashes_reduced=0.2)]
    program = pb.build_program(candidates, budget=200_000.0)
    rows = pb.hsip_export(program)
    assert rows[0]["location"] == "40.0,-75.0"


def test_degrades_on_garbage_candidates():
    program = pb.build_program(
        [
            {"treatment_cost": "oops", "annual_crashes_reduced": 1.0},
            {"treatment_cost": float("nan"), "annual_crashes_reduced": 1.0},
            None,
            _candidate(project_id="ok", treatment_cost=50_000.0, annual_crashes_reduced=0.1),
        ],
        budget=100_000.0,
    )
    assert program["status"] == "ok"
    assert program["scored_count"] == 1
    assert program["unscorable_count"] == 2  # two bad dicts; None is not counted as a candidate
    assert program["selected"][0]["project_id"] == "ok"
