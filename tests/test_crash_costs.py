from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs as cc


def test_kabco_table_shape_and_ordering():
    assert set(cc.KABCO_COMPREHENSIVE_COST) == set(cc.KABCO_LEVELS)
    costs = [cc.KABCO_COMPREHENSIVE_COST[level] for level in cc.KABCO_LEVELS]
    assert costs == sorted(costs, reverse=True)  # K is most expensive, O least


def test_severity_cost_is_case_insensitive_and_defaults_zero():
    assert cc.severity_cost("k") == cc.KABCO_COMPREHENSIVE_COST["K"]
    assert cc.severity_cost("A") == 655_000.0
    assert cc.severity_cost("unknown") == 0.0


def test_severity_weight_relative_to_fatal():
    assert cc.severity_weight("K") == pytest.approx(1.0)
    assert cc.severity_weight("A") < 0.1
    assert cc.severity_weight("K") > cc.severity_weight("B")


def test_expected_annual_cost_sums_counts():
    total = cc.expected_annual_cost({"K": 2, "A": 10})
    assert total == pytest.approx(2 * 11_295_400.0 + 10 * 655_000.0)
    assert cc.expected_annual_cost({}) == 0.0


def test_present_value_uniform_series():
    # discount rate 0 -> straight multiplication.
    assert cc.present_value(1000.0, 20, 0.0) == pytest.approx(20_000.0)
    # 3% over 20 years -> uniform-series present-worth factor ~= 14.8775.
    assert cc.present_value(1000.0, 20, 0.03) == pytest.approx(14877.5, abs=1.0)


def test_benefit_cost_ratio_and_net():
    result = cc.benefit_cost(100_000.0, 500_000.0, service_life_years=20, discount_rate=0.03)
    assert result["present_value_benefit"] == pytest.approx(1_487_747.0, abs=50.0)
    assert result["benefit_cost_ratio"] > 1.0
    assert result["net_benefit"] > 0.0
    assert result["cost_basis"] == cc.COST_SOURCE


def test_benefit_cost_zero_treatment_cost():
    result = cc.benefit_cost(100_000.0, 0.0)
    assert result["benefit_cost_ratio"] is None
    assert result["net_benefit"] == result["present_value_benefit"]
