from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import empirical_bayes as eb


def test_eb_weight_formula():
    # w = 1 / (1 + k * predicted_before)
    assert eb.eb_weight(10, 0.5) == pytest.approx(1 / 6)
    assert eb.eb_weight(0, 0.5) == pytest.approx(1.0)
    # Non-positive / unknown overdispersion -> rely wholly on the SPF (w = 1).
    assert eb.eb_weight(10, 0) == 1.0
    assert eb.eb_weight(10, None) == 1.0


def test_before_after_matches_hand_calculation():
    # pb=10, ob=12, pa=10, oa=6, k=0.5 -> CMF 0.48, SE 0.2186, 52% reduction.
    result = eb.before_after(
        [
            {
                "site_id": "A",
                "observed_before": 12,
                "observed_after": 6,
                "predicted_before": 10,
                "predicted_after": 10,
                "overdispersion": 0.5,
            }
        ]
    )
    assert result["status"] == "ok"
    assert result["cmf"] == pytest.approx(0.48, abs=1e-3)
    assert result["cmf_stderr"] == pytest.approx(0.2186, abs=1e-3)
    assert result["percent_reduction"] == pytest.approx(52.0, abs=0.1)
    assert result["significant_95"] is True  # CI upper 0.9085 < 1
    site = result["sites"][0]
    assert site["expected_after"] == pytest.approx(11.6667, abs=1e-3)
    assert site["weight"] == pytest.approx(1 / 6, abs=1e-4)


def test_no_treatment_effect_is_not_significant():
    result = eb.before_after(
        [
            {"observed_before": 10, "observed_after": 10, "predicted_before": 10,
             "predicted_after": 10, "overdispersion": 0.5}
        ]
    )
    assert result["significant_95"] is False
    assert 0.8 < result["cmf"] < 1.1


def test_multiple_sites_aggregate():
    sites = [
        {"observed_before": 20, "observed_after": 8, "predicted_before": 15,
         "predicted_after": 15, "overdispersion": 0.4},
        {"observed_before": 10, "observed_after": 5, "predicted_before": 8,
         "predicted_after": 9, "overdispersion": 0.4},
    ]
    result = eb.before_after(sites)
    assert result["status"] == "ok"
    assert result["site_count"] == 2
    assert result["cmf"] < 1.0  # both sites fell -> a reduction
    assert len(result["sites"]) == 2


def test_invalid_sites_are_skipped_and_empty_is_insufficient():
    # Negative observed and non-positive predicted_before are invalid -> skipped.
    result = eb.before_after(
        [
            {"observed_before": -1, "observed_after": 5, "predicted_before": 10,
             "predicted_after": 10, "overdispersion": 0.5},
            {"observed_before": 5, "observed_after": 5, "predicted_before": 0,
             "predicted_after": 10, "overdispersion": 0.5},
        ]
    )
    assert result["status"] == "insufficient"
    assert result["site_count"] == 0
