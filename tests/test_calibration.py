from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import calibration


def test_brier_score():
    assert calibration.brier_score([(0.0, 0.0), (1.0, 1.0)]) == 0.0  # perfect
    assert calibration.brier_score([(0.5, 1.0), (0.5, 0.0)]) == pytest.approx(0.25)
    assert calibration.brier_score([]) is None


def test_clean_pairs_drops_invalid():
    pairs = [
        (0.3, 1),
        (1.2, 1),        # prob out of range
        (0.4, 2),        # outcome not binary
        ("x", 0),        # non-numeric
        (float("nan"), 1),
        {"probability": 0.6, "outcome": 0},  # dict form
    ]
    clean = calibration.clean_pairs(pairs)
    assert clean == [(0.3, 1.0), (0.6, 0.0)]


def test_reliability_bins_shape_and_values():
    # All predictions 0.05 (bin 0) with a 50% observed rate.
    pairs = [(0.05, 1), (0.05, 0), (0.05, 1), (0.05, 0)]
    bins = calibration.reliability_bins(pairs, n_bins=10)
    assert len(bins) == 10
    first = bins[0]
    assert first["count"] == 4
    assert first["mean_predicted"] == pytest.approx(0.05)
    assert first["observed_frequency"] == pytest.approx(0.5)
    # Empty bins report None frequencies.
    assert bins[5]["count"] == 0 and bins[5]["observed_frequency"] is None


def test_probability_one_lands_in_last_bin():
    bins = calibration.reliability_bins([(1.0, 1)], n_bins=10)
    assert bins[9]["count"] == 1


def test_calibration_error_perfect_is_zero():
    # A well-calibrated bin: predicts 0.3 and the outcome occurs 3/10 of the time.
    pairs = [(0.3, 1)] * 3 + [(0.3, 0)] * 7
    assert calibration.calibration_error(pairs, n_bins=10) == pytest.approx(0.0)
    # Systematic over-confidence -> positive ECE.
    biased = [(0.9, 0)] * 10  # predicts 0.9 but never happens
    assert calibration.calibration_error(biased, n_bins=10) == pytest.approx(0.9)
    assert calibration.calibration_error([]) is None


def test_reliability_report():
    report = calibration.reliability_report([(0.2, 0), (0.8, 1)], n_bins=10)
    assert report["count"] == 2
    assert report["brier_score"] == pytest.approx(0.04)  # (0.2^2 + 0.2^2)/2
    assert report["n_bins"] == 10
    assert len(report["bins"]) == 10
    assert report["calibration_error"] == pytest.approx(0.2)
