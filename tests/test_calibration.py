from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import calibration


def test_brier_score_known_values():
    assert calibration.brier_score([1, 0], [1.0, 0.0]) == pytest.approx(0.0)
    assert calibration.brier_score([1, 0], [0.5, 0.5]) == pytest.approx(0.25)
    assert calibration.brier_score([], []) == 0.0


def test_reliability_bins_partition_all_samples():
    y_prob = np.array([0.05, 0.15, 0.95, 0.5])
    y_true = np.array([0, 0, 1, 1])
    bins = calibration.reliability_bins(y_true, y_prob, n_bins=10)
    assert len(bins) == 10
    assert sum(b["count"] for b in bins) == 4
    # Top bin includes the 1.0 edge; the 0.95 sample lands there.
    assert bins[-1]["count"] == 1


def test_ece_zero_when_perfectly_calibrated():
    # Predict 0 for negatives and 1 for positives: each bin matches observed.
    y_true = np.array([0, 0, 1, 1])
    y_prob = np.array([0.0, 0.0, 1.0, 1.0])
    assert calibration.expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.0)


def test_ece_large_when_overconfident():
    # Predict 0.9 for everything but nothing actually happens.
    y_true = np.zeros(10)
    y_prob = np.full(10, 0.9)
    assert calibration.expected_calibration_error(y_true, y_prob, n_bins=10) == pytest.approx(0.9)


def test_summarize_shape():
    summary = calibration.summarize([0, 1, 0, 1], [0.2, 0.8, 0.3, 0.7], n_bins=5)
    assert summary["count"] == 4
    assert summary["positive_rate"] == pytest.approx(0.5)
    assert {"brier_score", "ece", "reliability_bins", "mean_predicted"} <= set(summary)
    assert len(summary["reliability_bins"]) == 5
