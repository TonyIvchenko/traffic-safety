"""Pure calibration / reliability metrics for binary risk probabilities.

NumPy-only so they can be unit-tested without the training pipeline and reused
by scripts/evaluate_model.py.
"""

from __future__ import annotations

import numpy as np


def brier_score(y_true, y_prob) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    if y_true.size == 0:
        return 0.0
    return float(np.mean((y_prob - y_true) ** 2))


def reliability_bins(y_true, y_prob, n_bins: int = 10) -> list[dict]:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    bins: list[dict] = []
    for index in range(n_bins):
        lower, upper = float(edges[index]), float(edges[index + 1])
        if index == n_bins - 1:
            mask = (y_prob >= lower) & (y_prob <= upper)
        else:
            mask = (y_prob >= lower) & (y_prob < upper)
        count = int(mask.sum())
        bins.append(
            {
                "bin_lower": lower,
                "bin_upper": upper,
                "count": count,
                "mean_predicted": float(y_prob[mask].mean()) if count else None,
                "mean_observed": float(y_true[mask].mean()) if count else None,
            }
        )
    return bins


def expected_calibration_error(y_true, y_prob, n_bins: int = 10) -> float:
    y_true = np.asarray(y_true, dtype=np.float64)
    total = y_true.size
    if total == 0:
        return 0.0
    ece = 0.0
    for bucket in reliability_bins(y_true, y_prob, n_bins):
        if bucket["count"]:
            ece += (bucket["count"] / total) * abs(
                bucket["mean_observed"] - bucket["mean_predicted"]
            )
    return float(ece)


def summarize(y_true, y_prob, n_bins: int = 10) -> dict:
    y_true = np.asarray(y_true, dtype=np.float64)
    y_prob = np.asarray(y_prob, dtype=np.float64)
    count = int(y_true.size)
    return {
        "count": count,
        "positive_rate": float(y_true.mean()) if count else 0.0,
        "mean_predicted": float(y_prob.mean()) if count else 0.0,
        "brier_score": brier_score(y_true, y_prob),
        "ece": expected_calibration_error(y_true, y_prob, n_bins),
        "reliability_bins": reliability_bins(y_true, y_prob, n_bins),
    }
