"""Calibration and reliability metrics for probabilistic risk predictions.

Given ``(predicted_probability, outcome)`` pairs — outcome 0/1 — computes the
Brier score, a reliability table (predictions binned with their observed
frequency), and the Expected Calibration Error. These sit atop the model backtest
so consumers can judge whether a risk score of 0.3 really means ~30% of the time.

Pure and defensive: pairs with a non-numeric/out-of-range probability or a
non-binary outcome are dropped rather than raising. Pairs may be ``(prob, outcome)``
tuples or ``{"probability", "outcome"}`` dicts.
"""

from __future__ import annotations

import math


def _extract(pair):
    if isinstance(pair, dict):
        prob = pair.get("probability", pair.get("prob"))
        outcome = pair.get("outcome", pair.get("label"))
    elif isinstance(pair, (list, tuple)) and len(pair) >= 2:
        prob, outcome = pair[0], pair[1]
    else:
        return None
    try:
        probability = float(prob)
        observed = float(outcome)
    except (TypeError, ValueError):
        return None
    if not (math.isfinite(probability) and math.isfinite(observed)):
        return None
    if not (0.0 <= probability <= 1.0) or observed not in (0.0, 1.0):
        return None
    return (probability, observed)


def clean_pairs(pairs) -> list[tuple]:
    """The valid ``(probability, outcome)`` pairs (invalid ones dropped)."""
    return [pair for pair in (_extract(item) for item in (pairs or [])) if pair is not None]


def brier_score(pairs) -> float | None:
    """Mean squared error of probability vs outcome; None if there are no pairs."""
    clean = clean_pairs(pairs)
    if not clean:
        return None
    return sum((probability - observed) ** 2 for probability, observed in clean) / len(clean)


def reliability_bins(pairs, *, n_bins: int = 10) -> list[dict]:
    """Reliability table: each equal-width probability bin with its observed rate.

    Returns all ``n_bins`` bins (empty ones have count 0 and
    ``observed_frequency``/``mean_predicted`` None), each with ``p_low``/``p_high``.
    """
    bins = max(1, int(n_bins))
    clean = clean_pairs(pairs)
    buckets: list[dict] = [
        {
            "bin": index,
            "p_low": round(index / bins, 4),
            "p_high": round((index + 1) / bins, 4),
            "count": 0,
            "_pred_sum": 0.0,
            "_obs_sum": 0.0,
        }
        for index in range(bins)
    ]
    for probability, observed in clean:
        index = min(bins - 1, int(probability * bins))
        bucket = buckets[index]
        bucket["count"] += 1
        bucket["_pred_sum"] += probability
        bucket["_obs_sum"] += observed

    table = []
    for bucket in buckets:
        count = bucket["count"]
        table.append(
            {
                "bin": bucket["bin"],
                "p_low": bucket["p_low"],
                "p_high": bucket["p_high"],
                "count": count,
                "mean_predicted": round(bucket["_pred_sum"] / count, 4) if count else None,
                "observed_frequency": round(bucket["_obs_sum"] / count, 4) if count else None,
            }
        )
    return table


def calibration_error(pairs, *, n_bins: int = 10) -> float | None:
    """Expected Calibration Error: count-weighted mean |predicted - observed|."""
    clean = clean_pairs(pairs)
    if not clean:
        return None
    total = len(clean)
    error = 0.0
    for bucket in reliability_bins(clean, n_bins=n_bins):
        if bucket["count"]:
            error += (bucket["count"] / total) * abs(
                bucket["mean_predicted"] - bucket["observed_frequency"]
            )
    return error


def reliability_report(pairs, *, n_bins: int = 10) -> dict:
    """A combined report: sample count, Brier score, ECE, and the reliability table."""
    clean = clean_pairs(pairs)
    return {
        "count": len(clean),
        "brier_score": brier_score(clean),
        "calibration_error": calibration_error(clean, n_bins=n_bins),
        "n_bins": max(1, int(n_bins)),
        "bins": reliability_bins(clean, n_bins=n_bins),
    }
