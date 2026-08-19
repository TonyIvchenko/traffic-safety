"""Road Risk Advisory: an "AQI for driving".

Maps a model risk score (0-1) to a public 5-level advisory (Low, Moderate,
Elevated, High, Extreme) with a color and a short human-readable message. Pure
and deterministic; drivers (short factor phrases) are folded into the message so
the advisory reads like "Elevated — ... Key factors: wet roads, Friday PM rush."
"""

from __future__ import annotations

import math

ADVISORY_LEVELS = (
    {"index": 1, "name": "Low", "color": "#1a9850", "advice": "Normal driving conditions."},
    {"index": 2, "name": "Moderate", "color": "#a6d96a", "advice": "Typical risk — stay alert."},
    {
        "index": 3, "name": "Elevated", "color": "#fdae61",
        "advice": "Heightened crash risk — slow down and increase following distance.",
    },
    {
        "index": 4, "name": "High", "color": "#f46d43",
        "advice": "High crash risk — avoid distractions and consider delaying non-essential trips.",
    },
    {
        "index": 5, "name": "Extreme", "color": "#d7191c",
        "advice": "Extreme crash risk — travel only if necessary and use extreme caution.",
    },
)
# Upper bounds for levels 1-4; a score >= the last bound is level 5 (Extreme).
DEFAULT_ADVISORY_THRESHOLDS = (0.10, 0.20, 0.35, 0.50)


def _clamp(risk_score) -> float:
    try:
        value = float(risk_score)
    except (TypeError, ValueError):
        return 0.0
    if math.isnan(value):
        return 0.0
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def advisory_level(risk_score, *, thresholds=DEFAULT_ADVISORY_THRESHOLDS) -> dict:
    """The advisory level (dict) for a risk score."""
    score = _clamp(risk_score)
    for index, upper in enumerate(thresholds):
        if score < float(upper):
            return dict(ADVISORY_LEVELS[index])
    return dict(ADVISORY_LEVELS[len(ADVISORY_LEVELS) - 1])


def advisory(risk_score, *, drivers=None, thresholds=DEFAULT_ADVISORY_THRESHOLDS) -> dict:
    """Full advisory for a risk score: level, color, advice, and a message.

    ``drivers`` is an optional list of short factor phrases folded into the
    message (e.g. ``["wet roads", "Friday PM rush"]``).
    """
    score = _clamp(risk_score)
    level = advisory_level(score, thresholds=thresholds)
    factors = [str(driver).strip() for driver in (drivers or []) if str(driver).strip()]

    message = f"{level['name']} — {level['advice']}"
    if factors:
        message = f"{message} Key factors: {', '.join(factors)}."
    return {
        "risk_score": round(score, 4),
        "level": level["name"],
        "level_index": level["index"],
        "color": level["color"],
        "advice": level["advice"],
        "drivers": factors,
        "message": message,
    }


def advisory_scale(*, thresholds=DEFAULT_ADVISORY_THRESHOLDS) -> list[dict]:
    """The 5-level scale with each level's score range (for discovery/legends)."""
    bounds = [0.0, *[float(t) for t in thresholds], 1.0]
    scale = []
    for index, level in enumerate(ADVISORY_LEVELS):
        scale.append(
            {
                "index": level["index"],
                "name": level["name"],
                "color": level["color"],
                "min_risk": round(bounds[index], 4),
                "max_risk": round(bounds[index + 1], 4),
            }
        )
    return scale
