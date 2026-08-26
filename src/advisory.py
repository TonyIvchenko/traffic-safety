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

# Percentile bands for the RELATIVE advisory: a value below the 50th percentile of
# its reference is Low, then Moderate (<70th), Elevated (<85th), High (<95th), and
# Extreme (>=95th). These express "how does now compare to this place's normal".
RELATIVE_ADVISORY_THRESHOLDS = (0.50, 0.70, 0.85, 0.95)


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


# --------------------------------------------------------------------------- #
# Relative advisory: risk expressed against a reference distribution
# --------------------------------------------------------------------------- #


def _num_or_none(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or math.isinf(number) else number


def percentile_of(value, reference) -> float | None:
    """Empirical percentile (0..1) of ``value`` within ``reference``.

    Uses the midrank convention (fraction strictly below plus half the ties) so a
    flat/degenerate reference maps a value to 0.5 rather than jumping to an
    extreme. Returns None when ``value`` is non-numeric or the reference has no
    finite values (the caller can then fall back to an absolute reading).
    """
    numbers = [n for n in (_num_or_none(item) for item in (reference or [])) if n is not None]
    target = _num_or_none(value)
    if target is None or not numbers:
        return None
    below = sum(1 for n in numbers if n < target)
    equal = sum(1 for n in numbers if n == target)
    return (below + 0.5 * equal) / len(numbers)


def _level_for_fraction(fraction: float, thresholds) -> dict:
    for index, upper in enumerate(thresholds):
        if fraction < float(upper):
            return dict(ADVISORY_LEVELS[index])
    return dict(ADVISORY_LEVELS[len(ADVISORY_LEVELS) - 1])


def relative_descriptor(percentile) -> str:
    """A short phrase for where a percentile sits versus normal."""
    fraction = _num_or_none(percentile)
    if fraction is None:
        return "compared to typical is unknown"
    if fraction < 0.50:
        return "below normal"
    if fraction < 0.70:
        return "near normal"
    if fraction < 0.85:
        return "above normal"
    if fraction < 0.95:
        return "well above normal"
    return "near this location's weekly peak"


def relative_advisory(
    value,
    reference,
    *,
    thresholds=RELATIVE_ADVISORY_THRESHOLDS,
    drivers=None,
    absolute_thresholds=DEFAULT_ADVISORY_THRESHOLDS,
) -> dict:
    """Advisory for ``value`` expressed relative to a ``reference`` distribution.

    The level is chosen by ``value``'s percentile within ``reference`` (e.g. a
    location's own 168-hour weekly climatology), so the scale discriminates by
    time and place instead of saturating on absolute risk. When the reference is
    empty/degenerate the result falls back to the absolute :func:`advisory`
    (``basis`` == ``"absolute"``, ``percentile`` None).
    """
    score = _clamp(value)
    factors = [str(driver).strip() for driver in (drivers or []) if str(driver).strip()]
    pct = percentile_of(value, reference)

    if pct is None:
        result = advisory(score, drivers=factors, thresholds=absolute_thresholds)
        result["percentile"] = None
        result["basis"] = "absolute"
        result["relative_descriptor"] = None
        return result

    level = _level_for_fraction(pct, thresholds)
    descriptor = relative_descriptor(pct)
    message = f"{level['name']} — {level['advice']} Currently {descriptor} for this location."
    if factors:
        message = f"{message} Key factors: {', '.join(factors)}."
    return {
        "risk_score": round(score, 4),
        "percentile": round(pct, 4),
        "level": level["name"],
        "level_index": level["index"],
        "color": level["color"],
        "advice": level["advice"],
        "basis": "relative_to_local_weekly_normal",
        "relative_descriptor": descriptor,
        "drivers": factors,
        "message": message,
    }
