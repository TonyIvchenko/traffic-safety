"""Infer a likely crash-type profile for a road segment from its attributes.

Countermeasure matching needs to know *what kinds* of crashes a corridor is prone
to, not just how many. This module maps roadway context (rural/urban, functional
tier from MTFCC / functional system) plus optional VRU, nighttime, and wet-road
shares to a profile of crash-type relevance weights in [0, 1], using the same
crash-type vocabulary as data/reference/countermeasures.json.

It is a screening heuristic, not a fitted model — weights express relative
relevance so downstream matching can pick applicable treatments.
"""

from __future__ import annotations

CRASH_TYPES = [
    "run_off_road",
    "head_on",
    "sideswipe",
    "angle",
    "rear_end",
    "left_turn",
    "pedestrian",
    "bicycle",
    "nighttime",
    "wet_road",
]

# Base relevance weights per roadway context (missing types default to 0).
_BASE = {
    "rural_highway": {"run_off_road": 0.9, "head_on": 0.5, "sideswipe": 0.3, "rear_end": 0.3, "angle": 0.2},
    "rural_arterial": {"run_off_road": 0.8, "head_on": 0.5, "angle": 0.4, "rear_end": 0.3, "pedestrian": 0.2},
    "rural_local": {"run_off_road": 0.7, "angle": 0.4, "head_on": 0.3, "pedestrian": 0.3},
    "urban_highway": {"rear_end": 0.9, "sideswipe": 0.6, "run_off_road": 0.4, "angle": 0.3},
    "urban_arterial": {"angle": 0.8, "rear_end": 0.7, "pedestrian": 0.5, "left_turn": 0.4, "sideswipe": 0.3},
    "urban_local": {"pedestrian": 0.7, "angle": 0.6, "bicycle": 0.4, "rear_end": 0.3, "run_off_road": 0.2},
}
_LOCAL_MTFCC = {"S1400", "S1500", "S1630", "S1640", "S1740", "S1780"}


def _int(value):
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _num(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number  # NaN -> default


def _clamp(value: float) -> float:
    return 0.0 if value < 0.0 else 1.0 if value > 1.0 else value


def road_tier(mtfcc=None, func_sys=None) -> str:
    """'highway', 'arterial', or 'local' from MTFCC (preferred) or functional system."""
    code = str(mtfcc or "").upper()
    if code == "S1100":
        return "highway"
    if code == "S1200":
        return "arterial"
    if code in _LOCAL_MTFCC:
        return "local"
    func = _int(func_sys)
    if func in (1, 2):
        return "highway"
    if func in (3, 4):
        return "arterial"
    if func in (5, 6, 7):
        return "local"
    return "arterial"


def is_urban(rur_urb=None) -> bool:
    """True for urban context; unknown defaults to urban (where VRU risk concentrates)."""
    value = str(rur_urb).strip().lower()
    if value in ("2", "urban", "u"):
        return True
    if value in ("1", "rural", "r"):
        return False
    return True


def context_class(rur_urb=None, mtfcc=None, func_sys=None) -> str:
    return f"{'urban' if is_urban(rur_urb) else 'rural'}_{road_tier(mtfcc, func_sys)}"


def crash_type_profile(
    *,
    rur_urb=None,
    mtfcc=None,
    func_sys=None,
    vru_share: float = 0.0,
    nighttime_share=None,
    wet_share=None,
) -> dict[str, float]:
    """Crash-type relevance weights in [0, 1] for a segment.

    ``vru_share`` (fraction of crashes involving a pedestrian/cyclist) raises the
    pedestrian/bicycle weights; ``nighttime_share`` / ``wet_share`` set the
    condition weights when known.
    """
    profile = {crash_type: 0.0 for crash_type in CRASH_TYPES}
    profile.update(_BASE.get(context_class(rur_urb, mtfcc, func_sys), _BASE["urban_arterial"]))

    vru = _clamp(_num(vru_share))
    if vru > 0.0:
        profile["pedestrian"] = max(profile["pedestrian"], vru)
        profile["bicycle"] = max(profile["bicycle"], vru * 0.6)
    if nighttime_share is not None:
        profile["nighttime"] = _clamp(_num(nighttime_share))
    if wet_share is not None:
        profile["wet_road"] = _clamp(_num(wet_share))

    return {crash_type: round(_clamp(weight), 4) for crash_type, weight in profile.items()}


def dominant_crash_types(profile: dict, *, top_n: int = 3, min_weight: float = 0.2) -> list[str]:
    """The most-relevant crash types (weight >= ``min_weight``), highest first."""
    ranked = sorted(profile.items(), key=lambda item: item[1], reverse=True)
    return [crash_type for crash_type, weight in ranked if weight >= min_weight][: int(top_n)]
