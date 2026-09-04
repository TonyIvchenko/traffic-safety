"""Human-readable Road Risk Advisory messages.

Composes a short, context-aware sentence from an advisory block plus optional
location and time-of-week context, e.g.

    "Elevated for Los Angeles — busier than usual for a Friday evening;
     contributing factors: wet roads. Slow down and increase following distance."

Pure and defensive: :func:`compose` never raises and always returns a string
that begins with the advisory level, whatever fields are missing or malformed.
"""

from __future__ import annotations

WEEKDAYS = (
    "Monday",
    "Tuesday",
    "Wednesday",
    "Thursday",
    "Friday",
    "Saturday",
    "Sunday",
)

# Relative-descriptor -> clause template ({t} = the time-of-week phrase).
_RELATIVE_CLAUSES = {
    "below normal": "quieter than usual for a {t}",
    "near normal": "about as risky as a typical {t}",
    "above normal": "busier than usual for a {t}",
    "well above normal": "much busier than usual for a {t}",
    "near this location's weekly peak": "near the riskiest this area gets",
}


_BASE_CAVEATS = (
    "The level is relative to this location's own typical week (a percentile), not "
    "an absolute crash probability.",
    "Metro regions are approximate bounding boxes for screening, not official MSA "
    "boundaries.",
    "Risk reflects historical crash patterns and is guidance, not a guarantee of "
    "safety.",
)


def caveats(*, mode="climatology", basis="relative_to_local_weekly_normal", live_incomplete=False) -> list[str]:
    """Honest limitations for an advisory response, tailored to its context."""
    items = list(_BASE_CAVEATS)
    if basis == "absolute":
        items[0] = (
            "This location has no usable weekly variation (out of coverage or flat "
            "climatology), so the level uses an absolute risk scale."
        )
    if str(mode) == "live":
        items.append(
            "Live conditions come from third-party weather providers and can lag or "
            "be unavailable."
        )
        if live_incomplete:
            items.append(
                "Some sample points could not be scored; the reading reflects the "
                "remaining points."
            )
    return items


def _daypart(hour: int) -> str:
    if hour < 5:
        return "overnight"
    if hour < 10:
        return "morning"
    if hour < 12:
        return "late morning"
    if hour < 17:
        return "afternoon"
    if hour < 21:
        return "evening"
    return "night"


def time_of_week_phrase(frame_idx) -> str:
    """A phrase like "Friday evening" for an hour-of-week index (0-167), or ""."""
    try:
        idx = int(frame_idx)
    except (TypeError, ValueError):
        return ""
    idx = max(0, min(24 * 7 - 1, idx))
    return f"{WEEKDAYS[idx // 24]} {_daypart(idx % 24)}"


def _clean_drivers(drivers) -> list[str]:
    if not isinstance(drivers, list):
        return []
    return [str(driver).strip() for driver in drivers if str(driver).strip()]


def compose(advisory_block, *, location_name=None, frame_idx=None) -> str:
    """A context-aware advisory sentence built from an advisory block.

    ``advisory_block`` is a dict from :func:`advisory.relative_advisory` /
    :func:`advisory.advisory` (``level``, ``advice``, ``basis``,
    ``relative_descriptor``, ``drivers``). ``location_name`` and ``frame_idx``
    add place and time-of-week context. Always returns a string starting with the
    level; returns "" only for a non-dict input.
    """
    if not isinstance(advisory_block, dict):
        return ""
    level = (str(advisory_block.get("level") or "").strip()) or "Advisory"
    advice = str(advisory_block.get("advice") or "").strip()
    basis = advisory_block.get("basis")
    descriptor = advisory_block.get("relative_descriptor")
    drivers = _clean_drivers(advisory_block.get("drivers"))
    location = str(location_name).strip() if location_name and str(location_name).strip() else None

    subject = f"{level} for {location}" if location else level

    clauses: list[str] = []
    if basis == "relative_to_local_weekly_normal" and descriptor:
        phrase = time_of_week_phrase(frame_idx) if frame_idx is not None else ""
        template = _RELATIVE_CLAUSES.get(str(descriptor))
        if template and phrase:
            clauses.append(template.format(t=phrase))
        elif template:
            clauses.append(template.replace(" for a {t}", "").format(t=""))
        elif phrase:
            clauses.append(f"{descriptor} for a {phrase}")
        else:
            clauses.append(str(descriptor))
    if drivers:
        clauses.append(f"contributing factors: {', '.join(drivers)}")

    message = f"{subject} — {'; '.join(clauses)}" if clauses else subject
    return f"{message}. {advice}" if advice else f"{message}."
