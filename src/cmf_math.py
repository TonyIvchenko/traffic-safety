"""Crash Modification Factor (CMF) arithmetic.

A CMF multiplies expected crashes: ``after = before * CMF`` (CMF < 1 reduces
crashes). The crash reduction factor is ``CRF = 1 - CMF``.

Combining multiple treatments: the FHWA convention multiplies CMFs
(``method='multiply'``), which assumes independent effects and can overstate the
combined benefit. ``method='diminishing'`` applies the most effective treatment
fully and discounts each additional one (geometric ``decay``) — a conservative
screening default so stacked recommendations are not over-credited.
"""

from __future__ import annotations

DEFAULT_DECAY = 0.66  # each additional treatment counts ~2/3 as much


def _num(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number  # NaN -> default


def _clamp_cmf(cmf) -> float:
    """A CMF as a non-negative float; unparseable/NaN -> 1.0 (no effect)."""
    value = _num(cmf, default=1.0)
    return max(0.0, value)


def apply_cmf(expected_crashes, cmf) -> float:
    """Expected crashes remaining after applying a CMF."""
    return max(0.0, _num(expected_crashes)) * _clamp_cmf(cmf)


def crash_reduction_factor(cmf) -> float:
    """CRF = 1 - CMF (positive means crashes avoided)."""
    return round(1.0 - _clamp_cmf(cmf), 6)


def crashes_reduced(expected_crashes, cmf) -> float:
    expected = max(0.0, _num(expected_crashes))
    return round(expected - apply_cmf(expected, cmf), 6)


def combine_cmfs(cmfs, *, method: str = "multiply", decay: float = DEFAULT_DECAY) -> float:
    """Combined CMF for applying several treatments together.

    ``multiply``: product of CMFs (independence). ``diminishing``: most-effective
    treatment applied fully, each subsequent one discounted by ``decay**rank``.
    """
    values = [_clamp_cmf(c) for c in cmfs]
    if not values:
        return 1.0

    if method == "multiply":
        combined = 1.0
        for value in values:
            combined *= value
        return round(combined, 6)

    if method == "diminishing":
        combined = 1.0
        # Most effective first (lowest CMF); its full CRF is credited, later ones decayed.
        for rank, value in enumerate(sorted(values)):
            effective_crf = (1.0 - value) * (float(decay) ** rank)
            combined *= 1.0 - effective_crf
        return round(max(0.0, combined), 6)

    raise ValueError("method must be 'multiply' or 'diminishing'")


def combined_effect(
    expected_crashes,
    cmfs,
    *,
    method: str = "diminishing",
    decay: float = DEFAULT_DECAY,
) -> dict:
    """Expected/remaining/reduced crashes for a set of stacked treatments."""
    combined = combine_cmfs(cmfs, method=method, decay=decay)
    expected = max(0.0, _num(expected_crashes))
    remaining = apply_cmf(expected, combined)
    return {
        "combined_cmf": combined,
        "expected_crashes": round(expected, 4),
        "remaining_crashes": round(remaining, 4),
        "crashes_reduced": round(expected - remaining, 4),
    }
