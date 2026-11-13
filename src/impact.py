"""Impact dashboard rollups: the public-good headline for the whole service.

Synthesizes what the rest of the stack can quantify into one summary an agency or
the public can read at a glance:

* **Crashes averted** — if the recommended FHWA countermeasures were applied
  across the High Injury Network, how many crashes per year (and over the
  treatments' service life) would be prevented.
* **Investment targeted** — the construction dollars those treatments represent,
  the present value of the crashes they would avoid (FHWA comprehensive costs),
  and the resulting benefit-cost ratio.
* **Equity gaps** — how concentrated crashes are in disadvantaged communities, and
  an estimate of how much of the avertable harm falls there, so progress on the
  gap can be tracked.

This is a *screening* rollup: it reuses the same representative CMFs and crash
costs as the countermeasure and grant tools, and the equity split is an estimate
from the network-wide disadvantaged crash share, not a per-project join. Pure and
defensive — missing pieces degrade to zeros/None rather than raising, and nothing
here performs I/O.
"""

from __future__ import annotations

import math

import crash_costs

METHOD = "impact_screening_rollup"


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonneg(value):
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def prevention_impact(
    treatments,
    *,
    crash_cost=None,
    service_life_years=None,
    discount_rate=None,
) -> dict:
    """Aggregate the prevention potential of a set of recommended treatments.

    ``treatments`` is an iterable of dicts carrying ``annual_crashes_reduced`` and
    ``treatment_cost`` (the shape of a countermeasure ``recommended`` block). Sums
    the crashes averted and the investment, values the averted crashes at the FHWA
    comprehensive fatal-crash cost discounted over the service life, and returns a
    program-level benefit-cost.
    """
    unit_cost = crash_costs.severity_cost("K") if crash_cost is None else float(crash_cost)
    service_life = (
        crash_costs.DEFAULT_SERVICE_LIFE_YEARS if service_life_years is None else int(service_life_years)
    )
    rate = crash_costs.DEFAULT_DISCOUNT_RATE if discount_rate is None else float(discount_rate)

    treated = 0
    annual_crashes = 0.0
    investment = 0.0
    for treatment in treatments or []:
        if not isinstance(treatment, dict):
            continue
        reduced = _nonneg(treatment.get("annual_crashes_reduced"))
        if reduced is None:
            continue
        treated += 1
        annual_crashes += reduced
        investment += _nonneg(treatment.get("treatment_cost")) or 0.0

    annual_benefit = annual_crashes * unit_cost
    present_value = crash_costs.present_value(annual_benefit, service_life, rate)
    investment = round(investment, 2)
    present_value = round(present_value, 2)
    return {
        "segments_with_treatment": treated,
        "annual_crashes_averted": round(annual_crashes, 3),
        "crashes_averted_service_life": round(annual_crashes * service_life, 2),
        "investment_required": investment,
        "annual_benefit": round(annual_benefit, 2),
        "present_value_benefit": present_value,
        "net_benefit": round(present_value - investment, 2),
        "benefit_cost_ratio": round(present_value / investment, 3) if investment > 0 else None,
        "service_life_years": service_life,
        "discount_rate": rate,
        "fatal_crash_cost": round(unit_cost, 2),
        "cost_basis": crash_costs.COST_SOURCE,
    }


def equity_impact(annual_crashes_averted, equity_disparity) -> dict:
    """Equity framing of the impact, from the network equity disparity.

    Applies the network-wide disadvantaged crash share to the avertable crashes as
    an estimate of how much of the harm falls in disadvantaged communities, and
    carries the disparity ratios so the gap can be tracked over time.
    """
    disparity = equity_disparity if isinstance(equity_disparity, dict) else {}
    crash_share = _finite(disparity.get("crash_share_disadvantaged"))
    averted = _nonneg(annual_crashes_averted) or 0.0
    estimated = round(averted * crash_share, 3) if crash_share is not None else None
    return {
        "disadvantaged_crash_share": crash_share,
        "crash_disparity_ratio": disparity.get("crash_disparity_ratio"),
        "risk_disparity_ratio": disparity.get("risk_disparity_ratio"),
        "segment_share_disadvantaged": disparity.get("segment_share_disadvantaged"),
        "estimated_disadvantaged_crashes_averted_annual": estimated,
        "basis": (
            "network-wide disadvantaged crash share applied to avertable crashes "
            "(an estimate, not a per-project join)"
        ),
    }


def _headline(prevention: dict, equity: dict | None) -> str:
    annual = prevention.get("annual_crashes_averted") or 0.0
    bcr = prevention.get("benefit_cost_ratio")
    parts = [f"~{annual:g} crashes/year avertable across {prevention.get('segments_with_treatment', 0)} treated segments"]
    if bcr is not None:
        parts.append(f"at a {bcr:g}:1 benefit-cost")
    if equity is not None and equity.get("disadvantaged_crash_share") is not None:
        parts.append(f"{round(100 * equity['disadvantaged_crash_share'])}% of crashes in disadvantaged communities")
    return "; ".join(parts) + "."


def impact_summary(
    treatments,
    *,
    equity_disparity=None,
    coverage=None,
    crash_cost=None,
    service_life_years=None,
    discount_rate=None,
) -> dict:
    """Compose the full impact dashboard rollup from its parts."""
    prevention = prevention_impact(
        treatments,
        crash_cost=crash_cost,
        service_life_years=service_life_years,
        discount_rate=discount_rate,
    )
    equity = equity_impact(prevention["annual_crashes_averted"], equity_disparity) if equity_disparity is not None else None
    result = {
        "status": "ok",
        "method": METHOD,
        "headline": _headline(prevention, equity),
        "prevention": prevention,
    }
    if equity is not None:
        result["equity"] = equity
    if coverage is not None:
        result["coverage"] = dict(coverage)
    return result
