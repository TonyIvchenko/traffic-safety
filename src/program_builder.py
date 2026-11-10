"""Equity-weighted benefit-cost program builder.

Turns a pool of candidate safety projects into a funded *program*: the bundle an
agency would submit for an SS4A (Safe Streets and Roads for All) or HSIP (Highway
Safety Improvement Program) application. Each candidate is valued the HSIP way —
the crashes it would avoid, priced at FHWA comprehensive crash costs and
discounted over the treatment's service life (``crash_costs``) — screened for
cost-effectiveness, and then ranked and selected under a budget.

The ranking is *equity-weighted*: a project's present-value benefit is multiplied
by ``1 + svi_weight*svi + disadvantaged_boost`` so that, dollar for dollar,
projects serving more socially vulnerable (CDC/ATSDR SVI) and Justice40
disadvantaged communities rise up the list. With both weights at zero the ranking
reduces to the plain benefit-cost ratio, so the equity lens is explicit and
tunable rather than baked in.

Selection is a transparent greedy knapsack: eligible projects are ordered by
equity-weighted benefit per dollar and funded down the list until the budget is
exhausted (a project that does not fit is skipped, not a hard stop, so cheaper
lower-ranked projects can still be funded). The result carries program-level
rollups — total programmed cost, present-value benefit, program benefit-cost
ratio, crashes reduced, and how much of the money reaches disadvantaged
communities — plus an :func:`hsip_export` row shape ready to drop into an
application.

Pure and defensive: unscorable candidates are skipped, degenerate inputs never
divide by zero, and nothing here performs I/O.
"""

from __future__ import annotations

import math

import cmf_math
import crash_costs

DEFAULT_SVI_WEIGHT = 1.0  # how strongly SVI percentile boosts priority
DEFAULT_DISADVANTAGED_BOOST = 0.5  # extra boost for a Justice40 disadvantaged tract
DEFAULT_MIN_BENEFIT_COST_RATIO = 1.0  # HSIP cost-effectiveness screen (BCR >= 1)
SELECTION_METRIC = "equity_weighted_benefit_cost_ratio"


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonneg(value):
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def _pos(value):
    number = _finite(value)
    return number if number is not None and number > 0 else None


def _unit(value):
    """An SVI percentile clamped to [0, 1]; None when unparseable/non-finite."""
    number = _finite(value)
    if number is None:
        return None
    return min(1.0, max(0.0, number))


def equity_weight(
    svi_percentile,
    disadvantaged,
    *,
    svi_weight: float = DEFAULT_SVI_WEIGHT,
    disadvantaged_boost: float = DEFAULT_DISADVANTAGED_BOOST,
) -> float:
    """Benefit multiplier ``1 + svi_weight*svi + boost*disadvantaged``.

    Unknown SVI adds no boost (we do not inflate communities we cannot assess).
    Negative weights are floored at 0 so the multiplier never drops below 1.
    """
    svi = _unit(svi_percentile) or 0.0
    weight = max(0.0, _finite(svi_weight) or 0.0)
    boost = max(0.0, _finite(disadvantaged_boost) or 0.0)
    return 1.0 + weight * svi + (boost if bool(disadvantaged) else 0.0)


def _annual_crashes_reduced(candidate: dict):
    """Average annual crashes a candidate would avoid.

    Prefers an explicit ``annual_crashes_reduced``; otherwise derives it from an
    ``expected_annual_crashes`` count and a ``cmf`` (crash modification factor).
    Returns None when neither is supplied (the candidate is unscorable).
    """
    direct = _nonneg(candidate.get("annual_crashes_reduced"))
    if direct is not None:
        return direct
    expected = _nonneg(candidate.get("expected_annual_crashes"))
    cmf = _pos(candidate.get("cmf"))
    if expected is not None and cmf is not None:
        # crashes_reduced may be negative if cmf >= 1 (no safety benefit); kept
        # as-is so an ineffective treatment is screened out by its BCR downstream.
        return cmf_math.crashes_reduced(expected, cmf)
    return None


_PASSTHROUGH_FIELDS = (
    "project_id",
    "name",
    "segment_id",
    "lat",
    "lon",
    "countermeasure_id",
    "countermeasure",
)


def score_candidate(
    candidate: dict,
    *,
    crash_cost: float,
    service_life_years: int,
    discount_rate: float,
    svi_weight: float = DEFAULT_SVI_WEIGHT,
    disadvantaged_boost: float = DEFAULT_DISADVANTAGED_BOOST,
) -> dict | None:
    """Value one candidate project (benefit-cost + equity weighting), or None.

    Returns None when the candidate has no positive treatment cost or no benefit
    source (so it cannot be programmed). The returned record carries the HSIP
    benefit-cost fields plus the equity multiplier, the equity-weighted
    present-value benefit, and its per-dollar efficiency.
    """
    cost = _pos(candidate.get("treatment_cost"))
    reduced = _annual_crashes_reduced(candidate)
    if cost is None or reduced is None:
        return None

    unit_cost = _pos(candidate.get("crash_cost")) or float(crash_cost)
    annual_benefit = reduced * unit_cost
    economics = crash_costs.benefit_cost(annual_benefit, cost, service_life_years, discount_rate)

    svi = _unit(candidate.get("svi_percentile"))
    disadvantaged = bool(candidate.get("disadvantaged"))
    multiplier = equity_weight(
        svi, disadvantaged, svi_weight=svi_weight, disadvantaged_boost=disadvantaged_boost
    )
    present_value = float(economics["present_value_benefit"])
    equity_weighted_benefit = present_value * multiplier

    record = {field: candidate.get(field) for field in _PASSTHROUGH_FIELDS if candidate.get(field) is not None}
    record.update(
        {
            "treatment_cost": round(cost, 2),
            "annual_crashes_reduced": round(reduced, 4),
            "crash_cost_each": round(unit_cost, 2),
            "annual_benefit": round(annual_benefit, 2),
            "present_value_benefit": present_value,
            "net_benefit": economics["net_benefit"],
            "benefit_cost_ratio": economics["benefit_cost_ratio"],
            "svi_percentile": svi,
            "disadvantaged": disadvantaged,
            "equity_multiplier": round(multiplier, 4),
            "equity_weighted_benefit": round(equity_weighted_benefit, 2),
            "equity_weighted_bcr": round(equity_weighted_benefit / cost, 4) if cost > 0 else None,
        }
    )
    return record


def _program_equity(selected: list[dict], programmed_cost: float) -> dict:
    """How much of the programmed money and how many projects reach disadvantaged
    communities, plus the funded mean SVI."""
    count = len(selected)
    disadvantaged = [row for row in selected if row.get("disadvantaged")]
    disadvantaged_cost = round(sum(float(row["treatment_cost"]) for row in disadvantaged), 2)
    known_svi = [row["svi_percentile"] for row in selected if row.get("svi_percentile") is not None]
    return {
        "disadvantaged_project_count": len(disadvantaged),
        "disadvantaged_project_share": round(len(disadvantaged) / count, 4) if count else None,
        "disadvantaged_cost": disadvantaged_cost,
        "disadvantaged_cost_share": round(disadvantaged_cost / programmed_cost, 4)
        if programmed_cost > 0
        else None,
        "mean_svi_selected": round(sum(known_svi) / len(known_svi), 4) if known_svi else None,
    }


def build_program(
    candidates,
    *,
    budget=None,
    svi_weight: float = DEFAULT_SVI_WEIGHT,
    disadvantaged_boost: float = DEFAULT_DISADVANTAGED_BOOST,
    min_benefit_cost_ratio: float = DEFAULT_MIN_BENEFIT_COST_RATIO,
    service_life_years=None,
    discount_rate=None,
    crash_cost=None,
    max_projects=None,
) -> dict:
    """Assemble a funded safety program from candidate projects.

    Scores every candidate, screens out those below ``min_benefit_cost_ratio``,
    ranks the rest by equity-weighted benefit per dollar, and funds down the list
    under ``budget`` (and an optional ``max_projects`` cap). Returns the selected
    and deferred projects plus program-level rollups. ``budget``/``crash_cost`` of
    None mean "no budget cap" / "use the FHWA fatal-crash cost".
    """
    service_life = (
        crash_costs.DEFAULT_SERVICE_LIFE_YEARS if service_life_years is None else int(service_life_years)
    )
    rate = crash_costs.DEFAULT_DISCOUNT_RATE if discount_rate is None else float(discount_rate)
    unit_cost = crash_costs.severity_cost("K") if crash_cost is None else float(crash_cost)
    budget_value = _pos(budget)
    cap = int(max_projects) if max_projects is not None and int(max_projects) > 0 else None
    min_bcr = max(0.0, _finite(min_benefit_cost_ratio) or 0.0)

    candidate_list = [c for c in (candidates or []) if isinstance(c, dict)]
    scored: list[dict] = []
    for candidate in candidate_list:
        record = score_candidate(
            candidate,
            crash_cost=unit_cost,
            service_life_years=service_life,
            discount_rate=rate,
            svi_weight=svi_weight,
            disadvantaged_boost=disadvantaged_boost,
        )
        if record is not None:
            scored.append(record)

    eligible: list[dict] = []
    screened_out: list[dict] = []
    for record in scored:
        bcr = record.get("benefit_cost_ratio")
        if bcr is not None and bcr >= min_bcr:
            eligible.append(record)
        else:
            screened_out.append({**record, "status": "ineligible", "defer_reason": "below_min_benefit_cost_ratio"})

    # Greedy knapsack: best equity-weighted value per dollar first.
    eligible.sort(
        key=lambda r: (r.get("equity_weighted_bcr") or 0.0, r.get("equity_weighted_benefit") or 0.0),
        reverse=True,
    )

    selected: list[dict] = []
    unfunded: list[dict] = []
    programmed_cost = 0.0
    for record in eligible:
        cost = float(record["treatment_cost"])
        over_budget = budget_value is not None and (programmed_cost + cost) > budget_value
        over_cap = cap is not None and len(selected) >= cap
        if over_budget or over_cap:
            reason = "exceeds_budget" if over_budget else "project_cap_reached"
            unfunded.append({**record, "status": "deferred", "defer_reason": reason})
            continue
        programmed_cost += cost
        selected.append(
            {
                **record,
                "status": "selected",
                "rank": len(selected) + 1,
                "cumulative_cost": round(programmed_cost, 2),
            }
        )

    programmed_cost = round(programmed_cost, 2)
    present_value_benefit = round(sum(float(row["present_value_benefit"]) for row in selected), 2)
    annual_crashes_reduced = round(sum(float(row["annual_crashes_reduced"]) for row in selected), 4)
    deferred = unfunded + screened_out

    return {
        "status": "ok",
        "method": "equity_weighted_benefit_cost_program",
        "selection_metric": SELECTION_METRIC,
        "parameters": {
            "budget": budget_value,
            "svi_weight": float(svi_weight),
            "disadvantaged_boost": float(disadvantaged_boost),
            "min_benefit_cost_ratio": round(min_bcr, 4),
            "service_life_years": service_life,
            "discount_rate": rate,
            "fatal_crash_cost": round(unit_cost, 2),
            "max_projects": cap,
        },
        "candidate_count": len(candidate_list),
        "scored_count": len(scored),
        "eligible_count": len(eligible),
        "selected_count": len(selected),
        "deferred_count": len(deferred),
        "unscorable_count": len(candidate_list) - len(scored),
        "budget": budget_value,
        "programmed_cost": programmed_cost,
        "budget_remaining": round(budget_value - programmed_cost, 2) if budget_value is not None else None,
        "present_value_benefit": present_value_benefit,
        "program_net_benefit": round(present_value_benefit - programmed_cost, 2),
        "program_benefit_cost_ratio": round(present_value_benefit / programmed_cost, 3)
        if programmed_cost > 0
        else None,
        "annual_crashes_reduced": annual_crashes_reduced,
        "crashes_reduced_service_life": round(annual_crashes_reduced * service_life, 2),
        "equity": _program_equity(selected, programmed_cost),
        "selected": selected,
        "deferred": deferred,
        "cost_basis": crash_costs.COST_SOURCE,
    }


_EXPORT_FIELDS = (
    ("project_id", "project_id"),
    ("name", "project_name"),
    ("countermeasure", "countermeasure"),
    ("treatment_cost", "estimated_cost"),
    ("annual_crashes_reduced", "crashes_reduced_annual"),
    ("present_value_benefit", "present_value_benefit"),
    ("benefit_cost_ratio", "benefit_cost_ratio"),
    ("svi_percentile", "svi_percentile"),
    ("rank", "funding_rank"),
)


def hsip_export(program: dict) -> list[dict]:
    """Flat per-project rows for the selected program, in an SS4A/HSIP shape.

    Each row carries the location (segment id or lat/lon), the treatment, its
    estimated cost, the crashes it reduces, present-value benefit, benefit-cost
    ratio, whether it serves a disadvantaged community, and its funding rank —
    the columns a Safety Improvement Program application asks for.
    """
    rows: list[dict] = []
    for record in program.get("selected", []):
        segment = record.get("segment_id")
        if segment is not None:
            location = str(segment)
        elif record.get("lat") is not None and record.get("lon") is not None:
            location = f"{record['lat']},{record['lon']}"
        else:
            location = None
        row = {dest: record.get(source) for source, dest in _EXPORT_FIELDS}
        row["location"] = location
        row["serves_disadvantaged_community"] = bool(record.get("disadvantaged"))
        rows.append(row)
    return rows
