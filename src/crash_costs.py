"""Crash unit costs and benefit-cost helpers for safety analysis.

Comprehensive crash costs by KABCO injury severity, from FHWA "Crash Costs for
Highway Safety Analysis" (FHWA-SA-17-071, 2016 US dollars). Comprehensive costs
include quality-of-life valuation and are the standard basis for HSIP/SS4A
benefit-cost analysis. Callers may override the defaults for a different dollar
year or agency policy.

KABCO scale: K=fatal, A=suspected serious, B=suspected minor, C=possible injury,
O=no apparent injury (property damage only).
"""

from __future__ import annotations

KABCO_LEVELS = ("K", "A", "B", "C", "O")

KABCO_COMPREHENSIVE_COST = {
    "K": 11_295_400.0,  # Fatal
    "A": 655_000.0,     # Suspected serious injury
    "B": 198_500.0,     # Suspected minor injury
    "C": 125_600.0,     # Possible injury
    "O": 11_900.0,      # No apparent injury (property damage only)
}
COST_SOURCE = "FHWA-SA-17-071 comprehensive crash costs (2016 USD)"

DEFAULT_SERVICE_LIFE_YEARS = 20
DEFAULT_DISCOUNT_RATE = 0.03


def severity_cost(level: str, costs: dict | None = None) -> float:
    costs = costs or KABCO_COMPREHENSIVE_COST
    return float(costs.get(str(level).upper(), 0.0))


def severity_weight(level: str, costs: dict | None = None) -> float:
    """Severity weight relative to a fatal crash (K=1.0), by comprehensive cost."""
    costs = costs or KABCO_COMPREHENSIVE_COST
    fatal = float(costs.get("K", 0.0)) or 1.0
    return severity_cost(level, costs) / fatal


def expected_annual_cost(counts_by_severity: dict, costs: dict | None = None) -> float:
    """Sum of (count x unit cost) over KABCO levels present in the mapping."""
    costs = costs or KABCO_COMPREHENSIVE_COST
    return float(
        sum(severity_cost(level, costs) * float(count) for level, count in counts_by_severity.items())
    )


def present_value(
    annual_benefit: float,
    service_life_years: int = DEFAULT_SERVICE_LIFE_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> float:
    """Present value of a uniform annual benefit over the service life."""
    years = int(service_life_years)
    rate = float(discount_rate)
    if rate <= 0.0:
        return float(annual_benefit) * years
    factor = (1.0 - (1.0 + rate) ** -years) / rate  # uniform-series present-worth
    return float(annual_benefit) * factor


def benefit_cost(
    annual_crash_cost_reduction: float,
    treatment_cost: float,
    service_life_years: int = DEFAULT_SERVICE_LIFE_YEARS,
    discount_rate: float = DEFAULT_DISCOUNT_RATE,
) -> dict:
    """Benefit-cost of a countermeasure given its annual crash-cost reduction."""
    pv_benefit = present_value(annual_crash_cost_reduction, service_life_years, discount_rate)
    cost = float(treatment_cost)
    return {
        "annual_benefit": float(annual_crash_cost_reduction),
        "present_value_benefit": round(pv_benefit, 2),
        "treatment_cost": cost,
        "net_benefit": round(pv_benefit - cost, 2),
        "benefit_cost_ratio": round(pv_benefit / cost, 3) if cost > 0.0 else None,
        "service_life_years": int(service_life_years),
        "discount_rate": float(discount_rate),
        "cost_basis": COST_SOURCE,
    }
