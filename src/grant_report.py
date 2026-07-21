"""Assemble a jurisdiction's safety-analysis report — the SS4A / HSIP deliverable.

Pure structuring of already-computed inputs (a jurisdiction's crashes plus its
HIN- and systemic-scored segments) into a nested, JSON-serializable dict that the
HTML renderer and the /v1/grants endpoints consume. No I/O and no clock: the
caller passes ``generated_at`` and the data vintage.
"""

from __future__ import annotations

import math
from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs
from hin import hin_summary

METHODOLOGY = {
    "high_injury_network": (
        "Road segments ranked by severity-weighted fatal crashes per km; the "
        "High Injury Network is the smallest set of segments that carries the "
        "target share of severity-weighted crashes."
    ),
    "systemic": (
        "Fatal-crash rate per km by roadway class (route type x feature class); "
        "every segment inherits its class rate so roads that are dangerous by "
        "design are flagged even without a recorded crash."
    ),
    "severity_basis": "FARS fatal crashes (KABCO 'K').",
    "crash_costs": crash_costs.COST_SOURCE,
    "benefit_cost": (
        "Benefit-cost of treating the High Injury Network applies a placeholder "
        "crash-reduction factor to the comprehensive cost of fatal crashes on the "
        "treated corridors, discounted over the treatment service life; corridor-"
        "specific Crash Modification Factors replace the placeholder in the "
        "countermeasures analysis."
    ),
}

# Placeholder benefit-cost assumptions until the countermeasures feature swaps in
# Crash Modification Factors and treatment-specific costs.
DEFAULT_HIN_CRASH_REDUCTION = 0.30  # fraction of fatal crashes avoided by treatment
DEFAULT_TREATMENT_COST_PER_KM = 200_000.0  # order-of-magnitude corridor safety package

DATA_SOURCES = [
    {"name": "FARS", "publisher": "NHTSA", "use": "fatal crash locations and counts"},
    {"name": "TIGER/Line", "publisher": "U.S. Census Bureau",
     "use": "road network and county/tract boundaries"},
    {"name": "FHWA crash costs", "publisher": "FHWA",
     "use": "comprehensive KABCO crash costs for benefit-cost"},
]

_GEOID_LEVEL = {2: "state", 5: "county", 11: "tract"}
_HIN_FIELDS = [
    "segment_id", "fullname", "rttyp", "mtfcc", "length_km",
    "fatal_crashes", "hin_intensity", "hin_rank", "center_lat", "center_lon",
]
_SYSTEMIC_FIELDS = [
    "segment_id", "fullname", "rttyp", "mtfcc", "length_km",
    "systemic_score", "systemic_rate", "systemic_expected_crashes",
    "center_lat", "center_lon",
]


def _to_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(number) else number


def _clean_value(value):
    if value is None:
        return None
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, (np.floating, float)):
        number = float(value)
        return None if math.isnan(number) else round(number, 4)
    if isinstance(value, np.str_):
        return str(value)
    return value


def _records(frame, fields, *, top_n=None, sort_by=None, ascending=False) -> list[dict]:
    if sort_by and sort_by in frame.columns:
        frame = frame.sort_values(sort_by, ascending=ascending, kind="mergesort")
    if top_n is not None:
        frame = frame.head(int(top_n))
    columns = [column for column in fields if column in frame.columns]
    return [
        {column: _clean_value(record[column]) for column in columns}
        for record in frame[columns].to_dict(orient="records")
    ]


def geoid_level(geoid: str) -> str:
    return _GEOID_LEVEL.get(len(str(geoid)), "area")


def crash_summary(crashes: pd.DataFrame) -> dict:
    if len(crashes) == 0:
        return {
            "total_fatal_crashes": 0,
            "total_fatalities": 0,
            "years": [],
            "by_year": {},
            "by_mode": {},
        }
    by_year = {int(year): int(count) for year, count in crashes.groupby("year").size().items()}
    total_fatalities = (
        int(crashes["fatals"].sum()) if "fatals" in crashes.columns else int(len(crashes))
    )
    by_mode: dict[str, int] = {}
    if "ped_count" in crashes.columns:
        by_mode["pedestrian"] = int(crashes["ped_count"].sum())
    if "cyc_count" in crashes.columns:
        by_mode["cyclist"] = int(crashes["cyc_count"].sum())
    return {
        "total_fatal_crashes": int(len(crashes)),
        "total_fatalities": total_fatalities,
        "years": sorted(int(year) for year in crashes["year"].unique()),
        "by_year": dict(sorted(by_year.items())),
        "by_mode": by_mode,
    }


def top_hin_corridors(segments: pd.DataFrame, *, top_n: int = 10) -> list[dict]:
    if "hin" not in segments.columns:
        return []
    selected = segments[segments["hin"].astype(bool)]
    return _records(selected, _HIN_FIELDS, top_n=top_n, sort_by="hin_rank", ascending=True)


def top_systemic_locations(
    segments: pd.DataFrame, *, top_n: int = 10, max_history: float = 0.0
) -> list[dict]:
    """History-poor segments (<= max_history crashes) ranked by systemic score."""
    if "systemic_score" not in segments.columns:
        return []
    candidates = segments
    if "fatal_crashes" in segments.columns:
        candidates = segments[segments["fatal_crashes"].astype(float) <= float(max_history)]
    return _records(candidates, _SYSTEMIC_FIELDS, top_n=top_n, sort_by="systemic_score")


def hin_benefit_cost(
    corridors: list[dict],
    *,
    analysis_years: int,
    crash_reduction: float = DEFAULT_HIN_CRASH_REDUCTION,
    treatment_cost_per_km: float = DEFAULT_TREATMENT_COST_PER_KM,
    service_life_years: int | None = None,
    discount_rate: float | None = None,
    fatal_cost: float | None = None,
) -> dict | None:
    """Benefit-cost of treating the given HIN corridors.

    Annualizes the corridors' recorded fatal crashes over ``analysis_years``,
    values them at the FHWA comprehensive fatal-crash cost, applies a placeholder
    ``crash_reduction`` (until CMFs land), and compares the discounted benefit to
    a length-scaled treatment cost. Returns ``None`` when there is nothing to
    treat (no corridors or a non-positive analysis window).
    """
    corridors = [corridor for corridor in corridors if isinstance(corridor, dict)]
    years = int(analysis_years)
    if not corridors or years <= 0:
        return None

    fatal_crash_cost = crash_costs.severity_cost("K") if fatal_cost is None else float(fatal_cost)
    service_life = (
        crash_costs.DEFAULT_SERVICE_LIFE_YEARS if service_life_years is None else int(service_life_years)
    )
    rate = crash_costs.DEFAULT_DISCOUNT_RATE if discount_rate is None else float(discount_rate)

    total_fatal = sum(_to_float(corridor.get("fatal_crashes")) for corridor in corridors)
    total_length = sum(_to_float(corridor.get("length_km")) for corridor in corridors)
    annual_fatal = total_fatal / years
    annual_reduction = annual_fatal * fatal_crash_cost * float(crash_reduction)
    treatment_cost = total_length * float(treatment_cost_per_km)

    result = crash_costs.benefit_cost(annual_reduction, treatment_cost, service_life, rate)
    result.update(
        {
            "treated_corridors": len(corridors),
            "treated_length_km": round(total_length, 3),
            "annual_fatal_crashes_treated": round(annual_fatal, 4),
            "crash_reduction_factor": float(crash_reduction),
            "treatment_cost_per_km": float(treatment_cost_per_km),
            "analysis_years": years,
            "fatal_crash_cost": fatal_crash_cost,
            "basis": "placeholder crash reduction pending countermeasure CMFs",
        }
    )
    return result


def assemble_grant_report(
    *,
    geoid: str,
    name: str,
    crashes: pd.DataFrame,
    segments: pd.DataFrame,
    benefit_cost: dict | None = None,
    top_n: int = 10,
    generated_at: str | None = None,
    data_vintage: dict | None = None,
) -> dict:
    report = {
        "jurisdiction": {"geoid": str(geoid), "name": name, "level": geoid_level(geoid)},
        "generated_at_utc": generated_at,
        "data_vintage": data_vintage or {},
        "crash_summary": crash_summary(crashes),
        "high_injury_network": (
            hin_summary(segments)
            if {"hin", "weighted_crashes", "length_km"}.issubset(segments.columns)
            else {}
        ),
        "hin_corridors": top_hin_corridors(segments, top_n=top_n),
        "systemic_locations": top_systemic_locations(segments, top_n=top_n),
        "methodology": METHODOLOGY,
        "data_sources": DATA_SOURCES,
    }
    if benefit_cost is not None:
        report["benefit_cost"] = benefit_cost
    return report
