"""Match FHWA countermeasures to a road segment and its crash-type profile.

Loads the curated reference table (data/reference/countermeasures.json) and ranks
the treatments that (a) fit the segment's roadway (MTFCC + urban/rural context)
and (b) address the crash types the segment is prone to (from crash_typing). The
ranked list feeds the CMF benefit-cost math and the /v1/countermeasures endpoints.
"""

from __future__ import annotations

from functools import lru_cache
import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cmf_math
import crash_costs
import crash_typing

REFERENCE_PATH = REPO_DIR / "data" / "reference" / "countermeasures.json"
KM_TO_MILE = 0.621371

_PUBLIC_FIELDS = (
    "id", "name", "category", "cmf", "cmf_basis", "cmf_star_rating",
    "vru_focused", "typical_cost_usd", "cost_unit", "applicable_crash_types",
)
# A treatment tagged with "total" addresses overall crashes, so it stays broadly
# applicable wherever the roadway fits, at this baseline relevance.
_TOTAL_RELEVANCE = 0.5


@lru_cache(maxsize=4)
def load_countermeasures(path=None) -> list[dict]:
    data = json.loads(Path(path or REFERENCE_PATH).read_text(encoding="utf-8"))
    return list(data.get("countermeasures", []))


def get_countermeasure(cm_id: str, *, catalog=None) -> dict | None:
    for cm in catalog if catalog is not None else load_countermeasures():
        if cm.get("id") == cm_id:
            return cm
    return None


def _public(cm: dict) -> dict:
    return {field: cm[field] for field in _PUBLIC_FIELDS if field in cm}


def _roadway_matches(cm: dict, segment_attrs: dict) -> bool:
    roadway = cm.get("roadway", {}) or {}
    allowed_mtfcc = [str(m).upper() for m in roadway.get("mtfcc", [])]
    mtfcc = str(segment_attrs.get("mtfcc", "") or "").upper()
    if allowed_mtfcc and mtfcc and mtfcc not in allowed_mtfcc:
        return False
    contexts = roadway.get("context", [])
    if contexts:
        context = "urban" if crash_typing.is_urban(segment_attrs.get("rur_urb")) else "rural"
        if context not in contexts:
            return False
    return True


def _crash_relevance(applicable_crash_types, crash_profile: dict) -> float:
    scores = []
    for crash_type in applicable_crash_types:
        if crash_type == "total":
            scores.append(_TOTAL_RELEVANCE)
        else:
            scores.append(float(crash_profile.get(crash_type, 0.0)))
    return max(scores, default=0.0)


def applicable_countermeasures(
    segment_attrs: dict,
    crash_profile: dict,
    *,
    catalog=None,
    top_n: int = 5,
    min_score: float = 0.2,
) -> list[dict]:
    """Countermeasures that fit the segment, ranked by crash-type relevance.

    Each result carries ``match_score`` (best-addressed crash type's profile
    weight; "total" treatments use a baseline) and ``crash_reduction`` (1 - CMF).
    Treatments whose roadway does not match, or whose relevance is below
    ``min_score``, are dropped.
    """
    catalog = catalog if catalog is not None else load_countermeasures()
    matches = []
    for cm in catalog:
        if not _roadway_matches(cm, segment_attrs):
            continue
        score = _crash_relevance(cm.get("applicable_crash_types", []), crash_profile)
        if score < float(min_score):
            continue
        record = _public(cm)
        record["match_score"] = round(score, 4)
        record["crash_reduction"] = round(1.0 - float(cm["cmf"]), 4)
        matches.append(record)

    matches.sort(key=lambda m: (m["match_score"], m["crash_reduction"]), reverse=True)
    return matches[: int(top_n)]


# --- Benefit-cost of applying a countermeasure --------------------------------


def annualize_crashes(total_crashes, analysis_years) -> float:
    """Average annual crashes from a multi-year total."""
    years = int(analysis_years) if analysis_years else 0
    years = years if years > 0 else 1
    return max(0.0, float(total_crashes or 0.0)) / years


def treatment_cost(cm: dict, *, length_km=None) -> float:
    """Total install cost: per-mile treatments scale by segment length; per-
    intersection / per-location treatments are a flat unit cost."""
    cost = float(cm.get("typical_cost_usd", 0.0) or 0.0)
    if cm.get("cost_unit") == "per_mile":
        miles = max(0.0, float(length_km or 0.0)) * KM_TO_MILE
        return round(cost * miles, 2)
    return round(cost, 2)


def countermeasure_benefit_cost(
    cm: dict,
    *,
    expected_annual_crashes: float,
    crash_cost=None,
    length_km=None,
    service_life_years=None,
    discount_rate=None,
) -> dict:
    """Benefit-cost of one treatment on one segment.

    Values the annual crashes it would avoid (``CRF * expected``) at the FHWA
    comprehensive fatal-crash cost, discounts over the service life, and compares
    to the install cost. Returns the crash_costs.benefit_cost dict plus the
    treatment inputs.
    """
    cmf = float(cm["cmf"])
    reduced = cmf_math.crashes_reduced(expected_annual_crashes, cmf)
    unit_cost = crash_costs.severity_cost("K") if crash_cost is None else float(crash_cost)
    annual_benefit = reduced * unit_cost
    cost = treatment_cost(cm, length_km=length_km)
    service_life = (
        crash_costs.DEFAULT_SERVICE_LIFE_YEARS if service_life_years is None else int(service_life_years)
    )
    rate = crash_costs.DEFAULT_DISCOUNT_RATE if discount_rate is None else float(discount_rate)

    result = crash_costs.benefit_cost(annual_benefit, cost, service_life, rate)
    result.update(
        {
            "countermeasure_id": cm.get("id"),
            "cmf": cmf,
            "crash_reduction": round(1.0 - cmf, 4),
            "expected_annual_crashes": round(max(0.0, float(expected_annual_crashes or 0.0)), 4),
            "annual_crashes_reduced": round(reduced, 4),
            "crash_cost_each": unit_cost,
        }
    )
    return result
