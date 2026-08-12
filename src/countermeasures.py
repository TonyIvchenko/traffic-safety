"""Match FHWA countermeasures to a road segment and its crash-type profile.

Loads the curated reference table (data/reference/countermeasures.json) and ranks
the treatments that (a) fit the segment's roadway (MTFCC + urban/rural context)
and (b) address the crash types the segment is prone to (from crash_typing). The
ranked list feeds the CMF benefit-cost math and the /v1/countermeasures endpoints.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cmf_math
import crash_costs
import crash_typing

from scripts.common import HIGH_INJURY_NETWORK_PATH

REFERENCE_PATH = REPO_DIR / "data" / "reference" / "countermeasures.json"
KM_TO_MILE = 0.621371
DEFAULT_ANALYSIS_YEARS = 5
CM_SEGMENTS_PATH_ENV = "TRAFFIC_SAFETY_CM_SEGMENTS_PATH"
_BENEFIT_COST_FIELDS = (
    "annual_benefit", "present_value_benefit", "treatment_cost",
    "net_benefit", "benefit_cost_ratio", "service_life_years", "discount_rate",
)
_SEGMENT_FIELDS = (
    "segment_id", "fullname", "mtfcc", "rttyp", "length_km",
    "center_lat", "center_lon", "fatal_crashes", "hin_rank", "rur_urb", "func_sys",
)

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


def _num(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    return default if number != number else number  # NaN -> default


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
    return max(0.0, _num(total_crashes)) / years


def treatment_cost(cm: dict, *, length_km=None) -> float:
    """Total install cost: per-mile treatments scale by segment length; per-
    intersection / per-location treatments are a flat unit cost."""
    cost = _num(cm.get("typical_cost_usd"))
    if cm.get("cost_unit") == "per_mile":
        miles = max(0.0, _num(length_km)) * KM_TO_MILE
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
            "expected_annual_crashes": round(max(0.0, _num(expected_annual_crashes)), 4),
            "annual_crashes_reduced": round(reduced, 4),
            "crash_cost_each": unit_cost,
        }
    )
    return result


def recommend_countermeasures(
    segment_attrs: dict,
    *,
    analysis_years: int = DEFAULT_ANALYSIS_YEARS,
    top_n: int = 5,
    min_score: float = 0.2,
    vru_share: float = 0.0,
    crash_cost=None,
    catalog=None,
) -> list[dict]:
    """Applicable countermeasures for a segment, each with a benefit-cost, ranked
    by benefit-cost ratio (highest first)."""
    profile = crash_typing.crash_type_profile(
        mtfcc=segment_attrs.get("mtfcc"),
        rur_urb=segment_attrs.get("rur_urb"),
        func_sys=segment_attrs.get("func_sys"),
        vru_share=vru_share,
    )
    recs = applicable_countermeasures(
        segment_attrs, profile, catalog=catalog, top_n=top_n, min_score=min_score
    )
    expected = annualize_crashes(segment_attrs.get("fatal_crashes", 0.0), analysis_years)

    out = []
    for rec in recs:
        bc = countermeasure_benefit_cost(
            rec,
            expected_annual_crashes=expected,
            length_km=segment_attrs.get("length_km"),
            crash_cost=crash_cost,
        )
        out.append(
            {
                **rec,
                "expected_annual_crashes": bc["expected_annual_crashes"],
                "annual_crashes_reduced": bc["annual_crashes_reduced"],
                "benefit_cost": {field: bc[field] for field in _BENEFIT_COST_FIELDS},
            }
        )
    out.sort(key=lambda r: (r["benefit_cost"].get("benefit_cost_ratio") or 0.0), reverse=True)
    return out


def _json_scalar(value):
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass
    return value.item() if hasattr(value, "item") else value


def _coord(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if number != number else number


def _in_bbox(segment: dict, bbox) -> bool:
    lat = _coord(segment.get("center_lat"))
    lon = _coord(segment.get("center_lon"))
    if lat is None or lon is None:
        return False
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


class CountermeasureStore:
    """Segment lookup (from the HIN parquet) that serves ranked recommendations."""

    def __init__(self, segments_by_id: dict, *, analysis_years: int = DEFAULT_ANALYSIS_YEARS) -> None:
        self._segments = segments_by_id
        self._analysis_years = analysis_years

    def __len__(self) -> int:
        return len(self._segments)

    @classmethod
    def from_parquet(cls, path, *, analysis_years: int = DEFAULT_ANALYSIS_YEARS) -> "CountermeasureStore":
        path = Path(path)
        if not path.exists():
            return cls({}, analysis_years=analysis_years)
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return cls({}, analysis_years=analysis_years)
        columns = [column for column in _SEGMENT_FIELDS if column in frame.columns]
        segments: dict[str, dict] = {}
        for row in frame[columns].to_dict("records"):
            segment_id = str(row.get("segment_id", "")).strip()
            if segment_id and segment_id.lower() != "nan":
                segments[segment_id] = row
        return cls(segments, analysis_years=analysis_years)

    def get_segment(self, segment_id) -> dict | None:
        return self._segments.get(str(segment_id).strip())

    def recommend(self, segment_id, *, top_n: int = 5) -> dict | None:
        segment = self.get_segment(segment_id)
        if segment is None:
            return None
        recommendations = recommend_countermeasures(
            segment, analysis_years=self._analysis_years, top_n=top_n
        )
        return {
            "segment": {
                field: _json_scalar(segment[field])
                for field in _SEGMENT_FIELDS
                if field in segment
            },
            "analysis_years": self._analysis_years,
            "count": len(recommendations),
            "recommendations": recommendations,
        }

    def _hotspot_record(self, segment: dict, recs_per_segment: int) -> dict:
        recommendations = recommend_countermeasures(
            segment, analysis_years=self._analysis_years, top_n=recs_per_segment
        )
        record = {
            field: _json_scalar(segment[field])
            for field in _SEGMENT_FIELDS
            if field in segment
        }
        record["applicable_count"] = len(recommendations)
        top = recommendations[0] if recommendations else None
        record["recommended"] = (
            {
                "id": top.get("id"),
                "name": top.get("name"),
                "cmf": top.get("cmf"),
                "crash_reduction": top.get("crash_reduction"),
                "annual_crashes_reduced": top.get("annual_crashes_reduced"),
                "benefit_cost_ratio": top["benefit_cost"].get("benefit_cost_ratio"),
                "net_benefit": top["benefit_cost"].get("net_benefit"),
                "treatment_cost": top["benefit_cost"].get("treatment_cost"),
            }
            if top is not None
            else None
        )
        return record

    def hotspots(
        self,
        *,
        bbox=None,
        top_n: int = 50,
        recs_per_segment: int = 5,
        min_fatal_crashes: float = 0.0,
    ) -> list[dict]:
        """Highest-crash segments (optionally in ``bbox``), each with its
        top-benefit-cost recommended treatment."""
        segments = list(self._segments.values())
        if bbox is not None:
            segments = [segment for segment in segments if _in_bbox(segment, bbox)]
        threshold = float(min_fatal_crashes)
        segments = [s for s in segments if _num(s.get("fatal_crashes")) >= threshold]
        segments.sort(key=lambda s: _num(s.get("fatal_crashes")), reverse=True)
        return [self._hotspot_record(s, recs_per_segment) for s in segments[: int(top_n)]]


@lru_cache(maxsize=4)
def _load_store_cached(path_str: str, analysis_years: int) -> CountermeasureStore:
    return CountermeasureStore.from_parquet(path_str, analysis_years=analysis_years)


def load_countermeasure_store(path=None, *, analysis_years: int = DEFAULT_ANALYSIS_YEARS) -> CountermeasureStore:
    resolved = str(path or os.environ.get(CM_SEGMENTS_PATH_ENV) or HIGH_INJURY_NETWORK_PATH)
    return _load_store_cached(resolved, analysis_years)
