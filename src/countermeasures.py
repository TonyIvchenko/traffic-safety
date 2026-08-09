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

import crash_typing

REFERENCE_PATH = REPO_DIR / "data" / "reference" / "countermeasures.json"

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
