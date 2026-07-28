"""Tract-level equity lookup for the Justice40 / SVI overlay.

Loads the processed equity index (build_equity_index.py -> tract_equity.csv.gz)
and answers, for a census-tract GEOID: its CDC/ATSDR Social Vulnerability Index
(SVI) percentile, a vulnerability band, and the Justice40 "disadvantaged" flag.
Backs the /v1/equity endpoints and the equity overlay. The index path is
configurable via ``TRAFFIC_SAFETY_EQUITY_PATH`` so tests and deployments can
point at their own dataset.
"""

from __future__ import annotations

from functools import lru_cache
import math
import os
from pathlib import Path
import sys

import pandas as pd

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.common import TRACT_EQUITY_PATH

EQUITY_PATH_ENV = "TRAFFIC_SAFETY_EQUITY_PATH"
# CDC SVI quartile bands over the 0-1 percentile (lower-inclusive).
_SVI_BANDS = ((0.25, "low"), (0.50, "moderate"), (0.75, "high"))
_TRUE_STRINGS = {"true", "1", "yes", "t"}


def _percentile(value):
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return None if math.isnan(number) or number < 0.0 else number


def _round(value):
    return None if value is None else round(float(value), 4)


def _to_bool(value) -> bool:
    if isinstance(value, str):
        return value.strip().lower() in _TRUE_STRINGS
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return bool(value)


def svi_category(percentile) -> str:
    """CDC-quartile vulnerability band for an SVI percentile; 'unknown' if absent."""
    value = _percentile(percentile)
    if value is None:
        return "unknown"
    for upper, label in _SVI_BANDS:
        if value < upper:
            return label
    return "very_high"


class EquityIndex:
    """In-memory census-tract -> equity record lookup."""

    def __init__(self, records) -> None:
        self._records = dict(records)

    def __len__(self) -> int:
        return len(self._records)

    @classmethod
    def from_csv(cls, path) -> "EquityIndex":
        path = Path(path)
        if not path.exists():
            return cls({})
        try:
            # A missing/corrupt reference file degrades to an empty index (every
            # tract reads 'unknown') rather than 500ing every equity request.
            frame = pd.read_csv(path, dtype={"tract_geoid": str})
        except (OSError, ValueError):
            return cls({})
        records: dict[str, dict] = {}
        for row in frame.to_dict("records"):
            geoid = str(row.get("tract_geoid", "")).strip()
            if geoid and geoid.lower() != "nan":
                records[geoid] = {
                    "svi_percentile": _round(_percentile(row.get("svi_percentile"))),
                    "disadvantaged": _to_bool(row.get("disadvantaged")),
                }
        return cls(records)

    def get(self, geoid) -> dict | None:
        return self._records.get(str(geoid).strip())

    def equity_for_tract(self, geoid) -> dict:
        """Always returns a record; an unknown tract reads as not-disadvantaged."""
        geoid = str(geoid).strip()
        record = self._records.get(geoid)
        percentile = record["svi_percentile"] if record else None
        return {
            "tract_geoid": geoid,
            "svi_percentile": percentile,
            "svi_category": svi_category(percentile),
            "disadvantaged": bool(record["disadvantaged"]) if record else False,
            "in_index": record is not None,
        }


@lru_cache(maxsize=8)
def _load_cached(path_str: str) -> EquityIndex:
    return EquityIndex.from_csv(path_str)


def load_equity_index(path=None) -> EquityIndex:
    resolved = str(path or os.environ.get(EQUITY_PATH_ENV) or TRACT_EQUITY_PATH)
    return _load_cached(resolved)


def equity_for_tract(geoid, *, path=None) -> dict:
    return load_equity_index(path).equity_for_tract(geoid)


# --- Equity hotspot ranking (over the per-segment equity overlay) --------------

DEFAULT_SVI_WEIGHT = 1.0  # how strongly SVI percentile boosts the priority
DEFAULT_DISADVANTAGED_BOOST = 0.5  # extra boost for a Justice40 tract
HIGH_SVI_THRESHOLD = 0.75  # CDC top-quartile "high vulnerability"


def equity_priority_score(
    risk,
    svi_percentile,
    disadvantaged,
    *,
    svi_weight: float = DEFAULT_SVI_WEIGHT,
    disadvantaged_boost: float = DEFAULT_DISADVANTAGED_BOOST,
) -> float:
    """Risk weighted up by tract vulnerability so dangerous *and* underserved
    corridors rank highest: ``risk * (1 + svi_weight*svi + boost*disadvantaged)``.
    Unknown SVI adds no boost (we do not inflate areas we cannot assess)."""
    base = _percentile(risk) or 0.0
    svi = _percentile(svi_percentile) or 0.0
    multiplier = 1.0 + svi_weight * svi + (disadvantaged_boost if bool(disadvantaged) else 0.0)
    return round(base * multiplier, 6)


def rank_equity_hotspots(
    overlay: "pd.DataFrame",
    *,
    top_n: int = 50,
    min_risk: float = 0.0,
    only_disadvantaged: bool = False,
    min_svi: float | None = None,
    svi_weight: float = DEFAULT_SVI_WEIGHT,
    disadvantaged_boost: float = DEFAULT_DISADVANTAGED_BOOST,
    rank_by: str = "priority",
) -> "pd.DataFrame":
    """Rank overlay segments as equity hotspots.

    Filters (``min_risk``, ``only_disadvantaged``, ``min_svi``) then ranks by the
    equity-weighted priority (``rank_by='priority'``) or by raw ``risk``. Adds an
    ``equity_priority`` column and returns the top ``top_n`` rows.
    """
    frame = overlay.copy()
    if "risk" in frame.columns:
        risk = pd.to_numeric(frame["risk"], errors="coerce").fillna(0.0)
    else:
        risk = pd.Series(0.0, index=frame.index)
    if "svi_percentile" in frame.columns:
        svi = pd.to_numeric(frame["svi_percentile"], errors="coerce")
    else:
        svi = pd.Series(float("nan"), index=frame.index)
    if "disadvantaged" in frame.columns:
        disadvantaged = frame["disadvantaged"].astype(bool)
    else:
        disadvantaged = pd.Series(False, index=frame.index)

    frame["equity_priority"] = [
        equity_priority_score(
            r, s, d, svi_weight=svi_weight, disadvantaged_boost=disadvantaged_boost
        )
        for r, s, d in zip(risk, svi, disadvantaged)
    ]

    keep = risk >= float(min_risk)
    if only_disadvantaged:
        keep = keep & disadvantaged
    if min_svi is not None:
        keep = keep & (svi.fillna(-1.0) >= float(min_svi))
    frame = frame[keep.to_numpy()]

    sort_col = "risk" if str(rank_by).strip().lower() == "risk" else "equity_priority"
    if sort_col not in frame.columns:
        sort_col = "equity_priority"
    frame = frame.sort_values(
        sort_col, ascending=False, kind="mergesort", na_position="last"
    ).head(int(top_n))
    return frame.reset_index(drop=True)
