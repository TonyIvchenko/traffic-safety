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
