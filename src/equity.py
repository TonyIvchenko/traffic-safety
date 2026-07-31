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

from scripts.common import EQUITY_OVERLAY_PATH, TRACT_EQUITY_PATH

EQUITY_PATH_ENV = "TRAFFIC_SAFETY_EQUITY_PATH"
EQUITY_OVERLAY_PATH_ENV = "TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH"
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
        # _to_bool tolerates pandas nullable (boolean/Int64) NA that a bare
        # .astype(bool) would 500 on (e.g. an unfilled left-join / convert_dtypes).
        disadvantaged = frame["disadvantaged"].map(_to_bool).astype(bool)
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


# --- Equity overlay accessor (serves the per-segment overlay to the API) -------


def _json_scalar(value):
    """Coerce a pandas/numpy cell to a JSON-serializable Python value."""
    try:
        if pd.isna(value):
            return None
    except (TypeError, ValueError):
        pass  # array-valued cell: pd.isna is ambiguous
    if hasattr(value, "item"):
        try:
            return value.item()
        except (ValueError, AttributeError):
            pass  # size>1 array: fall through to tolist()
    if hasattr(value, "tolist"):
        return value.tolist()
    return value


def _overlay_in_bbox(frame: "pd.DataFrame", bbox) -> "pd.DataFrame":
    if "center_lat" not in frame.columns or "center_lon" not in frame.columns:
        return frame.iloc[0:0]
    min_lat, max_lat, min_lon, max_lon = bbox
    lat = pd.to_numeric(frame["center_lat"], errors="coerce")
    lon = pd.to_numeric(frame["center_lon"], errors="coerce")
    mask = (lat >= min_lat) & (lat <= max_lat) & (lon >= min_lon) & (lon <= max_lon)
    return frame[mask.fillna(False).to_numpy()]


_EMPTY_DISPARITY = {
    "segments": 0,
    "disadvantaged_segments": 0,
    "high_svi_segments": 0,
    "segment_share_disadvantaged": None,
    "crashes_total": 0.0,
    "crashes_in_disadvantaged": 0.0,
    "crash_share_disadvantaged": None,
    "crash_burden_disadvantaged": None,
    "crash_burden_other": None,
    "crash_disparity_ratio": None,
    "mean_risk_disadvantaged": None,
    "mean_risk_other": None,
    "risk_disparity_ratio": None,
}


def _ratio(numerator: float, denominator: float):
    return round(numerator / denominator, 3) if denominator else None


def equity_disparity(frame: "pd.DataFrame") -> dict:
    """Crash/risk equity disparity over a (pre-scoped) segment overlay frame.

    The headline ``crash_disparity_ratio`` is the crash burden *per segment* in
    disadvantaged tracts divided by that in the rest — >1 means disadvantaged
    corridors bear a disproportionate share.
    """
    n = int(len(frame))
    if n == 0:
        return dict(_EMPTY_DISPARITY)

    disadvantaged = (
        frame["disadvantaged"].map(_to_bool).astype(bool)
        if "disadvantaged" in frame.columns
        else pd.Series(False, index=frame.index)
    )
    crashes = (
        pd.to_numeric(frame["crashes"], errors="coerce").fillna(0.0)
        if "crashes" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    risk = (
        pd.to_numeric(frame["risk"], errors="coerce").fillna(0.0)
        if "risk" in frame.columns
        else pd.Series(0.0, index=frame.index)
    )
    svi = (
        pd.to_numeric(frame["svi_percentile"], errors="coerce")
        if "svi_percentile" in frame.columns
        else pd.Series(float("nan"), index=frame.index)
    )

    n_disadvantaged = int(disadvantaged.sum())
    n_other = n - n_disadvantaged
    crashes_disadvantaged = float(crashes[disadvantaged].sum())
    crashes_other = float(crashes[~disadvantaged].sum())
    crashes_total = crashes_disadvantaged + crashes_other
    burden_disadvantaged = crashes_disadvantaged / n_disadvantaged if n_disadvantaged else 0.0
    burden_other = crashes_other / n_other if n_other else 0.0
    risk_disadvantaged = float(risk[disadvantaged].mean()) if n_disadvantaged else 0.0
    risk_other = float(risk[~disadvantaged].mean()) if n_other else 0.0

    return {
        "segments": n,
        "disadvantaged_segments": n_disadvantaged,
        "high_svi_segments": int((svi >= HIGH_SVI_THRESHOLD).sum()),
        "segment_share_disadvantaged": round(n_disadvantaged / n, 4),
        "crashes_total": round(crashes_total, 3),
        "crashes_in_disadvantaged": round(crashes_disadvantaged, 3),
        "crash_share_disadvantaged": round(crashes_disadvantaged / crashes_total, 4)
        if crashes_total
        else None,
        "crash_burden_disadvantaged": round(burden_disadvantaged, 4),
        "crash_burden_other": round(burden_other, 4),
        "crash_disparity_ratio": _ratio(burden_disadvantaged, burden_other),
        "mean_risk_disadvantaged": round(risk_disadvantaged, 4),
        "mean_risk_other": round(risk_other, 4),
        "risk_disparity_ratio": _ratio(risk_disadvantaged, risk_other),
    }


class EquityOverlay:
    """Read-only accessor over the per-segment equity overlay parquet."""

    def __init__(self, frame) -> None:
        self._frame = frame

    def __len__(self) -> int:
        return len(self._frame)

    @classmethod
    def from_parquet(cls, path) -> "EquityOverlay":
        path = Path(path)
        if not path.exists():
            return cls(pd.DataFrame())
        try:
            frame = pd.read_parquet(path)
        except (OSError, ValueError):
            return cls(pd.DataFrame())
        return cls(frame)

    def hotspots(
        self,
        *,
        bbox=None,
        top_n: int = 50,
        min_risk: float = 0.0,
        only_disadvantaged: bool = False,
        min_svi: float | None = None,
        rank_by: str = "priority",
    ) -> list[dict]:
        """Ranked equity hotspots as JSON-safe records (optionally bbox-filtered)."""
        frame = self._frame
        if len(frame) == 0:
            return []
        if bbox is not None:
            frame = _overlay_in_bbox(frame, bbox)
        ranked = rank_equity_hotspots(
            frame,
            top_n=top_n,
            min_risk=min_risk,
            only_disadvantaged=only_disadvantaged,
            min_svi=min_svi,
            rank_by=rank_by,
        )
        return [
            {key: _json_scalar(value) for key, value in row.items()}
            for row in ranked.to_dict("records")
        ]

    def summary(self, *, geoid=None) -> dict:
        """Equity disparity for a jurisdiction (tract-GEOID prefix), or the whole
        dataset when ``geoid`` is None. The state/county prefix is stable across
        the 2010/2020 tract vintages, so it scopes correctly regardless."""
        frame = self._frame
        scope = str(geoid).strip() if geoid else None
        if scope:
            if "tract_geoid" in frame.columns and len(frame):
                mask = frame["tract_geoid"].astype("string").str.startswith(scope)
                frame = frame[mask.fillna(False).to_numpy()]
            else:
                frame = frame.iloc[0:0]
        result = equity_disparity(frame)
        result["geoid"] = scope
        return result


@lru_cache(maxsize=4)
def _load_overlay_cached(path_str: str) -> EquityOverlay:
    return EquityOverlay.from_parquet(path_str)


def load_equity_overlay(path=None) -> EquityOverlay:
    resolved = str(path or os.environ.get(EQUITY_OVERLAY_PATH_ENV) or EQUITY_OVERLAY_PATH)
    return _load_overlay_cached(resolved)
