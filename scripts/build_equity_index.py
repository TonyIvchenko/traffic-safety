"""Join the raw SVI and CEJST tract datasets into one equity index per tract.

Produces ``data/processed/equity/tract_equity.csv.gz`` with one row per census
tract:

- ``tract_geoid`` — 11-digit tract GEOID (zero-padded).
- ``svi_percentile`` — CDC/ATSDR overall Social Vulnerability Index percentile
  (``RPL_THEMES``, 0-1); CDC's ``-999`` "no data" sentinel becomes null.
- ``disadvantaged`` — CEJST Justice40 "Identified as disadvantaged" flag.

Column names drift between dataset vintages, so the parsers match against a list
of candidate headers rather than hard-coding one.

Tract vintage matters: the two sources are joined on the raw 11-digit GEOID, so
they must use the same census-tract boundaries. The defaults (SVI 2020 + CEJST
2.0) are both 2010-tract; pairing 2020-tract SVI 2022 with 2010-tract CEJST would
silently mis-match every re-tracted area and mislabel its disadvantaged status.
(Segments are tagged from TIGER 2024 = 2020 tracts, so the downstream equity
overlay covers tracts unchanged between 2010 and 2020 and leaves re-tracted areas
null until a 2010->2020 crosswalk is added.)

    python scripts/build_equity_index.py

Output: data/processed/equity/tract_equity.csv.gz
"""

from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.common import (
    TRACT_EQUITY_PATH,
    cdc_svi_path,
    cejst_path,
    ensure_dirs,
)

SVI_GEOID_CANDIDATES = ["FIPS", "GEOID", "TRACTFIPS", "TRACT"]
SVI_PERCENTILE_CANDIDATES = ["RPL_THEMES", "SPL_THEMES"]
CEJST_GEOID_CANDIDATES = [
    "Census tract 2010 ID", "Census tract 2020 ID", "GEOID10", "GEOID", "GEOID20",
]
CEJST_FLAG_CANDIDATES = ["Identified as disadvantaged", "SN_C"]

_TRUE_STRINGS = {"true", "1", "yes", "y", "t"}
TRACT_GEOID_LENGTHS = (10, 11)  # 11 digits, or 10 when a leading zero was dropped


def _first_present(columns, candidates: list[str]) -> str:
    available = set(columns)
    for candidate in candidates:
        if candidate in available:
            return candidate
    raise ValueError(f"none of {candidates} present in columns {list(columns)}")


def normalize_geoid(value) -> str | None:
    """Coerce a tract identifier to an 11-digit zero-padded string, or None."""
    if value is None:
        return None
    text = str(value).strip()
    if text.endswith(".0"):  # float-formatted integer id
        text = text[:-2]
    digits = "".join(char for char in text if char.isdigit())
    if len(digits) not in TRACT_GEOID_LENGTHS:
        return None
    return digits.zfill(11)


def _to_bool(value) -> bool:
    if isinstance(value, bool):
        return value
    if value is None:
        return False
    try:
        if pd.isna(value):
            return False
    except (TypeError, ValueError):
        pass
    return str(value).strip().lower() in _TRUE_STRINGS


def parse_svi(frame: pd.DataFrame, *, geoid_col=None, percentile_col=None) -> pd.DataFrame:
    """Tract GEOID + SVI percentile; CDC's -999 sentinel becomes null."""
    geoid_col = geoid_col or _first_present(frame.columns, SVI_GEOID_CANDIDATES)
    percentile_col = percentile_col or _first_present(frame.columns, SVI_PERCENTILE_CANDIDATES)
    out = pd.DataFrame(
        {
            "tract_geoid": frame[geoid_col].map(normalize_geoid),
            "svi_percentile": pd.to_numeric(frame[percentile_col], errors="coerce"),
        }
    )
    out["svi_percentile"] = out["svi_percentile"].where(out["svi_percentile"] >= 0.0)
    out = out.dropna(subset=["tract_geoid"]).drop_duplicates(subset=["tract_geoid"], keep="first")
    return out.reset_index(drop=True)


def parse_cejst(frame: pd.DataFrame, *, geoid_col=None, flag_col=None) -> pd.DataFrame:
    """Tract GEOID + Justice40 disadvantaged flag."""
    geoid_col = geoid_col or _first_present(frame.columns, CEJST_GEOID_CANDIDATES)
    flag_col = flag_col or _first_present(frame.columns, CEJST_FLAG_CANDIDATES)
    out = pd.DataFrame(
        {
            "tract_geoid": frame[geoid_col].map(normalize_geoid),
            "disadvantaged": frame[flag_col].map(_to_bool),
        }
    )
    out = out.dropna(subset=["tract_geoid"]).drop_duplicates(subset=["tract_geoid"], keep="first")
    return out.reset_index(drop=True)


def build_equity_table(svi: pd.DataFrame, cejst: pd.DataFrame) -> pd.DataFrame:
    """Outer-join SVI and CEJST on tract GEOID into the equity index."""
    merged = svi.merge(cejst, on="tract_geoid", how="outer")
    # Outer join leaves NaN for tracts absent from CEJST; _to_bool maps those
    # (and any string flag) to a clean bool without pandas' fillna downcast warning.
    merged["disadvantaged"] = merged["disadvantaged"].map(_to_bool).astype(bool)
    merged = merged.sort_values("tract_geoid").reset_index(drop=True)
    return merged[["tract_geoid", "svi_percentile", "disadvantaged"]]


def main() -> None:
    ensure_dirs()
    svi_raw = pd.read_csv(cdc_svi_path(), dtype=str, low_memory=False)
    cejst_raw = pd.read_csv(cejst_path(), dtype=str, low_memory=False)

    table = build_equity_table(parse_svi(svi_raw), parse_cejst(cejst_raw))
    TRACT_EQUITY_PATH.parent.mkdir(parents=True, exist_ok=True)
    table.to_csv(TRACT_EQUITY_PATH, index=False)

    disadvantaged = int(table["disadvantaged"].sum())
    with_svi = int(table["svi_percentile"].notna().sum())
    print(
        f"tracts={len(table)} disadvantaged={disadvantaged} "
        f"with_svi={with_svi} -> {TRACT_EQUITY_PATH}"
    )


if __name__ == "__main__":
    main()
