"""Systemic safety scoring: risk by roadway attributes, not crash history.

Hotspot (history-based) analysis misses roads that are dangerous by design but
have not *yet* had a severe crash. FHWA systemic safety instead measures the
crash rate of roadway attribute groups and assigns every segment sharing those
attributes a risk score. Here the grouping uses the attributes available on the
TIGER network — route type (``rttyp``) and feature class (``mtfcc``) — and the
rate is fatal crashes per km. Thin groups (little exposure) fall back to the
overall network rate so a couple of crashes on a short road class do not produce
a spurious sky-high rate.
"""

from __future__ import annotations

import pandas as pd

DEFAULT_GROUP_COLS = ("rttyp", "mtfcc")
DEFAULT_MIN_GROUP_KM = 50.0


def systemic_rates(
    frame: pd.DataFrame,
    *,
    crash_col: str = "fatal_crashes",
    length_col: str = "length_km",
    group_cols=DEFAULT_GROUP_COLS,
    min_group_km: float = DEFAULT_MIN_GROUP_KM,
) -> pd.DataFrame:
    """Per roadway-attribute-group fatal-crash rate + normalized systemic score.

    Returns the group columns plus ``group_crashes``, ``group_length_km``,
    ``reliable``, ``crash_rate_per_km`` (fatal crashes / km, with thin groups set
    to the overall rate) and ``systemic_score`` (0-1, riskiest group = 1.0).
    """
    group_cols = list(group_cols)
    total_crashes = float(frame[crash_col].sum())
    total_length = float(frame[length_col].sum())
    overall_rate = total_crashes / total_length if total_length > 0 else 0.0

    grouped = (
        frame.groupby(group_cols, dropna=False)
        .agg(group_crashes=(crash_col, "sum"), group_length_km=(length_col, "sum"))
        .reset_index()
    )
    grouped["reliable"] = grouped["group_length_km"] >= float(min_group_km)
    raw_rate = grouped["group_crashes"] / grouped["group_length_km"].clip(lower=1e-9)
    grouped["crash_rate_per_km"] = raw_rate.where(grouped["reliable"], overall_rate)

    max_rate = float(grouped["crash_rate_per_km"].max()) or 1.0
    grouped["systemic_score"] = (grouped["crash_rate_per_km"] / max_rate).clip(0.0, 1.0)
    return grouped


def apply_systemic_scores(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    group_cols=DEFAULT_GROUP_COLS,
    length_col: str = "length_km",
) -> pd.DataFrame:
    """Attach ``systemic_rate``, ``systemic_score`` and ``systemic_expected_crashes``.

    Segments whose attribute group is absent from ``rates`` fall back to the
    median group rate (score 0).
    """
    group_cols = list(group_cols)
    lookup = rates[[*group_cols, "crash_rate_per_km", "systemic_score"]]
    merged = frame.merge(lookup, on=group_cols, how="left")

    fallback_rate = float(rates["crash_rate_per_km"].median()) if len(rates) else 0.0
    merged["crash_rate_per_km"] = merged["crash_rate_per_km"].fillna(fallback_rate)
    merged["systemic_score"] = merged["systemic_score"].fillna(0.0)
    merged = merged.rename(columns={"crash_rate_per_km": "systemic_rate"})
    merged["systemic_expected_crashes"] = (
        merged["systemic_rate"] * merged[length_col].astype(float)
    )
    return merged
