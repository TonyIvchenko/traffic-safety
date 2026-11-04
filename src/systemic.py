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

import exposure

DEFAULT_GROUP_COLS = ("rttyp", "mtfcc")
DEFAULT_MIN_GROUP_KM = 50.0


def attach_vmt(
    frame: pd.DataFrame,
    *,
    length_col: str = "length_km",
    mtfcc_col: str = "mtfcc",
    aadt_col: str = "aadt",
    days: float = 365.0,
) -> pd.DataFrame:
    """Add ``aadt`` and ``vmt`` (annual vehicle-miles) columns for exposure scoring.

    Each segment keeps its own finite, non-negative ``aadt`` when present; else it
    inherits the functional-class default for its TIGER ``mtfcc``. ``vmt`` is
    ``AADT x length(mi) x days`` (0 where AADT or length is unknown). Vectorized,
    so it is cheap over a full national network.
    """
    out = frame.copy()
    if mtfcc_col in out.columns:
        class_series = out[mtfcc_col].map(lambda m: exposure.functional_class_for(mtfcc=m))
        default_aadt = class_series.map(lambda c: exposure.default_aadt(c) if c else None)
    else:
        default_aadt = pd.Series([None] * len(out), index=out.index)

    if aadt_col in out.columns:
        existing = pd.to_numeric(out[aadt_col], errors="coerce")
    else:
        existing = pd.Series([float("nan")] * len(out), index=out.index, dtype="float64")
    usable = existing.notna() & (existing >= 0)
    out["aadt"] = existing.where(usable, pd.to_numeric(default_aadt, errors="coerce"))

    if length_col in out.columns:
        length_mi = pd.to_numeric(out[length_col], errors="coerce") * exposure.MILES_PER_KM
    else:
        length_mi = pd.Series([0.0] * len(out), index=out.index)
    vmt = out["aadt"] * length_mi * float(days)
    out["vmt"] = vmt.where(vmt.notna() & (vmt >= 0), 0.0)
    return out


def systemic_rates(
    frame: pd.DataFrame,
    *,
    crash_col: str = "fatal_crashes",
    length_col: str = "length_km",
    group_cols=DEFAULT_GROUP_COLS,
    min_group_km: float = DEFAULT_MIN_GROUP_KM,
    vmt_col: str | None = None,
) -> pd.DataFrame:
    """Per roadway-attribute-group fatal-crash rate + normalized systemic score.

    Returns the group columns plus ``group_crashes``, ``group_length_km``,
    ``reliable``, ``crash_rate_per_km`` (fatal crashes / km, with thin groups set
    to the overall rate) and ``systemic_score`` (0-1, riskiest group = 1.0).

    When ``vmt_col`` names a vehicle-miles column (see :func:`attach_vmt`), the
    frame also gets ``group_vmt``, ``crash_rate_per_mvmt`` (crashes per million
    VMT — exposure-normalized, so a busy road is not flagged merely for carrying
    more traffic) and ``systemic_score_exposure`` (0-1 by that rate).
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

    if vmt_col is not None and vmt_col in frame.columns:
        total_vmt = float(frame[vmt_col].sum())
        overall_mvmt_rate = (total_crashes / total_vmt * exposure.MVMT) if total_vmt > 0 else 0.0
        vmt_by_group = (
            frame.groupby(group_cols, dropna=False)[vmt_col].sum().reset_index(name="group_vmt")
        )
        grouped = grouped.merge(vmt_by_group, on=group_cols, how="left")
        grouped["group_vmt"] = grouped["group_vmt"].fillna(0.0)
        # A group is exposure-reliable only with enough length *and* real VMT.
        reliable_exposure = grouped["reliable"] & (grouped["group_vmt"] > 0.0)
        raw_mvmt = grouped["group_crashes"] / grouped["group_vmt"].clip(lower=1e-9) * exposure.MVMT
        grouped["crash_rate_per_mvmt"] = raw_mvmt.where(reliable_exposure, overall_mvmt_rate)
        max_mvmt = float(grouped["crash_rate_per_mvmt"].max()) or 1.0
        grouped["systemic_score_exposure"] = (
            grouped["crash_rate_per_mvmt"] / max_mvmt
        ).clip(0.0, 1.0)

    return grouped


def apply_systemic_scores(
    frame: pd.DataFrame,
    rates: pd.DataFrame,
    *,
    group_cols=DEFAULT_GROUP_COLS,
    length_col: str = "length_km",
    vmt_col: str | None = None,
) -> pd.DataFrame:
    """Attach ``systemic_rate``, ``systemic_score`` and ``systemic_expected_crashes``.

    Segments whose attribute group is absent from ``rates`` fall back to the
    median group rate (score 0). When ``rates`` carries the exposure columns
    (built with ``vmt_col``), also attaches ``systemic_rate_mvmt``,
    ``systemic_score_exposure`` and, when ``vmt_col`` is present on ``frame``,
    ``systemic_expected_crashes_exposure``.
    """
    group_cols = list(group_cols)
    cols = [*group_cols, "crash_rate_per_km", "systemic_score"]
    has_exposure = "crash_rate_per_mvmt" in rates.columns
    if has_exposure:
        cols += ["crash_rate_per_mvmt", "systemic_score_exposure"]
    merged = frame.merge(rates[cols], on=group_cols, how="left")

    fallback_rate = float(rates["crash_rate_per_km"].median()) if len(rates) else 0.0
    merged["crash_rate_per_km"] = merged["crash_rate_per_km"].fillna(fallback_rate)
    merged["systemic_score"] = merged["systemic_score"].fillna(0.0)
    merged = merged.rename(columns={"crash_rate_per_km": "systemic_rate"})
    merged["systemic_expected_crashes"] = (
        merged["systemic_rate"] * merged[length_col].astype(float)
    )

    if has_exposure:
        fallback_mvmt = float(rates["crash_rate_per_mvmt"].median()) if len(rates) else 0.0
        merged["crash_rate_per_mvmt"] = merged["crash_rate_per_mvmt"].fillna(fallback_mvmt)
        merged["systemic_score_exposure"] = merged["systemic_score_exposure"].fillna(0.0)
        merged = merged.rename(columns={"crash_rate_per_mvmt": "systemic_rate_mvmt"})
        if vmt_col is not None and vmt_col in merged.columns:
            merged["systemic_expected_crashes_exposure"] = (
                merged["systemic_rate_mvmt"] * merged[vmt_col].astype(float) / exposure.MVMT
            )
    return merged
