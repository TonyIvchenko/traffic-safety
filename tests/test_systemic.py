from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import systemic


def _segments() -> pd.DataFrame:
    # Interstate: 20 fatal / 100 km = 0.20/km. Local: 2 / 100 = 0.02/km.
    # Thin "M/S1500" class has only 5 km of exposure (< min_group_km=50).
    return pd.DataFrame(
        {
            "segment_id": ["i1", "i2", "l1", "l2", "m1"],
            "rttyp": ["I", "I", "C", "C", "M"],
            "mtfcc": ["S1100", "S1100", "S1400", "S1400", "S1500"],
            "fatal_crashes": [12.0, 8.0, 1.0, 1.0, 5.0],
            "length_km": [50.0, 50.0, 50.0, 50.0, 5.0],
        }
    )


def test_systemic_rates_computes_rate_per_km():
    rates = systemic.systemic_rates(_segments(), min_group_km=50.0)
    by_class = {(r.rttyp, r.mtfcc): r for r in rates.itertuples()}

    assert by_class[("I", "S1100")].crash_rate_per_km == pytest.approx(0.20)
    assert by_class[("C", "S1400")].crash_rate_per_km == pytest.approx(0.02)
    # Interstate is the riskiest class -> systemic_score 1.0.
    assert by_class[("I", "S1100")].systemic_score == pytest.approx(1.0)
    assert by_class[("C", "S1400")].systemic_score == pytest.approx(0.1)


def test_thin_group_falls_back_to_overall_rate():
    rates = systemic.systemic_rates(_segments(), min_group_km=50.0)
    thin = rates[(rates["rttyp"] == "M")].iloc[0]
    # overall = (12+8+1+1+5) / (50+50+50+50+5) = 27 / 205 ~= 0.1317
    assert thin["reliable"] is False or bool(thin["reliable"]) is False
    assert thin["crash_rate_per_km"] == pytest.approx(27.0 / 205.0, abs=1e-4)


def test_apply_systemic_scores_attaches_columns():
    frame = _segments()
    rates = systemic.systemic_rates(frame, min_group_km=50.0)
    scored = systemic.apply_systemic_scores(frame, rates)

    interstate = scored[scored["segment_id"] == "i1"].iloc[0]
    assert interstate["systemic_rate"] == pytest.approx(0.20)
    assert interstate["systemic_score"] == pytest.approx(1.0)
    # expected = rate x segment length
    assert interstate["systemic_expected_crashes"] == pytest.approx(0.20 * 50.0)


def test_apply_systemic_scores_history_poor_segment_still_scored():
    frame = _segments()
    rates = systemic.systemic_rates(frame, min_group_km=50.0)
    # A brand-new interstate segment with zero crash history.
    new_segment = pd.DataFrame(
        {
            "segment_id": ["new"],
            "rttyp": ["I"],
            "mtfcc": ["S1100"],
            "fatal_crashes": [0.0],
            "length_km": [2.0],
        }
    )
    scored = systemic.apply_systemic_scores(new_segment, rates)
    row = scored.iloc[0]
    # No history, but its road class is high-risk -> non-zero systemic score.
    assert row["systemic_score"] == pytest.approx(1.0)
    assert row["systemic_expected_crashes"] == pytest.approx(0.20 * 2.0)


def test_apply_systemic_scores_unknown_group_uses_fallback():
    frame = _segments()
    rates = systemic.systemic_rates(frame, min_group_km=50.0)
    unknown = pd.DataFrame(
        {"segment_id": ["u"], "rttyp": ["Z"], "mtfcc": ["S9999"], "fatal_crashes": [0.0], "length_km": [1.0]}
    )
    scored = systemic.apply_systemic_scores(unknown, rates)
    assert scored.iloc[0]["systemic_score"] == 0.0
    assert scored.iloc[0]["systemic_rate"] == pytest.approx(float(rates["crash_rate_per_km"].median()))


# --------------------------------------------------------------------------- #
# Exposure normalization (H2.2): crashes per VMT, not just per km.
# --------------------------------------------------------------------------- #


def test_attach_vmt_fills_class_default_and_respects_explicit_aadt():
    frame = pd.DataFrame({"mtfcc": ["S1100", "S1400", "S9999"], "length_km": [1.0, 2.0, 1.0]})
    out = systemic.attach_vmt(frame)
    assert out.loc[0, "aadt"] == 45000  # S1100 -> freeway_expressway default
    assert out.loc[1, "aadt"] == 1000  # S1400 -> local default
    assert pd.isna(out.loc[2, "aadt"]) and out.loc[2, "vmt"] == 0.0  # unknown class
    assert out.loc[0, "vmt"] == pytest.approx(45000 * 1.0 * 0.621371 * 365)

    explicit = systemic.attach_vmt(
        pd.DataFrame({"mtfcc": ["S1400"], "length_km": [1.0], "aadt": [9999.0]})
    )
    assert explicit.loc[0, "aadt"] == 9999.0  # a segment's own AADT wins over the default


def _equal_per_km_segments() -> pd.DataFrame:
    # Two classes with an IDENTICAL per-km rate (0.20/km) but very different
    # traffic: a 200k-AADT freeway vs a 5k-AADT arterial.
    return pd.DataFrame(
        {
            "segment_id": ["f1", "f2", "a1", "a2"],
            "rttyp": ["I", "I", "S", "S"],
            "mtfcc": ["S1100", "S1100", "S1200", "S1200"],
            "fatal_crashes": [10.0, 10.0, 10.0, 10.0],
            "length_km": [50.0, 50.0, 50.0, 50.0],
            "aadt": [200000.0, 200000.0, 5000.0, 5000.0],
        }
    )


def test_exposure_rate_reranks_by_vmt():
    frame = systemic.attach_vmt(_equal_per_km_segments())
    rates = systemic.systemic_rates(frame, min_group_km=50.0, vmt_col="vmt")
    by_class = {(r.rttyp, r.mtfcc): r for r in rates.itertuples()}
    freeway = by_class[("I", "S1100")]
    arterial = by_class[("S", "S1200")]

    # Per km the two classes are indistinguishable...
    assert freeway.crash_rate_per_km == pytest.approx(arterial.crash_rate_per_km)
    assert freeway.systemic_score == pytest.approx(arterial.systemic_score)
    # ...but per VMT the light-traffic arterial is far riskier per unit of travel.
    assert arterial.crash_rate_per_mvmt > freeway.crash_rate_per_mvmt
    assert arterial.systemic_score_exposure == pytest.approx(1.0)
    assert freeway.systemic_score_exposure < arterial.systemic_score_exposure


def test_apply_systemic_scores_attaches_exposure_columns():
    frame = systemic.attach_vmt(_equal_per_km_segments())
    rates = systemic.systemic_rates(frame, min_group_km=50.0, vmt_col="vmt")
    scored = systemic.apply_systemic_scores(frame, rates, vmt_col="vmt")

    assert "systemic_rate_mvmt" in scored.columns
    assert "systemic_score_exposure" in scored.columns
    assert "systemic_expected_crashes_exposure" in scored.columns
    arterial = scored[scored["segment_id"] == "a1"].iloc[0]
    assert arterial["systemic_score_exposure"] == pytest.approx(1.0)
    assert arterial["systemic_expected_crashes_exposure"] > 0


def test_systemic_rates_without_vmt_is_unchanged():
    rates = systemic.systemic_rates(_equal_per_km_segments(), min_group_km=50.0)
    assert "crash_rate_per_mvmt" not in rates.columns  # exposure is opt-in
    assert "systemic_score_exposure" not in rates.columns
