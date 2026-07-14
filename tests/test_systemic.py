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
