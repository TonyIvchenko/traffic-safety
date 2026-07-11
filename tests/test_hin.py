from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import crash_costs
import hin


def _segments() -> pd.DataFrame:
    # Intensities: s1=10/1=10, s2=6/2=3, s3=4/4=1. Total weighted = 20.
    return pd.DataFrame(
        {
            "segment_id": ["s1", "s2", "s3"],
            "weighted_crashes": [10.0, 6.0, 4.0],
            "length_km": [1.0, 2.0, 4.0],
        }
    )


def test_severity_weighted_crashes_uses_fatal_as_unit():
    assert hin.severity_weighted_crashes({"K": 2}) == pytest.approx(2.0)
    mixed = hin.severity_weighted_crashes({"K": 2, "A": 10})
    assert mixed == pytest.approx(2.0 + 10 * crash_costs.severity_weight("A"))
    assert hin.severity_weighted_crashes({}) == 0.0


def test_build_hin_ranks_by_intensity_and_flags_target_share():
    result = hin.build_hin(_segments(), target_share=0.5)

    assert list(result["segment_id"]) == ["s1", "s2", "s3"]  # sorted by intensity desc
    assert list(result["hin_rank"]) == [1, 2, 3]
    assert result.loc[0, "hin_intensity"] == pytest.approx(10.0)
    # s1 alone reaches exactly 50% of weighted crashes.
    assert list(result["hin"]) == [True, False, False]
    assert result.loc[0, "hin_cumulative_share"] == pytest.approx(0.5)


def test_build_hin_includes_the_segment_that_crosses_the_target():
    result = hin.build_hin(_segments(), target_share=0.6)
    # s1 gets to 0.5; s2 crosses 0.6 and must be included.
    assert list(result["hin"]) == [True, True, False]
    captured = result.loc[result["hin"], "weighted_crashes"].sum()
    assert captured / result["weighted_crashes"].sum() >= 0.6


def test_build_hin_intensity_is_length_normalized():
    # A long segment with more raw crashes can rank below a short intense one.
    frame = pd.DataFrame(
        {
            "segment_id": ["long", "short"],
            "weighted_crashes": [9.0, 5.0],
            "length_km": [9.0, 0.5],  # intensity 1.0 vs 10.0
        }
    )
    result = hin.build_hin(frame, target_share=0.5)
    assert result.loc[0, "segment_id"] == "short"


def test_build_hin_handles_zero_crashes_and_zero_length():
    empty_crashes = pd.DataFrame(
        {"segment_id": ["a"], "weighted_crashes": [0.0], "length_km": [1.0]}
    )
    result = hin.build_hin(empty_crashes)
    assert list(result["hin"]) == [False]
    assert result.loc[0, "hin_cumulative_share"] == 0.0

    zero_length = pd.DataFrame(
        {"segment_id": ["a"], "weighted_crashes": [1.0], "length_km": [0.0]}
    )
    # Length is floored, so intensity stays finite.
    assert pd.notna(hin.build_hin(zero_length).loc[0, "hin_intensity"])


def test_hin_summary_reports_length_and_crash_shares():
    result = hin.build_hin(_segments(), target_share=0.5)
    summary = hin.hin_summary(result)

    assert summary["hin_segments"] == 1
    assert summary["network_segments"] == 3
    assert summary["hin_length_km"] == pytest.approx(1.0)
    assert summary["network_length_km"] == pytest.approx(7.0)
    assert summary["length_share"] == pytest.approx(1 / 7, abs=1e-4)
    assert summary["weighted_crash_share"] == pytest.approx(0.5)
