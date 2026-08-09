from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import cmf_math


def test_apply_cmf():
    assert cmf_math.apply_cmf(100, 0.7) == pytest.approx(70.0)
    assert cmf_math.apply_cmf(-5, 0.7) == 0.0  # negative expected clamped


def test_crash_reduction_factor():
    assert cmf_math.crash_reduction_factor(0.71) == pytest.approx(0.29)
    assert cmf_math.crash_reduction_factor(1.0) == 0.0


def test_crashes_reduced():
    assert cmf_math.crashes_reduced(100, 0.71) == pytest.approx(29.0)


def test_combine_multiply():
    assert cmf_math.combine_cmfs([0.5, 0.8], method="multiply") == pytest.approx(0.4)


def test_combine_empty_and_single():
    assert cmf_math.combine_cmfs([]) == 1.0
    assert cmf_math.combine_cmfs([0.6], method="multiply") == pytest.approx(0.6)
    assert cmf_math.combine_cmfs([0.6], method="diminishing") == pytest.approx(0.6)


def test_diminishing_is_more_conservative_than_multiply():
    cmfs = [0.71, 0.53]
    multiply = cmf_math.combine_cmfs(cmfs, method="multiply")
    diminishing = cmf_math.combine_cmfs(cmfs, method="diminishing")
    # Diminishing credits less combined reduction -> a higher combined CMF.
    assert diminishing > multiply
    # Manual: sorted [0.53, 0.71]; crf .47 full, .29*0.66=.1914; 0.53*0.8086=0.4286.
    assert diminishing == pytest.approx(0.4286, abs=1e-3)


def test_diminishing_is_order_independent():
    a = cmf_math.combine_cmfs([0.71, 0.53, 0.9], method="diminishing")
    b = cmf_math.combine_cmfs([0.9, 0.53, 0.71], method="diminishing")
    assert a == pytest.approx(b)


def test_invalid_method_raises():
    with pytest.raises(ValueError, match="method"):
        cmf_math.combine_cmfs([0.7], method="nope")


def test_bad_cmf_is_no_effect():
    # Non-numeric CMF -> 1.0 (no change).
    assert cmf_math.apply_cmf(100, "bad") == pytest.approx(100.0)
    assert cmf_math.combine_cmfs([0.5, None], method="multiply") == pytest.approx(0.5)


def test_combined_effect_is_consistent():
    effect = cmf_math.combined_effect(100, [0.71, 0.53], method="diminishing")
    assert effect["expected_crashes"] == 100.0
    assert effect["remaining_crashes"] == pytest.approx(effect["combined_cmf"] * 100, abs=1e-2)
    assert effect["crashes_reduced"] == pytest.approx(
        100 - effect["remaining_crashes"], abs=1e-2
    )
    assert effect["crashes_reduced"] > 0
