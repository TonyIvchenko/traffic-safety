from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import advisory


def test_advisory_level_bands():
    assert advisory.advisory_level(0.05)["name"] == "Low"
    assert advisory.advisory_level(0.10)["name"] == "Moderate"  # lower-inclusive boundary
    assert advisory.advisory_level(0.15)["name"] == "Moderate"
    assert advisory.advisory_level(0.20)["name"] == "Elevated"
    assert advisory.advisory_level(0.35)["name"] == "High"
    assert advisory.advisory_level(0.50)["name"] == "Extreme"
    assert advisory.advisory_level(0.90)["name"] == "Extreme"


def test_advisory_structure():
    result = advisory.advisory(0.4)
    assert result["level"] == "High"
    assert result["level_index"] == 4
    assert result["color"] == "#f46d43"
    assert result["risk_score"] == 0.4
    assert result["message"].startswith("High —")
    assert result["drivers"] == []


def test_advisory_folds_in_drivers():
    result = advisory.advisory(0.25, drivers=["wet roads", "Friday PM rush", ""])
    assert result["level"] == "Elevated"
    assert result["drivers"] == ["wet roads", "Friday PM rush"]  # blanks dropped
    assert "Key factors: wet roads, Friday PM rush." in result["message"]


def test_advisory_clamps_and_handles_bad_input():
    assert advisory.advisory(-1.0)["level"] == "Low"
    assert advisory.advisory(2.0)["level"] == "Extreme"
    assert advisory.advisory(2.0)["risk_score"] == 1.0
    assert advisory.advisory("bad")["level"] == "Low"  # non-numeric -> 0
    assert advisory.advisory(float("nan"))["level"] == "Low"


def test_advisory_custom_thresholds():
    # Everything above 0.5 is Extreme; a tight low band.
    thresholds = (0.05, 0.1, 0.2, 0.5)
    assert advisory.advisory_level(0.3, thresholds=thresholds)["name"] == "High"
    assert advisory.advisory_level(0.6, thresholds=thresholds)["name"] == "Extreme"


def test_advisory_scale():
    scale = advisory.advisory_scale()
    assert [level["name"] for level in scale] == [
        "Low", "Moderate", "Elevated", "High", "Extreme"
    ]
    assert scale[0]["min_risk"] == 0.0
    assert scale[-1]["max_risk"] == 1.0
    # Contiguous, non-overlapping bands.
    for lower, upper in zip(scale, scale[1:]):
        assert lower["max_risk"] == upper["min_risk"]


# --------------------------------------------------------------------------- #
# Relative advisory
# --------------------------------------------------------------------------- #

WEEK = [0.90 + 0.001 * i for i in range(24)]  # a saturated-but-varying reference


def test_percentile_of_positions():
    ref = [0.0, 0.25, 0.5, 0.75, 1.0]
    assert advisory.percentile_of(0.5, ref) == 0.5  # midrank of the median
    assert advisory.percentile_of(1.0, ref) == 0.9  # top value: (4 below + 0.5) / 5
    assert advisory.percentile_of(-1.0, ref) == 0.0  # below all
    assert advisory.percentile_of(2.0, ref) == 1.0  # above all


def test_percentile_of_degenerate_and_bad():
    assert advisory.percentile_of(0.5, []) is None  # empty reference
    assert advisory.percentile_of("x", [0.1, 0.2]) is None  # non-numeric value
    # A flat reference maps any value to the midrank 0.5 (no variation).
    assert advisory.percentile_of(0.99, [0.99, 0.99, 0.99]) == 0.5
    # Non-finite reference values are dropped.
    assert advisory.percentile_of(0.5, [float("nan"), 0.0, 1.0]) == 0.5


def test_relative_advisory_discriminates_within_reference():
    # A value near the top of its own weekly range reads Extreme even though the
    # absolute score (~0.92) would saturate any place at Extreme too.
    top = advisory.relative_advisory(max(WEEK), WEEK)
    assert top["level"] == "Extreme"
    assert top["basis"] == "relative_to_local_weekly_normal"
    assert top["percentile"] >= 0.95
    # A value near the bottom of the same range reads Low.
    low = advisory.relative_advisory(min(WEEK), WEEK)
    assert low["level"] == "Low"
    assert low["percentile"] < 0.5
    assert "below normal" in low["message"]


def test_relative_advisory_saturated_reference_is_moderate_not_extreme():
    flat = [0.996] * 50
    result = advisory.relative_advisory(0.996, flat)
    assert result["percentile"] == 0.5
    assert result["level"] == "Moderate"  # no weekly variation -> exactly typical


def test_relative_advisory_falls_back_to_absolute_when_no_reference():
    result = advisory.relative_advisory(0.4, [])
    assert result["basis"] == "absolute"
    assert result["percentile"] is None
    assert result["level"] == "High"  # absolute mapping of 0.4
    assert result["relative_descriptor"] is None


def test_relative_advisory_folds_drivers():
    result = advisory.relative_advisory(max(WEEK), WEEK, drivers=["wet roads", ""])
    assert result["drivers"] == ["wet roads"]
    assert "Key factors: wet roads." in result["message"]


def test_relative_descriptor_bands():
    assert advisory.relative_descriptor(0.2) == "below normal"
    assert advisory.relative_descriptor(0.6) == "near normal"
    assert advisory.relative_descriptor(0.8) == "above normal"
    assert advisory.relative_descriptor(0.9) == "well above normal"
    assert advisory.relative_descriptor(0.99) == "near this location's weekly peak"
