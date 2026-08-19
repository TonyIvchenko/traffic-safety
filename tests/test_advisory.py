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
