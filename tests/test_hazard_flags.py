from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import hazard_flags as hf


def test_ice_risk_requires_cold_and_moisture():
    assert hf.ice_risk(-3.0, 1.0) > 0.0        # sub-freezing + wet
    assert hf.ice_risk(10.0, 1.0) == 0.0       # warm
    assert hf.ice_risk(-3.0, 0.0, 40.0) == 0.0  # cold but dry
    assert hf.ice_risk(-2.0, 0.0, 95.0) > 0.0  # black ice: humid + freezing
    assert hf.ice_risk(None, 1.0) == 0.0


def test_ice_risk_scales_with_cold():
    assert hf.ice_risk(-5.0, 1.0) == pytest.approx(1.0)
    assert hf.ice_risk(-5.0, 1.0) > hf.ice_risk(0.0, 1.0)


def test_fog_risk_requires_saturation_and_calm():
    assert hf.fog_risk(10.0, 9.0, 96.0, 1.0) > 0.0   # small spread, humid, calm
    assert hf.fog_risk(20.0, 5.0, 40.0, 2.0) == 0.0  # dry, wide spread
    assert hf.fog_risk(10.0, 9.5, 96.0, 8.0) == 0.0  # too windy
    assert hf.fog_risk(None, 9.0, 96.0, 1.0) == 0.0


def test_assess_hazards_labels():
    hazards = hf.assess_hazards(
        {
            "temp_c": -2.0,
            "dewpoint_c": -2.5,
            "relative_humidity_pct": 97.0,
            "wind_speed_mps": 1.0,
            "wet_hour": 1.0,
        }
    )
    assert hazards["ice_risk"] > 0.0
    assert hazards["fog_risk"] > 0.0
    assert hazards["wet"] is True
    assert set(hazards["labels"]) == {"ice", "fog", "wet"}

    clear = hf.assess_hazards(
        {"temp_c": 22.0, "dewpoint_c": 8.0, "relative_humidity_pct": 40.0, "wind_speed_mps": 5.0, "wet_hour": 0.0}
    )
    assert clear["labels"] == []
