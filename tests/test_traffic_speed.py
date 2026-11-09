from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import traffic_speed as ts


def test_congestion_ratio_and_degrade():
    assert ts.congestion_ratio(30, 60) == 0.5
    assert ts.congestion_ratio(30, 0) is None  # zero free-flow
    assert ts.congestion_ratio(None, 60) is None
    assert ts.congestion_ratio(30, float("nan")) is None


def test_speed_risk_index_is_u_shaped():
    assert ts.speed_risk_index(20, 65) == ts.HEAVY_CONGESTION_RISK  # ratio ~0.31
    assert ts.speed_risk_index(45, 65) == ts.MODERATE_CONGESTION_RISK  # ratio ~0.69
    assert ts.speed_risk_index(60, 65) == ts.NEUTRAL_RISK  # near free-flow
    assert ts.speed_risk_index(80, 65) == ts.SPEEDING_RISK  # well above free-flow
    assert ts.speed_risk_index(None, 65) == ts.NEUTRAL_RISK  # unknown -> neutral


def test_period_for_hour():
    assert ts.period_for_hour(8) == "peak"
    assert ts.period_for_hour(17) == "peak"
    assert ts.period_for_hour(13) == "offpeak"
    assert ts.period_for_hour(None) == "peak"  # default


def test_store_lookup_peak_offpeak_and_fallback():
    store = ts.TrafficSpeedStore.from_records(
        [
            {"id": "x", "name": "Test Fwy", "lat": 34.0, "lon": -118.0,
             "reference_speed_mph": 65, "peak_speed_mph": 20, "offpeak_speed_mph": 60},
            {"id": "bad", "lat": None, "lon": -118.0, "reference_speed_mph": 60},  # dropped
        ]
    )
    assert len(store) == 1

    peak = store.lookup(34.001, -118.001, period="peak", max_km=3.0)
    assert peak["basis"] == "npmrds_segment"
    assert peak["observed_speed_mph"] == 20
    assert peak["speed_risk_index"] == ts.HEAVY_CONGESTION_RISK
    assert peak["segment_name"] == "Test Fwy"

    offpeak = store.lookup(34.001, -118.001, period="offpeak", max_km=3.0)
    assert offpeak["observed_speed_mph"] == 60
    assert offpeak["speed_risk_index"] == ts.NEUTRAL_RISK

    far = store.lookup(40.0, -75.0, max_km=3.0)
    assert far["basis"] == "unavailable"
    assert far["speed_risk_index"] == ts.NEUTRAL_RISK
    assert far["observed_speed_mph"] is None


def test_reference_store_loads():
    store = ts.load_traffic_speed_store()
    assert len(store) > 0
    la = store.lookup(34.0335, -118.2487, period="peak", max_km=3.0)
    assert la["basis"] == "npmrds_segment"
    assert la["speed_risk_index"] > 1.0  # the LA sample is congested at peak
