from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import vru_runtime

VRU_AVAILABLE = vru_runtime.VRU_MODEL_PATH.exists()


def test_empty_bundle_degrades_not_crashes():
    layer = vru_runtime.VRULayer({})
    assert layer.ready is False

    point = layer.score_point(34.05, -118.24, day_of_week=6, hour=20, month=9)
    assert point["layer"] == "vru"
    assert point["in_coverage"] is False
    assert point["risk_score"] == 0.0
    assert point["risk_level"] == "low"

    profile = layer.weekly_profile(34.05, -118.24, month=9)
    assert len(profile) == vru_runtime.WEEKLY_FRAMES
    assert set(profile) == {0.0}


@pytest.mark.skipif(not VRU_AVAILABLE, reason="VRU model bundle not present")
def test_vru_layer_scores_a_covered_cell():
    layer = vru_runtime.load_vru_layer()
    assert layer.ready is True
    assert layer.target == "pedestrian_or_cyclist_crash"

    lats = np.asarray(layer.bundle["candidate_lats"], dtype=float)
    lons = np.asarray(layer.bundle["candidate_lons"], dtype=float)
    lat, lon = float(lats[0]), float(lons[0])

    point = layer.score_point(lat, lon, day_of_week=6, hour=20, month=9)
    assert point["in_coverage"] is True
    assert 0.0 <= point["risk_score"] <= 1.0
    assert point["risk_level"] in {"low", "moderate", "high", "extreme"}

    profile = layer.weekly_profile(lat, lon, month=9)
    assert len(profile) == vru_runtime.WEEKLY_FRAMES
    assert max(profile) > 0.0  # a covered cell has real weekly variation


@pytest.mark.skipif(not VRU_AVAILABLE, reason="VRU model bundle not present")
def test_vru_out_of_coverage_point_is_zero():
    layer = vru_runtime.load_vru_layer()
    ocean = layer.score_point(0.0, 0.0, day_of_week=1, hour=0, month=1)
    assert ocean["in_coverage"] is False
    assert ocean["risk_score"] == 0.0
