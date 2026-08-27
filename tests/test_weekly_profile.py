"""The batched weekly risk profile must equal per-frame predict_traffic_safety.

weekly_risk_profile scores all 168 weekly frames in one predict_proba call; this
pins it to the identical values the per-call path produces (the whole point is
speed without changing outputs), plus the out-of-coverage and shape contracts.
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import predict

pytestmark = pytest.mark.skipif(not predict.MODEL_BUNDLE, reason="model bundle unavailable")

# In-coverage (downtown LA) and out-of-coverage (mid-Atlantic ocean).
IN_COVERAGE = (34.0522, -118.2437)
OUT_OF_COVERAGE = (0.0, -30.0)


def _per_call_profile(lat: float, lon: float, month: int) -> np.ndarray:
    return np.array(
        [
            predict.predict_traffic_safety(
                lat, lon, day_of_week=how // 24 + 1, hour=how % 24, month=month
            )["risk_score"]
            for how in range(168)
        ],
        dtype=np.float32,
    )


@pytest.mark.parametrize("month", [1, 9])
def test_batched_matches_per_call(month):
    lat, lon = IN_COVERAGE
    batched = predict.weekly_risk_profile(lat, lon, month=month)
    per_call = _per_call_profile(lat, lon, month)
    assert batched.shape == (168,)
    np.testing.assert_allclose(batched, per_call, atol=1e-6)


def test_profile_shape_and_bounds():
    profile = predict.weekly_risk_profile(*IN_COVERAGE, month=1)
    assert profile.shape == (168,)
    assert float(profile.min()) >= 0.0 and float(profile.max()) <= 1.0


def test_out_of_coverage_is_all_zero():
    profile = predict.weekly_risk_profile(*OUT_OF_COVERAGE, month=1)
    assert profile.shape == (168,)
    assert not profile.any()


def test_batched_is_faster_than_per_call():
    import time

    lat, lon = IN_COVERAGE
    t0 = time.perf_counter()
    predict.weekly_risk_profile(lat, lon, month=1)
    batched = time.perf_counter() - t0

    t0 = time.perf_counter()
    _per_call_profile(lat, lon, 1)
    per_call = time.perf_counter() - t0

    # A large margin so the assertion is robust to machine noise.
    assert batched < per_call / 5.0
