from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import exposure


def test_annual_vmt_and_degrade():
    # 50k AADT on a 1 km segment for a year: 50000 * (1 km in miles) * 365.
    vmt = exposure.annual_vmt(50000, 1.0)
    assert vmt == 50000 * exposure.MILES_PER_KM * 365
    # Missing / non-finite / negative inputs degrade to None, never raise.
    assert exposure.annual_vmt(None, 1.0) is None
    assert exposure.annual_vmt(50000, float("nan")) is None
    assert exposure.annual_vmt(-5, 1.0) is None
    assert exposure.annual_vmt(10**400, 1.0) is None  # OverflowError path


def test_crashes_per_mvmt_is_exposure_normalized():
    rate = exposure.crashes_per_mvmt(10, 50000, 1.0, years=5)
    assert rate is not None and rate > 0
    # A busier road with the same crash count is *less* risky per VMT.
    busier = exposure.crashes_per_mvmt(10, 100000, 1.0, years=5)
    assert busier < rate
    # No exposure or no time window -> None (caller falls back).
    assert exposure.crashes_per_mvmt(10, 0, 1.0, years=5) is None
    assert exposure.crashes_per_mvmt(10, 50000, 1.0, years=0) is None
    assert exposure.crashes_per_mvmt(-1, 50000, 1.0) is None


def test_functional_class_resolution_and_defaults():
    assert exposure.normalize_class("Principal Arterial") == "principal_arterial"
    assert exposure.normalize_class("nonsense") is None
    assert exposure.functional_class_for(mtfcc="S1100") == "freeway_expressway"
    assert exposure.default_aadt("interstate") == exposure.FUNCTIONAL_CLASS_AADT["interstate"]
    assert exposure.default_aadt("nonsense") is None


def test_store_nearest_match_and_fallback():
    store = exposure.AADTStore.from_records(
        [
            {"name": "Main Fwy", "lat": 34.0, "lon": -118.0, "aadt": 200000, "functional_class": "interstate", "length_km": 1.6},
            {"name": "bad", "lat": None, "lon": -118.0, "aadt": 5000},  # dropped: no lat
        ]
    )
    assert len(store) == 1  # the malformed record is filtered out

    # A point right on the segment matches it.
    hit = store.lookup(34.001, -118.001, max_km=3.0)
    assert hit["basis"] == "hpms_segment"
    assert hit["aadt"] == 200000
    assert hit["segment_name"] == "Main Fwy"
    assert hit["distance_km"] is not None

    # Far away with a known class -> class default; without one -> unavailable.
    far = store.lookup(40.0, -75.0, functional_class="local", max_km=3.0)
    assert far["basis"] == "class_default"
    assert far["aadt"] == exposure.FUNCTIONAL_CLASS_AADT["local"]
    unknown = store.lookup(40.0, -75.0, max_km=3.0)
    assert unknown["basis"] == "unavailable" and unknown["aadt"] is None


def test_reference_store_loads():
    store = exposure.load_aadt_store()
    assert len(store) > 0
    la = store.lookup(34.0335, -118.2487, max_km=3.0)  # near I-10 DTLA sample
    assert la["basis"] == "hpms_segment"
    assert la["aadt"] > 0
