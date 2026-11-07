from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import work_zones as wz

WHEN = datetime(2026, 11, 13, tzinfo=timezone.utc)


def _feature(wz_id, coords, impact, start, end, *, flat=False, road="Test Rd"):
    timing = {"start_date": start, "end_date": end}
    if flat:
        props = {"road_names": [road], "vehicle_impact": impact, **timing}
    else:
        props = {
            "core_details": {"road_names": [road], "direction": "northbound"},
            "vehicle_impact": impact,
            **timing,
        }
    return {"id": wz_id, "type": "Feature",
            "geometry": {"type": "LineString", "coordinates": coords}, "properties": props}


def test_flatten_coords_handles_geometry_types():
    assert wz._flatten_coords([-118.0, 34.0]) == [(-118.0, 34.0)]  # Point
    assert wz._flatten_coords([[-118.0, 34.0], [-118.1, 34.1]]) == [(-118.0, 34.0), (-118.1, 34.1)]
    multi = wz._flatten_coords([[[-118.0, 34.0]], [[-118.1, 34.1]]])  # MultiLineString
    assert (-118.1, 34.1) in multi


def test_active_filtering_by_window():
    store = wz.WorkZoneStore.from_records(
        [
            _feature("live", [[-118.0, 34.0], [-118.0, 34.0]], "some-lanes-closed",
                     "2026-01-01T00:00:00Z", "2027-01-01T00:00:00Z"),
            _feature("expired", [[-118.0, 34.0], [-118.0, 34.0]], "all-lanes-closed",
                     "2026-01-01T00:00:00Z", "2026-02-01T00:00:00Z"),
        ]
    )
    active_ids = {z["id"] for z in store.all(when=WHEN)}
    assert active_ids == {"live"}  # the expired event is filtered out
    # No time filter -> everything is returned.
    assert len(store.all()) == 2


def test_point_impact_multiplier_by_vehicle_impact():
    store = wz.WorkZoneStore.from_records(
        [
            _feature("closed", [[-118.00, 34.00], [-118.00, 34.00]], "all-lanes-closed",
                     "2026-01-01T00:00:00Z", "2027-12-31T00:00:00Z"),
        ]
    )
    on_zone = store.point_impact(34.0, -118.0, when=WHEN, radius_km=1.0)
    assert on_zone["in_work_zone"] is True
    assert on_zone["risk_multiplier"] == wz.IMPACT_RISK["all-lanes-closed"]
    assert on_zone["nearest"]["id"] == "closed"

    away = store.point_impact(40.0, -75.0, when=WHEN, radius_km=1.0)
    assert away == {"in_work_zone": False, "count": 0, "nearest": None, "risk_multiplier": 1.0}


def test_flat_layout_and_unknown_impact_default():
    store = wz.WorkZoneStore.from_records(
        [_feature("flat", [[-118.0, 34.0], [-118.0, 34.0]], "mystery", "2026-01-01T00:00:00Z", None,
                  flat=True, road="Flat Ave")]
    )
    zone = store.all()[0]
    assert zone["road_names"] == ["Flat Ave"]  # read from a flat (non-core_details) layout
    impact = store.point_impact(34.0, -118.0, when=WHEN)
    assert impact["risk_multiplier"] == wz.DEFAULT_IMPACT_RISK  # unknown impact -> default


def test_malformed_features_are_skipped():
    store = wz.WorkZoneStore.from_records(
        [
            None,
            "not a feature",
            {"type": "Feature", "geometry": None, "properties": {}},  # no coordinates
            _feature("ok", [[-118.0, 34.0], [-118.0, 34.0]], "some-lanes-closed",
                     "2026-01-01T00:00:00Z", None),
        ]
    )
    assert len(store) == 1
    assert store.all()[0]["id"] == "ok"


def test_missing_feed_file_yields_empty_store(tmp_path):
    store = wz.WorkZoneStore.from_geojson(tmp_path / "nope.json")
    assert len(store) == 0


def test_reference_feed_loads_and_excludes_expired():
    store = wz.load_work_zone_store()
    assert len(store) >= 5
    active = store.all(when=WHEN)
    assert all(z["id"] != "wz-sea-i5-006-expired" for z in active)
    # Near the LA I-10 sample event.
    la = store.point_impact(34.0340, -118.2450, when=WHEN, radius_km=1.5)
    assert la["in_work_zone"] is True
