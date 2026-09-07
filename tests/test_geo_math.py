from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import geo_math


def test_haversine_reexport_la_to_ny():
    # ~3936 km between downtown LA and Manhattan.
    km = geo_math.haversine_km(34.0522, -118.2437, 40.7128, -74.0060)
    assert 3900 <= km <= 3980


def test_destination_point_east_at_equator():
    lat, lon = geo_math.destination_point(0.0, 0.0, 90.0, 111.19492664)  # ~1 deg
    assert lat == pytest.approx(0.0, abs=1e-6)
    assert lon == pytest.approx(1.0, abs=1e-4)


def test_destination_point_round_trips_distance():
    dest = geo_math.destination_point(34.0, -118.0, 45.0, 50.0)
    back = geo_math.haversine_km(34.0, -118.0, dest[0], dest[1])
    assert back == pytest.approx(50.0, abs=1e-3)


def test_destination_point_normalizes_longitude():
    # Heading west across the antimeridian stays in [-180, 180].
    _, lon = geo_math.destination_point(0.0, -179.0, 270.0, 500.0)
    assert -180.0 <= lon <= 180.0
    assert lon > 0  # wrapped past -180


def test_bbox_around_contains_point_and_spans_radius():
    min_lat, max_lat, min_lon, max_lon = geo_math.bbox_around(34.0, -118.0, 10.0)
    assert min_lat < 34.0 < max_lat and min_lon < -118.0 < max_lon
    # ~10 km is ~0.18 deg of latitude on each side.
    assert (max_lat - min_lat) == pytest.approx(2 * 10.0 / 110.574, rel=1e-6)


FACILITIES = [
    {"id": "a", "lat": 34.00, "lon": -118.00, "kind": "hospital"},
    {"id": "b", "lat": 34.05, "lon": -118.05, "kind": "shelter"},
    {"id": "c", "lat": 35.00, "lon": -119.00, "kind": "hospital"},
    {"bad": "no coords"},
    {"id": "d", "lat": "x", "lon": None},  # non-numeric -> skipped
]


def test_nearest_k_sorted_and_annotated():
    result = geo_math.nearest_k(34.01, -118.01, FACILITIES, k=2)
    assert [f["id"] for f in result] == ["a", "b"]
    assert result[0]["distance_km"] <= result[1]["distance_km"]
    assert result[0]["kind"] == "hospital"  # original fields preserved


def test_nearest_k_skips_malformed_and_respects_k():
    assert len(geo_math.nearest_k(0, 0, FACILITIES, k=99)) == 3  # only 3 valid
    assert geo_math.nearest_k(0, 0, FACILITIES, k=0) == []
    assert geo_math.nearest_k(0, 0, [], k=3) == []


def test_within_radius_km():
    hits = geo_math.within_radius_km(34.01, -118.01, FACILITIES, radius_km=10.0)
    assert [f["id"] for f in hits] == ["a", "b"]  # c is ~130 km away
    assert all(h["distance_km"] <= 10.0 for h in hits)


def test_accepts_lat_lon_tuples():
    pts = [(34.0, -118.0), (40.0, -74.0)]
    nearest = geo_math.nearest_k(34.01, -118.01, pts, k=1)
    assert nearest[0]["lat"] == 34.0 and nearest[0]["lon"] == -118.0
    assert "distance_km" in nearest[0]
