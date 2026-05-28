from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segment_support as ss

# One degree of latitude in km for the configured earth radius.
ONE_DEG_KM = 3.141592653589793 / 180.0 * ss.EARTH_RADIUS_KM


def test_haversine_zero_and_one_degree():
    assert ss.haversine_km(0.0, 0.0, 0.0, 0.0) == 0.0
    assert ss.haversine_km(0.0, 0.0, 1.0, 0.0) == pytest.approx(ONE_DEG_KM, rel=1e-6)


def test_polyline_length_sums_segments():
    # points are (lon, lat)
    points = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
    assert ss.polyline_length_km(points) == pytest.approx(2.0 * ONE_DEG_KM, rel=1e-6)


def test_interpolate_point_midpoint():
    mid = ss.interpolate_point((0.0, 0.0), (10.0, 20.0), 0.5)
    assert mid == pytest.approx((5.0, 10.0))


def test_densify_polyline_splits_long_edges():
    points = [(0.0, 0.0), (0.0, 1.0)]  # ~111 km edge
    dense = ss.densify_polyline(points, max_edge_km=50.0)
    assert len(dense) > 2
    for idx in range(1, len(dense)):
        lon1, lat1 = dense[idx - 1]
        lon2, lat2 = dense[idx]
        assert ss.haversine_km(lat1, lon1, lat2, lon2) <= 50.0 + 1e-6


def test_densify_polyline_keeps_short_input():
    assert ss.densify_polyline([(0.0, 0.0)], max_edge_km=10.0) == [(0.0, 0.0)]


def test_segmentize_polyline_respects_max_length():
    points = [(0.0, 0.0), (0.0, 1.0), (0.0, 2.0)]
    segments = ss.segmentize_polyline(points, max_length_km=40.0)
    assert segments
    for segment in segments:
        assert ss.polyline_length_km(segment) <= 40.0 + 1e-6


def test_polyline_bounds_and_centroid():
    points = [(-1.0, 2.0), (3.0, -4.0), (1.0, 0.0)]
    min_lat, max_lat, min_lon, max_lon = ss.polyline_bounds(points)
    assert (min_lat, max_lat, min_lon, max_lon) == (-4.0, 2.0, -1.0, 3.0)

    centroid_lat, centroid_lon = ss.polyline_centroid(points)
    assert centroid_lat == pytest.approx((2.0 - 4.0 + 0.0) / 3.0)
    assert centroid_lon == pytest.approx((-1.0 + 3.0 + 1.0) / 3.0)


def test_to_local_xy_km_reference_offsets():
    assert ss.to_local_xy_km(0.0, 0.0, 0.0, 0.0) == (0.0, 0.0)
    x, y = ss.to_local_xy_km(0.0, 1.0, 0.0, 0.0)
    assert x == pytest.approx(111.32)
    assert y == pytest.approx(0.0)


def test_point_to_polyline_distance_on_and_off_line():
    polyline = [(0.0, 0.0), (0.0, 1.0)]  # vertical line along lon=0
    on_line = ss.point_to_polyline_distance_km(0.5, 0.0, polyline)
    assert on_line == pytest.approx(0.0, abs=1e-6)

    off_line = ss.point_to_polyline_distance_km(0.5, 0.1, polyline)
    assert off_line == pytest.approx(0.1 * 111.32, rel=1e-3)


def test_point_to_polyline_distance_requires_two_points():
    assert ss.point_to_polyline_distance_km(0.0, 0.0, [(0.0, 0.0)]) == float("inf")


def test_coords_json_round_trip():
    points = [(-118.25, 34.05), (-118.24, 34.06)]
    payload = ss.coords_to_json(points)
    restored = ss.coords_from_json(payload)
    assert restored == [pytest.approx(p) for p in points]
