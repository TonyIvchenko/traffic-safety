from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import road_tiles as rt


def test_road_kind_from_attrs_classifies_by_rttyp_and_mtfcc():
    assert rt.road_kind_from_attrs("I", None) == 2
    assert rt.road_kind_from_attrs(None, "S1100") == 2
    assert rt.road_kind_from_attrs("U", None) == 1
    assert rt.road_kind_from_attrs("S", None) == 1
    assert rt.road_kind_from_attrs(None, "S1200") == 1
    assert rt.road_kind_from_attrs("C", "S1400") == 0
    assert rt.road_kind_from_attrs(None, None) == 0
    # Inputs are normalized for case and whitespace.
    assert rt.road_kind_from_attrs(" i ", None) == 2


def test_mercator_world_pixel_maps_lon_lat_to_expected_pixels():
    scale = 256.0  # zoom 0
    center_x, center_y = rt.mercator_world_pixel(0.0, 0.0, 0)
    assert center_x == pytest.approx(scale / 2.0)
    assert center_y == pytest.approx(scale / 2.0)

    east_x, _ = rt.mercator_world_pixel(0.0, 180.0, 0)
    west_x, _ = rt.mercator_world_pixel(0.0, -180.0, 0)
    assert east_x == pytest.approx(scale)
    assert west_x == pytest.approx(0.0)

    # Northern latitudes map to smaller y (towards the top of the map).
    _, north_y = rt.mercator_world_pixel(45.0, 0.0, 0)
    assert north_y < center_y


def test_mercator_world_pixel_clamps_extreme_latitudes():
    _, near_pole = rt.mercator_world_pixel(89.9, 0.0, 0)
    _, clamp_pole = rt.mercator_world_pixel(85.05112878, 0.0, 0)
    assert near_pole == pytest.approx(clamp_pole, abs=1e-6)
    assert -1e-6 <= near_pole <= 256.0 + 1e-6


def test_encode_relative_path_round_trips_scaled_int16():
    points = [(0.0, 0.0), (1.0, 2.5), (-3.0, 4.0)]
    blob = rt.encode_relative_path(points)
    decoded = np.frombuffer(blob, dtype=np.int16).reshape(-1, 2)
    expected = np.array(
        [[round(x * rt.ROAD_TILE_COORD_SCALE), round(y * rt.ROAD_TILE_COORD_SCALE)] for x, y in points]
    )
    assert np.array_equal(decoded, expected)


def test_clip_segment_to_rect_inside_outside_and_crossing():
    inside = rt._clip_segment_to_rect(1.0, 1.0, 2.0, 2.0, 0.0, 0.0, 10.0, 10.0)
    assert inside == ((1.0, 1.0), (2.0, 2.0))

    outside = rt._clip_segment_to_rect(20.0, 20.0, 30.0, 30.0, 0.0, 0.0, 10.0, 10.0)
    assert outside is None

    crossing = rt._clip_segment_to_rect(-5.0, 5.0, 5.0, 5.0, 0.0, 0.0, 10.0, 10.0)
    assert crossing is not None
    (start, end) = crossing
    assert start == pytest.approx((0.0, 5.0))
    assert end == pytest.approx((5.0, 5.0))


def test_points_close_respects_tolerance():
    assert rt._points_close((1.0, 1.0), (1.0, 1.0 + 1e-9))
    assert not rt._points_close((1.0, 1.0), (1.0, 1.1))


def test_iter_tile_paths_emits_encoded_paths_within_tile_grid():
    coords = [(-118.0, 34.0), (-117.99, 34.01), (-117.98, 34.0)]
    zoom = 10
    paths = rt.iter_tile_paths(coords, zoom)

    assert paths, "expected at least one tile path"
    tile_count = 2**zoom
    for tile_x, tile_y, blob in paths:
        assert 0 <= tile_x < tile_count
        assert 0 <= tile_y < tile_count
        decoded = np.frombuffer(blob, dtype=np.int16)
        assert decoded.size % 2 == 0
        assert decoded.size >= 4  # at least two points per drawable path


def test_iter_tile_paths_ignores_degenerate_input():
    assert rt.iter_tile_paths([(-118.0, 34.0)], 10) == []
    assert rt.iter_tile_paths([], 10) == []


def _within_tile_box(point, tile_x=0, tile_y=0, tol=1e-6):
    x, y = point
    min_x, min_y = tile_x * 256.0, tile_y * 256.0
    return (min_x - tol <= x <= min_x + 256.0 + tol) and (
        min_y - tol <= y <= min_y + 256.0 + tol
    )


def test_clip_polyline_keeps_fully_interior_path():
    world_points = [(50.0, 50.0), (100.0, 100.0), (150.0, 150.0)]
    paths = rt._clip_polyline_to_tile(world_points, 0, 0)
    assert paths == [world_points]


def test_clip_polyline_drops_fully_exterior_path():
    world_points = [(300.0, 300.0), (400.0, 400.0)]
    assert rt._clip_polyline_to_tile(world_points, 0, 0) == []


def test_clip_polyline_trims_segment_at_boundary():
    world_points = [(128.0, 128.0), (400.0, 128.0)]
    paths = rt._clip_polyline_to_tile(world_points, 0, 0)
    assert len(paths) == 1
    assert paths[0][0] == pytest.approx((128.0, 128.0))
    assert paths[0][-1] == pytest.approx((256.0, 128.0))


def test_clip_polyline_splits_when_leaving_and_reentering():
    world_points = [
        (50.0, 128.0),   # inside
        (100.0, 128.0),  # inside
        (300.0, 400.0),  # outside (down-right)
        (310.0, 410.0),  # outside -> this segment is fully outside, forcing a break
        (120.0, 128.0),  # back inside
        (150.0, 128.0),  # inside
    ]
    paths = rt._clip_polyline_to_tile(world_points, 0, 0)
    assert len(paths) == 2
    for path in paths:
        assert len(path) >= 2
        assert all(_within_tile_box(point) for point in path)
