from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import region_index

# A 11x11 synthetic grid over lat/lon [0, 10] x [0, 10]: cell (row, col) has
# risk (row + col) / 20, so risk climbs toward the south-east corner.
COVERAGE = {"lat_min": 0.0, "lat_max": 10.0, "lon_min": 0.0, "lon_max": 10.0}


def _frame(offset: float = 0.0) -> list[list[float]]:
    return [[(row + col) / 20.0 + offset for col in range(11)] for row in range(11)]


def _cube(*offsets: float) -> list[list[list[float]]]:
    return [_frame(off) for off in (offsets or (0.0,))]


# bbox [min_lat, max_lat, min_lon, max_lon] = [2, 4, 2, 4] -> rows 6..8, cols 2..4.
BBOX_MID = [2.0, 4.0, 2.0, 4.0]


def test_sample_selects_expected_cells():
    values = region_index.sample_region_cells(_cube(), COVERAGE, BBOX_MID, frame_idx=0)
    assert len(values) == 9  # 3 rows x 3 cols
    assert min(values) == 0.40 and max(values) == 0.60


def test_reduce_statistics():
    values = region_index.sample_region_cells(_cube(), COVERAGE, BBOX_MID, frame_idx=0)
    assert region_index._reduce(values, "mean") == pytest.approx(0.50)
    assert region_index._reduce(values, "max") == 0.60
    assert region_index._reduce(values, "min") == 0.40
    assert abs(region_index._reduce(values, "p90") - 0.56) < 1e-9


def test_region_risk_score_defaults_to_p90():
    score = region_index.region_risk_score(_cube(), COVERAGE, BBOX_MID, frame_idx=0)
    assert abs(score - 0.56) < 1e-9


def test_frame_idx_none_samples_all_frames():
    values = region_index.sample_region_cells(_cube(0.0, 0.1), COVERAGE, BBOX_MID)
    assert len(values) == 18  # 9 cells x 2 frames
    # Second frame shifted up by 0.1, so the mean rises by 0.05.
    assert abs(region_index._reduce(values, "mean") - 0.55) < 1e-9


def test_region_index_maps_to_advisory():
    region = {"id": "mid", "name": "Mid", "bbox": BBOX_MID}
    idx = region_index.region_index(_cube(), COVERAGE, region, frame_idx=0)
    assert idx["region_id"] == "mid"
    assert idx["sample_count"] == 9
    assert abs(idx["risk_score"] - 0.56) < 1e-4
    assert idx["risk_mean"] == 0.5 and idx["risk_max"] == 0.6
    assert idx["advisory"]["level"] == "Extreme"  # 0.56 >= 0.50 threshold
    assert idx["bbox"] == BBOX_MID


def test_min_risk_filters_low_cells():
    values = region_index.sample_region_cells(
        _cube(), COVERAGE, BBOX_MID, frame_idx=0, min_risk=0.5
    )
    assert values and all(v >= 0.5 for v in values)


def test_index_all_regions_sorted_desc():
    hi = {"id": "hi", "name": "Hi", "bbox": [2.0, 4.0, 2.0, 4.0]}  # p90 ~0.56
    lo = {"id": "lo", "name": "Lo", "bbox": [8.0, 10.0, 0.0, 2.0]}  # low corner
    ranked = region_index.index_all_regions(
        _cube(), COVERAGE, region_catalog=[lo, hi], frame_idx=0
    )
    assert [r["region_id"] for r in ranked] == ["hi", "lo"]
    assert ranked[0]["advisory"]["level"] == "Extreme"
    assert ranked[1]["advisory"]["level"] in {"Low", "Moderate"}


def test_region_by_id_string_resolves_from_catalog():
    idx = region_index.region_index(_cube(), COVERAGE, "los_angeles", frame_idx=0)
    assert idx is not None and idx["region_id"] == "los_angeles"


def test_unknown_region_returns_none():
    assert region_index.region_index(_cube(), COVERAGE, "atlantis", frame_idx=0) is None


def test_degrades_on_empty_cube():
    idx = region_index.region_index([], COVERAGE, {"id": "x", "name": "X", "bbox": BBOX_MID})
    assert idx["sample_count"] == 0
    assert idx["risk_score"] == 0.0
    assert idx["advisory"]["level"] == "Low"


def test_degrades_on_bbox_outside_coverage():
    far = {"id": "far", "name": "Far", "bbox": [50.0, 60.0, 50.0, 60.0]}
    idx = region_index.region_index(_cube(), COVERAGE, far, frame_idx=0)
    assert idx["sample_count"] == 0
    assert idx["risk_score"] == 0.0


def test_nan_cells_are_dropped():
    frame = _frame()
    frame[6][2] = float("nan")
    frame[7][3] = float("inf")
    values = region_index.sample_region_cells([frame], COVERAGE, BBOX_MID, frame_idx=0)
    assert len(values) == 7  # 9 minus the two non-finite cells
    assert all(v == v for v in values)  # no NaN survived


def test_bad_frame_idx_and_bad_inputs_do_not_raise():
    assert region_index.sample_region_cells(_cube(), COVERAGE, BBOX_MID, frame_idx="x") == []
    assert region_index.region_risk_score(None, COVERAGE, BBOX_MID) == 0.0
    # Out-of-range frame clamps into the cube rather than erroring.
    assert region_index.region_risk_score(_cube(), COVERAGE, BBOX_MID, frame_idx=999) > 0.0


# --------------------------------------------------------------------------- #
# Live regional index
# --------------------------------------------------------------------------- #


def test_grid_sample_points_interior():
    points = region_index.grid_sample_points([2.0, 4.0, 2.0, 4.0], rows=3, cols=3)
    assert len(points) == 9
    # Every point strictly inside the bbox (no edge points).
    for p in points:
        assert 2.0 < p["lat"] < 4.0 and 2.0 < p["lon"] < 4.0
    # Centre of a 3x3 grid is the bbox centre.
    assert points[4] == {"lat": 3.0, "lon": 3.0}


def test_grid_sample_points_degenerate_bbox():
    assert region_index.grid_sample_points([4.0, 2.0, 2.0, 4.0]) == []  # min_lat > max_lat
    assert region_index.grid_sample_points("nope") == []


def _fixed_predictor(score):
    def predict(lat, lon):
        return {"risk_score": score, "cell_id": f"{lat:.2f},{lon:.2f}"}

    return predict


def test_live_region_index_aggregates():
    region = {"id": "mid", "name": "Mid", "bbox": [2.0, 4.0, 2.0, 4.0]}
    idx = region_index.live_region_index(_fixed_predictor(0.42), region, rows=3, cols=3)
    assert idx["mode"] == "live" and idx["status"] == "ok"
    assert idx["sample_points"] == 9 and idx["sample_count"] == 9 and idx["failed_points"] == 0
    assert idx["risk_score"] == pytest.approx(0.42)
    assert idx["advisory"]["level"] == "High"  # 0.42 in [0.35, 0.50)


def test_live_region_index_accepts_bare_number_and_clamps():
    region = {"id": "x", "name": "X", "bbox": [2.0, 4.0, 2.0, 4.0]}

    def predict(lat, lon):
        return 5.0  # out of range -> clamped to 1.0

    idx = region_index.live_region_index(predict, region)
    assert idx["risk_score"] == 1.0
    assert idx["advisory"]["level"] == "Extreme"


def test_live_region_index_counts_partial_failures():
    region = {"id": "x", "name": "X", "bbox": [2.0, 4.0, 2.0, 4.0]}
    calls = {"n": 0}

    def flaky(lat, lon):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            raise RuntimeError("provider hiccup")
        return {"risk_score": 0.3}

    idx = region_index.live_region_index(flaky, region, rows=3, cols=3)
    assert idx["status"] == "ok"
    assert idx["sample_count"] + idx["failed_points"] == 9
    assert idx["failed_points"] > 0


def test_live_region_index_unavailable_when_all_fail():
    region = {"id": "x", "name": "X", "bbox": [2.0, 4.0, 2.0, 4.0]}

    def down(lat, lon):
        raise RuntimeError("provider down")

    idx = region_index.live_region_index(down, region)
    assert idx["status"] == "unavailable"
    assert idx["sample_count"] == 0
    assert idx["advisory"] is None
    assert idx["risk_score"] == 0.0


def test_live_region_index_unknown_region_is_none():
    assert region_index.live_region_index(_fixed_predictor(0.2), "atlantis") is None


def test_compare_to_normal_branches():
    assert region_index.compare_to_normal(0.5, 0.2)["comparison"] == "worse than normal"
    assert region_index.compare_to_normal(0.2, 0.5)["comparison"] == "better than normal"
    assert region_index.compare_to_normal(0.30, 0.32)["comparison"] == "about normal"
    worse = region_index.compare_to_normal(0.5, 0.25)
    assert worse["delta"] == pytest.approx(0.25) and worse["ratio"] == pytest.approx(2.0)
    # Zero baseline -> ratio undefined (None), not a division error.
    assert region_index.compare_to_normal(0.4, 0.0)["ratio"] is None
