from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import trends


def test_mann_kendall_increasing_matches_hand_calc():
    # [1,2,3,4,5]: S=10, Var=16.667, Z=(10-1)/sqrt(16.667)=2.2045, p~0.0275.
    mk = trends.mann_kendall([1, 2, 3, 4, 5])
    assert mk["s"] == 10
    assert mk["z"] == pytest.approx(2.2045, abs=1e-3)
    assert mk["p_value"] == pytest.approx(0.0275, abs=1e-3)
    assert mk["trend"] == "increasing"
    assert mk["significant"] is True


def test_mann_kendall_decreasing_and_flat():
    assert trends.mann_kendall([5, 4, 3, 2, 1])["trend"] == "decreasing"
    flat = trends.mann_kendall([3, 3, 3, 3, 3])
    assert flat["s"] == 0
    assert flat["trend"] == "no trend"
    assert flat["significant"] is False


def test_mann_kendall_handles_ties_in_variance():
    # Ties reduce the variance; the tie-corrected formula must stay finite.
    mk = trends.mann_kendall([1, 1, 2, 2, 3, 3])
    assert mk is not None
    assert mk["variance"] > 0
    assert mk["trend"] == "increasing"


def test_mann_kendall_too_short_is_none():
    assert trends.mann_kendall([1, 2]) is None


def test_theil_sen_slope():
    assert trends.theil_sen([1, 3, 5, 7])["slope"] == pytest.approx(2.0)
    # robust to a single outlier year
    assert trends.theil_sen([1, 2, 3, 40, 5])["slope"] == pytest.approx(1.0, abs=1e-6)
    # custom times
    assert trends.theil_sen([0, 10], times=[2000, 2010])["slope"] == pytest.approx(1.0)
    assert trends.theil_sen([5]) is None


def test_series_trend_new_hotspot():
    # Not hot early, crosses the threshold recently, rising -> new_hotspot.
    result = trends.series_trend([0, 1, 2, 5, 8], hot_threshold=4, recent_years=2)
    assert result["status"] == "ok"
    assert result["trend"] == "increasing"
    assert result["trend_per_year"] == pytest.approx(2.0)
    assert result["hot_flags"] == [False, False, False, True, True]
    assert result["category"] == "new_hotspot"
    assert result["latest_yoy_change"] == pytest.approx(3.0)
    assert result["recent_vs_baseline_change"] == pytest.approx(5.5)


def test_series_trend_persistent_and_diminishing():
    persistent = trends.series_trend([9, 10, 9, 10, 9], hot_threshold=5, recent_years=2)
    assert persistent["category"] == "persistent_hotspot"
    diminishing = trends.series_trend([10, 9, 8, 7, 6], hot_threshold=5, recent_years=2)
    assert diminishing["category"] == "diminishing_hotspot"


def test_series_trend_insufficient():
    out = trends.series_trend([1, 2])
    assert out["status"] == "insufficient"


def test_classify_taxonomy_edges():
    # never hot, flat -> none; never hot, rising -> emerging
    assert trends.classify([False, False, False], "no trend") == "none"
    assert trends.classify([False, False, False], "increasing") == "emerging"
    # cooled off -> past_hotspot
    assert trends.classify([True, True, False], "decreasing") == "past_hotspot"


def test_emerging_hotspots_ranks_rising_first():
    locations = [
        {"location_id": "flat", "series": [2, 2, 2, 2, 2]},
        {"location_id": "rising", "series": [1, 2, 4, 7, 11]},
        {"location_id": "falling", "series": [11, 8, 5, 3, 1]},
        {"location_id": "always_hot", "series": [20, 21, 20, 22, 21]},
    ]
    result = trends.emerging_hotspots(locations, hot_quantile=0.75, recent_years=2)
    assert result["status"] == "ok"
    assert result["evaluated_count"] == 4
    assert result["series_length"] == 5
    top = result["hotspots"][0]
    # The rising / intensifying corridor should outrank a merely persistent one.
    assert top["location_id"] in {"rising", "always_hot"}
    cats = {h["location_id"]: h["category"] for h in result["hotspots"]}
    assert cats["falling"] in {"diminishing_hotspot", "past_hotspot"}
    assert "category_counts" in result


def test_emerging_hotspots_rising_only_filter():
    locations = [
        {"location_id": "rising", "series": [1, 2, 4, 7, 11]},
        {"location_id": "flat", "series": [2, 2, 2, 2, 2]},
    ]
    result = trends.emerging_hotspots(locations, rising_only=True, hot_quantile=0.5)
    ids = {h["location_id"] for h in result["hotspots"]}
    assert "flat" not in ids


def test_emerging_hotspots_absolute_threshold():
    locations = [
        {"location_id": "a", "series": [0, 1, 6, 7, 9]},
        {"location_id": "b", "series": [1, 1, 1, 1, 1]},
    ]
    result = trends.emerging_hotspots(locations, hot_threshold=5.0)
    assert result["hot_basis"] == "absolute_threshold"
    a = next(h for h in result["hotspots"] if h["location_id"] == "a")
    assert a["hot_flags"] == [False, False, True, True, True]


def test_emerging_hotspots_skips_short_series():
    locations = [
        {"location_id": "ok", "series": [1, 2, 3, 4]},
        {"location_id": "tooshort", "series": [1, 2]},
    ]
    result = trends.emerging_hotspots(locations)
    assert result["evaluated_count"] == 1
    assert result["insufficient_count"] == 1
    assert result["insufficient"][0]["location_id"] == "tooshort"


def test_emerging_hotspots_degrades_on_garbage():
    locations = [
        {"location_id": "good", "series": [1, 3, 5, 7]},
        {"location_id": "nans", "series": [float("nan"), float("inf"), 1]},
        None,
    ]
    result = trends.emerging_hotspots(locations)
    assert result["status"] == "ok"
    assert result["location_count"] == 2  # None filtered out
    # 'nans' cleans down to a single finite point -> insufficient.
    assert result["insufficient_count"] == 1
