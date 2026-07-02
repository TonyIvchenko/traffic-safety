from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import risk_eval


def _fake_predictors(risk: float = 0.5):
    calls = {"climatology": 0, "live": 0}

    def predict_point(*, lat, lon, day_of_week, hour, month):
        calls["climatology"] += 1
        return {"risk_score": risk, "risk_level": "moderate", "hazards": {"labels": []}}

    def predict_point_live(*, lat, lon, forecast_hours, provider):
        calls["live"] += 1
        return {
            "risk_score": risk,
            "risk_level": "moderate",
            "hazards": {"labels": []},
            "live_provider": "nws",
        }

    return predict_point, predict_point_live, calls


# Two points ~1.1 km apart in downtown LA (same H3 res-5 cell).
_NEAR = [(-118.2437, 34.0522), (-118.2437, 34.0622)]
# LA to ~900 km away (too many samples at 1 km spacing).
_FAR = [(-118.0, 34.0), (-110.0, 40.0)]


def test_score_route_accumulates_distance_and_steps():
    predict, predict_live, calls = _fake_predictors()
    result = risk_eval.score_route(
        _NEAR,
        mode="climatology",
        predict_point=predict,
        predict_point_live=predict_live,
        h3_resolution=5,
        sample_spacing_km=2.0,
    )
    assert result["sample_count"] == 2
    assert result["distance_km"] == pytest.approx(1.11, abs=0.05)
    assert result["route_risk_score_mean"] == pytest.approx(0.5)
    assert result["route_risk_level"] == "moderate"
    assert result["live_provider"] is None
    distances = [step["distance_km"] for step in result["steps"]]
    assert distances == sorted(distances)


def test_score_route_dedupes_predictions_by_cell():
    predict, predict_live, calls = _fake_predictors()
    risk_eval.score_route(
        _NEAR,
        mode="climatology",
        predict_point=predict,
        predict_point_live=predict_live,
        h3_resolution=5,
        sample_spacing_km=0.25,  # many samples, but one res-5 cell
    )
    assert calls["climatology"] == 1


def test_score_route_shared_cache_across_calls():
    predict, predict_live, calls = _fake_predictors()
    shared: dict = {}
    for _ in range(2):
        risk_eval.score_route(
            _NEAR,
            mode="live",
            predict_point=predict,
            predict_point_live=predict_live,
            h3_resolution=5,
            cell_cache=shared,
        )
    assert calls["live"] == 1  # second route reuses the shared cell cache
    assert risk_eval.score_route(
        _NEAR,
        mode="live",
        predict_point=predict,
        predict_point_live=predict_live,
        h3_resolution=5,
        cell_cache=shared,
    )["live_provider"] == "nws"


def test_score_route_rejects_bad_config():
    predict, predict_live, _ = _fake_predictors()
    with pytest.raises(risk_eval.RouteConfigError):
        risk_eval.score_route(
            _NEAR, mode="weird", predict_point=predict,
            predict_point_live=predict_live, h3_resolution=5,
        )
    with pytest.raises(risk_eval.RouteConfigError):
        risk_eval.score_route(
            _NEAR, mode="climatology", predict_point=predict,
            predict_point_live=predict_live, h3_resolution=5, sample_spacing_km=999.0,
        )


def test_score_route_rejects_too_long_routes():
    predict, predict_live, _ = _fake_predictors()
    with pytest.raises(risk_eval.RouteTooLongError) as excinfo:
        risk_eval.score_route(
            _FAR, mode="climatology", predict_point=predict,
            predict_point_live=predict_live, h3_resolution=5, sample_spacing_km=1.0,
        )
    assert excinfo.value.estimated > excinfo.value.max_samples
