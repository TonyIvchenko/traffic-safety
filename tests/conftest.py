"""Test configuration.

The trained model bundle (``models/traffic_safety.joblib``) and the risk overlay
(``tiles/overlay.npz``) are large build artifacts kept out of git, so a fresh
checkout (e.g. CI) has no model. The pure-logic suite runs fine without it; the
tests below exercise the risk model / live endpoints and are skipped when the
bundle is absent, so ``pytest`` is green on a plain checkout and complete locally.

Add a test here if it needs the trained model to pass.
"""

from __future__ import annotations

from pathlib import Path

import pytest

_MODEL_PATH = Path(__file__).resolve().parents[1] / "models" / "traffic_safety.joblib"
MODEL_AVAILABLE = _MODEL_PATH.exists()

# Node-id fragments (file::test) for tests that require the trained model.
MODEL_DEPENDENT = (
    # Model prediction + live-risk behaviour.
    "test_main.py::test_predict_traffic_safety_shape_and_values",
    "test_main.py::test_predict_traffic_safety_out_of_coverage_keeps_consistent_schema",
    "test_main.py::test_explain_prediction_attributes_factors",
    "test_main.py::test_health_endpoint_contract",
    "test_main.py::test_live_risk_endpoint_uses_mocked_snapshot",
    # /v1 endpoints that score the risk model.
    "test_api_v1.py::test_v1_health",
    "test_api_v1.py::test_v1_point_climatology",
    "test_api_v1.py::test_v1_point_explain",
    "test_api_v1.py::test_v1_point_omits_explanation_by_default",
    "test_api_v1.py::test_v1_point_live_uses_mocked_snapshot",
    "test_api_v1.py::test_v1_weekly_profile",
    "test_api_v1.py::test_v1_route_climatology",
    "test_api_v1.py::test_v1_route_geojson_output",
    "test_api_v1.py::test_v1_route_live_dedupes_weather_by_cell",
    "test_api_v1.py::test_v1_route_glare_annotation",
    "test_api_v1.py::test_v1_route_without_glare_datetime_has_no_glare_fields",
    "test_api_v1.py::test_v1_route_compare_recommends_lowest_risk",
    "test_api_v1.py::test_v1_route_compare_geojson",
    "test_api_v1.py::test_v1_route_compare_live_shares_weather_cache",
    "test_api_v1.py::test_v1_advisory_point_climatology",
    "test_api_v1.py::test_v1_advisory_point_live_folds_hazards_and_compares",
    "test_api_v1.py::test_v1_advisory_region_by_id",
    "test_api_v1.py::test_v1_advisory_region_live_and_compare",
    "test_api_v1.py::test_v1_advisory_national_json",
    "test_api_v1.py::test_v1_evacuate_compass_climatology",
    "test_api_v1.py::test_v1_evacuate_toward_facilities",
    "test_api_v1.py::test_v1_evacuate_geojson",
    "test_api_v1.py::test_v1_evacuate_live",
    "test_api_v1.py::test_v1_emergency_readiness_by_id",
    "test_api_v1.py::test_v1_profile_route_climatology",
    # Degrade-contract cases whose route scoring needs the model.
    "test_degrade_contract.py::test_get_never_5xx_or_nonstandard_json[/v1/emergency/evacuate",
)


def pytest_collection_modifyitems(config, items):
    if MODEL_AVAILABLE:
        return
    skip_no_model = pytest.mark.skip(
        reason="requires models/traffic_safety.joblib (build artifact absent in this environment)"
    )
    for item in items:
        if any(fragment in item.nodeid for fragment in MODEL_DEPENDENT):
            item.add_marker(skip_no_model)
