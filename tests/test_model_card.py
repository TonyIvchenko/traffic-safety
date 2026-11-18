from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import model_card


_METADATA = {
    "model_type": "HistGradientBoostingClassifier",
    "resolution": 5,
    "feature_names": ["lat", "lon", "hour_sin", "wet_hour"],
    "weather_feature_names": ["temp_c", "wet_hour"],
    "n_features": 4,
    "train_years": [2020, 2018, 2019, 2023, 2021, 2022],
    "eval_years": [2024],
    "negative_sampling_ratio": 4,
    "spatial_cells": 30141,
}


def _card(**overrides):
    kwargs = dict(
        model_version="0.2.0",
        model_ready=True,
        metadata=_METADATA,
        metrics={"val_roc_auc": 0.8936552, "val_average_precision": 0.7154109},
        report={"available": False},
        risk_thresholds={"moderate": 0.0831, "high": 0.2941, "extreme": 0.6181},
        coverage={"lat_min": 18.0, "lat_max": 72.0},
        frame_count=168,
        caveats=["Climatological, not a real-time collision warning."],
        data_sources={
            "training": [{"id": "fars"}, {"id": "us_accidents"}],
            "live": [{"id": "nws_api"}],
        },
    )
    kwargs.update(overrides)
    return model_card.build_model_card(**kwargs)


def test_card_has_all_mitchell_sections():
    card = _card()
    for section in (
        "model_details",
        "intended_use",
        "factors",
        "metrics",
        "training_data",
        "evaluation_data",
        "quantitative_analyses",
        "ethical_considerations",
        "caveats_and_recommendations",
    ):
        assert section in card, section
    assert card["framework"] == "model_cards_mitchell_2019"
    assert card["schema_version"] == model_card.CARD_SCHEMA_VERSION
    assert card["generated_at"]


def test_card_reports_real_details_and_metrics():
    card = _card()
    details = card["model_details"]
    assert details["version"] == "0.2.0"
    assert details["type"] == "HistGradientBoostingClassifier"
    assert details["ready"] is True
    assert details["spatial_unit"]["resolution"] == 5
    assert details["spatial_unit"]["cells_modeled"] == 30141
    assert details["temporal_unit"]["frames"] == 168
    assert details["features"]["count"] == 4
    assert "wet_hour" in details["features"]["names"]
    # Years are span-summarized and de-duplicated/sorted.
    assert details["training_window"] == {
        "start": 2018,
        "end": 2023,
        "years": [2018, 2019, 2020, 2021, 2022, 2023],
    }
    assert details["evaluation_window"]["start"] == 2024

    perf = card["metrics"]["performance"]
    assert perf["roc_auc"] == 0.8937
    assert perf["average_precision"] == 0.7154
    assert card["metrics"]["decision_thresholds"]["high"] == 0.2941


def test_card_merges_caveats_and_lists_use():
    card = _card()
    caveats = card["caveats_and_recommendations"]
    # The supplied advisory caveat is kept, and the model-specific ones appended.
    assert "Climatological, not a real-time collision warning." in caveats
    assert any("Crash Modification Factors" in c for c in caveats)
    assert card["intended_use"]["out_of_scope_uses"]
    assert card["ethical_considerations"]
    assert card["training_data"]["sources"] == [{"id": "fars"}, {"id": "us_accidents"}]
    assert card["evaluation_data"]["live_inference_sources"] == [{"id": "nws_api"}]


def test_card_pulls_metrics_from_report_when_bundle_lacks_them():
    card = _card(
        metrics={},
        report={"available": True, "metrics": {"roc_auc": 0.77, "brier_score": 0.03}},
    )
    perf = card["metrics"]["performance"]
    assert perf["roc_auc"] == 0.77
    assert perf["brier_score"] == 0.03
    assert card["quantitative_analyses"]["calibration_report_available"] is True


def test_card_degrades_when_everything_missing():
    card = model_card.build_model_card(model_version="missing", model_ready=False)
    assert card["model_details"]["ready"] is False
    assert card["model_details"]["type"] is None
    assert card["model_details"]["training_window"] is None
    assert card["model_details"]["features"]["names"] == []
    assert card["metrics"]["performance"] == {}
    assert card["metrics"]["decision_thresholds"] == {}
    # Editorial sections are always present even with no model.
    assert card["ethical_considerations"]
    assert card["caveats_and_recommendations"]


def test_year_span_ignores_garbage():
    assert model_card._year_span([]) is None
    assert model_card._year_span("nonsense") is None
    assert model_card._year_span([2019, "bad", None, 2021, float("nan")]) == {
        "start": 2019,
        "end": 2021,
        "years": [2019, 2021],
    }
