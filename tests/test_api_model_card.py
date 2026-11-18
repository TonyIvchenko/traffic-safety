from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_model_card_endpoint_contract():
    client = TestClient(MODULE.api)
    response = client.get("/v1/model/card")

    assert response.status_code == 200
    assert "max-age" in response.headers.get("cache-control", "")
    card = response.json()

    assert card["framework"] == "model_cards_mitchell_2019"
    for section in (
        "model_details",
        "intended_use",
        "metrics",
        "training_data",
        "ethical_considerations",
        "caveats_and_recommendations",
    ):
        assert section in card
    assert isinstance(card["model_details"]["features"], dict)
    assert card["intended_use"]["out_of_scope_uses"]

    # When the trained bundle is present locally, the real provenance surfaces.
    if card["model_details"]["ready"]:
        assert card["model_details"]["type"]
        assert card["metrics"]["performance"].get("roc_auc") is not None
        assert card["model_details"]["training_window"] is not None


def test_meta_advertises_model_card():
    client = TestClient(MODULE.api)
    meta = client.get("/v1/meta").json()
    assert meta["model_card"]["endpoint"] == "/v1/model/card"
    assert meta["model_card"]["framework"] == "model_cards_mitchell_2019"
