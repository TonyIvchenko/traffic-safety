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


def test_model_card_page_renders_shell_and_assets():
    client = TestClient(MODULE.api)
    response = client.get("/model-card")

    assert response.status_code == 200
    body = response.text
    assert 'id="card-shell"' in body
    assert 'id="card-details"' in body
    assert 'id="card-metrics"' in body
    assert 'id="card-ethics"' in body
    assert 'id="card-caveats"' in body
    # The page pulls in its own stylesheet and controller.
    assert "model-card.css" in body
    assert "model-card.js" in body
    # It points at the JSON resource it renders.
    assert 'href="/v1/model/card"' in body


def test_model_card_nav_link_present_on_other_pages():
    client = TestClient(MODULE.api)
    about = client.get("/about")
    assert about.status_code == 200
    assert 'href="/model-card"' in about.text


def test_model_card_static_assets_are_served():
    client = TestClient(MODULE.api)

    css = client.get("/traffic-safety-static/model-card.css")
    js = client.get("/traffic-safety-static/model-card.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert ".card-shell" in css.text

    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "model/card" in js.text  # the page consumes the model-card API
