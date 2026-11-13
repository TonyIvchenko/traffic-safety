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


def test_impact_page_renders_shell_and_assets():
    client = TestClient(MODULE.api)
    response = client.get("/impact")

    assert response.status_code == 200
    body = response.text
    assert 'id="impact-shell"' in body
    assert 'id="impact-crashes"' in body
    assert 'id="impact-investment"' in body
    assert 'id="impact-equity"' in body
    assert 'id="impact-headline"' in body
    # The page pulls in its own stylesheet and controller.
    assert "impact.css" in body
    assert "impact.js" in body


def test_impact_nav_link_present_on_other_pages():
    client = TestClient(MODULE.api)
    about = client.get("/about")
    assert about.status_code == 200
    assert 'href="/impact"' in about.text


def test_impact_static_assets_are_served():
    client = TestClient(MODULE.api)

    css = client.get("/traffic-safety-static/impact.css")
    js = client.get("/traffic-safety-static/impact.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert ".impact-shell" in css.text

    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "impact/summary" in js.text  # the page consumes the impact API
