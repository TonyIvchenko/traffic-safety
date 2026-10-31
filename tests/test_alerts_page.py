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


def test_alerts_page_renders_shell_and_assets():
    client = TestClient(MODULE.api)
    response = client.get("/alerts")

    assert response.status_code == 200
    body = response.text
    assert 'id="alerts-shell"' in body
    assert 'id="watch-form"' in body
    assert 'id="alerts-map"' in body
    # Citizen framing: relative-to-normal advisory, not raw absolute risk.
    assert "typical week" in body
    # The page pulls in its own stylesheet and controller.
    assert "alerts.css" in body
    assert "alerts.js" in body
    # The watch form offers all three delivery channels.
    for channel in ("poll", "email", "webhook"):
        assert f'value="{channel}"' in body


def test_alerts_nav_link_present_on_other_pages():
    client = TestClient(MODULE.api)
    about = client.get("/about")
    assert about.status_code == 200
    assert 'href="/alerts"' in about.text


def test_alerts_static_assets_are_served():
    client = TestClient(MODULE.api)

    css = client.get("/traffic-safety-static/alerts.css")
    js = client.get("/traffic-safety-static/alerts.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert ".alerts-shell" in css.text

    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    assert "advisory/national" in js.text  # the page consumes the public advisory API
