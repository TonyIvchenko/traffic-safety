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


def test_developers_page_renders_shell_and_assets():
    client = TestClient(MODULE.api)
    response = client.get("/developers")

    assert response.status_code == 200
    body = response.text
    assert 'id="dev-shell"' in body
    assert 'id="dev-tiers-body"' in body
    assert 'id="dev-key-input"' in body
    assert 'id="dev-key-check"' in body
    assert 'id="dev-key-result"' in body
    # The page pulls in its own stylesheet and controller.
    assert "developers.css" in body
    assert "developers.js" in body
    # Reference links point at the interactive docs.
    assert 'href="/v1/docs"' in body


def test_developers_nav_link_present_on_other_pages():
    client = TestClient(MODULE.api)
    about = client.get("/about")
    assert about.status_code == 200
    assert 'href="/developers"' in about.text


def test_developers_static_assets_are_served():
    client = TestClient(MODULE.api)

    css = client.get("/traffic-safety-static/developers.css")
    js = client.get("/traffic-safety-static/developers.js")

    assert css.status_code == 200
    assert "text/css" in css.headers["content-type"]
    assert ".dev-shell" in css.text

    assert js.status_code == 200
    assert "javascript" in js.headers["content-type"]
    # The page consumes the meta + self endpoints of the key API.
    assert "/meta" in js.text
    assert "keys/me" in js.text
