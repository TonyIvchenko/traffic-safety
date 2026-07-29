from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


@pytest.fixture()
def equity_client(tmp_path, monkeypatch):
    path = tmp_path / "tract_equity.csv"
    path.write_text(
        "tract_geoid,svi_percentile,disadvantaged\n06037920100,0.9,True\n", encoding="utf-8"
    )
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_PATH", str(path))
    return TestClient(MODULE.api)


def test_equity_point_known_tract(equity_client, monkeypatch):
    # geo_lookup is stubbed (no TIGER shapefiles in CI); the point maps to a
    # tract present in the fixture equity index.
    monkeypatch.setattr(MODULE, "_tract_of", lambda lat, lon: "06037920100")
    response = equity_client.get("/v1/equity/point?lat=34.05&lon=-118.24")
    assert response.status_code == 200
    payload = response.json()
    assert payload["tract_geoid"] == "06037920100"
    assert payload["svi_percentile"] == 0.9
    assert payload["svi_category"] == "very_high"
    assert payload["disadvantaged"] is True
    assert payload["in_index"] is True
    assert payload["lat"] == 34.05 and payload["lon"] == -118.24
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_equity_point_unresolved_tract_is_unknown(equity_client, monkeypatch):
    monkeypatch.setattr(MODULE, "_tract_of", lambda lat, lon: None)
    payload = equity_client.get("/v1/equity/point?lat=0&lon=0").json()
    assert payload["tract_geoid"] is None
    assert payload["in_index"] is False
    assert payload["disadvantaged"] is False
    assert payload["svi_category"] == "unknown"


def test_equity_point_tract_not_in_index(equity_client, monkeypatch):
    monkeypatch.setattr(MODULE, "_tract_of", lambda lat, lon: "99999999999")
    payload = equity_client.get("/v1/equity/point?lat=1&lon=1").json()
    assert payload["tract_geoid"] == "99999999999"
    assert payload["in_index"] is False
    assert payload["disadvantaged"] is False


def test_equity_point_requires_coordinates(equity_client):
    assert equity_client.get("/v1/equity/point?lat=34.0").status_code == 422


def test_equity_point_rejects_out_of_range(equity_client):
    assert equity_client.get("/v1/equity/point?lat=200&lon=-118").status_code == 422
