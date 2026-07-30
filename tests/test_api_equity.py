from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd
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


@pytest.fixture()
def hotspots_client(tmp_path, monkeypatch):
    overlay = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "risk": [0.8, 0.8, 0.9],
            "svi_percentile": [0.9, 0.1, None],
            "disadvantaged": [True, False, False],
            "center_lat": [34.0, 34.1, 36.0],
            "center_lon": [-118.2, -118.3, -119.0],
            "fullname": ["Main St", "1st Ave", "Rural Rd"],
        }
    )
    path = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(path))
    return TestClient(MODULE.api)


def test_equity_hotspots_json_ranks_by_priority(hotspots_client):
    response = hotspots_client.get("/v1/equity/hotspots")
    assert response.status_code == 200
    payload = response.json()
    assert payload["hotspots"][0]["segment_id"] == "a"  # disadvantaged + high SVI
    assert "equity_priority" in payload["hotspots"][0]
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_equity_hotspots_only_disadvantaged(hotspots_client):
    payload = hotspots_client.get("/v1/equity/hotspots?only_disadvantaged=true").json()
    assert {h["segment_id"] for h in payload["hotspots"]} == {"a"}


def test_equity_hotspots_bbox(hotspots_client):
    payload = hotspots_client.get(
        "/v1/equity/hotspots?min_lat=33.9&max_lat=34.2&min_lon=-118.5&max_lon=-118.1"
    ).json()
    assert {h["segment_id"] for h in payload["hotspots"]} == {"a", "b"}  # excludes rural 'c'


def test_equity_hotspots_geojson(hotspots_client):
    response = hotspots_client.get("/v1/equity/hotspots?format=geojson")
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert "center_lat" not in feature["properties"]


def test_equity_hotspots_partial_bbox_is_422(hotspots_client):
    assert hotspots_client.get("/v1/equity/hotspots?min_lat=34.0").status_code == 422


def test_equity_hotspots_empty_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(tmp_path / "missing.parquet"))
    payload = TestClient(MODULE.api).get("/v1/equity/hotspots").json()
    assert payload["count"] == 0


def test_equity_hotspots_geojson_tolerates_bad_centroid(tmp_path, monkeypatch):
    # A non-numeric centroid must be skipped, not 500 the GeoJSON response.
    overlay = pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "risk": [0.8, 0.7],
            "svi_percentile": [0.9, 0.8],
            "disadvantaged": [True, True],
            "center_lat": ["34.0", "bad"],  # 'b' has a non-numeric centroid
            "center_lon": ["-118.2", "-118.3"],
            "fullname": ["Main St", "1st Ave"],
        }
    )
    path = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(path))
    response = TestClient(MODULE.api).get("/v1/equity/hotspots?format=geojson")
    assert response.status_code == 200
    assert response.json()["count"] == 1  # 'b' skipped, 'a' kept
