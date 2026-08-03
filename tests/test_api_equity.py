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


def test_meta_advertises_equity(hotspots_client):
    payload = hotspots_client.get("/v1/meta").json()
    equity_meta = payload["equity"]
    assert equity_meta["enabled"] is True
    assert equity_meta["segments"] == 3  # the fixture overlay has 3 segments
    assert set(equity_meta["endpoints"]) >= {
        "/v1/equity/point",
        "/v1/equity/hotspots",
        "/v1/equity/summary",
        "/v1/equity/choropleth",
    }
    assert "svi" in equity_meta["data_vintage"]
    assert equity_meta["data_vintage"]["tract_boundaries"] == "2010 census tracts"


def test_meta_equity_segments_zero_when_no_overlay(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(tmp_path / "missing.parquet"))
    payload = TestClient(MODULE.api).get("/v1/meta").json()
    assert payload["equity"]["segments"] == 0


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


@pytest.fixture()
def summary_client(tmp_path, monkeypatch):
    overlay = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "tract_geoid": ["06037920100", "06037920200", "06037920300", "06059000100"],
            "disadvantaged": [True, True, False, False],
            "crashes": [10.0, 6.0, 2.0, 0.0],
            "risk": [0.8, 0.6, 0.4, 0.2],
            "svi_percentile": [0.9, 0.8, 0.3, 0.1],
        }
    )
    path = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(path))
    return TestClient(MODULE.api)


def test_equity_summary_national(summary_client):
    payload = summary_client.get("/v1/equity/summary").json()
    assert payload["geoid"] is None
    assert payload["segments"] == 4
    assert payload["crash_disparity_ratio"] == pytest.approx(8.0)
    assert summary_client.get("/v1/equity/summary").headers["cache-control"] == "public, max-age=3600"


def test_equity_summary_by_geoid(summary_client):
    payload = summary_client.get("/v1/equity/summary?geoid=06037").json()
    assert payload["geoid"] == "06037"
    assert payload["segments"] == 3
    assert payload["crash_disparity_ratio"] == pytest.approx(4.0)


def test_equity_summary_includes_weighted_burden(summary_client):
    payload = summary_client.get("/v1/equity/summary").json()
    burden = payload["weighted_burden"]
    assert "burden_ratio" in burden and "crash_weighted_svi" in burden
    assert burden["svi_weighted_crashes"] > 0


def test_equity_summary_unknown_geoid(summary_client):
    payload = summary_client.get("/v1/equity/summary?geoid=99").json()
    assert payload["segments"] == 0
    assert payload["crash_disparity_ratio"] is None


@pytest.fixture()
def choropleth_client(tmp_path, monkeypatch):
    overlay = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "tract_geoid": ["06037920100", "06037920100", "06037920200"],
            "svi_percentile": [0.9, 0.9, 0.2],
            "disadvantaged": [True, True, False],
            "risk": [0.8, 0.6, 0.4],
            "crashes": [5.0, 3.0, 1.0],
            "center_lat": [34.0, 34.02, 34.2],
            "center_lon": [-118.2, -118.22, -118.4],
        }
    )
    path = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(path))
    return TestClient(MODULE.api)


def test_equity_choropleth_geojson(choropleth_client):
    response = choropleth_client.get("/v1/equity/choropleth")
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["count"] == 2  # two tracts
    feature = next(
        f for f in payload["features"] if f["properties"]["tract_geoid"] == "06037920100"
    )
    assert feature["geometry"]["type"] == "Point"
    assert feature["properties"]["segment_count"] == 2
    assert feature["properties"]["fill"] == "#d7191c"  # very_high SVI band
    assert "center_lat" not in feature["properties"]


def test_equity_choropleth_json_format(choropleth_client):
    payload = choropleth_client.get("/v1/equity/choropleth?format=json").json()
    assert payload["count"] == 2
    assert {t["tract_geoid"] for t in payload["tracts"]} == {"06037920100", "06037920200"}


def test_equity_choropleth_bbox(choropleth_client):
    payload = choropleth_client.get(
        "/v1/equity/choropleth?min_lat=33.9&max_lat=34.1&min_lon=-118.3&max_lon=-118.1&format=json"
    ).json()
    assert {t["tract_geoid"] for t in payload["tracts"]} == {"06037920100"}


def test_equity_choropleth_partial_bbox_is_422(choropleth_client):
    assert choropleth_client.get("/v1/equity/choropleth?min_lat=34.0").status_code == 422


def test_choropleth_builder_uses_injected_polygon():
    import sys as _sys
    from pathlib import Path as _Path

    src = str(_Path(__file__).resolve().parents[1] / "src")
    if src not in _sys.path:
        _sys.path.insert(0, src)
    import api_v1

    polygon = {"type": "Polygon", "coordinates": [[[0, 0], [0, 1], [1, 1], [0, 0]]]}
    records = [{"tract_geoid": "06037920100", "svi_category": "high", "center_lat": 34.0, "center_lon": -118.2}]
    geojson = api_v1._equity_choropleth_geojson(records, geometry_of=lambda g: polygon)
    assert geojson["features"][0]["geometry"] == polygon  # true choropleth polygon
    assert geojson["features"][0]["properties"]["fill"] == "#fdae61"
