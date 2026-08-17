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


def _hin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["seg-1", "seg-2"],
            "fullname": ["Main St", "Rural Rd"],
            "mtfcc": ["S1200", "S1100"],
            "rur_urb": [2, 1],
            "length_km": [1.5, 2.0],
            "center_lat": [34.0, 36.0],
            "center_lon": [-118.2, -119.0],
            "fatal_crashes": [6.0, 10.0],
            "hin_rank": [2, 1],
        }
    )


@pytest.fixture()
def cm_client(tmp_path, monkeypatch):
    path = tmp_path / "hin.parquet"
    _hin_frame().to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_CM_SEGMENTS_PATH", str(path))
    return TestClient(MODULE.api)


def test_meta_advertises_countermeasures(cm_client):
    payload = cm_client.get("/v1/meta").json()
    block = payload["countermeasures"]
    assert block["enabled"] is True
    assert block["segments"] == 2  # the fixture HIN has two segments
    assert block["catalog_size"] >= 10
    assert block["catalog_version"]
    assert set(block["endpoints"]) == {
        "/v1/countermeasures/segment", "/v1/countermeasures/hotspots"
    }


def test_countermeasures_segment_returns_ranked_recommendations(cm_client):
    response = cm_client.get("/v1/countermeasures/segment?segment_id=seg-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["segment"]["segment_id"] == "seg-1"
    assert payload["count"] >= 1
    top = payload["recommendations"][0]
    assert "cmf" in top and "benefit_cost" in top
    assert top["benefit_cost"]["benefit_cost_ratio"] > 0
    # CMF confidence + uncertainty note are surfaced (F3.13).
    assert top["cmf_confidence"] in {"low", "moderate", "high", "unknown"}
    assert "cmf_note" in payload
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_countermeasures_segment_ranked_by_bcr(cm_client):
    recs = cm_client.get("/v1/countermeasures/segment?segment_id=seg-1&top_n=10").json()[
        "recommendations"
    ]
    ratios = [r["benefit_cost"]["benefit_cost_ratio"] or 0.0 for r in recs]
    assert ratios == sorted(ratios, reverse=True)


def test_countermeasures_segment_unknown_is_404(cm_client):
    assert cm_client.get("/v1/countermeasures/segment?segment_id=nope").status_code == 404


def test_countermeasures_segment_requires_id(cm_client):
    assert cm_client.get("/v1/countermeasures/segment").status_code == 422


def test_countermeasures_hotspots_json(cm_client):
    response = cm_client.get("/v1/countermeasures/hotspots")
    assert response.status_code == 200
    payload = response.json()
    # Ranked by fatal_crashes desc: seg-2 (10) before seg-1 (6).
    assert [h["segment_id"] for h in payload["hotspots"]] == ["seg-2", "seg-1"]
    assert payload["hotspots"][0]["recommended"] is not None
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_countermeasures_hotspots_geojson(cm_client):
    response = cm_client.get("/v1/countermeasures/hotspots?format=geojson")
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "Point"
    assert "recommended" in feature["properties"]
    assert "center_lat" not in feature["properties"]


def test_countermeasures_hotspots_bbox(cm_client):
    payload = cm_client.get(
        "/v1/countermeasures/hotspots?min_lat=33.9&max_lat=34.1&min_lon=-118.3&max_lon=-118.1"
    ).json()
    assert {h["segment_id"] for h in payload["hotspots"]} == {"seg-1"}


def test_countermeasures_hotspots_partial_bbox_is_422(cm_client):
    assert cm_client.get("/v1/countermeasures/hotspots?min_lat=34.0").status_code == 422
