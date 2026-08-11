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
def cm_client(tmp_path, monkeypatch):
    frame = pd.DataFrame(
        {
            "segment_id": ["seg-1"],
            "fullname": ["Main St"],
            "mtfcc": ["S1200"],
            "rur_urb": [2],
            "length_km": [1.5],
            "center_lat": [34.0],
            "center_lon": [-118.2],
            "fatal_crashes": [6.0],
            "hin_rank": [1],
        }
    )
    path = tmp_path / "hin.parquet"
    frame.to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_CM_SEGMENTS_PATH", str(path))
    return TestClient(MODULE.api)


def test_countermeasures_segment_returns_ranked_recommendations(cm_client):
    response = cm_client.get("/v1/countermeasures/segment?segment_id=seg-1")
    assert response.status_code == 200
    payload = response.json()
    assert payload["segment"]["segment_id"] == "seg-1"
    assert payload["count"] >= 1
    top = payload["recommendations"][0]
    assert "cmf" in top and "benefit_cost" in top
    assert top["benefit_cost"]["benefit_cost_ratio"] > 0
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
