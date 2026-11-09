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


def test_speed_point_reports_congestion_at_peak():
    client = TestClient(MODULE.api)
    resp = client.get("/v1/speed/point?lat=34.0335&lon=-118.2487&period=peak")
    assert resp.status_code == 200
    body = resp.json()
    assert body["basis"] == "npmrds_segment"
    assert body["period"] == "peak"
    assert 0.0 < body["congestion_ratio"] < 1.0  # congested
    assert body["speed_risk_index"] > 1.0
    assert body["source"] == "npmrds"
    assert resp.headers["cache-control"] == "public, max-age=3600"


def test_speed_point_hour_overrides_period():
    client = TestClient(MODULE.api)
    # Hour 13 is off-peak -> near free-flow on the LA sample -> neutral index.
    body = client.get("/v1/speed/point?lat=34.0335&lon=-118.2487&hour=13").json()
    assert body["period"] == "offpeak"
    assert body["speed_risk_index"] == 1.0


def test_speed_point_unavailable_is_neutral():
    client = TestClient(MODULE.api)
    body = client.get("/v1/speed/point?lat=39.0&lon=-100.0").json()
    assert body["basis"] == "unavailable"
    assert body["speed_risk_index"] == 1.0
    assert body["observed_speed_mph"] is None


def test_speed_point_validation():
    client = TestClient(MODULE.api)
    assert client.get("/v1/speed/point?lat=120&lon=-100").status_code == 422
    assert client.get("/v1/speed/point?lat=34&lon=-118&hour=99").status_code == 422


def test_meta_documents_traffic_speed():
    client = TestClient(MODULE.api)
    block = client.get("/v1/meta").json()["traffic_speed"]
    assert block["enabled"] is True
    assert block["speed_segments"] > 0
    assert "/v1/speed/point" in block["endpoints"]
    assert block["source"] == "npmrds"
