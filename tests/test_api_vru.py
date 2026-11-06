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
_LEVELS = {"low", "moderate", "high", "extreme"}


def test_vru_point_returns_valid_schema():
    # Works with or without the VRU bundle: a covered cell scores, an uncovered
    # point degrades to risk 0 — either way the schema and bounds hold.
    client = TestClient(MODULE.api)
    resp = client.get("/v1/vru/point?lat=34.05&lon=-118.24&day_of_week=6&hour=20&month=9")
    assert resp.status_code == 200
    body = resp.json()
    assert body["layer"] == "vru"
    assert body["target"] == "pedestrian_or_cyclist_crash"
    assert 0.0 <= body["risk_score"] <= 1.0
    assert body["risk_level"] in _LEVELS
    assert isinstance(body["in_coverage"], bool)
    assert body["caveats"]
    assert resp.headers["cache-control"] == "public, max-age=3600"


def test_vru_weekly_returns_profile_and_extremes():
    client = TestClient(MODULE.api)
    resp = client.get("/v1/vru/point/weekly?lat=34.05&lon=-118.24&month=9")
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["profile"]) == 168
    assert set(body["safest_hour"]) >= {"frame_idx", "frame_label", "risk_score"}
    assert set(body["riskiest_hour"]) >= {"frame_idx", "frame_label", "risk_score"}
    # The riskiest hour is at least as risky as the safest.
    assert body["riskiest_hour"]["risk_score"] >= body["safest_hour"]["risk_score"]


def test_vru_point_validates_coordinates():
    client = TestClient(MODULE.api)
    assert client.get("/v1/vru/point?lat=120&lon=-118").status_code == 422
    assert client.get("/v1/vru/point?lat=34&lon=-118&hour=99").status_code == 422


def test_meta_documents_vru_layer():
    client = TestClient(MODULE.api)
    vru = client.get("/v1/meta").json()["vru"]
    assert isinstance(vru["enabled"], bool)
    assert vru["target"] == "pedestrian_or_cyclist_crash"
    assert "/v1/vru/point" in vru["endpoints"]
    assert "/v1/vru/point/weekly" in vru["endpoints"]
