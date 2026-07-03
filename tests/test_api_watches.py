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
def client(tmp_path, monkeypatch):
    # Point the process-wide store at a throwaway DB for each test.
    monkeypatch.setenv("TRAFFIC_SAFETY_WATCH_DB", str(tmp_path / "watches.sqlite3"))
    return TestClient(MODULE.api)


_POINT_WATCH = {
    "kind": "point",
    "params": {"lat": 34.0522, "lon": -118.2437, "forecast_hours": 0},
    "threshold_level": "high",
}


def test_watch_lifecycle(client):
    created = client.post("/v1/watches", json=_POINT_WATCH)
    assert created.status_code == 201
    payload = created.json()
    watch_id, token = payload["id"], payload["token"]
    assert token
    assert payload["active"] is True
    assert payload["last_evaluated_at"] is None
    assert created.headers["cache-control"] == "no-store"

    polled = client.get(f"/v1/watches/{watch_id}?token={token}")
    assert polled.status_code == 200
    status = polled.json()
    assert status["kind"] == "point"
    assert "token" not in status  # token only returned at creation

    paused = client.patch(f"/v1/watches/{watch_id}?token={token}", json={"active": False})
    assert paused.status_code == 200
    assert paused.json()["active"] is False

    deleted = client.delete(f"/v1/watches/{watch_id}?token={token}")
    assert deleted.status_code == 200
    assert client.get(f"/v1/watches/{watch_id}?token={token}").status_code == 404


def test_watch_auth_errors(client):
    created = client.post("/v1/watches", json=_POINT_WATCH).json()
    assert client.get(f"/v1/watches/{created['id']}?token=wrong").status_code == 403
    assert client.get("/v1/watches/nonexistent?token=x").status_code == 404


def test_route_watch_normalizes_waypoints(client):
    body = {
        "kind": "route",
        "params": {"waypoints": [[-118.2437, 34.0522], [-118.40, 34.02]]},
    }
    created = client.post("/v1/watches", json=body)
    assert created.status_code == 201
    params = created.json()["params"]
    assert params["waypoints"] == [[-118.2437, 34.0522], [-118.40, 34.02]]
    assert params["sample_spacing_km"] == 2.0
    assert params["provider"] == "auto"


def test_area_watch_and_webhook_channel(client):
    body = {
        "kind": "area",
        "params": {"min_lat": 33.9, "max_lat": 34.2, "min_lon": -118.5, "max_lon": -118.1},
        "channel": "webhook",
        "webhook_url": "https://example.test/hook",
        "threshold_level": "extreme",
    }
    created = client.post("/v1/watches", json=body)
    assert created.status_code == 201
    payload = created.json()
    assert payload["webhook_secret"]  # returned once at creation
    assert payload["params"]["limit"] == 100


def test_watch_validation_errors(client):
    bad_kind = client.post("/v1/watches", json={"kind": "galaxy", "params": {}})
    assert bad_kind.status_code == 422

    missing_point = client.post("/v1/watches", json={"kind": "point", "params": {"lat": 34.0}})
    assert missing_point.status_code == 422

    bad_bbox = client.post(
        "/v1/watches",
        json={"kind": "area", "params": {"min_lat": 34.2, "max_lat": 33.9, "min_lon": -118.5, "max_lon": -118.1}},
    )
    assert bad_bbox.status_code == 422

    webhook_no_url = client.post(
        "/v1/watches", json={**_POINT_WATCH, "channel": "webhook"}
    )
    assert webhook_no_url.status_code == 422

    bad_provider = client.post(
        "/v1/watches",
        json={"kind": "point", "params": {"lat": 34.0, "lon": -118.0, "provider": "bogus"}},
    )
    assert bad_provider.status_code == 400


def test_meta_documents_watches(client):
    payload = client.get("/v1/meta").json()
    assert payload["watches"]["enabled"] is True
    assert set(payload["watches"]["kinds"]) == {"point", "route", "area"}
