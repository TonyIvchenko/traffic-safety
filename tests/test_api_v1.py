from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_v1_health():
    client = TestClient(MODULE.api)
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["api_version"] == "1.0"
    assert payload["model_ready"] is True
    assert response.headers["cache-control"] == "no-store"


def test_v1_meta_describes_coverage_and_providers():
    client = TestClient(MODULE.api)
    response = client.get("/v1/meta")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["coverage"]) == {"lat_min", "lat_max", "lon_min", "lon_max"}
    assert payload["timeline"]["frame_count"] == 168
    assert payload["risk_levels"] == ["low", "moderate", "high", "extreme"]
    assert isinstance(payload["live_providers"], list)
    assert payload["rate_limit"]["scope"] == "per_client_ip"
    assert "nws" in payload["providers_accepted"]


def test_v1_point_climatology():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["weather_source"] == "climatology"
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert "weather" in payload
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_point_live_uses_mocked_snapshot(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=22.0,
        dewpoint_c=15.0,
        relative_humidity_pct=63.0,
        wind_speed_mps=4.5,
        wet_hour=0.0,
        summary="Clear",
    )

    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", lambda **kwargs: snapshot)
    client = TestClient(MODULE.api)
    response = client.get("/v1/risk/point?lat=34.0522&lon=-118.2437&mode=live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_provider"] == "nws"
    assert payload["weather_source"] == "live_observation"
    assert response.headers["cache-control"] == "no-store"


def test_v1_point_live_rejects_unknown_provider():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/point?lat=34.0522&lon=-118.2437&mode=live&provider=bogus"
    )
    assert response.status_code == 400


def test_v1_point_rejects_bad_mode_and_coords():
    client = TestClient(MODULE.api)
    assert client.get("/v1/risk/point?lat=34&lon=-118&mode=weird").status_code == 422
    assert client.get("/v1/risk/point?lat=120&lon=-118").status_code == 422


def test_v1_openapi_lists_v1_paths():
    client = TestClient(MODULE.api)
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/risk/point" in paths
    assert "/v1/meta" in paths
    assert "/v1/health" in paths
