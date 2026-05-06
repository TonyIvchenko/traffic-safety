from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
import json
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None
    assert spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_health_endpoint_contract():
    client = TestClient(MODULE.api)

    response = client.get("/health")

    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["service"] == "Traffic Safety"
    assert payload["model_ready"] is True
    assert payload["overlay_ready"] is True
    assert payload["frames"] == 168
    assert isinstance(payload["live_providers"], list)
    assert any(provider["name"] == "nws" for provider in payload["live_providers"])


def test_predict_traffic_safety_shape_and_values():
    result = MODULE.predict_traffic_safety(
        lat=34.0522,
        lon=-118.2437,
        day_of_week=5,
        hour=17,
        month=9,
    )

    assert result["model_version"] == MODULE.MODEL_VERSION
    assert 0.0 <= result["risk_score"] <= 1.0
    assert result["risk_level"] in {"low", "moderate", "high", "extreme"}
    assert result["weather_source"] == "climatology"
    assert "weather" in result


def test_map_html_uses_local_static_assets():
    html = MODULE._map_html()

    assert '<div id="risk-map-shell"' in html
    assert 'data-config="' in html
    assert "Traffic Safety" in html


def test_document_html_includes_google_analytics_when_configured(monkeypatch):
    monkeypatch.setattr(MODULE, "GA_MEASUREMENT_ID", "G-R8ESFCMRPB")

    html = MODULE._document_html(title="Test", content="<p>ok</p>")

    assert "googletagmanager.com/gtag/js?id=G-R8ESFCMRPB" in html
    assert "gtag('config', 'G-R8ESFCMRPB');" in html


def test_bootstrap_loader_looks_for_local_map_script():
    assert "bootstrapTrafficSafetyMap" in MODULE._map_bootstrap_js()


def test_tiles_are_served():
    client = TestClient(MODULE.api)

    response = client.get("/tiles/0/4/0/0.png")

    assert response.status_code == 200
    assert "image/png" in response.headers["content-type"]
    assert len(response.content) > 0


def test_live_risk_endpoint_uses_mocked_snapshot(monkeypatch):
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

    def fake_fetch_live_weather(lat: float, lon: float, forecast_hours: int, provider: str):
        assert provider == "auto"
        assert forecast_hours == 0
        return snapshot

    monkeypatch.setattr(MODULE, "fetch_live_weather", fake_fetch_live_weather)
    client = TestClient(MODULE.api)

    response = client.get("/api/live-risk?lat=34.0522&lon=-118.2437&forecast_hours=0")

    assert response.status_code == 200
    payload = response.json()
    assert payload["live_provider"] == "nws"
    assert payload["weather_source"] == "live_observation"
    assert 0.0 <= payload["risk_score"] <= 1.0


def test_about_and_contact_pages_render():
    client = TestClient(MODULE.api)

    about_response = client.get("/about")
    contact_response = client.get("/contact")

    assert about_response.status_code == 200
    assert "Public national datasets feed the forecast stack" in about_response.text
    assert "Dual-layer modeling balances national coverage with road-level detail" in about_response.text
    assert "This is a research and engineering prototype" in about_response.text
    assert "Predictions should not be used for operational decision-making without further validation." in about_response.text
    assert "End-to-end pipeline" in about_response.text
    assert "How raw records become a national risk layer" in about_response.text
    assert "Road Risk Monitor" in about_response.text
    assert contact_response.status_code == 200
    assert 'id="contact-form"' in contact_response.text
    assert "Send a message" in contact_response.text


def test_contact_submission_is_logged_when_email_forwarding_is_not_configured(monkeypatch, tmp_path):
    submission_path = tmp_path / "contact_submissions.jsonl"
    monkeypatch.setattr(MODULE, "CONTACT_SUBMISSIONS_PATH", submission_path)
    monkeypatch.setattr(MODULE, "CONTACT_EMAIL", "")
    monkeypatch.setattr(MODULE, "SMTP_HOST", "")

    client = TestClient(MODULE.api)
    response = client.post(
        "/api/contact",
        json={
            "name": "Test User",
            "email": "test@example.com",
            "organization": "Road Lab",
            "subject": "Pilot",
            "message": "We want to run a deployment pilot.",
        },
    )

    assert response.status_code == 200
    payload = response.json()
    assert payload["kind"] == "warning"
    assert submission_path.exists()
    stored = [json.loads(line) for line in submission_path.read_text(encoding="utf-8").splitlines()]
    assert len(stored) == 1
    assert stored[0]["name"] == "Test User"
