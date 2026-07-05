from __future__ import annotations

from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import evaluate_watches as ew
import notify
import watch_store as ws

NOW = datetime(2026, 7, 2, 12, 0, tzinfo=timezone.utc)


@pytest.fixture()
def store(tmp_path):
    return ws.WatchStore(tmp_path / "watches.sqlite3")


@pytest.fixture()
def webhook_calls(monkeypatch):
    calls: list[dict] = []

    def fake_post(url, payload, secret=None):
        calls.append({"url": url, "payload": payload, "secret": secret})
        return {"delivered": True, "status_code": 200, "error": None}

    monkeypatch.setattr(notify, "post_webhook", fake_post)
    return calls


def _fake_point(level: str, score: float):
    def fake(*, lat, lon, forecast_hours, provider):
        return {
            "risk_score": score,
            "risk_level": level,
            "weather_source": "live_observation",
            "live_provider": "nws",
            "hazards": {"labels": ["wet"]},
        }

    return fake


def _point_watch(store, **kwargs):
    defaults = dict(
        kind="point",
        params={"lat": 34.05, "lon": -118.24, "forecast_hours": 0, "provider": "auto"},
        threshold_level="high",
        channel="webhook",
        webhook_url="https://example.test/hook",
    )
    defaults.update(kwargs)
    return store.create_watch(**defaults)


def test_breach_sends_signed_webhook_and_records(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    watch = _point_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 1, "breached": 1, "notified": 1, "errors": 0}
    assert len(webhook_calls) == 1
    call = webhook_calls[0]
    assert call["url"] == "https://example.test/hook"
    assert call["secret"] == watch["webhook_secret"]
    assert call["payload"]["risk_level"] == "extreme"
    assert call["payload"]["watch_id"] == watch["id"]

    record = store.get_watch(watch["id"])
    assert record["last_level"] == "extreme"
    assert record["last_breach_at"] == NOW.isoformat()
    assert record["last_notified_at"] == NOW.isoformat()


def test_cooldown_suppresses_repeat_notifications(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    _point_watch(store, cooldown_minutes=60)

    ew.run_once(store, now=NOW)
    ew.run_once(store, now=NOW + timedelta(minutes=10))  # inside cooldown
    assert len(webhook_calls) == 1

    ew.run_once(store, now=NOW + timedelta(minutes=61))  # cooldown elapsed
    assert len(webhook_calls) == 2


def test_below_threshold_updates_status_without_notifying(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("moderate", 0.2))
    watch = _point_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 0 and summary["notified"] == 0
    assert webhook_calls == []
    record = store.get_watch(watch["id"])
    assert record["last_level"] == "moderate"
    assert record["last_breach_at"] is None


def test_poll_channel_never_notifies_but_tracks_breach(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.9))
    watch = _point_watch(store, channel="poll", webhook_url=None)

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 1 and summary["notified"] == 0
    assert webhook_calls == []
    assert store.get_watch(watch["id"])["last_breach_at"] == NOW.isoformat()


def test_dry_run_skips_webhooks(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.9))
    _point_watch(store)
    summary = ew.run_once(store, dry_run=True, now=NOW)
    assert summary["breached"] == 1 and summary["notified"] == 0
    assert webhook_calls == []


def test_evaluation_error_is_isolated(store, webhook_calls, monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(ew, "predict_traffic_safety_live", boom)
    watch = _point_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 0, "breached": 0, "notified": 0, "errors": 1}
    assert store.get_watch(watch["id"])["last_evaluated_at"] is None


def test_route_watch_uses_score_route(store, webhook_calls, monkeypatch):
    def fake_score_route(points, **kwargs):
        assert len(points) == 2
        return {
            "route_risk_score_max": 0.88,
            "route_risk_level": "extreme",
            "distance_km": 12.0,
            "high_risk_fraction": 0.5,
            "riskiest_point": {"lat": 34.0, "lon": -118.3},
        }

    monkeypatch.setattr(ew.risk_eval, "score_route", fake_score_route)
    store.create_watch(
        kind="route",
        params={"waypoints": [[-118.24, 34.05], [-118.40, 34.02]], "sample_spacing_km": 2.0},
        threshold_level="extreme",
        channel="webhook",
        webhook_url="https://example.test/route-hook",
    )

    summary = ew.run_once(store, now=NOW)
    assert summary["notified"] == 1
    assert webhook_calls[0]["payload"]["detail"]["riskiest_point"]["lat"] == 34.0


def test_area_watch_maps_top_segment_to_level(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(
        ew.segment_runtime,
        "score_segments_in_bbox",
        lambda **kwargs: {
            "count": 2,
            "segments": [
                {"segment_id": "a", "fullname": "Main St", "risk_score": 0.95},
                {"segment_id": "b", "fullname": "Side St", "risk_score": 0.10},
            ],
        },
    )
    monkeypatch.setattr(ew, "_risk_level", lambda p: "extreme" if p > 0.5 else "low")
    store.create_watch(
        kind="area",
        params={"min_lat": 33.9, "max_lat": 34.2, "min_lon": -118.5, "max_lon": -118.1},
        threshold_level="high",
        channel="webhook",
        webhook_url="https://example.test/area-hook",
    )

    summary = ew.run_once(store, now=NOW)
    assert summary["notified"] == 1
    payload = webhook_calls[0]["payload"]
    assert payload["risk_level"] == "extreme"
    assert payload["detail"]["riskiest_segment"]["segment_id"] == "a"


def test_area_watch_with_no_segments_is_low(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(
        ew.segment_runtime,
        "score_segments_in_bbox",
        lambda **kwargs: {"count": 0, "segments": []},
    )
    watch = store.create_watch(
        kind="area",
        params={"min_lat": 1, "max_lat": 2, "min_lon": 3, "max_lon": 4},
        channel="poll",
    )
    summary = ew.run_once(store, now=NOW)
    assert summary["breached"] == 0
    assert store.get_watch(watch["id"])["last_level"] == "low"
