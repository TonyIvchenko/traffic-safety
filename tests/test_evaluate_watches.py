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


@pytest.fixture(autouse=True)
def reference_profile(monkeypatch):
    # A reference weekly climatology spanning 0.1..0.9 so scores map to known
    # relative percentiles (0.2 -> Low, 0.9 -> Extreme) without the real model.
    # Point/route watches resolve a reference point; area watches here carry no
    # centroid, so they fall back to the absolute level.
    profile = [0.1 + 0.8 * i / 167 for i in range(168)]
    monkeypatch.setattr(ew, "weekly_risk_profile", lambda lat, lon, month: list(profile))


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


def _fake_point_by_hour(scores_by_hour: dict[int, float], *, default: float = 0.0):
    """A point predictor whose score depends on the forecast offset (for horizons)."""

    def fake(*, lat, lon, forecast_hours, provider):
        score = scores_by_hour.get(int(forecast_hours), default)
        return {
            "risk_score": score,
            "risk_level": "extreme" if score >= 0.5 else "low",
            "weather_source": "forecast",
            "live_provider": "nws",
            "hazards": {"labels": []},
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
    # 0.2 sits in the bottom quintile of the reference profile -> relative "low".
    assert record["last_level"] == "low"
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


# --------------------------------------------------------------------------- #
# Relative-breach semantics (H1.1): watches fire on how unusual the risk is for
# the watched place/time, not on absolute risk that saturates in dense metros.
# --------------------------------------------------------------------------- #


def test_relative_breach_maps_level_index_to_threshold_word():
    assert ew.relative_breach(4, "high") is True       # High meets "high"
    assert ew.relative_breach(5, "high") is True       # Extreme exceeds "high"
    assert ew.relative_breach(3, "high") is False      # Elevated is below "high"
    assert ew.relative_breach(5, "extreme") is True
    assert ew.relative_breach(4, "extreme") is False
    assert ew.relative_breach(2, "moderate") is True
    assert ew.relative_breach(1, "low") is True
    assert ew.relative_breach("MODERATE", "moderate") is False  # non-numeric index
    assert ew.relative_breach(3, "nonsense") is False           # unknown threshold


def test_relative_assessment_scores_against_reference_profile():
    hot = ew.relative_assessment(0.93, {"lat": 34.0, "lon": -118.0}, month=7)
    assert hot["basis"] == "relative_to_local_weekly_normal"
    assert hot["level"] == "Extreme"
    cold = ew.relative_assessment(0.2, {"lat": 34.0, "lon": -118.0}, month=7)
    assert cold["level"] == "Low"
    # No reference point -> None, so the caller falls back to the absolute level.
    assert ew.relative_assessment(0.5, None, month=7) is None


def test_breach_payload_carries_relative_context(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    _point_watch(store)

    ew.run_once(store, now=NOW)

    payload = webhook_calls[0]["payload"]
    assert payload["basis"] == "relative_to_local_weekly_normal"
    assert payload["percentile"] == pytest.approx(1.0)
    assert payload["relative_descriptor"] == "near this location's weekly peak"
    assert payload["risk_level"] == "extreme"
    # The raw absolute reading is preserved alongside the relative decision.
    assert payload["detail"]["absolute_risk_level"] == "extreme"


def test_locally_normal_risk_does_not_breach_despite_high_absolute(store, webhook_calls, monkeypatch):
    # A dense-metro point whose absolute level is "extreme" but whose score sits
    # in the bottom quartile of its own weekly normal must NOT fire: that is the
    # whole point of relative alerting.
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.3))
    watch = _point_watch(store)  # threshold "high"

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 1, "breached": 0, "notified": 0, "errors": 0}
    assert webhook_calls == []
    assert store.get_watch(watch["id"])["last_level"] == "low"


# --------------------------------------------------------------------------- #
# Forecast-horizon watches (H1.2): scan the next N hours and alert ahead of time.
# --------------------------------------------------------------------------- #


def _email_watch(store, **kwargs):
    defaults = dict(
        kind="point",
        params={"lat": 34.05, "lon": -118.24, "forecast_hours": 0, "provider": "auto"},
        threshold_level="high",
        channel="email",
        email="driver@example.test",
    )
    defaults.update(kwargs)
    return store.create_watch(**defaults)


def test_email_channel_breach_sends_email(store, monkeypatch):
    sent = []

    def fake_send(to_address, subject, body):
        sent.append((to_address, subject, body))
        return {"delivered": True, "error": None}

    monkeypatch.setattr(ew.notify, "send_email", fake_send)
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    watch = _email_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 1, "breached": 1, "notified": 1, "errors": 0}
    assert len(sent) == 1
    to_address, subject, body = sent[0]
    assert to_address == "driver@example.test"
    assert "risk alert" in subject.lower()
    assert watch["id"] in body
    assert store.get_watch(watch["id"])["last_notified_at"] == NOW.isoformat()


def test_email_delivery_failure_is_not_counted_as_notified(store, monkeypatch):
    monkeypatch.setattr(
        ew.notify, "send_email", lambda *a, **k: {"delivered": False, "error": "smtp_not_configured"}
    )
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    watch = _email_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 1 and summary["notified"] == 0
    assert store.get_watch(watch["id"])["last_notified_at"] is None


# --------------------------------------------------------------------------- #
# Region watches (H1.4): breach on a metro's live risk vs its own weekly normal.
# --------------------------------------------------------------------------- #


def _region_watch(store, **kwargs):
    defaults = dict(
        kind="region",
        params={"region_id": "los_angeles", "forecast_hours": 0, "provider": "auto"},
        threshold_level="high",
        channel="webhook",
        webhook_url="https://example.test/region-hook",
    )
    defaults.update(kwargs)
    return store.create_watch(**defaults)


def test_region_watch_breaches_on_regional_relative_level(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    _region_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 1 and summary["notified"] == 1
    payload = webhook_calls[0]["payload"]
    assert payload["basis"] == "relative_to_local_weekly_normal"
    assert payload["risk_level"] == "extreme"
    assert payload["detail"]["region_id"] == "los_angeles"
    assert payload["detail"]["region_name"].startswith("Los Angeles")
    assert payload["detail"]["sample_count"] >= 1


def test_region_watch_calm_region_does_not_breach(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("moderate", 0.2))
    watch = _region_watch(store)

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 0 and summary["notified"] == 0
    assert webhook_calls == []
    assert store.get_watch(watch["id"])["last_level"] == "low"


def test_region_watch_unknown_region_is_error_not_breach(store, webhook_calls, monkeypatch):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point("extreme", 0.93))
    # The store does not validate region_id (the API does), so this reaches the
    # evaluator and must be isolated as an error rather than a false alert.
    _region_watch(store, params={"region_id": "atlantis", "provider": "auto"})

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 0, "breached": 0, "notified": 0, "errors": 1}
    assert webhook_calls == []


def test_horizon_offsets_span_window_and_cap_samples():
    assert ew._horizon_offsets(12) == [0, 2, 4, 6, 8, 10, 12]
    assert ew._horizon_offsets(6) == [0, 1, 2, 3, 4, 5, 6]
    offsets = ew._horizon_offsets(48)
    assert len(offsets) <= ew.MAX_HORIZON_SAMPLES
    assert offsets[0] == 0 and offsets[-1] == 48  # window edges always sampled


def _horizon_point_watch(store, scores, monkeypatch, *, horizon=12):
    monkeypatch.setattr(ew, "predict_traffic_safety_live", _fake_point_by_hour(scores, default=0.2))
    return _point_watch(
        store,
        params={"lat": 34.05, "lon": -118.24, "forecast_hours": 0,
                "horizon_hours": horizon, "provider": "auto"},
    )


def test_horizon_watch_alerts_on_earliest_future_breach(store, webhook_calls, monkeypatch):
    # Calm now, but risk spikes 8 hours out: fire with 8h of lead time.
    scores = {0: 0.2, 2: 0.2, 4: 0.2, 6: 0.3, 8: 0.93, 10: 0.93, 12: 0.93}
    _horizon_point_watch(store, scores, monkeypatch)

    summary = ew.run_once(store, now=NOW)

    assert summary == {"evaluated": 1, "breached": 1, "notified": 1, "errors": 0}
    payload = webhook_calls[0]["payload"]
    assert payload["lead_time_hours"] == 8
    assert payload["horizon_hours"] == 12
    assert payload["risk_level"] == "extreme"
    assert [f["hour"] for f in payload["forecast"]] == [0, 2, 4, 6, 8, 10, 12]
    assert payload["forecast"][0]["breached"] is False
    assert payload["forecast"][4]["breached"] is True  # the 8-hour sample


def test_horizon_watch_quiet_window_does_not_breach(store, webhook_calls, monkeypatch):
    watch = _horizon_point_watch(store, {}, monkeypatch)  # every hour defaults to 0.2

    summary = ew.run_once(store, now=NOW)

    assert summary["breached"] == 0 and summary["notified"] == 0
    assert webhook_calls == []
    assert store.get_watch(watch["id"])["last_level"] == "low"
