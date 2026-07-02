from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import watch_store as ws


@pytest.fixture()
def store(tmp_path):
    return ws.WatchStore(tmp_path / "watches.sqlite3")


def test_create_and_get_round_trip(store):
    created = store.create_watch(
        kind="point",
        params={"lat": 34.05, "lon": -118.24, "forecast_hours": 0},
        threshold_level="high",
    )
    assert created["id"] and created["token"]
    assert created["channel"] == "poll"
    assert created["webhook_secret"] is None  # poll channel has no secret

    fetched = store.get_watch(created["id"])
    assert fetched["params"]["lat"] == 34.05
    assert fetched["active"] is True
    assert fetched["last_evaluated_at"] is None


def test_token_authorization(store):
    created = store.create_watch(kind="point", params={"lat": 1.0, "lon": 2.0})
    assert store.get_watch_authorized(created["id"], created["token"]) is not None
    assert store.get_watch_authorized(created["id"], "wrong-token") is None
    assert store.get_watch_authorized("missing-id", created["token"]) is None


def test_webhook_channel_requires_url_and_gets_secret(store):
    with pytest.raises(ValueError, match="webhook_url"):
        store.create_watch(kind="point", params={}, channel="webhook")

    created = store.create_watch(
        kind="area",
        params={"min_lat": 1, "max_lat": 2, "min_lon": 3, "max_lon": 4},
        channel="webhook",
        webhook_url="https://example.test/hook",
    )
    assert created["webhook_secret"]
    assert created["webhook_url"] == "https://example.test/hook"


def test_validation_rejects_bad_values(store):
    with pytest.raises(ValueError):
        store.create_watch(kind="nonsense", params={})
    with pytest.raises(ValueError):
        store.create_watch(kind="point", params={}, channel="carrier-pigeon")
    with pytest.raises(ValueError):
        store.create_watch(kind="point", params={}, threshold_level="apocalyptic")
    with pytest.raises(ValueError):
        store.create_watch(kind="point", params={}, cooldown_minutes=-5)


def test_pause_resume_and_list_active(store):
    created = store.create_watch(kind="point", params={"lat": 1.0, "lon": 2.0})
    assert [w["id"] for w in store.list_active()] == [created["id"]]

    paused = store.set_active(created["id"], created["token"], False)
    assert paused["active"] is False
    assert store.list_active() == []
    assert store.set_active(created["id"], "bad-token", True) is None

    store.set_active(created["id"], created["token"], True)
    assert len(store.list_active()) == 1


def test_delete_requires_token(store):
    created = store.create_watch(kind="point", params={})
    assert store.delete_watch(created["id"], "bad-token") is False
    assert store.delete_watch(created["id"], created["token"]) is True
    assert store.get_watch(created["id"]) is None


def test_record_evaluation_updates_bookkeeping(store):
    created = store.create_watch(kind="point", params={})
    store.record_evaluation(created["id"], level="moderate", breached=False, notified=False)
    record = store.get_watch(created["id"])
    assert record["last_level"] == "moderate"
    assert record["last_evaluated_at"] is not None
    assert record["last_breach_at"] is None
    assert record["last_notified_at"] is None

    store.record_evaluation(
        created["id"], level="extreme", breached=True, notified=True, now_iso="2026-07-01T00:00:00+00:00"
    )
    record = store.get_watch(created["id"])
    assert record["last_level"] == "extreme"
    assert record["last_breach_at"] == "2026-07-01T00:00:00+00:00"
    assert record["last_notified_at"] == "2026-07-01T00:00:00+00:00"


def test_level_at_least():
    assert ws.level_at_least("extreme", "high") is True
    assert ws.level_at_least("high", "high") is True
    assert ws.level_at_least("moderate", "high") is False
    assert ws.level_at_least("garbage", "high") is False


def test_public_view_hides_secrets_by_default(store):
    created = store.create_watch(
        kind="point", params={}, channel="webhook", webhook_url="https://example.test/h"
    )
    view = ws.public_view(created)
    assert "token" not in view and "webhook_secret" not in view
    with_secrets = ws.public_view(created, include_secrets=True)
    assert with_secrets["token"] == created["token"]
    assert with_secrets["webhook_secret"] == created["webhook_secret"]
