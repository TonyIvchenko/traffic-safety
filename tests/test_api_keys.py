from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api_keys


# --- key material --------------------------------------------------------

def test_generate_key_has_prefix_and_is_hashable():
    raw, prefix = api_keys.generate_key()
    assert raw.startswith("rsk_")
    assert raw.startswith(prefix)
    assert len(prefix) == len("rsk_") + 6
    digest = api_keys.hash_key(raw)
    assert len(digest) == 64
    # Hashing is stable and whitespace-insensitive; raw is never recoverable.
    assert api_keys.hash_key(f"  {raw}  ") == digest
    assert digest != raw


def test_extract_key_reads_both_header_forms():
    assert api_keys.extract_key("rsk_abc", None) == "rsk_abc"
    assert api_keys.extract_key(None, "Bearer rsk_xyz") == "rsk_xyz"
    assert api_keys.extract_key(None, "bearer rsk_xyz") == "rsk_xyz"
    assert api_keys.extract_key(None, "Basic nope") is None
    assert api_keys.extract_key(None, None) is None
    # X-API-Key wins when both are present.
    assert api_keys.extract_key("rsk_header", "Bearer rsk_auth") == "rsk_header"


# --- tiers ---------------------------------------------------------------

def test_tier_table_override_anonymous_rate():
    table = api_keys.tier_table(anonymous_rate_per_min=300)
    assert table["anonymous"]["rate_per_min"] == 300
    # The base table is untouched (copy, not alias).
    assert api_keys.TIERS["anonymous"]["rate_per_min"] == 120
    # A nonsense override is ignored.
    assert api_keys.tier_table(anonymous_rate_per_min=0)["anonymous"]["rate_per_min"] == 120
    assert api_keys.tier_table(anonymous_rate_per_min="x")["anonymous"]["rate_per_min"] == 120


def test_resolve_tier_falls_back_to_default():
    assert api_keys.resolve_tier("standard") == "standard"
    assert api_keys.resolve_tier("STANDARD") == "standard"
    assert api_keys.resolve_tier("bogus") == api_keys.DEFAULT_TIER
    assert api_keys.resolve_tier(None) == api_keys.DEFAULT_TIER


def test_issuable_tiers_excludes_anonymous():
    issuable = api_keys.issuable_tiers()
    assert "anonymous" not in issuable
    assert set(issuable) == {"free", "standard", "enterprise"}


# --- store ---------------------------------------------------------------

def test_store_create_authenticate_revoke(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    record = store.create_key(tier="standard", label="Acme")
    raw = record["api_key"]
    assert raw.startswith("rsk_")
    assert record["tier"] == "standard"
    assert record["label"] == "Acme"

    authed = store.authenticate(raw)
    assert authed is not None and authed["id"] == record["id"]
    assert "api_key" not in authed  # the raw key is never read back from storage

    # Unknown / empty keys authenticate to nothing.
    assert store.authenticate("rsk_wrong") is None
    assert store.authenticate("") is None
    assert store.authenticate(None) is None

    # A revoked key stops authenticating but is still listable.
    assert store.set_active(record["id"], False) is not None
    assert store.authenticate(raw) is None
    assert len(store.list_keys()) == 1

    assert store.delete(record["id"]) is True
    assert store.delete(record["id"]) is False
    assert store.list_keys() == []


def test_store_rejects_non_issuable_tier(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    for bad in ("anonymous", "bogus"):
        try:
            store.create_key(tier=bad)
        except ValueError:
            pass
        else:  # pragma: no cover - should not happen
            raise AssertionError(f"expected ValueError for tier {bad!r}")


def test_public_view_hides_secret_unless_requested(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    record = store.create_key(tier="free")
    assert "api_key" in api_keys.public_view(record, include_secret=True)
    assert "api_key" not in api_keys.public_view(record)
    assert "key_hash" not in api_keys.public_view(record, include_secret=True)


# --- usage meter ---------------------------------------------------------

class _Clock:
    def __init__(self, t: float = 1_700_000_000.0) -> None:
        self.t = t

    def __call__(self) -> float:
        return self.t


def test_usage_meter_counts_and_rolls_over_daily():
    clock = _Clock()
    meter = api_keys.UsageMeter(clock=clock)
    assert meter.record("key:a") == 1
    assert meter.record("key:a") == 2
    assert meter.day_count("key:a") == 2
    assert meter.day_count("key:b") == 0

    usage = meter.usage("key:a")
    assert usage["requests_today"] == 2
    assert usage["requests_total"] == 2
    assert usage["last_request_at"] is not None

    # Next UTC day: the day count resets but the lifetime total carries over.
    clock.t += 86_400
    assert meter.day_count("key:a") == 0
    assert meter.record("key:a") == 1
    assert meter.usage("key:a")["requests_total"] == 3


def test_resolve_identity_prefers_valid_key(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    record = store.create_key(tier="standard")
    ident = api_keys.resolve_identity(record["api_key"], store)
    assert ident.key_id == record["id"]
    assert ident.tier == "standard"
    assert ident.is_anonymous is False

    anon = api_keys.resolve_identity("rsk_bad", store)
    assert anon.is_anonymous is True
    assert anon.tier == api_keys.ANONYMOUS_TIER
    assert api_keys.resolve_identity(None, store).is_anonymous is True


# --- middleware ----------------------------------------------------------

def _app(store, meter, *, table, require_key=False, registry=None):
    app = FastAPI()

    @app.get("/v1/ping")
    def ping():
        return {"ok": True}

    @app.get("/open")
    def open_route():
        return {"ok": True}

    api_keys.install_api_key_middleware(
        app,
        store=store,
        meter=meter,
        table=table,
        require_key=require_key,
        registry=registry,
    )
    return app


def test_middleware_tiers_rate_limit_and_sets_headers(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    key = store.create_key(tier="standard")["api_key"]
    table = {
        "anonymous": {"label": "Anonymous", "rate_per_min": 2, "daily_quota": None, "issuable": False},
        "standard": {"label": "Standard", "rate_per_min": 100, "daily_quota": None, "issuable": True},
    }
    client = TestClient(_app(store, api_keys.UsageMeter(), table=table))

    # Anonymous: capacity == rate == 2, so the third request is throttled.
    a1 = client.get("/v1/ping")
    a2 = client.get("/v1/ping")
    a3 = client.get("/v1/ping")
    assert a1.status_code == 200
    assert a1.headers["X-API-Key-Tier"] == "anonymous"
    assert a1.headers["X-RateLimit-Limit"] == "2"
    assert a2.status_code == 200
    assert a3.status_code == 429
    assert int(a3.headers["Retry-After"]) >= 1

    # The keyed client has its own, much larger bucket and tier header.
    keyed = client.get("/v1/ping", headers={"X-API-Key": key})
    assert keyed.status_code == 200
    assert keyed.headers["X-API-Key-Tier"] == "standard"
    assert keyed.headers["X-RateLimit-Limit"] == "100"


def test_middleware_enforces_daily_quota(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    key = store.create_key(tier="free")["api_key"]
    table = {
        "anonymous": {"label": "Anonymous", "rate_per_min": 1000, "daily_quota": None, "issuable": False},
        "free": {"label": "Free", "rate_per_min": 1000, "daily_quota": 2, "issuable": True},
    }
    client = TestClient(_app(store, api_keys.UsageMeter(), table=table))
    headers = {"X-API-Key": key}

    ok1 = client.get("/v1/ping", headers=headers)
    ok2 = client.get("/v1/ping", headers=headers)
    blocked = client.get("/v1/ping", headers=headers)
    assert ok1.status_code == 200
    assert ok1.headers["X-RateLimit-Quota"] == "2"
    assert ok1.headers["X-RateLimit-Quota-Remaining"] == "1"
    assert ok2.status_code == 200
    assert ok2.headers["X-RateLimit-Quota-Remaining"] == "0"
    assert blocked.status_code == 429
    assert "quota" in blocked.json()["detail"].lower()


def test_middleware_require_key_rejects_anonymous(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    key = store.create_key(tier="free")["api_key"]
    client = TestClient(
        _app(store, api_keys.UsageMeter(), table=api_keys.TIERS, require_key=True)
    )
    assert client.get("/v1/ping").status_code == 401
    assert client.get("/v1/ping", headers={"X-API-Key": "rsk_bad"}).status_code == 401
    assert client.get("/v1/ping", headers={"X-API-Key": key}).status_code == 200


def test_middleware_lets_cors_preflight_through(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    app = _app(store, api_keys.UsageMeter(), table=api_keys.TIERS, require_key=True)

    @app.options("/v1/ping")
    def ping_options():
        return {"ok": True}

    client = TestClient(app)
    # A keyless OPTIONS preflight is not rejected even with require_key on.
    response = client.options("/v1/ping")
    assert response.status_code == 200
    assert "X-API-Key-Tier" not in response.headers


def test_middleware_leaves_non_v1_paths_alone(tmp_path):
    store = api_keys.ApiKeyStore(tmp_path / "keys.sqlite3")
    client = TestClient(
        _app(store, api_keys.UsageMeter(), table=api_keys.TIERS, require_key=True)
    )
    # /open is outside the prefix, so even require_key doesn't gate it.
    response = client.get("/open")
    assert response.status_code == 200
    assert "X-API-Key-Tier" not in response.headers


def test_config_from_env_disabled_by_default(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_API_KEYS_ENABLED", raising=False)
    assert api_keys.api_key_config_from_env() is None

    monkeypatch.setenv("TRAFFIC_SAFETY_API_KEYS_ENABLED", "1")
    monkeypatch.setenv("TRAFFIC_SAFETY_API_ADMIN_TOKEN", "sekret")
    monkeypatch.setenv("TRAFFIC_SAFETY_API_REQUIRE_KEY", "yes")
    config = api_keys.api_key_config_from_env()
    assert config is not None
    assert config["admin_token"] == "sekret"
    assert config["require_key"] is True
