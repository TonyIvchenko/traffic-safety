from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient

MAIN_PATH = Path(__file__).resolve().parents[1] / "src" / "main.py"


def _load_main(name: str):
    spec = importlib.util.spec_from_file_location(name, MAIN_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _enable(monkeypatch, tmp_path, *, admin_token="test-admin-token", require_key=False):
    monkeypatch.setenv("TRAFFIC_SAFETY_API_KEYS_ENABLED", "1")
    monkeypatch.setenv("TRAFFIC_SAFETY_API_ADMIN_TOKEN", admin_token)
    monkeypatch.setenv("TRAFFIC_SAFETY_API_REQUIRE_KEY", "1" if require_key else "0")
    monkeypatch.setenv("TRAFFIC_SAFETY_API_KEY_DB", str(tmp_path / "api_keys.sqlite3"))


def test_key_lifecycle_end_to_end(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    module = _load_main("ts_main_keys_enabled")
    client = TestClient(module.api)
    admin = {"X-Admin-Token": "test-admin-token"}

    # meta advertises the feature as enabled with its tiers.
    meta = client.get("/v1/meta").json()["api_keys"]
    assert meta["enabled"] is True
    assert meta["auth_header"] == "X-API-Key"
    assert any(t["name"] == "standard" for t in meta["tiers"])

    # Mint a key (admin only); the raw key is returned exactly once.
    created = client.post("/v1/keys", json={"tier": "standard", "label": "Acme"}, headers=admin)
    assert created.status_code == 201
    body = created.json()
    key = body["api_key"]
    assert key.startswith("rsk_")
    assert body["tier"] == "standard"

    # The listing (admin) shows it without the secret, with usage attached.
    listed = client.get("/v1/keys", headers=admin)
    assert listed.status_code == 200
    payload = listed.json()
    assert payload["count"] == 1
    assert "api_key" not in payload["keys"][0]
    assert "usage" in payload["keys"][0]

    # /keys/me authenticates with the key itself and reports tier + limits.
    me = client.get("/v1/keys/me", headers={"X-API-Key": key})
    assert me.status_code == 200
    me_body = me.json()
    assert me_body["tier"] == "standard"
    assert me_body["limits"]["rate_per_min"] == module.api_keys.TIERS["standard"]["rate_per_min"]
    assert me_body["usage"]["requests_today"] >= 1
    assert me.headers["X-API-Key-Tier"] == "standard"

    # Revoke it: the key no longer authenticates, but the row remains.
    revoked = client.post(f"/v1/keys/{body['id']}/revoke", headers=admin)
    assert revoked.status_code == 200
    assert revoked.json()["active"] is False
    assert client.get("/v1/keys/me", headers={"X-API-Key": key}).status_code == 401


def test_admin_endpoints_require_admin_token(monkeypatch, tmp_path):
    _enable(monkeypatch, tmp_path)
    module = _load_main("ts_main_keys_admin_gate")
    client = TestClient(module.api)

    # No token / wrong token are both rejected.
    assert client.post("/v1/keys", json={"tier": "free"}).status_code == 401
    assert (
        client.post("/v1/keys", json={"tier": "free"}, headers={"X-Admin-Token": "nope"}).status_code
        == 401
    )
    # An unknown key on the self endpoint is a 401, not a 500.
    assert client.get("/v1/keys/me", headers={"X-API-Key": "rsk_unknown"}).status_code == 401
    # A non-issuable tier is a clean 422.
    bad = client.post(
        "/v1/keys", json={"tier": "anonymous"}, headers={"X-Admin-Token": "test-admin-token"}
    )
    assert bad.status_code == 422


def test_feature_off_by_default(monkeypatch, tmp_path):
    for var in (
        "TRAFFIC_SAFETY_API_KEYS_ENABLED",
        "TRAFFIC_SAFETY_API_ADMIN_TOKEN",
        "TRAFFIC_SAFETY_API_REQUIRE_KEY",
    ):
        monkeypatch.delenv(var, raising=False)
    module = _load_main("ts_main_keys_disabled")
    client = TestClient(module.api)

    meta = client.get("/v1/meta").json()
    assert meta["api_keys"]["enabled"] is False
    # The scope string stays as the legacy value when keys are off.
    assert meta["rate_limit"]["scope"] == "per_client_ip"
    # Management endpoints answer 404 when the feature is disabled.
    assert client.get("/v1/keys", headers={"X-Admin-Token": "x"}).status_code == 404
    assert client.get("/v1/keys/me", headers={"X-API-Key": "rsk_x"}).status_code == 404
