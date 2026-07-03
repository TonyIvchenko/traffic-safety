from __future__ import annotations

import hashlib
import hmac
import json
from pathlib import Path
from types import SimpleNamespace
import sys

import requests

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import notify


def test_sign_payload_matches_manual_hmac():
    body = b'{"alert": true}'
    expected = hmac.new(b"topsecret", body, hashlib.sha256).hexdigest()
    assert notify.sign_payload(body, "topsecret") == f"sha256={expected}"


def test_verify_signature_round_trip():
    body = b'{"x": 1}'
    signature = notify.sign_payload(body, "s3cr3t")
    assert notify.verify_signature(body, "s3cr3t", signature) is True
    assert notify.verify_signature(body, "wrong", signature) is False
    assert notify.verify_signature(b'{"x": 2}', "s3cr3t", signature) is False


def test_post_webhook_success_signs_and_delivers(monkeypatch):
    seen = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        seen.update(url=url, data=data, headers=headers, timeout=timeout)
        return SimpleNamespace(status_code=204)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    result = notify.post_webhook(
        "https://example.test/hook", {"watch_id": "w1", "risk_level": "extreme"}, "shh"
    )

    assert result == {"delivered": True, "status_code": 204, "error": None}
    assert seen["url"] == "https://example.test/hook"
    assert seen["timeout"] == notify.WEBHOOK_TIMEOUT_SECONDS
    # The signature covers the exact bytes sent.
    assert notify.verify_signature(seen["data"], "shh", seen["headers"]["X-Signature"])
    assert json.loads(seen["data"])["watch_id"] == "w1"


def test_post_webhook_without_secret_omits_signature(monkeypatch):
    seen = {}

    def fake_post(url, data=None, headers=None, timeout=None):
        seen.update(headers=headers)
        return SimpleNamespace(status_code=200)

    monkeypatch.setattr(notify.requests, "post", fake_post)
    notify.post_webhook("https://example.test/hook", {"a": 1})
    assert "X-Signature" not in seen["headers"]


def test_post_webhook_handles_http_error_status(monkeypatch):
    monkeypatch.setattr(
        notify.requests, "post", lambda *a, **k: SimpleNamespace(status_code=500)
    )
    result = notify.post_webhook("https://example.test/hook", {}, "s")
    assert result["delivered"] is False
    assert result["status_code"] == 500


def test_post_webhook_handles_network_error(monkeypatch):
    def boom(*a, **k):
        raise requests.ConnectionError("unreachable")

    monkeypatch.setattr(notify.requests, "post", boom)
    result = notify.post_webhook("https://example.test/hook", {}, "s")
    assert result["delivered"] is False
    assert result["status_code"] is None
    assert "unreachable" in result["error"]
