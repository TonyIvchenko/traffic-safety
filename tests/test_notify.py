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


# --------------------------------------------------------------------------- #
# Email delivery (H1.3)
# --------------------------------------------------------------------------- #

_SMTP_ENV = (
    "TRAFFIC_SAFETY_SMTP_HOST",
    "TRAFFIC_SAFETY_SMTP_PORT",
    "TRAFFIC_SAFETY_SMTP_USERNAME",
    "TRAFFIC_SAFETY_SMTP_PASSWORD",
    "TRAFFIC_SAFETY_SMTP_FROM",
    "TRAFFIC_SAFETY_SMTP_STARTTLS",
)


def _clear_smtp_env(monkeypatch):
    for name in _SMTP_ENV:
        monkeypatch.delenv(name, raising=False)


def test_email_not_configured_without_host(monkeypatch):
    _clear_smtp_env(monkeypatch)
    assert notify.email_configured() is False
    result = notify.send_email("driver@example.test", "Subj", "Body")
    assert result == {"delivered": False, "error": "smtp_not_configured"}


def test_smtp_settings_from_env(monkeypatch):
    _clear_smtp_env(monkeypatch)
    monkeypatch.setenv("TRAFFIC_SAFETY_SMTP_HOST", "smtp.example.test")
    monkeypatch.setenv("TRAFFIC_SAFETY_SMTP_USERNAME", "mailer")
    monkeypatch.setenv("TRAFFIC_SAFETY_SMTP_STARTTLS", "0")
    cfg = notify.smtp_settings()
    assert cfg["host"] == "smtp.example.test"
    assert cfg["port"] == 587  # default
    assert cfg["sender"] == "mailer"  # falls back to the username
    assert cfg["starttls"] is False
    assert notify.email_configured() is True


def test_send_email_uses_smtp_and_reports_delivered(monkeypatch):
    sent = {}

    class FakeSMTP:
        def __init__(self, host, port, timeout=None):
            sent.update(host=host, port=port, timeout=timeout)

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def starttls(self, context=None):
            sent["starttls"] = True

        def login(self, username, password):
            sent["login"] = (username, password)

        def send_message(self, message):
            sent["to"] = message["To"]
            sent["from"] = message["From"]
            sent["subject"] = message["Subject"]
            sent["body"] = message.get_content()

    monkeypatch.setattr(notify.smtplib, "SMTP", FakeSMTP)
    cfg = {
        "host": "smtp.example.test",
        "port": 2525,
        "username": "mailer",
        "password": "pw",
        "sender": "alerts@example.test",
        "starttls": True,
    }
    result = notify.send_email("driver@example.test", "High risk", "Drive safe", settings=cfg)

    assert result == {"delivered": True, "error": None}
    assert sent["host"] == "smtp.example.test" and sent["port"] == 2525
    assert sent["starttls"] is True
    assert sent["login"] == ("mailer", "pw")
    assert sent["to"] == "driver@example.test"
    assert sent["from"] == "alerts@example.test"
    assert sent["subject"] == "High risk"
    assert "Drive safe" in sent["body"]


def test_send_email_reports_smtp_failure(monkeypatch):
    import smtplib as _smtplib

    def boom(*a, **k):
        raise _smtplib.SMTPException("relay refused")

    monkeypatch.setattr(notify.smtplib, "SMTP", boom)
    cfg = {"host": "h", "port": 25, "username": None, "password": None,
           "sender": "a@b.test", "starttls": False}
    result = notify.send_email("x@y.test", "s", "b", settings=cfg)
    assert result["delivered"] is False
    assert "relay refused" in result["error"]


def test_send_email_requires_recipient():
    cfg = {"host": "h", "port": 25, "username": None, "password": None,
           "sender": "a@b.test", "starttls": False}
    assert notify.send_email("", "s", "b", settings=cfg) == {
        "delivered": False, "error": "no_recipient"
    }
