"""Outbound alert delivery: HMAC-signed webhook POSTs and SMTP email.

Webhook receivers verify authenticity by recomputing
``sha256=HMAC_SHA256(secret, raw_body)`` and comparing it to the
``X-Signature`` header.

Email delivery uses an SMTP relay configured entirely from the environment
(``TRAFFIC_SAFETY_SMTP_*``); when no relay is configured the sender degrades to
a ``smtp_not_configured`` result rather than raising, so a sweep over watches
never crashes on an unconfigured host.
"""

from __future__ import annotations

from email.message import EmailMessage
import hashlib
import hmac
import json
import os
import smtplib
import ssl

import requests

WEBHOOK_TIMEOUT_SECONDS = 10
EMAIL_TIMEOUT_SECONDS = 15
SIGNATURE_HEADER = "X-Signature"
USER_AGENT = "road-risk-monitor-alerts/1.0"
_SMTP_FALSEY = {"", "0", "false", "no", "off"}


def sign_payload(body: bytes, secret: str) -> str:
    digest = hmac.new(str(secret).encode("utf-8"), body, hashlib.sha256).hexdigest()
    return f"sha256={digest}"


def verify_signature(body: bytes, secret: str, signature: str) -> bool:
    return hmac.compare_digest(sign_payload(body, secret), str(signature))


def post_webhook(url: str, payload: dict, secret: str | None = None) -> dict:
    """POST an alert payload; returns a delivery record (never raises)."""
    body = json.dumps(payload, ensure_ascii=True).encode("utf-8")
    headers = {"Content-Type": "application/json", "User-Agent": USER_AGENT}
    if secret:
        headers[SIGNATURE_HEADER] = sign_payload(body, secret)
    try:
        response = requests.post(
            url, data=body, headers=headers, timeout=WEBHOOK_TIMEOUT_SECONDS
        )
        return {
            "delivered": 200 <= response.status_code < 300,
            "status_code": response.status_code,
            "error": None,
        }
    except requests.RequestException as exc:
        return {"delivered": False, "status_code": None, "error": str(exc)}


def smtp_settings() -> dict | None:
    """SMTP relay config from the environment, or None when no host is set."""
    host = os.getenv("TRAFFIC_SAFETY_SMTP_HOST", "").strip()
    if not host:
        return None
    username = os.getenv("TRAFFIC_SAFETY_SMTP_USERNAME", "").strip()
    sender = os.getenv("TRAFFIC_SAFETY_SMTP_FROM", "").strip() or username or "alerts@roadriskmonitor.local"
    starttls = os.getenv("TRAFFIC_SAFETY_SMTP_STARTTLS", "1").strip().lower() not in _SMTP_FALSEY
    try:
        port = int(os.getenv("TRAFFIC_SAFETY_SMTP_PORT", "587") or 587)
    except (TypeError, ValueError):
        port = 587
    return {
        "host": host,
        "port": port,
        "username": username or None,
        "password": os.getenv("TRAFFIC_SAFETY_SMTP_PASSWORD", "") or None,
        "sender": sender,
        "starttls": starttls,
    }


def email_configured() -> bool:
    """Whether an SMTP relay is configured for email delivery."""
    return smtp_settings() is not None


def send_email(to_address: str, subject: str, body: str, *, settings: dict | None = None) -> dict:
    """Send a plain-text alert email; returns a delivery record (never raises)."""
    cfg = settings or smtp_settings()
    if cfg is None:
        return {"delivered": False, "error": "smtp_not_configured"}
    recipient = str(to_address or "").strip()
    if not recipient:
        return {"delivered": False, "error": "no_recipient"}

    message = EmailMessage()
    message["Subject"] = str(subject)
    message["From"] = cfg["sender"]
    message["To"] = recipient
    message.set_content(str(body))
    try:
        with smtplib.SMTP(cfg["host"], cfg["port"], timeout=EMAIL_TIMEOUT_SECONDS) as server:
            if cfg["starttls"]:
                server.starttls(context=ssl.create_default_context())
            if cfg["username"]:
                server.login(cfg["username"], cfg["password"] or "")
            server.send_message(message)
        return {"delivered": True, "error": None}
    except (OSError, smtplib.SMTPException) as exc:
        return {"delivered": False, "error": str(exc)}
