"""Outbound alert delivery: HMAC-signed webhook POSTs.

The receiver verifies authenticity by recomputing
``sha256=HMAC_SHA256(secret, raw_body)`` and comparing it to the
``X-Signature`` header.
"""

from __future__ import annotations

import hashlib
import hmac
import json

import requests

WEBHOOK_TIMEOUT_SECONDS = 10
SIGNATURE_HEADER = "X-Signature"
USER_AGENT = "road-risk-monitor-alerts/1.0"


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
