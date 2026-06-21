from __future__ import annotations

from pathlib import Path
import sys

from fastapi import FastAPI
from fastapi.testclient import TestClient

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import api_ratelimit as rl


class _Clock:
    def __init__(self) -> None:
        self.t = 0.0

    def __call__(self) -> float:
        return self.t

    def advance(self, dt: float) -> None:
        self.t += dt


def test_token_bucket_allows_up_to_capacity_then_blocks():
    clock = _Clock()
    limiter = rl.TokenBucketRateLimiter(rate_per_min=60, burst=3, clock=clock)

    first = limiter.check("ip-a")
    second = limiter.check("ip-a")
    third = limiter.check("ip-a")
    fourth = limiter.check("ip-a")

    assert [d.allowed for d in (first, second, third)] == [True, True, True]
    assert [d.remaining for d in (first, second, third)] == [2, 1, 0]
    assert fourth.allowed is False
    assert fourth.remaining == 0
    assert fourth.limit == 60
    assert fourth.retry_after > 0.0


def test_token_bucket_refills_over_time():
    clock = _Clock()
    limiter = rl.TokenBucketRateLimiter(rate_per_min=60, burst=1, clock=clock)

    assert limiter.check("ip-a").allowed is True
    assert limiter.check("ip-a").allowed is False  # bucket empty
    # 60/min == 1 token/sec, so one second restores a token.
    clock.advance(1.0)
    assert limiter.check("ip-a").allowed is True


def test_token_bucket_keys_are_isolated():
    clock = _Clock()
    limiter = rl.TokenBucketRateLimiter(rate_per_min=60, burst=1, clock=clock)

    assert limiter.check("ip-a").allowed is True
    assert limiter.check("ip-a").allowed is False
    # A different client has its own bucket.
    assert limiter.check("ip-b").allowed is True


def test_rate_limiter_from_env(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_RATE_LIMIT_ENABLED", raising=False)
    monkeypatch.delenv("TRAFFIC_SAFETY_RATE_LIMIT_PER_MIN", raising=False)
    limiter = rl.rate_limiter_from_env()
    assert limiter is not None
    assert limiter.rate_per_min == rl.DEFAULT_RATE_PER_MIN

    monkeypatch.setenv("TRAFFIC_SAFETY_RATE_LIMIT_PER_MIN", "0")
    assert rl.rate_limiter_from_env() is None

    monkeypatch.setenv("TRAFFIC_SAFETY_RATE_LIMIT_PER_MIN", "300")
    monkeypatch.setenv("TRAFFIC_SAFETY_RATE_LIMIT_ENABLED", "off")
    assert rl.rate_limiter_from_env() is None


def _app_with_limiter(limiter) -> FastAPI:
    app = FastAPI()

    @app.get("/v1/ping")
    def ping():
        return {"ok": True}

    @app.get("/open")
    def open_route():
        return {"ok": True}

    rl.install_rate_limit_middleware(app, limiter, path_prefix="/v1")
    return app


def test_middleware_throttles_v1_and_sets_headers():
    limiter = rl.TokenBucketRateLimiter(rate_per_min=60, burst=2)
    client = TestClient(_app_with_limiter(limiter))

    ok1 = client.get("/v1/ping")
    ok2 = client.get("/v1/ping")
    blocked = client.get("/v1/ping")

    assert ok1.status_code == 200
    assert ok1.headers["X-RateLimit-Limit"] == "60"
    assert "X-RateLimit-Remaining" in ok1.headers
    assert ok2.status_code == 200
    assert blocked.status_code == 429
    assert int(blocked.headers["Retry-After"]) >= 1
    assert blocked.headers["X-RateLimit-Remaining"] == "0"


def test_middleware_leaves_non_v1_paths_unthrottled():
    limiter = rl.TokenBucketRateLimiter(rate_per_min=60, burst=1)
    client = TestClient(_app_with_limiter(limiter))

    for _ in range(5):
        response = client.get("/open")
        assert response.status_code == 200
        assert "X-RateLimit-Limit" not in response.headers


def test_install_returns_false_when_disabled():
    app = FastAPI()
    assert rl.install_rate_limit_middleware(app, None) is False
