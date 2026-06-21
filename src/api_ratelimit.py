"""In-process token-bucket rate limiting for the public ``/v1`` API.

This is intentionally dependency-free: limits are kept in memory and enforced
per client IP. That means limits are *per server instance* — fine for a single
process, but not shared across replicas. Swap in a Redis-backed limiter if the
deployment ever scales horizontally.
"""

from __future__ import annotations

from dataclasses import dataclass
import os
import threading
import time
from typing import Callable

from starlette.requests import Request
from starlette.responses import JSONResponse, Response

DEFAULT_RATE_PER_MIN = 120
DEFAULT_PATH_PREFIX = "/v1"

_FALSY = {"0", "false", "no", "off", ""}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSY


def _env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    try:
        return int(value.strip())
    except ValueError:
        return default


@dataclass(frozen=True)
class RateLimitDecision:
    allowed: bool
    limit: int
    remaining: int
    retry_after: float


class TokenBucketRateLimiter:
    """Token bucket keyed by an arbitrary identity string (e.g. client IP).

    ``rate_per_min`` tokens are added per minute up to ``burst`` capacity; each
    accepted request consumes one token. The ``clock`` is injectable so tests can
    advance time deterministically.
    """

    def __init__(
        self,
        rate_per_min: int,
        burst: int | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        self.rate_per_min = int(rate_per_min)
        self.capacity = int(burst) if burst else max(1, int(rate_per_min))
        self._refill_per_sec = self.rate_per_min / 60.0
        self._clock = clock
        self._lock = threading.Lock()
        self._state: dict[str, tuple[float, float]] = {}

    def check(self, key: str) -> RateLimitDecision:
        with self._lock:
            now = self._clock()
            tokens, last = self._state.get(key, (float(self.capacity), now))
            tokens = min(self.capacity, tokens + max(0.0, now - last) * self._refill_per_sec)

            if tokens >= 1.0:
                tokens -= 1.0
                self._state[key] = (tokens, now)
                return RateLimitDecision(True, self.rate_per_min, int(tokens), 0.0)

            self._state[key] = (tokens, now)
            needed = 1.0 - tokens
            retry_after = needed / self._refill_per_sec if self._refill_per_sec > 0 else 60.0
            return RateLimitDecision(False, self.rate_per_min, 0, retry_after)


def rate_limiter_from_env() -> TokenBucketRateLimiter | None:
    """Build a limiter from environment config, or ``None`` when disabled."""
    if not _env_flag("TRAFFIC_SAFETY_RATE_LIMIT_ENABLED", default=True):
        return None
    rate = _env_int("TRAFFIC_SAFETY_RATE_LIMIT_PER_MIN", DEFAULT_RATE_PER_MIN)
    if rate <= 0:
        return None
    burst = _env_int("TRAFFIC_SAFETY_RATE_LIMIT_BURST", rate)
    return TokenBucketRateLimiter(rate_per_min=rate, burst=burst)


def client_key(request: Request) -> str:
    client = request.client
    return client.host if client and client.host else "unknown"


def _apply_headers(response: Response, decision: RateLimitDecision) -> None:
    response.headers["X-RateLimit-Limit"] = str(decision.limit)
    response.headers["X-RateLimit-Remaining"] = str(decision.remaining)


def install_rate_limit_middleware(
    app,
    limiter: TokenBucketRateLimiter | None,
    path_prefix: str = DEFAULT_PATH_PREFIX,
) -> bool:
    """Register an HTTP middleware that throttles requests under ``path_prefix``.

    Returns ``True`` if a limiter was installed, ``False`` when disabled.
    """
    if limiter is None:
        return False

    @app.middleware("http")
    async def _rate_limit(request: Request, call_next):
        if not request.url.path.startswith(path_prefix):
            return await call_next(request)

        decision = limiter.check(client_key(request))
        if not decision.allowed:
            response = JSONResponse(
                {"detail": "rate limit exceeded; slow down and retry"},
                status_code=429,
            )
            response.headers["Retry-After"] = str(max(1, int(round(decision.retry_after))))
            _apply_headers(response, decision)
            return response

        response = await call_next(request)
        _apply_headers(response, decision)
        return response

    return True
