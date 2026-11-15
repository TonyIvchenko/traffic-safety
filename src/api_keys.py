"""API keys, tiering, and usage metering for the public ``/v1`` API.

This is the Horizon-4 platform foundation. It is entirely **opt-in**: unless
``TRAFFIC_SAFETY_API_KEYS_ENABLED`` is set, none of this is wired into the app
and the service behaves exactly as before (anonymous access with the in-memory
IP rate limiter from :mod:`api_ratelimit`).

When enabled, requests under ``/v1`` are attributed to an identity:

* a valid API key (presented as ``X-API-Key`` or ``Authorization: Bearer``) →
  that key's **tier**, which sets its per-minute rate limit and daily quota;
* otherwise → the ``anonymous`` tier, keyed by client IP (the public-good
  default; set ``TRAFFIC_SAFETY_API_REQUIRE_KEY`` to reject anonymous traffic).

Design choices mirror the rest of the service:

* Keys are stored **hashed** (SHA-256); the raw key is shown once at creation
  and is unrecoverable afterwards — the store never holds a usable secret.
* The key *lifecycle* lives in SQLite (low-frequency writes, like
  :mod:`watch_store`); per-request *usage* is counted in memory by
  :class:`UsageMeter`, so the hot path does no disk I/O. Like the rate limiter,
  usage is therefore **per server instance** — aggregate across replicas via the
  exported snapshot if the deployment ever scales horizontally.
* Every pure helper degrades rather than crashes on malformed input.
"""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
import hashlib
import hmac
import os
from pathlib import Path
import secrets
import sqlite3
import threading
import time
from typing import Callable

from api_ratelimit import TokenBucketRateLimiter

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DEFAULT_DB_PATH = REPO_DIR / "data" / "api_keys.sqlite3"

API_KEY_HEADER = "X-API-Key"
KEY_PREFIX = "rsk"
_PREFIX_DISPLAY_CHARS = 6

# Tier table. ``rate_per_min`` feeds a token-bucket limiter; ``daily_quota`` is a
# rolling per-UTC-day cap (``None`` = unlimited). ``anonymous`` is the identity
# for traffic without a valid key and is never issuable as a key.
ANONYMOUS_TIER = "anonymous"
DEFAULT_TIER = "free"
TIERS: dict[str, dict] = {
    "anonymous": {"label": "Anonymous", "rate_per_min": 120, "daily_quota": 10_000, "issuable": False},
    "free": {"label": "Free", "rate_per_min": 240, "daily_quota": 50_000, "issuable": True},
    "standard": {"label": "Standard", "rate_per_min": 1_200, "daily_quota": 1_000_000, "issuable": True},
    "enterprise": {"label": "Enterprise", "rate_per_min": 6_000, "daily_quota": None, "issuable": True},
}

_FALSY = {"0", "false", "no", "off", ""}


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in _FALSY


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


# --- tiers ---------------------------------------------------------------

def tier_table(anonymous_rate_per_min: int | None = None) -> dict[str, dict]:
    """A copy of :data:`TIERS`, optionally overriding the anonymous rate.

    Lets a deployment keep anonymous throughput continuous with whatever the
    existing IP limiter used, while keyed tiers get their own higher limits.
    """
    table = {name: dict(spec) for name, spec in TIERS.items()}
    if anonymous_rate_per_min is not None:
        try:
            rate = int(anonymous_rate_per_min)
        except (TypeError, ValueError):
            rate = None
        if rate is not None and rate > 0:
            table[ANONYMOUS_TIER]["rate_per_min"] = rate
    return table


def resolve_tier(name: object, table: dict[str, dict] | None = None) -> str:
    """Normalise a tier name to a known tier, falling back to the default."""
    table = table or TIERS
    candidate = str(name or "").strip().lower()
    return candidate if candidate in table else DEFAULT_TIER


def issuable_tiers(table: dict[str, dict] | None = None) -> list[str]:
    table = table or TIERS
    return [name for name, spec in table.items() if spec.get("issuable")]


# --- key material --------------------------------------------------------

def generate_key() -> tuple[str, str]:
    """Return ``(raw_key, display_prefix)``. The raw key is shown only once."""
    raw = f"{KEY_PREFIX}_{secrets.token_urlsafe(32)}"
    return raw, raw[: len(KEY_PREFIX) + 1 + _PREFIX_DISPLAY_CHARS]


def hash_key(raw: object) -> str:
    return hashlib.sha256(str(raw or "").strip().encode("utf-8")).hexdigest()


def extract_key(header_api_key: str | None, authorization: str | None) -> str | None:
    """Pull a raw key out of the ``X-API-Key`` or ``Authorization: Bearer`` header."""
    if header_api_key and header_api_key.strip():
        return header_api_key.strip()
    if authorization and authorization.strip():
        parts = authorization.strip().split(None, 1)
        if len(parts) == 2 and parts[0].lower() == "bearer" and parts[1].strip():
            return parts[1].strip()
    return None


def identity_key(key_id: str) -> str:
    return f"key:{key_id}"


def ip_identity(host: str) -> str:
    return f"ip:{host or 'unknown'}"


# --- store ---------------------------------------------------------------

_SCHEMA = """
CREATE TABLE IF NOT EXISTS api_keys (
    id TEXT PRIMARY KEY,
    key_hash TEXT NOT NULL UNIQUE,
    prefix TEXT NOT NULL,
    label TEXT,
    tier TEXT NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL
);
"""


class ApiKeyStore:
    """SQLite store of API-key *lifecycle* (create / list / revoke / delete).

    Only the SHA-256 hash of each key is stored; the raw key is returned once
    from :meth:`create_key` and cannot be recovered afterwards.
    """

    def __init__(self, db_path: Path | str = DEFAULT_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        with self._connect() as connection:
            connection.execute("PRAGMA journal_mode=WAL")
            connection.executescript(_SCHEMA)

    def _connect(self) -> sqlite3.Connection:
        connection = sqlite3.connect(self.db_path, timeout=10)
        connection.row_factory = sqlite3.Row
        return connection

    def create_key(self, *, tier: str = DEFAULT_TIER, label: str | None = None) -> dict:
        tier = str(tier or "").strip().lower() or DEFAULT_TIER
        spec = TIERS.get(tier)
        if spec is None or not spec.get("issuable"):
            raise ValueError(f"tier must be one of {issuable_tiers()}")
        raw, prefix = generate_key()
        record = {
            "id": secrets.token_hex(8),
            "key_hash": hash_key(raw),
            "prefix": prefix,
            "label": (label or "").strip() or None,
            "tier": tier,
            "active": True,
            "created_at": utc_now_iso(),
            "api_key": raw,  # one-time only; never persisted in clear
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO api_keys (id, key_hash, prefix, label, tier, active, created_at)
                VALUES (?, ?, ?, ?, ?, 1, ?)
                """,
                (
                    record["id"],
                    record["key_hash"],
                    record["prefix"],
                    record["label"],
                    record["tier"],
                    record["created_at"],
                ),
            )
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["active"] = bool(record["active"])
        return record

    def authenticate(self, raw_key: object) -> dict | None:
        """Return the active record for ``raw_key``, or ``None``.

        Looks up by hash; the extra constant-time compare guards against any
        theoretical hash-prefix match surviving the unique index.
        """
        if not str(raw_key or "").strip():
            return None
        digest = hash_key(raw_key)
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM api_keys WHERE key_hash = ? AND active = 1", (digest,)
            ).fetchone()
        if row is None:
            return None
        record = self._row_to_record(row)
        if not hmac.compare_digest(record["key_hash"], digest):
            return None
        return record

    def get(self, key_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM api_keys WHERE id = ?", (str(key_id),)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def list_keys(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM api_keys ORDER BY created_at"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def set_active(self, key_id: str, active: bool) -> dict | None:
        if self.get(key_id) is None:
            return None
        with self._connect() as connection:
            connection.execute(
                "UPDATE api_keys SET active = ? WHERE id = ?",
                (1 if active else 0, str(key_id)),
            )
        return self.get(key_id)

    def delete(self, key_id: str) -> bool:
        if self.get(key_id) is None:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM api_keys WHERE id = ?", (str(key_id),))
        return True


_DEFAULT_STORE: ApiKeyStore | None = None


def get_default_store() -> ApiKeyStore:
    """Process-wide store; ``TRAFFIC_SAFETY_API_KEY_DB`` overrides the path."""
    global _DEFAULT_STORE
    path = Path(os.getenv("TRAFFIC_SAFETY_API_KEY_DB", str(DEFAULT_DB_PATH)))
    if _DEFAULT_STORE is None or _DEFAULT_STORE.db_path != path:
        _DEFAULT_STORE = ApiKeyStore(path)
    return _DEFAULT_STORE


def public_view(record: dict, *, include_secret: bool = False) -> dict:
    """Shape a stored key for API responses; the raw key only at creation."""
    view = {
        "id": record["id"],
        "prefix": record["prefix"],
        "label": record.get("label"),
        "tier": record["tier"],
        "active": record.get("active", True),
        "created_at": record["created_at"],
    }
    if include_secret and record.get("api_key"):
        view["api_key"] = record["api_key"]
    return view


# --- usage metering (in-memory, per instance) ---------------------------

class UsageMeter:
    """Thread-safe per-identity request counter, bucketed by UTC day.

    Counts live in memory only (no disk I/O in the request path). ``clock`` is
    injectable so tests can roll the day over deterministically.
    """

    def __init__(self, clock: Callable[[], float] = time.time) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        # identity -> {"date": "YYYY-MM-DD", "count": int, "total": int, "last": iso}
        self._state: dict[str, dict] = {}

    def _today(self) -> str:
        return datetime.fromtimestamp(self._clock(), tz=timezone.utc).strftime("%Y-%m-%d")

    def record(self, identity: str) -> int:
        """Increment today's count for ``identity``; return the new day count."""
        today = self._today()
        with self._lock:
            entry = self._state.get(identity)
            if entry is None or entry["date"] != today:
                entry = {"date": today, "count": 0, "total": entry["total"] if entry else 0}
                self._state[identity] = entry
            entry["count"] += 1
            entry["total"] += 1
            entry["last"] = utc_now_iso()
            return entry["count"]

    def day_count(self, identity: str) -> int:
        today = self._today()
        with self._lock:
            entry = self._state.get(identity)
            return int(entry["count"]) if entry and entry["date"] == today else 0

    def usage(self, identity: str) -> dict:
        today = self._today()
        with self._lock:
            entry = self._state.get(identity)
            if not entry:
                return {"date": today, "requests_today": 0, "requests_total": 0, "last_request_at": None}
            return {
                "date": today,
                "requests_today": int(entry["count"]) if entry["date"] == today else 0,
                "requests_total": int(entry.get("total", 0)),
                "last_request_at": entry.get("last"),
            }

    def snapshot(self) -> dict[str, dict]:
        with self._lock:
            return {identity: dict(entry) for identity, entry in self._state.items()}


# --- identity resolution -------------------------------------------------

@dataclass(frozen=True)
class Identity:
    key_id: str | None
    tier: str
    label: str | None

    @property
    def is_anonymous(self) -> bool:
        return self.key_id is None

    def meter_key(self, ip_host: str) -> str:
        return identity_key(self.key_id) if self.key_id else ip_identity(ip_host)


def resolve_identity(
    raw_key: str | None,
    store: ApiKeyStore | None,
    table: dict[str, dict] | None = None,
) -> Identity:
    table = table or TIERS
    if raw_key and store is not None:
        record = store.authenticate(raw_key)
        if record is not None:
            tier = resolve_tier(record.get("tier"), table)
            return Identity(record["id"], tier, record.get("label"))
    return Identity(None, ANONYMOUS_TIER, None)


class _LimiterRegistry:
    """One token-bucket limiter per distinct per-minute rate.

    Identities within a rate are isolated by their own bucket key, so two tiers
    that happen to share a rate still don't share buckets across tiers because
    the identity string encodes the key id / IP, not the tier.
    """

    def __init__(self, clock: Callable[[], float] = time.monotonic) -> None:
        self._clock = clock
        self._lock = threading.Lock()
        self._by_rate: dict[int, TokenBucketRateLimiter] = {}

    def for_rate(self, rate_per_min: int) -> TokenBucketRateLimiter:
        rate = int(rate_per_min)
        with self._lock:
            limiter = self._by_rate.get(rate)
            if limiter is None:
                limiter = TokenBucketRateLimiter(rate_per_min=rate, clock=self._clock)
                self._by_rate[rate] = limiter
            return limiter


# --- config + middleware -------------------------------------------------

def api_key_config_from_env() -> dict | None:
    """Return the API-key config, or ``None`` when the feature is disabled."""
    if not _env_flag("TRAFFIC_SAFETY_API_KEYS_ENABLED", default=False):
        return None
    admin_token = (os.getenv("TRAFFIC_SAFETY_API_ADMIN_TOKEN") or "").strip() or None
    return {
        "require_key": _env_flag("TRAFFIC_SAFETY_API_REQUIRE_KEY", default=False),
        "admin_token": admin_token,
        "db_path": os.getenv("TRAFFIC_SAFETY_API_KEY_DB", str(DEFAULT_DB_PATH)),
    }


def install_api_key_middleware(
    app,
    *,
    store: ApiKeyStore,
    meter: UsageMeter,
    table: dict[str, dict] | None = None,
    require_key: bool = False,
    path_prefix: str = "/v1",
    registry: _LimiterRegistry | None = None,
) -> _LimiterRegistry:
    """Install the opt-in identity / tiered-rate-limit / usage middleware.

    Replaces the plain IP rate limiter for ``path_prefix`` traffic: anonymous
    requests are still IP-keyed at the anonymous tier, while keyed requests get
    their tier's rate and daily quota. Returns the limiter registry (for tests).
    """
    from starlette.requests import Request
    from starlette.responses import JSONResponse

    table = table or TIERS
    registry = registry or _LimiterRegistry()

    def _tier_spec(name: str) -> dict:
        return table.get(name) or table.get(DEFAULT_TIER) or {"rate_per_min": 120, "daily_quota": None}

    @app.middleware("http")
    async def _api_key_gate(request: Request, call_next):
        if not request.url.path.startswith(path_prefix):
            return await call_next(request)
        # CORS preflight carries no auth by design — let the CORS layer answer it
        # (and don't spend the caller's quota on a preflight).
        if request.method == "OPTIONS":
            return await call_next(request)

        raw_key = extract_key(
            request.headers.get(API_KEY_HEADER), request.headers.get("authorization")
        )
        identity = resolve_identity(raw_key, store, table)

        if require_key and identity.is_anonymous:
            return JSONResponse(
                {"detail": f"an API key is required; present it as {API_KEY_HEADER}"},
                status_code=401,
            )

        spec = _tier_spec(identity.tier)
        client = request.client
        ip_host = client.host if client and client.host else "unknown"
        meter_key = identity.meter_key(ip_host)

        # Per-minute rate limit (token bucket), keyed by identity within its tier.
        rate = int(spec.get("rate_per_min") or 0)
        decision = None
        if rate > 0:
            decision = registry.for_rate(rate).check(meter_key)
            if not decision.allowed:
                response = JSONResponse(
                    {"detail": "rate limit exceeded; slow down and retry"},
                    status_code=429,
                )
                response.headers["Retry-After"] = str(max(1, int(round(decision.retry_after))))
                response.headers["X-RateLimit-Limit"] = str(decision.limit)
                response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
                response.headers["X-API-Key-Tier"] = identity.tier
                return response

        # Daily quota (rolling per-UTC-day), counted in memory.
        today_count = meter.record(meter_key)
        quota = spec.get("daily_quota")
        if quota is not None and today_count > int(quota):
            response = JSONResponse(
                {"detail": "daily quota exceeded for this tier; try again tomorrow or upgrade"},
                status_code=429,
            )
            response.headers["Retry-After"] = "3600"
            response.headers["X-API-Key-Tier"] = identity.tier
            response.headers["X-RateLimit-Quota"] = str(quota)
            response.headers["X-RateLimit-Quota-Remaining"] = "0"
            return response

        response = await call_next(request)
        response.headers["X-API-Key-Tier"] = identity.tier
        if decision is not None:
            response.headers["X-RateLimit-Limit"] = str(decision.limit)
            response.headers["X-RateLimit-Remaining"] = str(decision.remaining)
        if quota is not None:
            response.headers["X-RateLimit-Quota"] = str(quota)
            response.headers["X-RateLimit-Quota-Remaining"] = str(max(0, int(quota) - today_count))
        return response

    return registry
