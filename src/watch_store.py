"""SQLite-backed store for risk watches (the proactive-alert subscriptions).

A watch is an anonymous subscription identified by ``id`` and authorized by a
per-watch ``token`` (returned once at creation). ``params`` holds the watched
geometry/time settings as JSON; the evaluator CLI updates the ``last_*``
bookkeeping columns on each run.
"""

from __future__ import annotations

from datetime import datetime, timezone
import hmac
import json
import os
from pathlib import Path
import secrets
import sqlite3

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
DEFAULT_DB_PATH = REPO_DIR / "data" / "watches.sqlite3"

WATCH_KINDS = {"point", "route", "area"}
WATCH_CHANNELS = {"poll", "webhook"}
LEVEL_ORDER = {"low": 0, "moderate": 1, "high": 2, "extreme": 3}

_SCHEMA = """
CREATE TABLE IF NOT EXISTS watches (
    id TEXT PRIMARY KEY,
    token TEXT NOT NULL,
    kind TEXT NOT NULL,
    params TEXT NOT NULL,
    threshold_level TEXT NOT NULL,
    channel TEXT NOT NULL,
    webhook_url TEXT,
    webhook_secret TEXT,
    cooldown_minutes INTEGER NOT NULL,
    active INTEGER NOT NULL DEFAULT 1,
    created_at TEXT NOT NULL,
    last_evaluated_at TEXT,
    last_level TEXT,
    last_breach_at TEXT,
    last_notified_at TEXT
);
"""


def utc_now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def level_at_least(level: str, threshold: str) -> bool:
    return LEVEL_ORDER.get(str(level), -1) >= LEVEL_ORDER.get(str(threshold), 99)


class WatchStore:
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

    def create_watch(
        self,
        *,
        kind: str,
        params: dict,
        threshold_level: str = "high",
        channel: str = "poll",
        webhook_url: str | None = None,
        cooldown_minutes: int = 60,
    ) -> dict:
        kind = str(kind).strip().lower()
        channel = str(channel).strip().lower()
        threshold_level = str(threshold_level).strip().lower()
        if kind not in WATCH_KINDS:
            raise ValueError(f"kind must be one of {sorted(WATCH_KINDS)}")
        if channel not in WATCH_CHANNELS:
            raise ValueError(f"channel must be one of {sorted(WATCH_CHANNELS)}")
        if threshold_level not in LEVEL_ORDER:
            raise ValueError(f"threshold_level must be one of {sorted(LEVEL_ORDER)}")
        if channel == "webhook" and not (webhook_url or "").strip():
            raise ValueError("webhook channel requires webhook_url")
        if int(cooldown_minutes) < 0:
            raise ValueError("cooldown_minutes must be >= 0")

        record = {
            "id": secrets.token_hex(8),
            "token": secrets.token_urlsafe(24),
            "kind": kind,
            "params": dict(params),
            "threshold_level": threshold_level,
            "channel": channel,
            "webhook_url": (webhook_url or "").strip() or None,
            "webhook_secret": secrets.token_urlsafe(24) if channel == "webhook" else None,
            "cooldown_minutes": int(cooldown_minutes),
            "active": True,
            "created_at": utc_now_iso(),
            "last_evaluated_at": None,
            "last_level": None,
            "last_breach_at": None,
            "last_notified_at": None,
        }
        with self._connect() as connection:
            connection.execute(
                """
                INSERT INTO watches (
                    id, token, kind, params, threshold_level, channel, webhook_url,
                    webhook_secret, cooldown_minutes, active, created_at
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record["id"],
                    record["token"],
                    record["kind"],
                    json.dumps(record["params"]),
                    record["threshold_level"],
                    record["channel"],
                    record["webhook_url"],
                    record["webhook_secret"],
                    record["cooldown_minutes"],
                    1,
                    record["created_at"],
                ),
            )
        return record

    @staticmethod
    def _row_to_record(row: sqlite3.Row) -> dict:
        record = dict(row)
        record["params"] = json.loads(record["params"])
        record["active"] = bool(record["active"])
        return record

    def get_watch(self, watch_id: str) -> dict | None:
        with self._connect() as connection:
            row = connection.execute(
                "SELECT * FROM watches WHERE id = ?", (str(watch_id),)
            ).fetchone()
        return self._row_to_record(row) if row else None

    def get_watch_authorized(self, watch_id: str, token: str) -> dict | None:
        record = self.get_watch(watch_id)
        if record is None or not hmac.compare_digest(record["token"], str(token)):
            return None
        return record

    def list_active(self) -> list[dict]:
        with self._connect() as connection:
            rows = connection.execute(
                "SELECT * FROM watches WHERE active = 1 ORDER BY created_at"
            ).fetchall()
        return [self._row_to_record(row) for row in rows]

    def set_active(self, watch_id: str, token: str, active: bool) -> dict | None:
        if self.get_watch_authorized(watch_id, token) is None:
            return None
        with self._connect() as connection:
            connection.execute(
                "UPDATE watches SET active = ? WHERE id = ?",
                (1 if active else 0, str(watch_id)),
            )
        return self.get_watch(watch_id)

    def delete_watch(self, watch_id: str, token: str) -> bool:
        if self.get_watch_authorized(watch_id, token) is None:
            return False
        with self._connect() as connection:
            connection.execute("DELETE FROM watches WHERE id = ?", (str(watch_id),))
        return True

    def record_evaluation(
        self,
        watch_id: str,
        *,
        level: str,
        breached: bool,
        notified: bool,
        now_iso: str | None = None,
    ) -> None:
        now = now_iso or utc_now_iso()
        with self._connect() as connection:
            connection.execute(
                """
                UPDATE watches SET
                    last_evaluated_at = ?,
                    last_level = ?,
                    last_breach_at = CASE WHEN ? THEN ? ELSE last_breach_at END,
                    last_notified_at = CASE WHEN ? THEN ? ELSE last_notified_at END
                WHERE id = ?
                """,
                (now, str(level), int(bool(breached)), now, int(bool(notified)), now, str(watch_id)),
            )


_DEFAULT_STORE: WatchStore | None = None


def get_default_store() -> WatchStore:
    """Process-wide store; TRAFFIC_SAFETY_WATCH_DB overrides the DB path."""
    global _DEFAULT_STORE
    path = Path(os.getenv("TRAFFIC_SAFETY_WATCH_DB", str(DEFAULT_DB_PATH)))
    if _DEFAULT_STORE is None or _DEFAULT_STORE.db_path != path:
        _DEFAULT_STORE = WatchStore(path)
    return _DEFAULT_STORE


def public_view(record: dict, *, include_secrets: bool = False) -> dict:
    """Shape a stored record for API responses; secrets only at creation time."""
    view = {
        "id": record["id"],
        "kind": record["kind"],
        "params": record["params"],
        "threshold_level": record["threshold_level"],
        "channel": record["channel"],
        "webhook_url": record["webhook_url"],
        "cooldown_minutes": record["cooldown_minutes"],
        "active": record["active"],
        "created_at": record["created_at"],
        "last_evaluated_at": record["last_evaluated_at"],
        "last_level": record["last_level"],
        "last_breach_at": record["last_breach_at"],
        "last_notified_at": record["last_notified_at"],
    }
    if include_secrets:
        view["token"] = record["token"]
        view["webhook_secret"] = record["webhook_secret"]
    return view
