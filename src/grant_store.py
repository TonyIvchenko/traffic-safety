"""Serve precomputed per-county grant datasets to the /v1/grants endpoints.

build_grant_dataset.py writes one ``<GEOID>.json`` per county under a reports
directory; this store reads them on demand so an operator can drop in a
refreshed dataset without restarting the service. The directory is configurable
via ``TRAFFIC_SAFETY_GRANT_DIR`` (mirroring the watch store's env override) so
tests and deployments can point at their own data.

GEOIDs are validated as digit strings before touching the filesystem, which
also blocks path-traversal via the query parameter.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

DEFAULT_GRANT_DIR = Path(__file__).resolve().parents[1] / "data" / "reports" / "grant"
GRANT_DIR_ENV = "TRAFFIC_SAFETY_GRANT_DIR"
GEOID_MAX_LEN = 11  # 2=state, 5=county, 11=tract


def valid_geoid(geoid: str) -> bool:
    text = str(geoid)
    return text.isdigit() and 1 <= len(text) <= GEOID_MAX_LEN


def summarize(report: dict) -> dict:
    """Compact headline view of a full grant report (drops the big tables)."""
    return {
        "jurisdiction": report.get("jurisdiction", {}),
        "data_vintage": report.get("data_vintage", {}),
        "generated_at_utc": report.get("generated_at_utc"),
        "crash_summary": report.get("crash_summary", {}),
        "high_injury_network": report.get("high_injury_network", {}),
        "hin_corridor_count": len(report.get("hin_corridors", []) or []),
        "systemic_location_count": len(report.get("systemic_locations", []) or []),
        "has_benefit_cost": "benefit_cost" in report,
    }


class GrantStore:
    """Read-only accessor over a directory of ``<GEOID>.json`` grant reports."""

    def __init__(self, directory) -> None:
        self._dir = Path(directory)

    @property
    def directory(self) -> Path:
        return self._dir

    def available_geoids(self) -> list[str]:
        if not self._dir.is_dir():
            return []
        return sorted(path.stem for path in self._dir.glob("*.json") if valid_geoid(path.stem))

    def count(self) -> int:
        return len(self.available_geoids())

    def get_report(self, geoid: str) -> dict | None:
        if not valid_geoid(geoid):
            return None
        path = self._dir / f"{geoid}.json"
        if not path.is_file():
            return None
        try:
            data = json.loads(path.read_text(encoding="utf-8"))
        except (ValueError, OSError):
            return None
        # Enforce the dict contract: a valid-but-non-object payload (list, str,
        # number) must degrade to None so callers never .get() on the wrong type.
        return data if isinstance(data, dict) else None

    def summary(self, geoid: str) -> dict | None:
        report = self.get_report(geoid)
        return summarize(report) if report is not None else None


def get_default_store() -> GrantStore:
    return GrantStore(os.environ.get(GRANT_DIR_ENV) or DEFAULT_GRANT_DIR)
