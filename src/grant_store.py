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
    return text.isascii() and text.isdigit() and 1 <= len(text) <= GEOID_MAX_LEN


def _as_float(value) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def corridor_in_bbox(corridor, bbox) -> bool:
    """True when a corridor's centroid lies within ``(min_lat, max_lat, min_lon, max_lon)``.

    Malformed corridors (non-dict, or missing/non-numeric centroid) are skipped
    rather than raising — the store degrades on bad dropped-in files.
    """
    if not isinstance(corridor, dict):
        return False
    min_lat, max_lat, min_lon, max_lon = bbox
    lat = _as_float(corridor.get("center_lat"))
    lon = _as_float(corridor.get("center_lon"))
    if lat is None or lon is None:
        return False
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


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

    def hin_corridors(self, geoid: str) -> list[dict] | None:
        """The county's High Injury Network corridors (report order), or None."""
        report = self.get_report(geoid)
        if report is None:
            return None
        return list(report.get("hin_corridors", []) or [])

    def iter_reports(self):
        for geoid in self.available_geoids():
            report = self.get_report(geoid)
            if report is not None:
                yield geoid, report

    def hin_corridors_in_bbox(self, bbox, *, top_n: int | None = None) -> list[dict]:
        """HIN corridors across all counties whose centroid falls in ``bbox``.

        Ranked by severity intensity (crashes/km) so results are comparable
        across jurisdictions; each corridor is tagged with its ``geoid``. Reads
        every county file — the F1.12 index parquet is the fast-serving path.
        """
        collected: list[dict] = []
        for geoid, report in self.iter_reports():
            jurisdiction = report.get("jurisdiction", {}) or {}
            report_geoid = jurisdiction.get("geoid", geoid)
            for corridor in report.get("hin_corridors", []) or []:
                if corridor_in_bbox(corridor, bbox):
                    collected.append({**corridor, "geoid": report_geoid})
        # corridor_in_bbox already guaranteed every entry is a dict; coerce the
        # ranking key defensively so a non-numeric hin_intensity ranks last, not 500s.
        collected.sort(key=lambda corridor: -(_as_float(corridor.get("hin_intensity")) or 0.0))
        if top_n is not None:
            collected = collected[: int(top_n)]
        return collected


def get_default_store() -> GrantStore:
    return GrantStore(os.environ.get(GRANT_DIR_ENV) or DEFAULT_GRANT_DIR)
