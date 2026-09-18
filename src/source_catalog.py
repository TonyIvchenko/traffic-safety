"""Loader for the upstream data-source catalog (``source_catalog.json``).

Resolves the source ids each output dataset (:mod:`datasets`) references into full
provenance records (url, access, coverage, notes), so the ``/v1/datasets`` catalog
can present where its data comes from.

Degrade-not-crash: a missing/corrupt/misshaped catalog yields an empty source
list rather than raising. The path is configurable via
``TRAFFIC_SAFETY_SOURCE_CATALOG_PATH``.
"""

from __future__ import annotations

from functools import lru_cache
import json
import os
from pathlib import Path

REPO_DIR = Path(__file__).resolve().parents[1]
DEFAULT_SOURCE_CATALOG_PATH = REPO_DIR / "source_catalog.json"
SOURCE_CATALOG_PATH_ENV = "TRAFFIC_SAFETY_SOURCE_CATALOG_PATH"

_EMPTY = {"recommended_baseline": {}, "sources": []}


@lru_cache(maxsize=4)
def _load_cached(path_str: str) -> dict:
    path = Path(path_str)
    if not path.exists():
        return dict(_EMPTY)
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return dict(_EMPTY)
    if not isinstance(payload, dict):
        return dict(_EMPTY)
    sources = payload.get("sources")
    payload["sources"] = sources if isinstance(sources, list) else []
    payload.setdefault("recommended_baseline", {})
    return payload


def load_source_catalog(path=None) -> dict:
    """The parsed source catalog (cached), or an empty catalog if unavailable."""
    resolved = path or os.environ.get(SOURCE_CATALOG_PATH_ENV) or DEFAULT_SOURCE_CATALOG_PATH
    return _load_cached(str(resolved))


def get_source(source_id, *, path=None) -> dict | None:
    """A full source record by id (case-insensitive), or None if unknown."""
    key = str(source_id).strip().lower()
    for source in load_source_catalog(path).get("sources", []):
        if isinstance(source, dict) and str(source.get("id", "")).strip().lower() == key:
            return dict(source)
    return None


def resolve_sources(source_ids, *, path=None) -> list[dict]:
    """Resolve source ids to full records; unknown ids become a ``known: False`` marker."""
    resolved = []
    for source_id in source_ids or []:
        record = get_source(source_id, path=path)
        resolved.append(record if record is not None else {"id": str(source_id), "known": False})
    return resolved


def dataset_provenance(dataset, *, path=None) -> dict:
    """A dataset dict with its ``sources`` id list replaced by full source records."""
    if not isinstance(dataset, dict):
        return {}
    return {**dataset, "sources": resolve_sources(dataset.get("sources", []), path=path)}
