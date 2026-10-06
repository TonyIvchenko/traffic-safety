"""Degrade-not-crash contract net across the /v1 surface.

A curated battery of edge/boundary requests that must NEVER produce an HTTP 500 or
emit non-standard JSON (NaN/Infinity, which breaks strict parsers). FastAPI may
reject out-of-range inputs with 422 and resolution may 404 — both are fine; a 5xx
is not. This locks in the hardening-sweep findings as durable regression coverage.
"""

from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

from fastapi.testclient import TestClient
import pytest


def _load_main():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main_contract", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = _load_main()
CLIENT = TestClient(MAIN.api)


def _reject_constant(value):
    # Fires only for bare JSON NaN/Infinity/-Infinity literals, not quoted strings.
    raise ValueError(f"non-standard JSON constant: {value}")


def _assert_standard_json(url, response):
    if "json" in response.headers.get("content-type", ""):
        json.loads(response.text, parse_constant=_reject_constant)  # raises on NaN/Infinity

# (path, query) climatology/offline cases — no live weather providers hit.
CASES = [
    # Advisory — boundary coords, out-of-coverage, every format/geometry.
    ("/v1/advisory/point", "lat=0&lon=0"),
    ("/v1/advisory/point", "lat=90&lon=180&day_of_week=7&hour=23&month=12"),
    ("/v1/advisory/point", "lat=-90&lon=-180&day_of_week=1&hour=0&month=1"),
    ("/v1/advisory/region", "region=los_angeles&day_of_week=7&hour=23&month=6"),
    ("/v1/advisory/region", "lat=0&lon=0"),  # uncovered -> 404
    ("/v1/advisory/region", "region=&lat=34.05&lon=-118.24"),  # blank region -> point
    ("/v1/advisory/national", "day_of_week=5&hour=17&min_level=5"),
    ("/v1/advisory/national", "format=geojson&geometry=bbox"),
    ("/v1/advisory/national", "format=geojson&geometry=point&month=12"),
    # Emergency — far-from-everything, long routes, uncovered region.
    ("/v1/emergency/nearest-facility", "lat=0&lon=0&k=25"),
    ("/v1/emergency/nearest-facility", "lat=34.05&lon=-118.24&kind=hospital&max_km=500"),
    ("/v1/emergency/evacuate", "lat=34.05&lon=-118.24&distance_km=200&sample_spacing_km=0.5"),
    ("/v1/emergency/evacuate", "lat=0&lon=0&target=facilities&facility_radius_km=1"),  # none -> 404
    ("/v1/emergency/evacuate", "lat=34.05&lon=-118.24&format=geojson"),
    ("/v1/emergency/readiness", "region=miami&day_of_week=3&hour=3"),
    ("/v1/emergency/readiness", "lat=0&lon=0"),  # uncovered -> 404
    # Open data / catalog / download.
    ("/v1/datasets", ""),
    ("/v1/datasets/critical_facilities", ""),
    ("/v1/datasets/nope", ""),  # 404
    ("/v1/datasets/critical_facilities/download", "format=geojson"),
    ("/v1/datasets/high_injury_network/download", ""),  # query-scoped -> 422
    # Unified profile.
    ("/v1/profile/point", "lat=0&lon=0"),
    ("/v1/profile/point", "lat=34.05&lon=-118.24&segment_radius_km=50&day_of_week=7&hour=23"),
    # Discovery.
    ("/v1/meta", ""),
    ("/v1/health", ""),
]


@pytest.mark.parametrize("path,query", CASES)
def test_get_never_5xx_or_nonstandard_json(path, query):
    url = f"{path}?{query}" if query else path
    response = CLIENT.get(url)
    assert response.status_code < 500, f"{url} -> {response.status_code}: {response.text[:200]}"
    _assert_standard_json(url, response)


def test_post_validation_data_edge_bodies():
    bodies = [
        {"dataset": "critical_facilities", "records": []},
        {"dataset": "critical_facilities", "records": [{}]},
        {"dataset": "high_injury_network", "records": [{"segment_id": "x"}]},
        {"dataset": "equity_overlay", "records": [{"segment_id": "x", "svi_percentile": 5.0}]},
        {"dataset": "critical_facilities", "records": [{"id": "x", "name": "y", "kind": "hospital", "lat": 10**400}]},
    ]
    for body in bodies:
        response = CLIENT.post("/v1/validation/data", json=body)
        assert response.status_code < 500, f"{body} -> {response.status_code}"
        _assert_standard_json("/v1/validation/data", response)
