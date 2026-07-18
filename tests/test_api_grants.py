from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import pytest
from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()

SAMPLE_REPORT = {
    "jurisdiction": {"geoid": "06037", "name": "Los Angeles County", "level": "county"},
    "generated_at_utc": "2026-07-14T00:00:00+00:00",
    "data_vintage": {"fars_years": [2018, 2024], "crash_source": "FARS fatal crashes"},
    "crash_summary": {"total_fatal_crashes": 42, "total_fatalities": 45, "by_year": {2022: 20}},
    "high_injury_network": {"hin_segments": 2, "length_share": 0.4, "weighted_crash_share": 0.9},
    "hin_corridors": [{"segment_id": "a", "hin_rank": 1}, {"segment_id": "b", "hin_rank": 2}],
    "systemic_locations": [{"segment_id": "c", "systemic_score": 0.9}],
    "methodology": {"high_injury_network": "Ranked by weighted crashes per km."},
    "data_sources": [{"name": "FARS", "publisher": "NHTSA", "use": "fatal crash counts"}],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    (tmp_path / "06037.json").write_text(json.dumps(SAMPLE_REPORT), encoding="utf-8")
    return TestClient(MODULE.api)


def test_grants_summary_returns_headline(client):
    response = client.get("/v1/grants/summary?geoid=06037")
    assert response.status_code == 200
    payload = response.json()
    assert payload["jurisdiction"]["name"] == "Los Angeles County"
    assert payload["crash_summary"]["total_fatal_crashes"] == 42
    assert payload["high_injury_network"]["hin_segments"] == 2
    assert payload["hin_corridor_count"] == 2
    assert payload["systemic_location_count"] == 1
    assert payload["has_benefit_cost"] is False
    # The summary is compact — full corridor tables are not inlined.
    assert "hin_corridors" not in payload
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_grants_summary_unknown_geoid_returns_404(client):
    response = client.get("/v1/grants/summary?geoid=99999")
    assert response.status_code == 404
    assert "99999" in response.json()["detail"]


def test_grants_summary_bad_geoid_is_404_not_traversal(client):
    # A non-digit GEOID never touches the filesystem -> treated as not found.
    response = client.get("/v1/grants/summary", params={"geoid": "../../etc/passwd"})
    assert response.status_code == 404


def test_grants_summary_requires_geoid(client):
    response = client.get("/v1/grants/summary")
    assert response.status_code == 422


def test_grants_summary_non_object_file_is_404_not_500(tmp_path, monkeypatch):
    # A dropped-in file with valid-but-non-object JSON must degrade to 404,
    # never crash the handler with a 500.
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    (tmp_path / "06037.json").write_text("[1, 2, 3]", encoding="utf-8")
    response = TestClient(MODULE.api).get("/v1/grants/summary?geoid=06037")
    assert response.status_code == 404
