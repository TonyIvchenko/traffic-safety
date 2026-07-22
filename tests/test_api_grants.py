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
    "hin_corridors": [
        {"segment_id": "a", "fullname": "Main St", "hin_rank": 1, "hin_intensity": 5.0,
         "length_km": 1.0, "fatal_crashes": 5.0, "center_lat": 34.05, "center_lon": -118.24},
        {"segment_id": "b", "fullname": "1st Ave", "hin_rank": 2, "hin_intensity": 3.0,
         "length_km": 1.0, "fatal_crashes": 3.0, "center_lat": 34.10, "center_lon": -118.30},
    ],
    "systemic_locations": [{"segment_id": "c", "systemic_score": 0.9}],
    "methodology": {"high_injury_network": "Ranked by weighted crashes per km."},
    "data_sources": [{"name": "FARS", "publisher": "NHTSA", "use": "fatal crash counts"}],
}


@pytest.fixture()
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    (tmp_path / "06037.json").write_text(json.dumps(SAMPLE_REPORT), encoding="utf-8")
    return TestClient(MODULE.api)


def test_meta_advertises_grants(client):
    payload = client.get("/v1/meta").json()
    grants = payload["grants"]
    assert grants["enabled"] is True
    assert grants["jurisdictions"] == 1  # the fixture wrote one county dataset
    assert set(grants["endpoints"]) == {
        "/v1/grants/summary", "/v1/grants/hin", "/v1/grants/report"
    }
    assert "html" in grants["formats"]


def test_meta_grants_jurisdictions_zero_when_empty(tmp_path, monkeypatch):
    # No dropped-in datasets -> count degrades to 0, never raises.
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    payload = TestClient(MODULE.api).get("/v1/meta").json()
    assert payload["grants"]["jurisdictions"] == 0


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


def test_grants_hin_by_geoid_json(client):
    response = client.get("/v1/grants/hin?geoid=06037")
    assert response.status_code == 200
    payload = response.json()
    assert payload["geoid"] == "06037"
    assert payload["count"] == 2
    assert [c["segment_id"] for c in payload["corridors"]] == ["a", "b"]
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_grants_hin_by_geoid_geojson(client):
    response = client.get("/v1/grants/hin?geoid=06037&format=geojson")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert payload["count"] == 2
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "Point"
    # GeoJSON is [lon, lat]; centroids move into geometry, not properties.
    assert feature["geometry"]["coordinates"] == [-118.24, 34.05]
    assert feature["properties"]["hin_rank"] == 1
    assert "center_lat" not in feature["properties"]


def test_grants_hin_unknown_geoid_404(client):
    assert client.get("/v1/grants/hin?geoid=99999").status_code == 404


def test_grants_hin_by_bbox(client):
    over = client.get("/v1/grants/hin?min_lat=34.0&max_lat=34.2&min_lon=-118.4&max_lon=-118.2")
    assert over.status_code == 200
    body = over.json()
    assert body["count"] == 2
    assert {c["segment_id"] for c in body["corridors"]} == {"a", "b"}
    # A bbox elsewhere returns an empty collection (not a 404).
    empty = client.get("/v1/grants/hin?min_lat=40.0&max_lat=41.0&min_lon=-100.0&max_lon=-99.0")
    assert empty.status_code == 200
    assert empty.json()["count"] == 0


def test_grants_hin_requires_geoid_or_bbox(client):
    assert client.get("/v1/grants/hin").status_code == 422


def test_grants_hin_partial_bbox_is_422(client):
    assert client.get("/v1/grants/hin?min_lat=34.0").status_code == 422


def test_grants_hin_inverted_bbox_is_422(client):
    response = client.get(
        "/v1/grants/hin?min_lat=34.2&max_lat=34.0&min_lon=-118.4&max_lon=-118.2"
    )
    assert response.status_code == 422


def test_grants_hin_top_n_caps_results(client):
    response = client.get("/v1/grants/hin?geoid=06037&top_n=1")
    assert response.json()["count"] == 1


def test_grants_hin_bbox_malformed_corridor_is_200_not_500(tmp_path, monkeypatch):
    # A dropped-in file with a non-dict corridor and a non-numeric centroid must
    # degrade (skip the bad rows), not crash the bbox scan.
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    report = {
        "jurisdiction": {"geoid": "06037", "name": "LA", "level": "county"},
        "hin_corridors": [
            {"segment_id": "a", "hin_intensity": 5.0, "center_lat": 34.05, "center_lon": -118.24},
            "oops",
            {"segment_id": "x", "hin_intensity": "high", "center_lat": "north", "center_lon": -118.3},
        ],
    }
    (tmp_path / "06037.json").write_text(json.dumps(report), encoding="utf-8")
    client = TestClient(MODULE.api)
    response = client.get("/v1/grants/hin?min_lat=34.0&max_lat=34.2&min_lon=-118.4&max_lon=-118.2")
    assert response.status_code == 200
    assert [c["segment_id"] for c in response.json()["corridors"]] == ["a"]


def test_grants_hin_geoid_geojson_malformed_corridor_is_200_not_500(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    report = {
        "jurisdiction": {"geoid": "06037", "name": "LA", "level": "county"},
        "hin_corridors": [
            None,  # non-dict
            {"segment_id": "bad", "center_lat": "n/a", "center_lon": -118.3},  # non-numeric
            {"segment_id": "good", "center_lat": 34.05, "center_lon": -118.24},
        ],
    }
    (tmp_path / "06037.json").write_text(json.dumps(report), encoding="utf-8")
    client = TestClient(MODULE.api)
    response = client.get("/v1/grants/hin?geoid=06037&format=geojson")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 1
    assert payload["features"][0]["properties"]["segment_id"] == "good"


def test_grants_report_json_returns_full_report(client):
    response = client.get("/v1/grants/report?geoid=06037")
    assert response.status_code == 200
    payload = response.json()
    # The full report (not the compact summary): tables + methodology present.
    assert payload["jurisdiction"]["name"] == "Los Angeles County"
    assert isinstance(payload["hin_corridors"], list) and len(payload["hin_corridors"]) == 2
    assert "methodology" in payload
    assert "data_sources" in payload
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_grants_report_html_is_downloadable_document(client):
    response = client.get("/v1/grants/report?geoid=06037&format=html")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/html")
    assert response.headers["content-disposition"] == (
        'attachment; filename="safety-analysis-06037.html"'
    )
    body = response.text
    assert body.startswith("<!doctype html>")
    assert "Los Angeles County" in body


def test_grants_report_unknown_geoid_404(client):
    assert client.get("/v1/grants/report?geoid=99999").status_code == 404


def test_grants_report_requires_geoid(client):
    assert client.get("/v1/grants/report").status_code == 422


def test_grants_report_unknown_format_defaults_to_json(client):
    # Matches the sibling json/geojson convention: only the special format is
    # special-cased; anything else falls through to JSON.
    response = client.get("/v1/grants/report?geoid=06037&format=xml")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["jurisdiction"]["geoid"] == "06037"


def test_grants_report_html_malformed_report_is_200_not_500(tmp_path, monkeypatch):
    # A structurally malformed dropped-in report must still render (degraded),
    # never crash the HTML path with a 500.
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(tmp_path))
    report = {
        "jurisdiction": {"geoid": "06037", "name": "LA", "level": "county"},
        "crash_summary": {"by_year": {"2022": None}},  # non-numeric value
        "hin_corridors": [{"hin_rank": 1, "fullname": "Main St"}, None],  # non-dict row
        "systemic_locations": "nope",  # wrong container type
        "data_sources": [42],  # non-dict row
    }
    (tmp_path / "06037.json").write_text(json.dumps(report), encoding="utf-8")
    response = TestClient(MODULE.api).get("/v1/grants/report?geoid=06037&format=html")
    assert response.status_code == 200
    assert response.text.startswith("<!doctype html>")
