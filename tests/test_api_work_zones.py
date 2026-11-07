from __future__ import annotations

import importlib.util
from pathlib import Path

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_work_zones_near_finds_active_zone():
    client = TestClient(MODULE.api)
    # Near the LA I-10 reference work zone, at a time inside its window.
    resp = client.get(
        "/v1/work-zones/near?lat=34.0340&lon=-118.2450&radius_km=1.5&when=2026-11-13T00:00:00Z"
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["in_work_zone"] is True
    assert body["work_zone_count"] >= 1
    assert body["risk_multiplier"] > 1.0
    assert body["work_zones"]
    assert body["as_of_utc"].startswith("2026-11-13")
    assert resp.headers["cache-control"] == "no-store"


def test_work_zones_near_rural_point_is_clear():
    client = TestClient(MODULE.api)
    body = client.get("/v1/work-zones/near?lat=39.0&lon=-100.0").json()
    assert body["in_work_zone"] is False
    assert body["risk_multiplier"] == 1.0
    assert body["work_zones"] == []


def test_work_zones_respects_the_when_window():
    client = TestClient(MODULE.api)
    # Before any reference event starts -> nothing active, even near a zone.
    body = client.get(
        "/v1/work-zones/near?lat=34.0340&lon=-118.2450&when=2020-01-01T00:00:00Z"
    ).json()
    assert body["in_work_zone"] is False


def test_work_zones_validation():
    client = TestClient(MODULE.api)
    assert client.get("/v1/work-zones/near?lat=120&lon=-100").status_code == 422
    assert client.get("/v1/work-zones/near?lat=34&lon=-118&when=not-a-date").status_code == 422


def test_meta_documents_work_zones():
    client = TestClient(MODULE.api)
    block = client.get("/v1/meta").json()["work_zones"]
    assert block["enabled"] is True
    assert block["source"] == "wzdx"
    assert "/v1/work-zones/near" in block["endpoints"]
    assert block["loaded_total"] >= block["active_now"]
