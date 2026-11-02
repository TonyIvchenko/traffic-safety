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


def test_exposure_point_matches_a_mapped_segment():
    client = TestClient(MODULE.api)
    # Near the I-10 DTLA reference segment.
    resp = client.get("/v1/exposure/point?lat=34.0335&lon=-118.2487")
    assert resp.status_code == 200
    body = resp.json()
    assert body["basis"] == "hpms_segment"
    assert body["aadt"] and body["aadt"] > 0
    assert body["distance_km"] is not None
    assert body["annual_vmt_per_km"] and body["annual_vmt_per_km"] > 0
    assert body["source"] == "fhwa_hpms"
    assert resp.headers["cache-control"] == "public, max-age=3600"


def test_exposure_point_class_default_and_unavailable():
    client = TestClient(MODULE.api)
    # Rural point far from any mapped segment: a class hint gives a default.
    default = client.get(
        "/v1/exposure/point?lat=45.0&lon=-100.0&functional_class=local"
    ).json()
    assert default["basis"] == "class_default"
    assert default["aadt"] == MODULE._get_aadt_store().lookup(45.0, -100.0, functional_class="local")["aadt"]

    # Same point, no class hint -> unavailable (no fabricated number).
    unavailable = client.get("/v1/exposure/point?lat=45.0&lon=-100.0").json()
    assert unavailable["basis"] == "unavailable"
    assert unavailable["aadt"] is None
    assert unavailable["annual_vmt_per_km"] is None


def test_exposure_point_validates_coordinates():
    client = TestClient(MODULE.api)
    assert client.get("/v1/exposure/point?lat=120&lon=-100").status_code == 422
    assert client.get("/v1/exposure/point?lat=45&lon=-100&match_radius_km=0").status_code == 422


def test_meta_documents_exposure():
    client = TestClient(MODULE.api)
    exposure_meta = client.get("/v1/meta").json()["exposure"]
    assert exposure_meta["enabled"] is True
    assert exposure_meta["aadt_segments"] > 0
    assert "/v1/exposure/point" in exposure_meta["endpoints"]
    assert "interstate" in exposure_meta["functional_classes"]
    assert exposure_meta["source"] == "fhwa_hpms"
