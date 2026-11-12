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


def test_trends_series_classifies_new_hotspot():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/trends/series",
        json={"series": [0, 1, 2, 5, 8], "hot_threshold": 4, "recent_years": 2},
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["method"] == "mann_kendall_theil_sen"
    assert body["trend"] == "increasing"
    assert body["category"] == "new_hotspot"
    assert body["caveats"]
    assert resp.headers["cache-control"] == "no-store"


def test_trends_series_rejects_short_and_misaligned():
    client = TestClient(MODULE.api)
    short = client.post("/v1/trends/series", json={"series": [1, 2]})
    assert short.status_code == 422  # pydantic min_length
    misaligned = client.post(
        "/v1/trends/series", json={"series": [1, 2, 3], "years": [2020, 2021]}
    )
    assert misaligned.status_code == 422  # years must align


def test_trends_hotspots_ranks_and_tallies():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/trends/hotspots",
        json={
            "hot_quantile": 0.75,
            "recent_years": 2,
            "locations": [
                {"location_id": "rising", "series": [1, 2, 4, 7, 11]},
                {"location_id": "flat", "series": [2, 2, 2, 2, 2]},
                {"location_id": "falling", "series": [11, 8, 5, 3, 1]},
                {"location_id": "hot", "series": [20, 21, 20, 22, 21]},
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["evaluated_count"] == 4
    assert body["hotspots"][0]["location_id"] in {"rising", "hot"}
    assert "category_counts" in body
    assert resp.headers["cache-control"] == "no-store"


def test_trends_hotspots_rising_only():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/trends/hotspots",
        json={
            "rising_only": True,
            "hot_quantile": 0.5,
            "locations": [
                {"location_id": "rising", "series": [1, 2, 4, 7, 11]},
                {"location_id": "flat", "series": [2, 2, 2, 2, 2]},
            ],
        },
    )
    ids = {h["location_id"] for h in resp.json()["hotspots"]}
    assert "flat" not in ids


def test_meta_documents_trends():
    client = TestClient(MODULE.api)
    block = client.get("/v1/meta").json()["trends"]
    assert block["enabled"] is True
    assert block["method"] == "mann_kendall_emerging_hotspot"
    assert "/v1/trends/hotspots" in block["endpoints"]
    assert "intensifying_hotspot" in block["categories"]
