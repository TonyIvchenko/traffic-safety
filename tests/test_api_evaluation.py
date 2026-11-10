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


def test_before_after_evaluation_returns_cmf():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/evaluation/before-after",
        json={
            "sites": [
                {
                    "site_id": "A",
                    "observed_before": 12,
                    "observed_after": 6,
                    "predicted_before": 10,
                    "predicted_after": 10,
                    "overdispersion": 0.5,
                }
            ]
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["method"] == "empirical_bayes_before_after"
    assert abs(body["cmf"] - 0.48) < 0.01
    assert body["significant_95"] is True
    assert body["caveats"]
    assert resp.headers["cache-control"] == "no-store"


def test_request_level_overdispersion_default_applies():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/evaluation/before-after",
        json={
            "overdispersion": 0.5,
            "sites": [
                {"observed_before": 12, "observed_after": 6, "predicted_before": 10, "predicted_after": 10}
            ],
        },
    )
    assert resp.status_code == 200
    assert abs(resp.json()["cmf"] - 0.48) < 0.01


def test_missing_overdispersion_is_rejected():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/evaluation/before-after",
        json={
            "sites": [
                {"observed_before": 12, "observed_after": 6, "predicted_before": 10, "predicted_after": 10}
            ]
        },
    )
    assert resp.status_code == 422


def test_invalid_inputs_are_rejected():
    client = TestClient(MODULE.api)
    # predicted_before must be > 0
    bad_pred = client.post(
        "/v1/evaluation/before-after",
        json={"sites": [{"observed_before": 5, "observed_after": 5, "predicted_before": 0,
                          "predicted_after": 5, "overdispersion": 0.5}]},
    )
    assert bad_pred.status_code == 422
    # at least one site required
    empty = client.post("/v1/evaluation/before-after", json={"sites": []})
    assert empty.status_code == 422


def test_meta_documents_evaluation():
    client = TestClient(MODULE.api)
    block = client.get("/v1/meta").json()["evaluation"]
    assert block["enabled"] is True
    assert block["method"] == "empirical_bayes_before_after"
    assert "/v1/evaluation/before-after" in block["endpoints"]
