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


def _candidate(**overrides) -> dict:
    base = {"project_id": "P1", "treatment_cost": 100_000.0, "annual_crashes_reduced": 0.1}
    base.update(overrides)
    return base


def test_program_build_returns_funded_bundle():
    client = TestClient(MODULE.api)
    resp = client.post(
        "/v1/program/build",
        json={
            "budget": 150_000.0,
            "candidates": [
                _candidate(project_id="cheap", treatment_cost=50_000.0, annual_crashes_reduced=0.2,
                           svi_percentile=0.9, disadvantaged=True, segment_id="S-1",
                           countermeasure="Road diet"),
                _candidate(project_id="pricey", treatment_cost=200_000.0, annual_crashes_reduced=0.3),
            ],
        },
    )
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["method"] == "equity_weighted_benefit_cost_program"
    assert body["selected_count"] == 1
    assert body["selected"][0]["project_id"] == "cheap"
    assert body["caveats"]
    assert resp.headers["cache-control"] == "no-store"
    # SS4A/HSIP export rides along.
    assert body["export"][0]["location"] == "S-1"
    assert body["export"][0]["serves_disadvantaged_community"] is True
    assert body["export"][0]["funding_rank"] == 1


def test_program_build_equity_weighting_reorders():
    client = TestClient(MODULE.api)
    payload_candidates = [
        _candidate(project_id="rich", treatment_cost=100_000.0, annual_crashes_reduced=0.12, svi_percentile=0.1),
        _candidate(project_id="poor", treatment_cost=100_000.0, annual_crashes_reduced=0.10,
                   svi_percentile=0.95, disadvantaged=True),
    ]
    plain = client.post(
        "/v1/program/build",
        json={"budget": 100_000.0, "svi_weight": 0.0, "disadvantaged_boost": 0.0,
              "candidates": payload_candidates},
    ).json()
    assert plain["selected"][0]["project_id"] == "rich"
    weighted = client.post(
        "/v1/program/build",
        json={"budget": 100_000.0, "svi_weight": 2.0, "disadvantaged_boost": 1.0,
              "candidates": payload_candidates},
    ).json()
    assert weighted["selected"][0]["project_id"] == "poor"


def test_program_build_rejects_empty_and_unscorable():
    client = TestClient(MODULE.api)
    # pydantic: at least one candidate
    empty = client.post("/v1/program/build", json={"candidates": []})
    assert empty.status_code == 422
    # pydantic: treatment_cost must be > 0
    bad_cost = client.post(
        "/v1/program/build",
        json={"candidates": [{"treatment_cost": 0, "annual_crashes_reduced": 1.0}]},
    )
    assert bad_cost.status_code == 422
    # business rule: a candidate with no benefit source cannot be scored -> 422
    no_benefit = client.post(
        "/v1/program/build",
        json={"candidates": [{"treatment_cost": 100000.0}]},
    )
    assert no_benefit.status_code == 422


def test_meta_documents_program():
    client = TestClient(MODULE.api)
    block = client.get("/v1/meta").json()["program"]
    assert block["enabled"] is True
    assert block["method"] == "equity_weighted_benefit_cost_program"
    assert "/v1/program/build" in block["endpoints"]
