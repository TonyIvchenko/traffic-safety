from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

REPO_DIR = Path(__file__).resolve().parents[1]
MODEL_AVAILABLE = (REPO_DIR / "models" / "traffic_safety.joblib").exists()
pytestmark = pytest.mark.skipif(not MODEL_AVAILABLE, reason="main model bundle not present")


def load_module():
    module_path = REPO_DIR / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_context_applies_work_zone_and_speed_hazards():
    client = TestClient(MODULE.api)
    # LA I-10 DTLA at a peak hour: active work zone (some-lanes-closed, 1.3) and a
    # congested peak speed (1.25) -> combined 1.625.
    resp = client.get(
        "/v1/risk/point?lat=34.0335&lon=-118.2487&day_of_week=3&hour=8&month=9&context=true"
    )
    assert resp.status_code == 200
    body = resp.json()
    ctx = body["context"]
    assert ctx["applied"] is True
    assert ctx["combined_multiplier"] == pytest.approx(1.625)
    sources = {factor["source"] for factor in ctx["factors"]}
    assert sources == {"work_zone", "traffic_speed"}
    # The adjustment never lowers risk and never exceeds 1.0; the base is untouched.
    assert ctx["base_risk_score"] == pytest.approx(body["risk_score"])
    assert ctx["adjusted_risk_score"] >= ctx["base_risk_score"]
    assert ctx["adjusted_risk_score"] <= 1.0


def test_context_absent_when_not_requested():
    client = TestClient(MODULE.api)
    body = client.get(
        "/v1/risk/point?lat=34.0335&lon=-118.2487&day_of_week=3&hour=8&month=9"
    ).json()
    assert body.get("context") is None  # opt-in only


def test_context_neutral_where_no_hazards():
    client = TestClient(MODULE.api)
    # A rural point far from any mapped work zone or speed segment.
    body = client.get(
        "/v1/risk/point?lat=41.5&lon=-101.0&day_of_week=3&hour=8&month=9&context=true"
    ).json()
    ctx = body["context"]
    assert ctx["applied"] is False
    assert ctx["factors"] == []
    assert ctx["combined_multiplier"] == 1.0
    assert ctx["adjusted_risk_score"] == pytest.approx(ctx["base_risk_score"])
