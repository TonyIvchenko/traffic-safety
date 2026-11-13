from __future__ import annotations

import importlib.util
from pathlib import Path

import pandas as pd
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


def _hin_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["seg-1", "seg-2"],
            "fullname": ["Main St", "Rural Rd"],
            "mtfcc": ["S1200", "S1100"],
            "rur_urb": [2, 1],
            "length_km": [1.5, 2.0],
            "center_lat": [34.0, 36.0],
            "center_lon": [-118.2, -119.0],
            "fatal_crashes": [6.0, 10.0],
            "hin_rank": [2, 1],
        }
    )


def _equity_overlay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["seg-1", "seg-2"],
            "tract_geoid": ["06037000100", "06037000200"],
            "center_lat": [34.0, 36.0],
            "center_lon": [-118.2, -119.0],
            "svi_percentile": [0.9, 0.2],
            "disadvantaged": [True, False],
            "crashes": [8.0, 2.0],
            "risk": [0.6, 0.3],
        }
    )


@pytest.fixture()
def impact_client(tmp_path, monkeypatch):
    hin_path = tmp_path / "hin.parquet"
    _hin_frame().to_parquet(hin_path, index=False)
    overlay_path = tmp_path / "equity_overlay.parquet"
    _equity_overlay_frame().to_parquet(overlay_path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_CM_SEGMENTS_PATH", str(hin_path))
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(overlay_path))
    return TestClient(MODULE.api)


def test_impact_summary_rolls_up_prevention_and_equity(impact_client):
    resp = impact_client.get("/v1/impact/summary")
    assert resp.status_code == 200
    body = resp.json()
    assert body["status"] == "ok"
    assert body["method"] == "impact_screening_rollup"
    prevention = body["prevention"]
    # Both fixture segments get a recommended treatment -> crashes averted > 0.
    assert prevention["segments_with_treatment"] >= 1
    assert prevention["annual_crashes_averted"] > 0
    assert prevention["present_value_benefit"] > 0
    # Equity block present because the overlay is non-empty.
    assert "equity" in body
    assert body["equity"]["disadvantaged_crash_share"] is not None
    assert body["coverage"]["hin_segments"] == 2
    assert body["coverage"]["equity_segments"] == 2
    assert body["caveats"]
    assert resp.headers["cache-control"] == "public, max-age=3600"


def test_impact_summary_respects_overrides(impact_client):
    base = impact_client.get("/v1/impact/summary").json()
    longer = impact_client.get("/v1/impact/summary?service_life_years=30").json()
    # A longer service life values the same annual benefit higher.
    assert longer["prevention"]["service_life_years"] == 30
    assert longer["prevention"]["present_value_benefit"] > base["prevention"]["present_value_benefit"]


def test_meta_documents_impact(impact_client):
    block = impact_client.get("/v1/meta").json()["impact"]
    assert block["enabled"] is True
    assert block["method"] == "impact_screening_rollup"
    assert "/v1/impact/summary" in block["endpoints"]
