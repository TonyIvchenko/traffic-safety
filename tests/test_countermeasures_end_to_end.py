"""End-to-end verification of the countermeasure recommendations on synthetic data.

Walks the chain: a synthetic High Injury Network parquet -> CountermeasureStore
-> crash-typing + CMF matching + benefit-cost -> the /v1/countermeasures endpoints
and /v1/meta, and confirms the offline report picks the same best-benefit-cost
treatment the endpoint returns. A real run needs the HIN parquet (absent in CI),
so this is durable regression coverage.
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import SimpleNamespace

import pandas as pd
import pytest
from fastapi.testclient import TestClient

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_countermeasure_report as bcr


def load_main():
    module_path = REPO_DIR / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_main()
ANALYSIS_YEARS = 5  # CountermeasureStore default


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["rural-1", "urban-1", "urban-2"],
            "fullname": ["County Rd", "Main St", "1st Ave"],
            "mtfcc": ["S1100", "S1200", "S1200"],
            "rttyp": ["S", "U", "U"],
            "rur_urb": [1, 2, 2],
            "length_km": [3.0, 1.5, 1.0],
            "center_lat": [36.0, 34.0, 34.1],
            "center_lon": [-119.0, -118.2, -118.3],
            "fatal_crashes": [12.0, 8.0, 4.0],
            "hin_rank": [1, 2, 3],
        }
    )


@pytest.fixture()
def stack(tmp_path, monkeypatch):
    path = tmp_path / "hin.parquet"
    _segments().to_parquet(path, index=False)
    monkeypatch.setenv("TRAFFIC_SAFETY_CM_SEGMENTS_PATH", str(path))
    return SimpleNamespace(segments=_segments(), client=TestClient(MAIN.api))


def test_segment_endpoint_recommends_with_benefit_cost(stack):
    payload = stack.client.get("/v1/countermeasures/segment?segment_id=rural-1").json()
    assert payload["segment"]["segment_id"] == "rural-1"
    assert payload["count"] >= 1
    top = payload["recommendations"][0]
    assert top["benefit_cost"]["benefit_cost_ratio"] > 0
    assert top["cmf_confidence"] in {"low", "moderate", "high", "unknown"}
    # A rural highway's top treatments address run-off-road crashes.
    assert any(
        "run_off_road" in r["applicable_crash_types"] for r in payload["recommendations"]
    )
    assert "cmf_note" in payload


def test_hotspots_ranked_by_fatal_crashes(stack):
    payload = stack.client.get("/v1/countermeasures/hotspots").json()
    assert [h["segment_id"] for h in payload["hotspots"]] == ["rural-1", "urban-1", "urban-2"]
    assert all(h["recommended"] is not None for h in payload["hotspots"])


def test_meta_reports_segment_count(stack):
    meta = stack.client.get("/v1/meta").json()["countermeasures"]
    assert meta["segments"] == 3
    assert meta["catalog_size"] >= 10


def test_report_matches_endpoint_recommendation(stack):
    # The offline report's best-BCR pick equals the endpoint's top recommendation.
    report = bcr.build_countermeasure_report(_segments(), analysis_years=ANALYSIS_YEARS)
    report_rural = report[report["segment_id"] == "rural-1"].iloc[0]
    endpoint_top = stack.client.get(
        "/v1/countermeasures/segment?segment_id=rural-1"
    ).json()["recommendations"][0]
    assert report_rural["recommended_id"] == endpoint_top["id"]
    assert report_rural["benefit_cost_ratio"] == pytest.approx(
        endpoint_top["benefit_cost"]["benefit_cost_ratio"]
    )
