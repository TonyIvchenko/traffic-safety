"""End-to-end verification of the equity / Justice40 overlay on synthetic data.

Walks the chain: raw segments + a tract equity index -> build_equity_overlay ->
persisted overlay parquet -> EquityOverlay accessor -> the four /v1/equity
endpoints + /v1/meta, asserting the SVI / disadvantaged / risk numbers stay
consistent across every layer. A real run needs TIGER + FARS + SVI/CEJST data
(absent in CI), so this stands in as durable regression coverage.
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

import build_equity_overlay as beo
import equity


def load_main():
    module_path = REPO_DIR / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_main()


@pytest.fixture()
def equity_stack(tmp_path, monkeypatch):
    index = equity.EquityIndex(
        {
            "06037920100": {"svi_percentile": 0.9, "disadvantaged": True},
            "06037920200": {"svi_percentile": 0.2, "disadvantaged": False},
            "06059000100": {"svi_percentile": 0.6, "disadvantaged": False},
        }
    )
    segments = pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "tract_geoid": ["06037920100", "06037920200", "06059000100", "06037920100"],
            "risk": [0.8, 0.7, 0.6, 0.5],
            "crashes": [10.0, 2.0, 1.0, 4.0],
            "center_lat": [34.0, 34.1, 33.7, 34.02],
            "center_lon": [-118.2, -118.3, -117.8, -118.22],
            "fullname": ["Main St", "1st Ave", "Oak Rd", "Elm St"],
        }
    )
    overlay = beo.build_equity_overlay(segments, index)

    overlay_path = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(overlay_path, index=False)
    index_path = tmp_path / "tract_equity.csv"
    pd.DataFrame(
        {
            "tract_geoid": ["06037920100", "06037920200", "06059000100"],
            "svi_percentile": [0.9, 0.2, 0.6],
            "disadvantaged": [True, False, False],
        }
    ).to_csv(index_path, index=False)

    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH", str(overlay_path))
    monkeypatch.setenv("TRAFFIC_SAFETY_EQUITY_PATH", str(index_path))
    return SimpleNamespace(overlay=overlay, client=TestClient(MAIN.api))


def test_overlay_join_is_consistent(equity_stack):
    rows = {r["segment_id"]: r for r in equity_stack.overlay.to_dict("records")}
    # Tract 06037920100 (segments a, d) is disadvantaged, high SVI.
    assert bool(rows["a"]["disadvantaged"]) is True
    assert rows["a"]["svi_percentile"] == 0.9
    assert bool(rows["d"]["disadvantaged"]) is True
    assert bool(rows["b"]["disadvantaged"]) is False
    assert all(bool(r["in_equity_index"]) for r in rows.values())


def test_point_matches_index(equity_stack, monkeypatch):
    monkeypatch.setattr(MAIN, "_tract_of", lambda lat, lon: "06037920100")
    payload = equity_stack.client.get("/v1/equity/point?lat=34.0&lon=-118.2").json()
    assert payload["tract_geoid"] == "06037920100"
    assert payload["svi_percentile"] == 0.9
    assert payload["disadvantaged"] is True
    assert payload["svi_category"] == "very_high"


def test_hotspots_rank_disadvantaged_first(equity_stack):
    payload = equity_stack.client.get("/v1/equity/hotspots").json()
    assert payload["hotspots"][0]["segment_id"] == "a"  # highest equity priority
    only = equity_stack.client.get("/v1/equity/hotspots?only_disadvantaged=true").json()
    assert {h["segment_id"] for h in only["hotspots"]} == {"a", "d"}


def test_summary_disparity(equity_stack):
    payload = equity_stack.client.get("/v1/equity/summary").json()
    assert payload["segments"] == 4
    assert payload["disadvantaged_segments"] == 2  # a, d
    # Disadvantaged corridors carry 14 crashes over 2 segments vs 3 over 2 -> ratio > 1.
    assert payload["crash_disparity_ratio"] > 1.0
    assert payload["weighted_burden"]["burden_ratio"] is not None


def test_choropleth_tract_rollup(equity_stack):
    payload = equity_stack.client.get("/v1/equity/choropleth").json()
    assert payload["type"] == "FeatureCollection"
    by_tract = {f["properties"]["tract_geoid"]: f for f in payload["features"]}
    assert set(by_tract) == {"06037920100", "06037920200", "06059000100"}
    la = by_tract["06037920100"]["properties"]  # segments a + d
    assert la["segment_count"] == 2
    assert la["crashes"] == 14.0
    assert la["disadvantaged"] is True
    assert la["svi_percentile"] == 0.9


def test_meta_reports_overlay_size(equity_stack):
    payload = equity_stack.client.get("/v1/meta").json()
    assert payload["equity"]["segments"] == 4
    assert payload["equity"]["data_vintage"]["tract_boundaries"] == "2010 census tracts"
