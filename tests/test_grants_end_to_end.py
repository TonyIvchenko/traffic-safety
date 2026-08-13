"""End-to-end verification of the federal safety-grant pipeline on one county.

Walks the whole chain on synthetic fixtures — HIN construction, systemic
scoring, per-county report assembly + benefit-cost, disk persistence, the grant
store, HTML rendering, the live /v1/grants endpoints, and the rollup index — and
asserts the headline numbers stay consistent across every layer. A real run
needs the TIGER county shapefile + FARS extracts (absent in CI), so this stands
in as durable regression coverage of how the pieces connect.
"""

from __future__ import annotations

import importlib.util
import json
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

import build_grant_dataset as bgd
import build_grant_index as bgi
import grant_html
import grant_store
import hin as hin_mod
import systemic


def load_main():
    module_path = REPO_DIR / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MAIN = load_main()

ANALYSIS_YEARS = 5  # crashes span 2019..2023


def _network() -> pd.DataFrame:
    # A tiny synthetic primary/secondary network across two counties. 06037's
    # Main St + 1st Ave carry the fatal crashes -> they form the HIN; the quiet
    # local roads (0 crashes) are systemic-only candidates.
    rows = [
        # segment_id, county,   fullname,   rttyp, mtfcc,   length_km, fatal, lat,   lon
        ("a", "06037", "Main St", "U", "S1200", 1.0, 6.0, 34.05, -118.24),
        ("b", "06037", "1st Ave", "S", "S1200", 1.0, 5.0, 34.06, -118.25),
        ("c", "06037", "Quiet Ln", "C", "S1400", 2.0, 0.0, 34.07, -118.26),
        ("d", "06059", "Oak Rd", "I", "S1100", 1.5, 3.0, 33.70, -117.80),
        ("e", "06059", "Elm St", "C", "S1400", 1.0, 0.0, 33.71, -117.81),
    ]
    frame = pd.DataFrame(
        rows,
        columns=[
            "segment_id", "county_geoid", "fullname", "rttyp", "mtfcc",
            "length_km", "fatal_crashes", "center_lat", "center_lon",
        ],
    )
    frame["weighted_crashes"] = frame["fatal_crashes"]  # every FARS crash is a K crash
    frame["state_fips"] = frame["county_geoid"].str[:2]
    frame["linearid"] = frame["segment_id"]
    return frame


def _crashes() -> pd.DataFrame:
    rows = []
    la_years = [2019, 2019, 2020, 2020, 2021, 2021, 2022, 2022, 2023, 2023, 2023]  # 11
    rows += [{"county_geoid": "06037", "year": year, "fatals": 1, "lat": 34.05, "lon": -118.24}
             for year in la_years]
    rows += [{"county_geoid": "06059", "year": year, "fatals": 1, "lat": 33.70, "lon": -117.80}
             for year in (2019, 2021, 2023)]  # 3
    return pd.DataFrame(rows)


@pytest.fixture()
def built(tmp_path):
    """Run the offline half of the pipeline and persist per-county reports."""
    network = hin_mod.build_hin(
        _network(), weighted_col="weighted_crashes", length_col="length_km", target_share=0.5
    )
    rates = systemic.systemic_rates(network)
    network = systemic.apply_systemic_scores(network, rates)

    reports = bgd.build_county_reports(
        network,
        _crashes(),
        names={"06037": "Los Angeles County", "06059": "Orange County"},
        analysis_years=ANALYSIS_YEARS,
        benefit_cost_config={"crash_reduction": 0.30, "treatment_cost_per_km": 200_000.0},
    )
    bgd.write_county_reports(reports, tmp_path)
    return SimpleNamespace(dir=tmp_path, reports=reports)


def test_pipeline_builds_a_coherent_county_report(built):
    assert set(built.reports) == {"06037", "06059"}
    la = built.reports["06037"]
    assert la["jurisdiction"]["name"] == "Los Angeles County"
    assert la["crash_summary"]["total_fatal_crashes"] == 11
    assert la["high_injury_network"]["hin_segments"] == 2
    assert [c["segment_id"] for c in la["hin_corridors"]] == ["a", "b"]
    # Systemic screening surfaces the history-poor local road in the county.
    assert "c" in {loc["segment_id"] for loc in la["systemic_locations"]}

    # Benefit-cost now defaults to corridor-specific CMFs (F3.12).
    benefit_cost = la["benefit_cost"]
    assert benefit_cost["treated_corridors"] == 2
    assert 0.0 < benefit_cost["mean_crash_reduction"] < 1.0
    assert benefit_cost["benefit_cost_ratio"] > 0
    assert benefit_cost["basis"].startswith("corridor-specific")
    json.dumps(la)  # the whole report survives a JSON round-trip


def test_store_reads_persisted_reports(built):
    store = grant_store.GrantStore(built.dir)
    assert store.count() == 2
    summary = store.summary("06037")
    assert summary["high_injury_network"]["hin_segments"] == 2
    assert summary["has_benefit_cost"] is True
    assert store.hin_corridors("06037")[0]["segment_id"] == "a"


def test_html_renders_from_stored_report(built):
    store = grant_store.GrantStore(built.dir)
    document = grant_html.render_report(store.get_report("06037"))
    assert document.startswith("<!doctype html>")
    assert "Los Angeles County" in document
    assert "Main St" in document  # the top HIN corridor appears in the deliverable


def test_api_serves_the_county_end_to_end(built, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_GRANT_DIR", str(built.dir))
    client = TestClient(MAIN.api)
    report = built.reports["06037"]

    summary = client.get("/v1/grants/summary?geoid=06037")
    assert summary.status_code == 200
    assert summary.json()["high_injury_network"]["hin_segments"] == 2

    corridors = client.get("/v1/grants/hin?geoid=06037")
    assert [c["segment_id"] for c in corridors.json()["corridors"]] == ["a", "b"]

    geojson = client.get("/v1/grants/hin?geoid=06037&format=geojson")
    assert geojson.json()["count"] == 2
    assert geojson.json()["features"][0]["geometry"]["type"] == "Point"

    html = client.get("/v1/grants/report?geoid=06037&format=html")
    assert html.status_code == 200
    assert html.headers["content-type"].startswith("text/html")
    assert "Los Angeles County" in html.text

    meta = client.get("/v1/meta")
    assert meta.json()["grants"]["jurisdictions"] == 2

    # The API summary agrees with the on-disk report it is serving.
    assert summary.json()["has_benefit_cost"] is True
    assert report["benefit_cost"]["benefit_cost_ratio"] > 0


def test_index_matches_the_reports(built, tmp_path):
    store = grant_store.GrantStore(built.dir)
    index = bgi.build_index(store.iter_reports())
    assert set(index["geoid"]) == {"06037", "06059"}

    la_row = index[index["geoid"] == "06037"].iloc[0]
    report = built.reports["06037"]
    # Every headline number is consistent between the report and the index row.
    assert int(la_row["total_fatal_crashes"]) == report["crash_summary"]["total_fatal_crashes"]
    assert int(la_row["hin_segments"]) == report["high_injury_network"]["hin_segments"]
    assert float(la_row["benefit_cost_ratio"]) == pytest.approx(
        report["benefit_cost"]["benefit_cost_ratio"]
    )

    # And the index survives the Parquet round-trip used for serving.
    out = tmp_path / "index.parquet"
    index.to_parquet(out, index=False)
    assert int(pd.read_parquet(out).query("geoid == '06037'").iloc[0]["hin_segments"]) == 2
