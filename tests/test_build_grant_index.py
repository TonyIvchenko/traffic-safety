from __future__ import annotations

import json
from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_grant_index as bgi
import grant_store


def _report(geoid="06037", name="Los Angeles County", *, benefit_cost=True) -> dict:
    report = {
        "jurisdiction": {"geoid": geoid, "name": name, "level": "county"},
        "generated_at_utc": "2026-07-14T00:00:00+00:00",
        "data_vintage": {"fars_years": [2018, 2024]},
        "crash_summary": {"total_fatal_crashes": 42, "total_fatalities": 45},
        "high_injury_network": {
            "hin_segments": 2, "hin_length_km": 2.0, "length_share": 0.4,
            "weighted_crash_share": 0.9,
        },
        "hin_corridors": [{"segment_id": "a"}, {"segment_id": "b"}],
        "systemic_locations": [{"segment_id": "c"}],
    }
    if benefit_cost:
        report["benefit_cost"] = {"benefit_cost_ratio": 2.5, "net_benefit": 900000.0}
    return report


def test_index_row_flattens_report():
    row = bgi.index_row("06037", _report())
    assert row["geoid"] == "06037"
    assert row["name"] == "Los Angeles County"
    assert row["level"] == "county"
    assert row["fars_year_start"] == 2018
    assert row["fars_year_end"] == 2024
    assert row["total_fatal_crashes"] == 42
    assert row["hin_segments"] == 2
    assert row["hin_corridor_count"] == 2
    assert row["systemic_location_count"] == 1
    assert row["benefit_cost_ratio"] == 2.5
    assert row["net_benefit"] == 900000.0


def test_index_row_without_benefit_cost():
    row = bgi.index_row("06037", _report(benefit_cost=False))
    assert row["benefit_cost_ratio"] is None
    assert row["net_benefit"] is None


def test_index_row_tolerates_malformed_report():
    # Wrong nested container types must degrade to None, not raise.
    row = bgi.index_row("06037", {
        "jurisdiction": [1, 2],
        "data_vintage": "nope",
        "crash_summary": [1],
        "high_injury_network": "bad",
        "hin_corridors": {"x": 1},
        "benefit_cost": [1],
    })
    assert row["geoid"] == "06037"  # falls back to the passed geoid
    assert row["name"] is None
    assert row["fars_year_start"] is None
    assert row["total_fatal_crashes"] is None
    assert row["hin_corridor_count"] == 0


def test_build_index_columns_and_sort():
    reports = [("06059", _report("06059", "Orange County")), ("06037", _report("06037"))]
    frame = bgi.build_index(reports)
    assert list(frame.columns) == bgi.INDEX_COLUMNS
    assert list(frame["geoid"]) == ["06037", "06059"]  # sorted by GEOID


def test_build_index_empty_has_columns():
    frame = bgi.build_index([])
    assert list(frame.columns) == bgi.INDEX_COLUMNS
    assert len(frame) == 0


def test_index_row_scalar_guards_bad_leaf():
    # A nested object where a scalar is expected degrades to None (not the dict).
    row = bgi.index_row("06037", {
        "jurisdiction": {"geoid": "06037", "name": {"nested": "obj"}, "level": "county"},
        "crash_summary": {"total_fatal_crashes": {"n": 1}, "total_fatalities": [1, 2]},
    })
    assert row["name"] is None
    assert row["total_fatal_crashes"] is None
    assert row["total_fatalities"] is None


def test_build_index_bad_leaf_does_not_abort_parquet(tmp_path):
    # One county with a malformed leaf must not kill the whole national build.
    reports = [
        ("06037", _report("06037")),
        ("06059", {
            "jurisdiction": {"geoid": "06059", "name": "Orange County", "level": "county"},
            "crash_summary": {"total_fatal_crashes": {"n": 1}},  # bad leaf
        }),
    ]
    frame = bgi.build_index(reports)
    out = tmp_path / "index.parquet"
    frame.to_parquet(out, index=False)  # must not raise
    loaded = pd.read_parquet(out)
    assert set(loaded["geoid"]) == {"06037", "06059"}
    oc = loaded[loaded["geoid"] == "06059"].iloc[0]
    assert pd.isna(oc["total_fatal_crashes"])  # bad leaf coerced to null
    la = loaded[loaded["geoid"] == "06037"].iloc[0]
    assert int(la["total_fatal_crashes"]) == 42  # good county unaffected


def test_build_index_dtype_schema_stable_empty_vs_populated(tmp_path):
    # Empty and populated indexes must share an identical Parquet schema, so a
    # consumer appending/concatenating shards never hits a type mismatch.
    empty = bgi.build_index([])
    populated = bgi.build_index([("06037", _report("06037"))])
    assert list(empty.dtypes) == list(populated.dtypes)

    empty_path, full_path = tmp_path / "empty.parquet", tmp_path / "full.parquet"
    empty.to_parquet(empty_path, index=False)
    populated.to_parquet(full_path, index=False)
    import pyarrow.parquet as pq

    assert pq.read_schema(empty_path).equals(pq.read_schema(full_path))


def test_build_index_all_null_optional_column_is_stable(tmp_path):
    # benefit_cost_ratio is legitimately absent across all counties -> still a
    # float column, not a pyarrow null column.
    frame = bgi.build_index([("06037", _report("06037", benefit_cost=False))])
    assert str(frame["benefit_cost_ratio"].dtype) == "float64"
    frame.to_parquet(tmp_path / "idx.parquet", index=False)  # must not raise


def test_build_index_from_store_round_trips_parquet(tmp_path):
    for geoid, name in (("06037", "Los Angeles County"), ("06059", "Orange County")):
        (tmp_path / f"{geoid}.json").write_text(json.dumps(_report(geoid, name)), encoding="utf-8")
    store = grant_store.GrantStore(tmp_path)
    frame = bgi.build_index(store.iter_reports())
    out = tmp_path / "index.parquet"
    frame.to_parquet(out, index=False)

    loaded = pd.read_parquet(out)
    assert set(loaded["geoid"]) == {"06037", "06059"}
    la = loaded[loaded["geoid"] == "06037"].iloc[0]
    assert la["name"] == "Los Angeles County"
    assert int(la["total_fatal_crashes"]) == 42
