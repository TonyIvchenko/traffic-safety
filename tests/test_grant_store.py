from __future__ import annotations

import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import grant_store


def _report(geoid="06037", name="Los Angeles County", corridors=None) -> dict:
    if corridors is None:
        corridors = [
            {"segment_id": "a", "hin_rank": 1, "hin_intensity": 5.0,
             "center_lat": 34.05, "center_lon": -118.24},
            {"segment_id": "b", "hin_rank": 2, "hin_intensity": 3.0,
             "center_lat": 34.10, "center_lon": -118.30},
        ]
    return {
        "jurisdiction": {"geoid": geoid, "name": name, "level": "county"},
        "generated_at_utc": "2026-07-14T00:00:00+00:00",
        "data_vintage": {"fars_years": [2018, 2024]},
        "crash_summary": {"total_fatal_crashes": 42, "total_fatalities": 45},
        "high_injury_network": {"hin_segments": 2, "length_share": 0.4, "weighted_crash_share": 0.9},
        "hin_corridors": corridors,
        "systemic_locations": [{"segment_id": "c"}],
    }


def _write(directory: Path, report: dict) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    (directory / f"{report['jurisdiction']['geoid']}.json").write_text(
        json.dumps(report), encoding="utf-8"
    )


def test_valid_geoid_accepts_fips_rejects_traversal():
    assert grant_store.valid_geoid("48")
    assert grant_store.valid_geoid("06037")
    assert grant_store.valid_geoid("06037920100")
    assert not grant_store.valid_geoid("../secret")
    assert not grant_store.valid_geoid("06037.json")
    assert not grant_store.valid_geoid("")
    assert not grant_store.valid_geoid("060379201001")  # 12 digits, too long


def test_available_geoids_lists_valid_json_only(tmp_path):
    _write(tmp_path, _report("06037"))
    _write(tmp_path, _report("06059"))
    (tmp_path / "notes.txt").write_text("ignore me", encoding="utf-8")
    (tmp_path / "index.json").write_text("{}", encoding="utf-8")  # non-numeric stem ignored
    store = grant_store.GrantStore(tmp_path)
    assert store.available_geoids() == ["06037", "06059"]
    assert store.count() == 2


def test_get_report_reads_json(tmp_path):
    _write(tmp_path, _report("06037"))
    store = grant_store.GrantStore(tmp_path)
    report = store.get_report("06037")
    assert report["jurisdiction"]["name"] == "Los Angeles County"


def test_get_report_missing_returns_none(tmp_path):
    store = grant_store.GrantStore(tmp_path)
    assert store.get_report("99999") is None


def test_get_report_invalid_geoid_returns_none(tmp_path):
    _write(tmp_path, _report("06037"))
    store = grant_store.GrantStore(tmp_path)
    assert store.get_report("../06037") is None


def test_get_report_corrupt_json_returns_none(tmp_path):
    tmp_path.mkdir(parents=True, exist_ok=True)
    (tmp_path / "06037.json").write_text("{not valid json", encoding="utf-8")
    store = grant_store.GrantStore(tmp_path)
    assert store.get_report("06037") is None


def test_get_report_non_object_json_degrades_to_none(tmp_path):
    # Valid JSON that is not an object must not leak through as a non-dict and
    # blow up summarize()'s .get() calls (would otherwise surface as HTTP 500).
    tmp_path.mkdir(parents=True, exist_ok=True)
    for payload in ("[1, 2, 3]", '"pending"', "42", "null"):
        (tmp_path / "06037.json").write_text(payload, encoding="utf-8")
        store = grant_store.GrantStore(tmp_path)
        assert store.get_report("06037") is None, payload
        assert store.summary("06037") is None, payload


def test_summary_compacts_report(tmp_path):
    _write(tmp_path, _report("06037"))
    store = grant_store.GrantStore(tmp_path)
    summary = store.summary("06037")
    assert summary["jurisdiction"]["geoid"] == "06037"
    assert summary["hin_corridor_count"] == 2
    assert summary["systemic_location_count"] == 1
    assert summary["has_benefit_cost"] is False
    assert "hin_corridors" not in summary  # tables dropped from the summary
    assert summary["high_injury_network"]["hin_segments"] == 2


def test_summary_missing_returns_none(tmp_path):
    store = grant_store.GrantStore(tmp_path)
    assert store.summary("06037") is None


def test_missing_directory_is_empty(tmp_path):
    store = grant_store.GrantStore(tmp_path / "does-not-exist")
    assert store.available_geoids() == []
    assert store.count() == 0
    assert store.get_report("06037") is None


def test_get_default_store_honors_env(tmp_path, monkeypatch):
    monkeypatch.setenv(grant_store.GRANT_DIR_ENV, str(tmp_path))
    _write(tmp_path, _report("06037"))
    store = grant_store.get_default_store()
    assert store.directory == tmp_path
    assert store.get_report("06037") is not None


def test_corridor_in_bbox():
    bbox = (34.0, 34.2, -118.4, -118.2)
    assert grant_store.corridor_in_bbox({"center_lat": 34.05, "center_lon": -118.24}, bbox)
    assert not grant_store.corridor_in_bbox({"center_lat": 40.0, "center_lon": -118.24}, bbox)
    # Missing coordinates never match (no crash).
    assert not grant_store.corridor_in_bbox({"segment_id": "x"}, bbox)


def test_hin_corridors_returns_report_order(tmp_path):
    _write(tmp_path, _report("06037"))
    store = grant_store.GrantStore(tmp_path)
    corridors = store.hin_corridors("06037")
    assert [c["segment_id"] for c in corridors] == ["a", "b"]


def test_hin_corridors_missing_returns_none(tmp_path):
    store = grant_store.GrantStore(tmp_path)
    assert store.hin_corridors("99999") is None


def test_hin_corridors_in_bbox_filters_and_ranks_by_intensity(tmp_path):
    _write(tmp_path, _report("06037"))  # a@(34.05,-118.24) i=5, b@(34.10,-118.30) i=3
    _write(tmp_path, _report(
        "06059", "Orange County",
        corridors=[{"segment_id": "d", "hin_rank": 1, "hin_intensity": 9.0,
                    "center_lat": 33.70, "center_lon": -117.80}],
    ))
    store = grant_store.GrantStore(tmp_path)
    # Bbox over LA only -> a, b; ranked by intensity desc; tagged with geoid.
    la = store.hin_corridors_in_bbox((34.0, 34.2, -118.4, -118.2))
    assert [c["segment_id"] for c in la] == ["a", "b"]
    assert all(c["geoid"] == "06037" for c in la)
    # Wider bbox includes Orange County's d (intensity 9) -> ranked first.
    both = store.hin_corridors_in_bbox((33.0, 35.0, -119.0, -117.0))
    assert [c["segment_id"] for c in both] == ["d", "a", "b"]


def test_hin_corridors_in_bbox_respects_top_n(tmp_path):
    _write(tmp_path, _report("06037"))
    store = grant_store.GrantStore(tmp_path)
    assert len(store.hin_corridors_in_bbox((34.0, 34.2, -118.4, -118.2), top_n=1)) == 1


def test_corridor_in_bbox_rejects_malformed():
    bbox = (34.0, 34.2, -118.4, -118.2)
    assert not grant_store.corridor_in_bbox("oops", bbox)  # non-dict
    assert not grant_store.corridor_in_bbox({"center_lat": "north", "center_lon": -118.3}, bbox)
    assert not grant_store.corridor_in_bbox({"center_lat": [1], "center_lon": -118.3}, bbox)


def test_hin_corridors_in_bbox_skips_malformed_corridors(tmp_path):
    # A bad corridor in ANY county file must not 500 the whole bbox scan.
    _write(tmp_path, _report(
        "06037",
        corridors=[
            {"segment_id": "a", "hin_intensity": 5.0, "center_lat": 34.05, "center_lon": -118.24},
            "oops",  # non-dict entry
            {"segment_id": "x", "hin_intensity": 9.0, "center_lat": "north", "center_lon": -118.3},
            {"segment_id": "y", "hin_intensity": "high", "center_lat": 34.11, "center_lon": -118.29},
        ],
    ))
    store = grant_store.GrantStore(tmp_path)
    result = store.hin_corridors_in_bbox((34.0, 34.2, -118.4, -118.2))
    # 'a' (5.0) and 'y' (non-numeric intensity -> 0.0) kept; 'oops' and 'x' skipped.
    assert [c["segment_id"] for c in result] == ["a", "y"]
