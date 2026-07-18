from __future__ import annotations

import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import grant_store


def _report(geoid="06037", name="Los Angeles County") -> dict:
    return {
        "jurisdiction": {"geoid": geoid, "name": name, "level": "county"},
        "generated_at_utc": "2026-07-14T00:00:00+00:00",
        "data_vintage": {"fars_years": [2018, 2024]},
        "crash_summary": {"total_fatal_crashes": 42, "total_fatalities": 45},
        "high_injury_network": {"hin_segments": 2, "length_share": 0.4, "weighted_crash_share": 0.9},
        "hin_corridors": [{"segment_id": "a"}, {"segment_id": "b"}],
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
