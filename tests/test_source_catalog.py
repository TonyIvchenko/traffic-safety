from __future__ import annotations

import json
from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import datasets
import source_catalog


def test_load_default_catalog():
    catalog = source_catalog.load_source_catalog()
    ids = {s["id"] for s in catalog["sources"]}
    assert {"fars", "tiger_line", "noaa_ncei"} <= ids


def test_get_source_case_insensitive_and_unknown():
    fars = source_catalog.get_source("FARS")
    assert fars["id"] == "fars" and fars["url"].startswith("http")
    assert source_catalog.get_source("nope") is None


def test_resolve_sources_marks_unknown():
    resolved = source_catalog.resolve_sources(["fars", "made_up"])
    assert resolved[0]["id"] == "fars" and "url" in resolved[0]
    assert resolved[1] == {"id": "made_up", "known": False}


def test_dataset_provenance_resolves_records():
    dataset = datasets.get_dataset("high_injury_network")
    enriched = source_catalog.dataset_provenance(dataset)
    assert all(isinstance(s, dict) and "id" in s for s in enriched["sources"])
    # Every real dataset's sources resolve to known records.
    assert all(s.get("known", True) for s in enriched["sources"])
    assert any(s["id"] == "fars" for s in enriched["sources"])


def test_every_dataset_provenance_fully_resolves():
    for dataset in datasets.DATASETS:
        enriched = source_catalog.dataset_provenance(dataset)
        assert all(source.get("known", True) for source in enriched["sources"]), dataset["id"]


def test_degrades_on_missing_and_corrupt(tmp_path, monkeypatch):
    monkeypatch.setenv(source_catalog.SOURCE_CATALOG_PATH_ENV, str(tmp_path / "nope.json"))
    assert source_catalog.load_source_catalog()["sources"] == []
    corrupt = tmp_path / "corrupt.json"
    corrupt.write_text("{not json", encoding="utf-8")
    assert source_catalog.load_source_catalog(corrupt)["sources"] == []


def test_dataset_provenance_bad_input():
    assert source_catalog.dataset_provenance(None) == {}
