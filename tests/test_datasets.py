from __future__ import annotations

import json
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import datasets

SOURCE_CATALOG = REPO_DIR / "source_catalog.json"


def test_list_and_get():
    catalog = datasets.list_datasets()
    assert len(catalog) >= 6
    assert datasets.get_dataset("high_injury_network")["title"] == "High Injury Network"
    assert datasets.get_dataset("HIGH_INJURY_NETWORK")["id"] == "high_injury_network"  # case-insensitive
    assert datasets.get_dataset("nope") is None


def test_ids_unique_and_records_well_formed():
    ids = datasets.dataset_ids()
    assert len(ids) == len(set(ids))
    valid_formats = set(datasets.FORMAT_MEDIA_TYPES) | {"html"}
    for dataset in datasets.DATASETS:
        assert {"id", "title", "category", "description", "formats", "endpoints", "sources", "license"} <= set(dataset)
        assert dataset["formats"] and all(fmt in valid_formats for fmt in dataset["formats"])
        assert dataset["endpoints"] and dataset["sources"]


def test_supports_format():
    assert datasets.supports_format("high_injury_network", "geojson") is True
    assert datasets.supports_format("high_injury_network", "GEOJSON") is True
    assert datasets.supports_format("high_injury_network", "parquet") is False
    assert datasets.supports_format("nope", "json") is False


def test_catalog_view():
    view = datasets.catalog()
    assert view["count"] == len(datasets.DATASETS)
    assert "geojson" in view["formats"]
    assert len(view["datasets"]) == view["count"]


def test_source_ids_reference_the_source_catalog():
    # Every provenance source id must be a real entry in source_catalog.json.
    payload = json.loads(SOURCE_CATALOG.read_text(encoding="utf-8"))
    known = {source["id"] for source in payload["sources"]}
    for dataset in datasets.DATASETS:
        for source_id in dataset["sources"]:
            assert source_id in known, f"{dataset['id']} references unknown source {source_id}"
