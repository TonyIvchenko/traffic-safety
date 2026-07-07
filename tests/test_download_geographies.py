from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

import pytest
import requests

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import common
import download_geographies as dg


def test_county_url_and_path():
    assert common.tiger_county_url(2024) == (
        "https://www2.census.gov/geo/tiger/TIGER2024/COUNTY/tl_2024_us_county.zip"
    )
    assert common.tiger_county_path(2024).name == "tl_2024_us_county.zip"
    assert common.tiger_county_path(2024).parent == common.CENSUS_RAW_DIR


def test_tract_url_and_path_are_per_state():
    assert common.tiger_tract_url("06", 2024) == (
        "https://www2.census.gov/geo/tiger/TIGER2024/TRACT/tl_2024_06_tract.zip"
    )
    assert common.tiger_tract_path("36", 2024).name == "tl_2024_36_tract.zip"


def _response(content: bytes, error: Exception | None = None) -> SimpleNamespace:
    def raise_for_status():
        if error is not None:
            raise error

    return SimpleNamespace(content=content, raise_for_status=raise_for_status)


def test_download_file_writes_atomically(monkeypatch, tmp_path):
    output_path = tmp_path / "census" / "tl_2024_us_county.zip"
    monkeypatch.setattr(dg.requests, "get", lambda *a, **k: _response(b"shp-bytes"))

    dg.download_file("http://example/county.zip", output_path, force=False)

    assert output_path.read_bytes() == b"shp-bytes"
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_leaves_no_partial_on_error(monkeypatch, tmp_path):
    output_path = tmp_path / "tract.zip"
    monkeypatch.setattr(
        dg.requests, "get", lambda *a, **k: _response(b"x", error=requests.HTTPError("404"))
    )
    with pytest.raises(requests.HTTPError):
        dg.download_file("http://example/tract.zip", output_path)
    assert not output_path.exists()
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_skips_existing(monkeypatch, tmp_path):
    output_path = tmp_path / "county.zip"
    output_path.write_bytes(b"cached")
    monkeypatch.setattr(
        dg.requests, "get", lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download"))
    )
    dg.download_file("http://example/county.zip", output_path)
    assert output_path.read_bytes() == b"cached"
