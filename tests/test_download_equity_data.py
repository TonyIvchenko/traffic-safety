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
import download_equity_data as ded


def test_svi_url_and_path_use_year():
    assert common.cdc_svi_url(2022) == "https://svi.cdc.gov/Documents/Data/2022/csv/SVI2022_US.csv"
    assert common.cdc_svi_path(2022).name == "SVI2022_US.csv"
    assert common.cdc_svi_path(2022).parent == common.EQUITY_RAW_DIR


def test_cejst_url_and_path_use_version():
    assert common.cejst_url("2.0") == (
        "https://static-data-screeningtool.geoplatform.gov/data-versions/"
        "2.0/data/score/downloadable/2.0-communities.csv"
    )
    assert common.cejst_path("2.0").name == "cejst_2.0_communities.csv"
    assert common.cejst_path("2.0").parent == common.EQUITY_RAW_DIR


def test_default_year_and_version_thread_through():
    assert common.cdc_svi_url() == common.cdc_svi_url(common.CDC_SVI_YEAR)
    assert common.cejst_url() == common.cejst_url(common.CEJST_VERSION)


def _response(content: bytes, error: Exception | None = None) -> SimpleNamespace:
    def raise_for_status():
        if error is not None:
            raise error

    return SimpleNamespace(content=content, raise_for_status=raise_for_status)


def test_download_file_writes_atomically(monkeypatch, tmp_path):
    output_path = tmp_path / "equity" / "SVI2022_US.csv"
    monkeypatch.setattr(ded.requests, "get", lambda *a, **k: _response(b"svi,csv"))

    ded.download_file("http://example/svi.csv", output_path, force=False)

    assert output_path.read_bytes() == b"svi,csv"
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_leaves_no_partial_on_error(monkeypatch, tmp_path):
    output_path = tmp_path / "cejst.csv"
    monkeypatch.setattr(
        ded.requests, "get", lambda *a, **k: _response(b"x", error=requests.HTTPError("404"))
    )
    with pytest.raises(requests.HTTPError):
        ded.download_file("http://example/cejst.csv", output_path)
    assert not output_path.exists()
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_skips_existing(monkeypatch, tmp_path):
    output_path = tmp_path / "svi.csv"
    output_path.write_bytes(b"cached")
    monkeypatch.setattr(
        ded.requests, "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download")),
    )
    ded.download_file("http://example/svi.csv", output_path)
    assert output_path.read_bytes() == b"cached"
