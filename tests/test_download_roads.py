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

import download_roads


def _response(content: bytes, error: Exception | None = None) -> SimpleNamespace:
    def raise_for_status():
        if error is not None:
            raise error

    return SimpleNamespace(content=content, raise_for_status=raise_for_status)


def test_download_file_writes_atomically(monkeypatch, tmp_path):
    output_path = tmp_path / "roads" / "tl_2024_06_prisecroads.zip"
    monkeypatch.setattr(
        download_roads.requests, "get", lambda *a, **k: _response(b"shapefile-bytes")
    )

    download_roads.download_file("http://example/roads.zip", output_path, force=False)

    assert output_path.read_bytes() == b"shapefile-bytes"
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_leaves_no_partial_on_http_error(monkeypatch, tmp_path):
    output_path = tmp_path / "roads.zip"
    monkeypatch.setattr(
        download_roads.requests,
        "get",
        lambda *a, **k: _response(b"x", error=requests.HTTPError("404")),
    )

    with pytest.raises(requests.HTTPError):
        download_roads.download_file("http://example/roads.zip", output_path, force=False)

    assert not output_path.exists()
    assert not (output_path.parent / (output_path.name + ".part")).exists()


def test_download_file_skips_existing(monkeypatch, tmp_path):
    output_path = tmp_path / "roads.zip"
    output_path.write_bytes(b"cached")
    monkeypatch.setattr(
        download_roads.requests,
        "get",
        lambda *a, **k: (_ for _ in ()).throw(AssertionError("should not download")),
    )

    download_roads.download_file("http://example/roads.zip", output_path, force=False)

    assert output_path.read_bytes() == b"cached"
