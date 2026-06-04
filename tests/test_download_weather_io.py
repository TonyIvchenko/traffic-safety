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

import download_weather


def _response(content: bytes, error: Exception | None = None) -> SimpleNamespace:
    def raise_for_status():
        if error is not None:
            raise error

    return SimpleNamespace(content=content, raise_for_status=raise_for_status)


def test_download_file_writes_atomically(monkeypatch, tmp_path):
    path = tmp_path / "isd" / "723815-99999-2020.gz"
    monkeypatch.setattr(
        download_weather.requests, "get", lambda *a, **k: _response(b"gzip-bytes")
    )

    download_weather.download_file("http://example/isd.gz", path)

    assert path.read_bytes() == b"gzip-bytes"
    assert not (path.parent / (path.name + ".part")).exists()


def test_download_file_leaves_no_partial_on_error(monkeypatch, tmp_path):
    path = tmp_path / "isd.gz"
    monkeypatch.setattr(
        download_weather.requests,
        "get",
        lambda *a, **k: _response(b"x", error=requests.ConnectionError("dns")),
    )

    with pytest.raises(requests.ConnectionError):
        download_weather.download_file("http://example/isd.gz", path)

    assert not path.exists()
    assert not (path.parent / (path.name + ".part")).exists()
