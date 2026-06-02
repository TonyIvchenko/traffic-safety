from __future__ import annotations

from pathlib import Path
import sys

import pytest
import requests

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import download_data


class _FakeResponse:
    def __init__(self, chunks: list[bytes], error: Exception | None = None):
        self._chunks = chunks
        self._error = error

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False

    def raise_for_status(self):
        if self._error is not None:
            raise self._error

    def iter_content(self, chunk_size: int):
        yield from self._chunks


def test_download_file_writes_atomically(monkeypatch, tmp_path):
    destination = tmp_path / "nested" / "FARS.zip"
    monkeypatch.setattr(
        download_data.requests,
        "get",
        lambda *a, **k: _FakeResponse([b"abc", b"def"]),
    )

    download_data.download_file("http://example/file", destination)

    assert destination.read_bytes() == b"abcdef"
    # The temporary part file must not linger after a successful download.
    assert not (destination.parent / (destination.name + ".part")).exists()


def test_download_file_leaves_no_partial_file_on_error(monkeypatch, tmp_path):
    destination = tmp_path / "FARS.zip"

    def boom(*a, **k):
        return _FakeResponse([b"abc"], error=requests.HTTPError("500"))

    monkeypatch.setattr(download_data.requests, "get", boom)

    with pytest.raises(requests.HTTPError):
        download_data.download_file("http://example/file", destination)

    assert not destination.exists()
    assert not (destination.parent / (destination.name + ".part")).exists()


def test_download_file_skips_existing(monkeypatch, tmp_path):
    destination = tmp_path / "FARS.zip"
    destination.write_bytes(b"cached")

    def fail(*a, **k):
        raise AssertionError("should not download when cache exists")

    monkeypatch.setattr(download_data.requests, "get", fail)

    download_data.download_file("http://example/file", destination)

    assert destination.read_bytes() == b"cached"
