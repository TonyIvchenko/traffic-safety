from __future__ import annotations

from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import service_startup as ss


def test_display_http_host_maps_bind_all_to_loopback():
    assert ss.display_http_host("") == "127.0.0.1"
    assert ss.display_http_host("0.0.0.0") == "127.0.0.1"
    assert ss.display_http_host("::") == "127.0.0.1"


def test_display_http_host_keeps_explicit_addresses():
    assert ss.display_http_host("192.168.1.5") == "192.168.1.5"
    assert ss.display_http_host("localhost") == "localhost"


def test_format_service_startup_message():
    assert (
        ss.format_service_startup("Traffic Safety", "http://x:8080")
        == "Starting Traffic Safety on http://x:8080"
    )


def test_format_http_service_startup_uses_loopback_for_bind_all():
    assert (
        ss.format_http_service_startup("Traffic Safety", "0.0.0.0", 8080)
        == "Starting Traffic Safety on http://127.0.0.1:8080"
    )
    assert (
        ss.format_http_service_startup("Traffic Safety", "10.0.0.2", 9000)
        == "Starting Traffic Safety on http://10.0.0.2:9000"
    )
