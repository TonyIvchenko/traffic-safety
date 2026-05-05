#!/usr/bin/env python3
from __future__ import annotations


def display_http_host(host: str) -> str:
    return "127.0.0.1" if host in {"", "0.0.0.0"} else host


def format_service_startup(service_name: str, url: str) -> str:
    return f"Starting {service_name} on {url}"


def format_http_service_startup(service_name: str, host: str, port: int) -> str:
    return format_service_startup(
        service_name, f"http://{display_http_host(host)}:{port}"
    )


def print_http_service_startup(service_name: str, host: str, port: int) -> None:
    print(format_http_service_startup(service_name, host, port), flush=True)
