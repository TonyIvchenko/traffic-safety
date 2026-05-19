from __future__ import annotations

from datetime import timezone
from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import live_weather as lw


def test_env_flag_parses_truthy_and_falsy_values(monkeypatch):
    monkeypatch.delenv("SOME_FLAG", raising=False)
    assert lw._env_flag("SOME_FLAG", default=True) is True
    assert lw._env_flag("SOME_FLAG", default=False) is False
    for falsy in ("0", "false", "no", "off", ""):
        monkeypatch.setenv("SOME_FLAG", falsy)
        assert lw._env_flag("SOME_FLAG", default=True) is False
    for truthy in ("1", "true", "yes", "on"):
        monkeypatch.setenv("SOME_FLAG", truthy)
        assert lw._env_flag("SOME_FLAG", default=False) is True


def test_parse_datetime_normalizes_to_timezone_aware():
    zulu = lw._parse_datetime("2024-09-06T17:00:00Z")
    assert zulu.tzinfo is not None
    assert zulu.utcoffset() == timezone.utc.utcoffset(zulu)

    naive = lw._parse_datetime("2024-09-06T17:00:00")
    assert naive.tzinfo == timezone.utc

    with pytest.raises(lw.LiveWeatherProviderError):
        lw._parse_datetime(None)


def test_wet_hour_from_summary_uses_keywords_and_probability():
    assert lw._wet_hour_from_summary("Heavy rain likely", None) == 1.0
    assert lw._wet_hour_from_summary("Clear skies", None) == 0.0
    assert lw._wet_hour_from_summary("Clear", 45.0) == 1.0
    assert lw._wet_hour_from_summary("Clear", 10.0) == 0.0
    assert lw._wet_hour_from_summary(None, None) == 0.0


def test_quantitative_value_handles_scalars_dicts_and_none():
    assert lw._quantitative_value(None) is None
    assert lw._quantitative_value(12) == 12.0
    assert lw._quantitative_value({"value": 3.5}) == 3.5
    assert lw._quantitative_value({"value": None}) is None
    assert lw._quantitative_value("nope") is None


def test_coerce_relative_humidity_prefers_explicit_then_computes():
    assert lw._coerce_relative_humidity(150.0, None, None) == 100.0
    assert lw._coerce_relative_humidity(-5.0, None, None) == 0.0
    assert lw._coerce_relative_humidity(None, None, None) == 0.0
    computed = lw._coerce_relative_humidity(None, 20.0, 20.0)
    assert computed == pytest.approx(100.0, abs=1e-2)


def test_coerce_wind_speed_clamps_and_defaults():
    assert lw._coerce_wind_speed_mps(None) == 0.0
    assert lw._coerce_wind_speed_mps(-3.0) == 0.0
    assert lw._coerce_wind_speed_mps(7.5) == 7.5


def test_provider_priority_dedupes_and_falls_back(monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_LIVE_PROVIDERS", "tomorrow, nws , nws, bogus")
    assert lw._provider_priority() == ["tomorrow", "nws", "bogus"]

    monkeypatch.setenv("TRAFFIC_SAFETY_LIVE_PROVIDERS", "   ")
    assert lw._provider_priority() == ["nws", "openweather", "tomorrow"]


def test_resolve_adapter_auto_prefers_first_available_provider(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_ENABLE_NWS", raising=False)
    monkeypatch.delenv("TRAFFIC_SAFETY_LIVE_PROVIDERS", raising=False)
    adapter = lw.resolve_live_weather_adapter("auto")
    assert adapter.name == "nws"


def test_resolve_adapter_rejects_unknown_provider():
    with pytest.raises(lw.LiveWeatherProviderError, match="unknown"):
        lw.resolve_live_weather_adapter("does-not-exist")


def test_resolve_adapter_rejects_disabled_provider(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_ENABLE_TOMORROW_IO", raising=False)
    with pytest.raises(lw.LiveWeatherProviderError, match="disabled"):
        lw.resolve_live_weather_adapter("tomorrow")


def test_resolve_adapter_rejects_unconfigured_paid_provider(monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_ENABLE_OPENWEATHER", "1")
    monkeypatch.delenv("OPENWEATHER_API_KEY", raising=False)
    with pytest.raises(lw.LiveWeatherProviderError, match="not configured"):
        lw.resolve_live_weather_adapter("openweather")


def test_provider_statuses_reports_free_nws_as_available(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_ENABLE_NWS", raising=False)
    statuses = {status.name: status for status in lw.provider_statuses()}
    assert "nws" in statuses
    assert statuses["nws"].paid is False
    assert statuses["nws"].available is True
