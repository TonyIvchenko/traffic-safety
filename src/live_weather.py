from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from functools import lru_cache
import os
from typing import Any
from zoneinfo import ZoneInfo

import requests

from model_support import (
    fahrenheit_to_celsius,
    parse_wind_speed_string_mps,
    relative_humidity_from_temp_dewpoint,
)


NWS_BASE_URL = "https://api.weather.gov"
TOMORROW_BASE_URL = "https://api.tomorrow.io/v4/weather"
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/3.0/onecall"
DEFAULT_USER_AGENT = os.getenv(
    "TRAFFIC_SAFETY_USER_AGENT",
    "playground-traffic-safety/0.2",
)
WET_KEYWORDS = (
    "rain",
    "shower",
    "storm",
    "snow",
    "sleet",
    "hail",
    "drizzle",
    "ice",
    "freezing",
    "thunder",
)


class LiveWeatherProviderError(RuntimeError):
    pass


@dataclass(frozen=True)
class LiveWeatherSnapshot:
    provider: str
    provider_label: str
    observed_or_forecast: str
    timestamp_local: datetime
    forecast_hours: int
    temp_c: float
    dewpoint_c: float
    relative_humidity_pct: float
    wind_speed_mps: float
    wet_hour: float
    summary: str


@dataclass(frozen=True)
class ProviderStatus:
    name: str
    label: str
    paid: bool
    enabled: bool
    configured: bool
    available: bool


def _env_flag(name: str, default: bool) -> bool:
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() not in {"0", "false", "no", "off", ""}


def _parse_datetime(value: str | None) -> datetime:
    if not value:
        raise LiveWeatherProviderError("missing timestamp from weather provider")
    parsed = datetime.fromisoformat(value.replace("Z", "+00:00"))
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed


def _wet_hour_from_summary(summary: str | None, precipitation_probability: float | None) -> float:
    summary_text = (summary or "").lower()
    if precipitation_probability is not None and precipitation_probability >= 30.0:
        return 1.0
    return 1.0 if any(keyword in summary_text for keyword in WET_KEYWORDS) else 0.0


def _quantitative_value(payload: Any) -> float | None:
    if payload is None:
        return None
    if isinstance(payload, (int, float)):
        return float(payload)
    if isinstance(payload, dict):
        value = payload.get("value")
        if value is None:
            return None
        return float(value)
    return None


def _coerce_relative_humidity(
    humidity_pct: float | None,
    temp_c: float | None,
    dewpoint_c: float | None,
) -> float:
    if humidity_pct is not None:
        return max(0.0, min(100.0, float(humidity_pct)))
    if temp_c is None or dewpoint_c is None:
        return 0.0
    return float(relative_humidity_from_temp_dewpoint(temp_c, dewpoint_c).item())


def _coerce_wind_speed_mps(value: float | None) -> float:
    if value is None:
        return 0.0
    return max(0.0, float(value))


def _provider_priority() -> list[str]:
    raw = os.getenv("TRAFFIC_SAFETY_LIVE_PROVIDERS", "nws,openweather,tomorrow")
    ordered = [part.strip().lower() for part in raw.split(",") if part.strip()]
    deduped = []
    for name in ordered:
        if name not in deduped:
            deduped.append(name)
    return deduped or ["nws", "openweather", "tomorrow"]


def _request_json(
    url: str,
    *,
    params: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> dict[str, Any]:
    request_headers = {
        "User-Agent": DEFAULT_USER_AGENT,
        "Accept": "application/geo+json, application/json",
    }
    if headers:
        request_headers.update(headers)
    response = requests.get(url, params=params, headers=request_headers, timeout=20)
    response.raise_for_status()
    return response.json()


class BaseLiveWeatherAdapter:
    name = "base"
    label = "Base"
    paid = False
    requires_key = False
    flag_name = ""
    key_name = ""
    default_enabled = True

    def enabled(self) -> bool:
        if not self.flag_name:
            return True
        return _env_flag(self.flag_name, self.default_enabled)

    def configured(self) -> bool:
        if not self.requires_key:
            return True
        return bool(os.getenv(self.key_name, "").strip())

    def available(self) -> bool:
        return self.enabled() and self.configured()

    def fetch(self, lat: float, lon: float, forecast_hours: int) -> LiveWeatherSnapshot:
        raise NotImplementedError


class NWSWeatherAdapter(BaseLiveWeatherAdapter):
    name = "nws"
    label = "NWS"
    paid = False
    requires_key = False
    flag_name = "TRAFFIC_SAFETY_ENABLE_NWS"

    def _localize(self, timestamp: datetime, time_zone_name: str | None) -> datetime:
        if not time_zone_name:
            return timestamp
        try:
            return timestamp.astimezone(ZoneInfo(time_zone_name))
        except Exception:
            return timestamp

    def _forecast_snapshot(
        self,
        forecast_hourly_url: str,
        forecast_hours: int,
        time_zone_name: str | None,
    ) -> LiveWeatherSnapshot:
        payload = _request_json(forecast_hourly_url)
        periods = payload.get("properties", {}).get("periods") or []
        if not periods:
            raise LiveWeatherProviderError("NWS forecast returned no hourly periods")

        idx = min(max(0, int(forecast_hours)), len(periods) - 1)
        period = periods[idx]
        temperature = period.get("temperature")
        temp_unit = str(period.get("temperatureUnit", "")).upper()
        temp_c = (
            fahrenheit_to_celsius(temperature)
            if temp_unit == "F"
            else float(temperature or 0.0)
        )
        dewpoint_c = _quantitative_value(period.get("dewpoint"))
        humidity_pct = _quantitative_value(period.get("relativeHumidity"))
        wind_speed_mps = parse_wind_speed_string_mps(period.get("windSpeed"))
        precipitation_probability = _quantitative_value(
            period.get("probabilityOfPrecipitation")
        )
        summary = " ".join(
            part.strip()
            for part in [
                str(period.get("shortForecast", "")).strip(),
                str(period.get("detailedForecast", "")).strip(),
            ]
            if part and part.strip()
        )
        return LiveWeatherSnapshot(
            provider=self.name,
            provider_label=self.label,
            observed_or_forecast="forecast",
            timestamp_local=self._localize(
                _parse_datetime(period.get("startTime")),
                time_zone_name,
            ),
            forecast_hours=int(forecast_hours),
            temp_c=float(temp_c or 0.0),
            dewpoint_c=float(dewpoint_c or 0.0),
            relative_humidity_pct=_coerce_relative_humidity(
                humidity_pct,
                temp_c,
                dewpoint_c,
            ),
            wind_speed_mps=_coerce_wind_speed_mps(wind_speed_mps),
            wet_hour=_wet_hour_from_summary(summary, precipitation_probability),
            summary=summary,
        )

    def _observation_snapshot(
        self,
        observation_stations_url: str,
        time_zone_name: str | None,
    ) -> LiveWeatherSnapshot | None:
        stations_payload = _request_json(observation_stations_url)
        features = stations_payload.get("features") or []
        for station in features[:5]:
            properties = station.get("properties") or {}
            station_id = properties.get("stationIdentifier")
            if not station_id:
                station_id = str(station.get("id", "")).rstrip("/").split("/")[-1]
            if not station_id:
                continue
            try:
                latest = _request_json(
                    f"{NWS_BASE_URL}/stations/{station_id}/observations/latest"
                )
            except requests.HTTPError:
                continue
            props = latest.get("properties") or {}
            temp_c = _quantitative_value(props.get("temperature"))
            dewpoint_c = _quantitative_value(props.get("dewpoint"))
            humidity_pct = _quantitative_value(props.get("relativeHumidity"))
            wind_speed_mps = _quantitative_value(props.get("windSpeed"))
            precipitation_last_hour = _quantitative_value(props.get("precipitationLastHour"))
            summary = str(props.get("textDescription", "")).strip()
            if (
                temp_c is None
                and dewpoint_c is None
                and humidity_pct is None
                and wind_speed_mps is None
                and not summary
            ):
                continue
            return LiveWeatherSnapshot(
                provider=self.name,
                provider_label=self.label,
                observed_or_forecast="observation",
                timestamp_local=self._localize(
                    _parse_datetime(props.get("timestamp")),
                    time_zone_name,
                ),
                forecast_hours=0,
                temp_c=float(temp_c or 0.0),
                dewpoint_c=float(dewpoint_c or 0.0),
                relative_humidity_pct=_coerce_relative_humidity(
                    humidity_pct,
                    temp_c,
                    dewpoint_c,
                ),
                wind_speed_mps=_coerce_wind_speed_mps(wind_speed_mps),
                wet_hour=1.0
                if (precipitation_last_hour or 0.0) > 0.0
                else _wet_hour_from_summary(summary, None),
                summary=summary,
            )
        return None

    def fetch(self, lat: float, lon: float, forecast_hours: int) -> LiveWeatherSnapshot:
        points_payload = _request_json(f"{NWS_BASE_URL}/points/{lat},{lon}")
        properties = points_payload.get("properties") or {}
        forecast_hourly_url = properties.get("forecastHourly")
        observation_stations_url = properties.get("observationStations")
        time_zone_name = properties.get("timeZone")
        if not forecast_hourly_url:
            raise LiveWeatherProviderError("NWS points lookup returned no forecast URL")

        if int(forecast_hours) <= 0 and observation_stations_url:
            observation = self._observation_snapshot(
                observation_stations_url,
                time_zone_name,
            )
            if observation is not None:
                return observation
        return self._forecast_snapshot(
            forecast_hourly_url,
            int(forecast_hours),
            time_zone_name,
        )


class TomorrowIoWeatherAdapter(BaseLiveWeatherAdapter):
    name = "tomorrow"
    label = "Tomorrow.io"
    paid = True
    requires_key = True
    flag_name = "TRAFFIC_SAFETY_ENABLE_TOMORROW_IO"
    key_name = "TOMORROW_IO_API_KEY"
    default_enabled = False

    def _headers(self) -> dict[str, str]:
        return {"apikey": os.getenv(self.key_name, "").strip()}

    def _snapshot_from_values(
        self,
        *,
        timestamp_local: datetime,
        values: dict[str, Any],
        forecast_hours: int,
        observed_or_forecast: str,
    ) -> LiveWeatherSnapshot:
        temp_c = values.get("temperature")
        dewpoint_c = values.get("dewPoint")
        humidity_pct = values.get("humidity")
        wind_speed_mps = values.get("windSpeed")
        precipitation_probability = values.get("precipitationProbability")
        weather_code = values.get("weatherCode")
        summary = f"weatherCode={weather_code}" if weather_code is not None else ""
        return LiveWeatherSnapshot(
            provider=self.name,
            provider_label=self.label,
            observed_or_forecast=observed_or_forecast,
            timestamp_local=timestamp_local,
            forecast_hours=int(forecast_hours),
            temp_c=float(temp_c or 0.0),
            dewpoint_c=float(dewpoint_c or 0.0),
            relative_humidity_pct=_coerce_relative_humidity(
                float(humidity_pct) if humidity_pct is not None else None,
                float(temp_c) if temp_c is not None else None,
                float(dewpoint_c) if dewpoint_c is not None else None,
            ),
            wind_speed_mps=_coerce_wind_speed_mps(
                float(wind_speed_mps) if wind_speed_mps is not None else None
            ),
            wet_hour=_wet_hour_from_summary(
                summary,
                float(precipitation_probability)
                if precipitation_probability is not None
                else None,
            ),
            summary=summary,
        )

    def fetch(self, lat: float, lon: float, forecast_hours: int) -> LiveWeatherSnapshot:
        location = f"{lat},{lon}"
        if int(forecast_hours) <= 0:
            payload = _request_json(
                f"{TOMORROW_BASE_URL}/realtime",
                params={"location": location},
                headers=self._headers(),
            )
            data = payload.get("data") or {}
            return self._snapshot_from_values(
                timestamp_local=_parse_datetime(data.get("time")),
                values=data.get("values") or {},
                forecast_hours=0,
                observed_or_forecast="observation",
            )

        payload = _request_json(
            f"{TOMORROW_BASE_URL}/forecast",
            params={"location": location, "timesteps": "1h", "units": "metric"},
            headers=self._headers(),
        )
        hourly = payload.get("timelines", {}).get("hourly") or []
        if not hourly:
            raise LiveWeatherProviderError("Tomorrow.io forecast returned no hourly entries")
        idx = min(max(0, int(forecast_hours)), len(hourly) - 1)
        point = hourly[idx]
        return self._snapshot_from_values(
            timestamp_local=_parse_datetime(point.get("time")),
            values=point.get("values") or {},
            forecast_hours=int(forecast_hours),
            observed_or_forecast="forecast",
        )


class OpenWeatherWeatherAdapter(BaseLiveWeatherAdapter):
    name = "openweather"
    label = "OpenWeather"
    paid = True
    requires_key = True
    flag_name = "TRAFFIC_SAFETY_ENABLE_OPENWEATHER"
    key_name = "OPENWEATHER_API_KEY"
    default_enabled = False

    def fetch(self, lat: float, lon: float, forecast_hours: int) -> LiveWeatherSnapshot:
        payload = _request_json(
            OPENWEATHER_BASE_URL,
            params={
                "lat": lat,
                "lon": lon,
                "appid": os.getenv(self.key_name, "").strip(),
                "units": "metric",
            },
        )
        timezone_offset = int(payload.get("timezone_offset", 0))
        local_tz = timezone(timedelta(seconds=timezone_offset))
        if int(forecast_hours) <= 0:
            current = payload.get("current") or {}
            weather_summary = ""
            if current.get("weather"):
                weather_summary = str(current["weather"][0].get("description", "")).strip()
            precipitation_flag = 1.0 if any(key in current for key in ("rain", "snow")) else 0.0
            return LiveWeatherSnapshot(
                provider=self.name,
                provider_label=self.label,
                observed_or_forecast="observation",
                timestamp_local=datetime.fromtimestamp(
                    int(current.get("dt", 0)),
                    tz=timezone.utc,
                ).astimezone(local_tz),
                forecast_hours=0,
                temp_c=float(current.get("temp", 0.0) or 0.0),
                dewpoint_c=float(current.get("dew_point", 0.0) or 0.0),
                relative_humidity_pct=_coerce_relative_humidity(
                    float(current.get("humidity")) if current.get("humidity") is not None else None,
                    float(current.get("temp")) if current.get("temp") is not None else None,
                    float(current.get("dew_point")) if current.get("dew_point") is not None else None,
                ),
                wind_speed_mps=_coerce_wind_speed_mps(
                    float(current.get("wind_speed"))
                    if current.get("wind_speed") is not None
                    else None
                ),
                wet_hour=precipitation_flag or _wet_hour_from_summary(weather_summary, None),
                summary=weather_summary,
            )

        hourly = payload.get("hourly") or []
        if not hourly:
            raise LiveWeatherProviderError("OpenWeather returned no hourly forecast entries")
        idx = min(max(0, int(forecast_hours)), len(hourly) - 1)
        point = hourly[idx]
        weather_summary = ""
        if point.get("weather"):
            weather_summary = str(point["weather"][0].get("description", "")).strip()
        return LiveWeatherSnapshot(
            provider=self.name,
            provider_label=self.label,
            observed_or_forecast="forecast",
            timestamp_local=datetime.fromtimestamp(
                int(point.get("dt", 0)),
                tz=timezone.utc,
            ).astimezone(local_tz),
            forecast_hours=int(forecast_hours),
            temp_c=float(point.get("temp", 0.0) or 0.0),
            dewpoint_c=float(point.get("dew_point", 0.0) or 0.0),
            relative_humidity_pct=_coerce_relative_humidity(
                float(point.get("humidity")) if point.get("humidity") is not None else None,
                float(point.get("temp")) if point.get("temp") is not None else None,
                float(point.get("dew_point")) if point.get("dew_point") is not None else None,
            ),
            wind_speed_mps=_coerce_wind_speed_mps(
                float(point.get("wind_speed")) if point.get("wind_speed") is not None else None
            ),
            wet_hour=_wet_hour_from_summary(
                weather_summary,
                float(point.get("pop", 0.0)) * 100.0 if point.get("pop") is not None else None,
            ),
            summary=weather_summary,
        )


@lru_cache(maxsize=1)
def provider_registry() -> dict[str, BaseLiveWeatherAdapter]:
    # Adapters are stateless and read environment flags/keys on each call, so a
    # single shared instance per provider is safe and avoids rebuilding them on
    # every request.
    return {
        "nws": NWSWeatherAdapter(),
        "tomorrow": TomorrowIoWeatherAdapter(),
        "openweather": OpenWeatherWeatherAdapter(),
    }


def provider_statuses() -> list[ProviderStatus]:
    registry = provider_registry()
    statuses = []
    for name in _provider_priority():
        adapter = registry.get(name)
        if adapter is None:
            continue
        statuses.append(
            ProviderStatus(
                name=adapter.name,
                label=adapter.label,
                paid=adapter.paid,
                enabled=adapter.enabled(),
                configured=adapter.configured(),
                available=adapter.available(),
            )
        )
    return statuses


def resolve_live_weather_adapter(provider: str = "auto") -> BaseLiveWeatherAdapter:
    registry = provider_registry()
    requested = provider.strip().lower()
    if requested == "auto":
        for name in _provider_priority():
            adapter = registry.get(name)
            if adapter is not None and adapter.available():
                return adapter
        raise LiveWeatherProviderError("no live weather providers are available")

    adapter = registry.get(requested)
    if adapter is None:
        raise LiveWeatherProviderError(f"unknown live weather provider '{provider}'")
    if not adapter.enabled():
        raise LiveWeatherProviderError(f"provider '{provider}' is disabled by feature flag")
    if not adapter.configured():
        raise LiveWeatherProviderError(f"provider '{provider}' is not configured")
    return adapter


def fetch_live_weather(lat: float, lon: float, forecast_hours: int, provider: str = "auto") -> LiveWeatherSnapshot:
    adapter = resolve_live_weather_adapter(provider)
    return adapter.fetch(lat=float(lat), lon=float(lon), forecast_hours=int(forecast_hours))
