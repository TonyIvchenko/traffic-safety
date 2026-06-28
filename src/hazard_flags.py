"""Derived weather-hazard flags (ice, fog) from the features we already have.

No new data source: everything is computed from temperature, dewpoint, relative
humidity, wind, and the wet-hour flag that the scorers already return.
"""

from __future__ import annotations

ICE_TEMP_C = 1.0          # at/below this (with moisture) ice becomes plausible
ICE_FULL_C = -5.0         # fully saturated ice risk at/below this
ICE_HUMIDITY_MIN = 90.0   # allows black-ice risk without active precipitation

FOG_SPREAD_C = 2.5        # temp - dewpoint gap below which fog is plausible
FOG_HUMIDITY_MIN = 90.0
FOG_WIND_MAX_MPS = 3.0


def ice_risk(temp_c, wet_hour, relative_humidity_pct: float = 0.0) -> float:
    if temp_c is None:
        return 0.0
    moisture = (float(wet_hour or 0.0) > 0.0) or (float(relative_humidity_pct or 0.0) >= ICE_HUMIDITY_MIN)
    if not moisture or float(temp_c) > ICE_TEMP_C:
        return 0.0
    span = ICE_TEMP_C - ICE_FULL_C
    return round(min(1.0, max(0.0, (ICE_TEMP_C - float(temp_c)) / span)), 3)


def fog_risk(temp_c, dewpoint_c, relative_humidity_pct, wind_speed_mps) -> float:
    if temp_c is None or dewpoint_c is None:
        return 0.0
    spread = float(temp_c) - float(dewpoint_c)
    humidity = float(relative_humidity_pct or 0.0)
    wind = float(wind_speed_mps or 0.0)
    if spread > FOG_SPREAD_C or humidity < FOG_HUMIDITY_MIN or wind > FOG_WIND_MAX_MPS:
        return 0.0
    spread_factor = max(0.0, (FOG_SPREAD_C - max(0.0, spread)) / FOG_SPREAD_C)
    wind_factor = max(0.0, (FOG_WIND_MAX_MPS - wind) / FOG_WIND_MAX_MPS)
    return round(spread_factor * wind_factor, 3)


def assess_hazards(weather: dict) -> dict:
    temp_c = weather.get("temp_c")
    dewpoint_c = weather.get("dewpoint_c")
    humidity = weather.get("relative_humidity_pct")
    wind = weather.get("wind_speed_mps")
    wet = float(weather.get("wet_hour", 0.0) or 0.0) > 0.0

    ice = ice_risk(temp_c, weather.get("wet_hour", 0.0), humidity)
    fog = fog_risk(temp_c, dewpoint_c, humidity, wind)
    labels: list[str] = []
    if ice > 0.0:
        labels.append("ice")
    if fog > 0.0:
        labels.append("fog")
    if wet:
        labels.append("wet")
    return {"ice_risk": ice, "fog_risk": fog, "wet": wet, "labels": labels}
