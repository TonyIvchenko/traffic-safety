"""Deterministic sun-glare assessment from solar geometry.

No data source or model needed: given a location, a UTC time, and a travel
bearing, compute where the sun is (NOAA solar-position algorithm) and whether a
driver would be looking into a low sun. Low-angle sun glare is a well-documented
crash factor, especially near sunrise/sunset.
"""

from __future__ import annotations

from datetime import datetime, timezone
import math

# Glare when the sun is above the horizon but low, and roughly ahead of travel.
GLARE_ELEV_MAX_DEG = 12.0
GLARE_AZIMUTH_TOL_DEG = 25.0


def parse_utc(value: str) -> datetime:
    text = str(value).strip().replace("Z", "+00:00")
    parsed = datetime.fromisoformat(text)
    if parsed.tzinfo is None:
        return parsed.replace(tzinfo=timezone.utc)
    return parsed.astimezone(timezone.utc)


def _julian_day(when_utc: datetime) -> float:
    when = when_utc.astimezone(timezone.utc)
    year, month = when.year, when.month
    day_fraction = when.day + (when.hour + when.minute / 60.0 + when.second / 3600.0) / 24.0
    if month <= 2:
        year -= 1
        month += 12
    a = year // 100
    b = 2 - a + a // 4
    return int(365.25 * (year + 4716)) + int(30.6001 * (month + 1)) + day_fraction + b - 1524.5


def solar_position(lat: float, lon: float, when_utc: datetime) -> tuple[float, float]:
    """Return (azimuth_deg from North clockwise, elevation_deg) of the sun."""
    when = when_utc.astimezone(timezone.utc)
    jd = _julian_day(when)
    t = (jd - 2451545.0) / 36525.0

    mean_long = (280.46646 + t * (36000.76983 + t * 0.0003032)) % 360.0
    mean_anom = 357.52911 + t * (35999.05029 - 0.0001537 * t)
    eccent = 0.016708634 - t * (0.000042037 + 0.0000001267 * t)
    m_rad = math.radians(mean_anom)
    sun_eq = (
        math.sin(m_rad) * (1.914602 - t * (0.004817 + 0.000014 * t))
        + math.sin(2 * m_rad) * (0.019993 - 0.000101 * t)
        + math.sin(3 * m_rad) * 0.000289
    )
    true_long = mean_long + sun_eq
    app_long = true_long - 0.00569 - 0.00478 * math.sin(math.radians(125.04 - 1934.136 * t))
    mean_obliq = 23.0 + (26.0 + (21.448 - t * (46.815 + t * (0.00059 - t * 0.001813))) / 60.0) / 60.0
    obliq_corr = mean_obliq + 0.00256 * math.cos(math.radians(125.04 - 1934.136 * t))
    declination = math.degrees(
        math.asin(math.sin(math.radians(obliq_corr)) * math.sin(math.radians(app_long)))
    )

    var_y = math.tan(math.radians(obliq_corr / 2.0)) ** 2
    eq_time = 4.0 * math.degrees(
        var_y * math.sin(2 * math.radians(mean_long))
        - 2 * eccent * math.sin(m_rad)
        + 4 * eccent * var_y * math.sin(m_rad) * math.cos(2 * math.radians(mean_long))
        - 0.5 * var_y * var_y * math.sin(4 * math.radians(mean_long))
        - 1.25 * eccent * eccent * math.sin(2 * m_rad)
    )

    minutes = when.hour * 60.0 + when.minute + when.second / 60.0
    true_solar_time = (minutes + eq_time + 4.0 * lon) % 1440.0
    hour_angle = true_solar_time / 4.0 - 180.0

    lat_rad = math.radians(lat)
    decl_rad = math.radians(declination)
    ha_rad = math.radians(hour_angle)
    cos_zenith = math.sin(lat_rad) * math.sin(decl_rad) + math.cos(lat_rad) * math.cos(decl_rad) * math.cos(ha_rad)
    cos_zenith = max(-1.0, min(1.0, cos_zenith))
    zenith = math.degrees(math.acos(cos_zenith))
    elevation = 90.0 - zenith

    az_denom = math.cos(lat_rad) * math.sin(math.radians(zenith))
    if abs(az_denom) > 1e-9:
        az_arg = (math.sin(lat_rad) * math.cos(math.radians(zenith)) - math.sin(decl_rad)) / az_denom
        az_arg = max(-1.0, min(1.0, az_arg))
        az = math.degrees(math.acos(az_arg))
        azimuth = (az + 180.0) % 360.0 if hour_angle > 0 else (540.0 - az) % 360.0
    else:
        azimuth = 180.0 if lat > 0 else 0.0
    return azimuth, elevation


def bearing_deg(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    phi1, phi2 = math.radians(lat1), math.radians(lat2)
    dlon = math.radians(lon2 - lon1)
    y = math.sin(dlon) * math.cos(phi2)
    x = math.cos(phi1) * math.sin(phi2) - math.sin(phi1) * math.cos(phi2) * math.cos(dlon)
    return (math.degrees(math.atan2(y, x)) + 360.0) % 360.0


def angular_difference(a: float, b: float) -> float:
    return abs((a - b + 180.0) % 360.0 - 180.0)


def glare_assessment(lat: float, lon: float, when_utc: datetime, bearing: float) -> dict:
    azimuth, elevation = solar_position(lat, lon, when_utc)
    alignment = angular_difference(bearing % 360.0, azimuth)
    glare = 0.0 < elevation <= GLARE_ELEV_MAX_DEG and alignment <= GLARE_AZIMUTH_TOL_DEG
    if glare:
        elev_factor = (GLARE_ELEV_MAX_DEG - elevation) / GLARE_ELEV_MAX_DEG
        align_factor = (GLARE_AZIMUTH_TOL_DEG - alignment) / GLARE_AZIMUTH_TOL_DEG
        severity = round(max(0.0, elev_factor) * max(0.0, align_factor), 3)
        window = "sunrise" if azimuth < 180.0 else "sunset"
    else:
        severity = 0.0
        window = None
    return {
        "glare": bool(glare),
        "severity": severity,
        "sun_elevation": round(elevation, 2),
        "sun_azimuth": round(azimuth, 2),
        "alignment_deg": round(alignment, 2),
        "window": window,
    }
