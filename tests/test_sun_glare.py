from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import sun_glare as sg


def test_solar_noon_equinox_equator_is_near_overhead():
    az, el = sg.solar_position(0.0, 0.0, datetime(2024, 3, 20, 12, 0, tzinfo=timezone.utc))
    assert el > 85.0  # sun nearly overhead at solar noon on the equator at equinox


def test_sun_below_horizon_at_local_midnight():
    _, el = sg.solar_position(34.05, -118.24, datetime(2024, 6, 21, 8, 0, tzinfo=timezone.utc))
    assert el < 0.0  # ~1am PDT


def test_bearing_cardinal_directions():
    assert sg.bearing_deg(0.0, 0.0, 0.0, 1.0) == pytest.approx(90.0, abs=0.5)  # east
    assert sg.bearing_deg(0.0, 0.0, 1.0, 0.0) == pytest.approx(0.0, abs=0.5)  # north


def test_angular_difference_wraps():
    assert sg.angular_difference(350.0, 10.0) == pytest.approx(20.0)
    assert sg.angular_difference(90.0, 270.0) == pytest.approx(180.0)


def test_glare_when_facing_low_morning_sun():
    when = datetime(2024, 6, 21, 13, 30, tzinfo=timezone.utc)  # low sun in the ENE
    facing_sun = sg.glare_assessment(34.05, -118.24, when, 90.0)
    away_from_sun = sg.glare_assessment(34.05, -118.24, when, 270.0)
    assert facing_sun["glare"] is True
    assert facing_sun["window"] == "sunrise"
    assert facing_sun["severity"] > 0.0
    assert away_from_sun["glare"] is False


def test_no_glare_when_sun_is_high():
    when = datetime(2024, 6, 21, 20, 0, tzinfo=timezone.utc)  # ~1pm PDT, high sun
    assert sg.glare_assessment(34.05, -118.24, when, 180.0)["glare"] is False


def test_parse_utc_handles_z_and_naive():
    assert sg.parse_utc("2024-06-21T13:30:00Z").tzinfo is not None
    assert sg.parse_utc("2024-06-21T13:30:00").hour == 13
