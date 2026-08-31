from __future__ import annotations

from pathlib import Path
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import advisory
import advisory_messages as am


def test_time_of_week_phrase():
    assert am.time_of_week_phrase(0) == "Monday overnight"  # Mon 00:00
    assert am.time_of_week_phrase((5 - 1) * 24 + 17) == "Friday evening"  # Fri 17:00
    assert am.time_of_week_phrase((6 - 1) * 24 + 2) == "Saturday overnight"  # Sat 02:00
    assert am.time_of_week_phrase((3 - 1) * 24 + 12) == "Wednesday afternoon"  # Wed 12:00
    # Out-of-range clamps; bad input -> "".
    assert am.time_of_week_phrase(9999) == "Sunday night"
    assert am.time_of_week_phrase("x") == ""


def test_compose_relative_with_location_time_and_drivers():
    block = {
        "level": "Elevated",
        "advice": "Slow down and increase following distance.",
        "basis": "relative_to_local_weekly_normal",
        "relative_descriptor": "above normal",
        "drivers": ["wet roads", ""],
    }
    message = am.compose(block, location_name="Los Angeles", frame_idx=(5 - 1) * 24 + 17)
    assert message.startswith("Elevated for Los Angeles")
    assert "busier than usual for a Friday evening" in message
    assert "contributing factors: wet roads" in message
    assert message.endswith("Slow down and increase following distance.")


def test_compose_point_without_location_starts_with_level():
    block = {
        "level": "High",
        "advice": "Avoid distractions.",
        "basis": "relative_to_local_weekly_normal",
        "relative_descriptor": "well above normal",
        "drivers": [],
    }
    message = am.compose(block, frame_idx=(6 - 1) * 24 + 2)
    assert message.startswith("High")
    assert "much busier than usual for a Saturday overnight" in message


def test_compose_absolute_basis_has_no_relative_clause():
    block = {
        "level": "Low",
        "advice": "Normal driving conditions.",
        "basis": "absolute",
        "relative_descriptor": None,
        "drivers": [],
    }
    message = am.compose(block, location_name="Nowhere", frame_idx=0)
    assert message == "Low for Nowhere. Normal driving conditions."


def test_compose_weekly_peak_descriptor():
    block = {
        "level": "Extreme",
        "advice": "Travel only if necessary.",
        "basis": "relative_to_local_weekly_normal",
        "relative_descriptor": "near this location's weekly peak",
        "drivers": [],
    }
    message = am.compose(block, location_name="Miami", frame_idx=(7 - 1) * 24 + 23)
    assert message.startswith("Extreme for Miami")
    assert "near the riskiest this area gets" in message


def test_compose_is_defensive():
    assert am.compose(None) == ""
    assert am.compose({}) == "Advisory."  # no level -> generic, still a string
    # Missing frame_idx but a descriptor still yields a clause.
    msg = am.compose(
        {"level": "Moderate", "basis": "relative_to_local_weekly_normal",
         "relative_descriptor": "near normal", "advice": ""}
    )
    assert msg.startswith("Moderate")


def test_compose_message_starts_with_level_from_real_advisory():
    # End-to-end with an actual advisory block.
    block = advisory.relative_advisory(0.9, [0.1 + i / 100.0 for i in range(20)])
    message = am.compose(block, location_name="Denver", frame_idx=100)
    assert message.startswith(block["level"])
