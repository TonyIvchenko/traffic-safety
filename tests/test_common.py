from __future__ import annotations

import math
from pathlib import Path
import sys

import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts import common


def test_cyclical_encoders_hit_known_angles():
    sin_m, cos_m = common.month_sin_cos(3)  # 2*pi*3/12 = pi/2
    assert sin_m == pytest.approx(1.0)
    assert cos_m == pytest.approx(0.0, abs=1e-9)

    sin_h, cos_h = common.hour_sin_cos(6)  # 2*pi*6/24 = pi/2
    assert sin_h == pytest.approx(1.0)
    assert cos_h == pytest.approx(0.0, abs=1e-9)

    sin_d, cos_d = common.dow_sin_cos(1)  # 2*pi*0/7 = 0
    assert sin_d == pytest.approx(0.0, abs=1e-9)
    assert cos_d == pytest.approx(1.0)


def test_cyclical_encoders_are_unit_vectors():
    for hour in range(24):
        sin_h, cos_h = common.hour_sin_cos(hour)
        assert math.hypot(sin_h, cos_h) == pytest.approx(1.0)


def test_local_hour_of_week_label_boundaries():
    assert common.local_hour_of_week_label(0) == "Mon 00:00"
    assert common.local_hour_of_week_label(25) == "Tue 01:00"
    assert common.local_hour_of_week_label(167) == "Sun 23:00"


def test_weekly_frame_labels_cover_full_week():
    labels = common.weekly_frame_labels()
    assert len(labels) == 24 * 7
    assert labels[0] == "Mon 00:00"
    assert labels[-1] == "Sun 23:00"


def test_weekly_ticks_align_to_day_boundaries():
    ticks = common.weekly_ticks()
    assert [tick["label"] for tick in ticks] == common.WEEKDAY_LABELS
    assert [tick["frame_idx"] for tick in ticks] == [idx * 24 for idx in range(7)]


def test_source_url_builders():
    assert common.fars_zip_url(2022) == (
        "https://static.nhtsa.gov/nhtsa/downloads/FARS/2022/National/FARS2022NationalCSV.zip"
    )
    assert common.tiger_prisecroads_url("06") == (
        "https://www2.census.gov/geo/tiger/TIGER2024/PRISECROADS/tl_2024_06_prisecroads.zip"
    )
    assert common.noaa_isd_lite_url("723815-99999", 2020) == (
        "https://www.ncei.noaa.gov/pub/data/noaa/isd-lite/2020/723815-99999-2020.gz"
    )


def test_source_path_builders():
    assert common.fars_zip_path(2022).name == "FARS2022NationalCSV.zip"
    isd_path = common.noaa_isd_lite_path("723815-99999", 2020)
    assert isd_path.name == "723815-99999-2020.gz"
    assert isd_path.parent.name == "2020"


def test_current_month_is_valid():
    assert 1 <= common.current_month() <= 12
