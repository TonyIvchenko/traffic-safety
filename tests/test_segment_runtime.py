from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path
from types import SimpleNamespace
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segment_runtime as sr


def _rep_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "station_index": [5],
            "LAT": [34.0],
            "LON": [-118.0],
            "utc_offset_hours": [-8],
        }
    ).set_index("station_index")


def test_station_live_contexts_uses_live_snapshot(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        temp_c=21.0,
        relative_humidity_pct=55.0,
        wind_speed_mps=3.0,
        wet_hour=0.0,
    )
    monkeypatch.setattr(sr, "fetch_live_weather", lambda **kwargs: snapshot)

    contexts = sr._station_live_contexts(
        np.array([5], dtype=np.int16),
        rep_by_index=_rep_frame(),
        forecast_hours=0,
        provider="auto",
    )

    assert contexts[5]["provider"] == "nws"
    assert contexts[5]["temp_c"] == 21.0


def test_station_live_contexts_falls_back_when_station_missing(monkeypatch):
    # fetch should never be called for an unknown station index.
    def fail(**kwargs):
        raise AssertionError("fetch_live_weather should not be called for missing station")

    monkeypatch.setattr(sr, "fetch_live_weather", fail)

    contexts = sr._station_live_contexts(
        np.array([99], dtype=np.int16),
        rep_by_index=_rep_frame(),
        forecast_hours=0,
        provider="auto",
    )

    assert 99 in contexts
    assert contexts[99]["provider"] == "climatology"
    assert contexts[99]["temp_c"] is None
    assert 0 <= contexts[99]["hour_of_week"] < 168


def test_station_live_contexts_falls_back_when_fetch_fails(monkeypatch):
    def boom(**kwargs):
        raise RuntimeError("provider down")

    monkeypatch.setattr(sr, "fetch_live_weather", boom)

    contexts = sr._station_live_contexts(
        np.array([5, 5], dtype=np.int16),
        rep_by_index=_rep_frame(),
        forecast_hours=3,
        provider="auto",
    )

    assert contexts[5]["provider"] == "climatology"
    assert contexts[5]["wind_speed_mps"] is None
