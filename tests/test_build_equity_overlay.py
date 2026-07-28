from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_equity_overlay as beo
import equity


def _equity_index() -> "equity.EquityIndex":
    return equity.EquityIndex(
        {
            "06037920100": {"svi_percentile": 0.9, "disadvantaged": True},
            "06037920200": {"svi_percentile": 0.2, "disadvantaged": False},
        }
    )


def _segments() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "tract_geoid": ["06037920100", "06037920200", "06099000100"],  # c: not in index
            "fullname": ["Main St", "1st Ave", "Rural Rd"],
            "center_lat": [34.0, 34.1, 36.0],
            "center_lon": [-118.2, -118.3, -119.0],
            "risk": [0.7, 0.3, 0.5],
            "crashes": [5.0, 1.0, 0.0],
        }
    )


def test_segment_forecast_risk_scales_max_over_horizon():
    scores = np.array([[0, 0], [255, 128], [64, 255]], dtype=np.uint8)  # global segments
    risk = beo.segment_forecast_risk([1, 2], scores)
    assert risk[0] == 1.0  # 255/255
    assert risk[1] == 1.0  # max(64,255)/255
    assert risk.dtype == np.float64


def test_segment_forecast_risk_none_or_empty():
    assert list(beo.segment_forecast_risk([], None)) == []
    assert list(beo.segment_forecast_risk([0, 1], None)) == [0.0, 0.0]


def test_build_equity_overlay_joins_equity():
    overlay = beo.build_equity_overlay(_segments(), _equity_index())
    assert list(overlay.columns) == beo.OVERLAY_COLUMNS
    by_id = {row["segment_id"]: row for row in overlay.to_dict("records")}

    assert by_id["a"]["svi_percentile"] == 0.9
    assert by_id["a"]["svi_category"] == "very_high"
    assert bool(by_id["a"]["disadvantaged"]) is True
    assert bool(by_id["a"]["in_equity_index"]) is True
    assert by_id["a"]["risk"] == 0.7
    assert by_id["a"]["crashes"] == 5.0

    assert bool(by_id["b"]["disadvantaged"]) is False
    assert by_id["b"]["svi_category"] == "low"


def test_build_equity_overlay_unknown_tract_reads_unknown():
    overlay = beo.build_equity_overlay(_segments(), _equity_index())
    c = {row["segment_id"]: row for row in overlay.to_dict("records")}["c"]
    assert pd.isna(c["svi_percentile"])
    assert c["svi_category"] == "unknown"
    assert bool(c["disadvantaged"]) is False
    assert bool(c["in_equity_index"]) is False


def test_build_equity_overlay_missing_tract_value_is_unknown():
    segments = _segments()
    segments.loc[0, "tract_geoid"] = None  # segment 'a' has no tract
    overlay = beo.build_equity_overlay(segments, _equity_index())
    a = {row["segment_id"]: row for row in overlay.to_dict("records")}["a"]
    assert bool(a["in_equity_index"]) is False
    assert a["svi_category"] == "unknown"


def test_build_equity_overlay_schema_and_dtypes_stable(tmp_path):
    overlay = beo.build_equity_overlay(_segments(), _equity_index())
    assert str(overlay["svi_percentile"].dtype) == "float64"
    assert overlay["disadvantaged"].dtype == bool
    assert overlay["in_equity_index"].dtype == bool
    # Round-trips to Parquet (stable dtypes, incl. an all-unknown column).
    out = tmp_path / "segment_equity.parquet"
    overlay.to_parquet(out, index=False)
    assert set(pd.read_parquet(out)["segment_id"]) == {"a", "b", "c"}


def test_build_equity_overlay_missing_optional_columns():
    # Only segment_id + tract_geoid provided -> optional cols filled/stable.
    minimal = pd.DataFrame({"segment_id": ["a"], "tract_geoid": ["06037920100"]})
    overlay = beo.build_equity_overlay(minimal, _equity_index())
    assert list(overlay.columns) == beo.OVERLAY_COLUMNS
    assert pd.isna(overlay.iloc[0]["risk"])
    assert bool(overlay.iloc[0]["disadvantaged"]) is True
