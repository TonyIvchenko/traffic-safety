from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import equity


def test_svi_category_quartile_bands():
    assert equity.svi_category(0.10) == "low"
    assert equity.svi_category(0.25) == "moderate"  # lower-inclusive boundary
    assert equity.svi_category(0.49) == "moderate"
    assert equity.svi_category(0.50) == "high"
    assert equity.svi_category(0.74) == "high"
    assert equity.svi_category(0.75) == "very_high"
    assert equity.svi_category(1.0) == "very_high"


def test_svi_category_unknown_for_missing_or_bad():
    assert equity.svi_category(None) == "unknown"
    assert equity.svi_category(-1.0) == "unknown"  # -999 sentinel already nulled upstream
    assert equity.svi_category(float("nan")) == "unknown"
    assert equity.svi_category("n/a") == "unknown"


def test_equity_for_tract_known():
    index = equity.EquityIndex(
        {"06037920100": {"svi_percentile": 0.82, "disadvantaged": True}}
    )
    result = index.equity_for_tract("06037920100")
    assert result["svi_percentile"] == 0.82
    assert result["svi_category"] == "very_high"
    assert result["disadvantaged"] is True
    assert result["in_index"] is True


def test_equity_for_tract_unknown_tract():
    result = equity.EquityIndex({}).equity_for_tract("06037920100")
    assert result["tract_geoid"] == "06037920100"
    assert result["svi_percentile"] is None
    assert result["svi_category"] == "unknown"
    assert result["disadvantaged"] is False
    assert result["in_index"] is False


def test_from_csv_reads_records(tmp_path):
    path = tmp_path / "tract_equity.csv"
    pd.DataFrame(
        {
            "tract_geoid": ["06037920100", "06037920200"],
            "svi_percentile": [0.8, None],
            "disadvantaged": [True, False],
        }
    ).to_csv(path, index=False)

    index = equity.EquityIndex.from_csv(path)
    assert len(index) == 2
    a = index.equity_for_tract("06037920100")
    assert a["svi_percentile"] == 0.8 and a["disadvantaged"] is True
    b = index.equity_for_tract("06037920200")
    assert b["svi_percentile"] is None  # missing SVI -> None
    assert b["disadvantaged"] is False


def test_from_csv_does_not_treat_false_string_as_truthy(tmp_path):
    # to_csv writes bools as text; "False" must not read back as disadvantaged.
    path = tmp_path / "e.csv"
    path.write_text(
        "tract_geoid,svi_percentile,disadvantaged\n06037920100,0.5,False\n", encoding="utf-8"
    )
    assert equity.EquityIndex.from_csv(path).equity_for_tract("06037920100")["disadvantaged"] is False


def test_from_csv_preserves_leading_zero_geoid(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text(
        "tract_geoid,svi_percentile,disadvantaged\n01073001100,0.9,True\n", encoding="utf-8"
    )
    assert equity.EquityIndex.from_csv(path).get("01073001100") is not None


def test_from_csv_missing_file_is_empty(tmp_path):
    assert len(equity.EquityIndex.from_csv(tmp_path / "nope.csv")) == 0


def test_from_csv_corrupt_file_degrades_to_empty(tmp_path):
    # A corrupt reference file must not crash the loader (would 500 the API).
    path = tmp_path / "e.csv"
    path.write_bytes(b'"a,b\n\x00\x01 unterminated')
    index = equity.EquityIndex.from_csv(path)
    assert len(index) == 0
    assert index.equity_for_tract("06037920100")["in_index"] is False


def test_from_csv_missing_columns_do_not_crash(tmp_path):
    path = tmp_path / "e.csv"
    path.write_text("tract_geoid\n06037920100\n", encoding="utf-8")  # no svi/flag columns
    result = equity.EquityIndex.from_csv(path).equity_for_tract("06037920100")
    assert result["in_index"] is True
    assert result["svi_percentile"] is None
    assert result["disadvantaged"] is False


def test_load_equity_index_honors_env(tmp_path, monkeypatch):
    path = tmp_path / "eq.csv"
    path.write_text(
        "tract_geoid,svi_percentile,disadvantaged\n06037920100,0.9,True\n", encoding="utf-8"
    )
    monkeypatch.setenv(equity.EQUITY_PATH_ENV, str(path))
    result = equity.equity_for_tract("06037920100")
    assert result["disadvantaged"] is True
    assert result["svi_category"] == "very_high"


# --- Equity hotspot ranking ---------------------------------------------------


def _overlay() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "risk": [0.8, 0.8, 0.9, 0.2],
            "svi_percentile": [0.9, 0.1, None, 0.95],
            "disadvantaged": [True, False, False, True],
        }
    )


def test_equity_priority_score_boosts_vulnerable():
    # Same risk, but a disadvantaged high-SVI tract scores much higher.
    served = equity.equity_priority_score(0.8, 0.1, False)  # 0.8 * 1.1
    underserved = equity.equity_priority_score(0.8, 0.9, True)  # 0.8 * (1+0.9+0.5)
    assert served == pytest.approx(0.88)
    assert underserved == pytest.approx(1.92)
    assert underserved > served


def test_equity_priority_unknown_svi_no_boost():
    assert equity.equity_priority_score(0.5, None, False) == pytest.approx(0.5)


def test_rank_equity_hotspots_priority_boosts_disadvantaged():
    ranked = equity.rank_equity_hotspots(_overlay())
    # 'a' (risk .8, svi .9, disadv) outranks 'c' (risk .9 but unknown svi).
    assert ranked.iloc[0]["segment_id"] == "a"
    assert "equity_priority" in ranked.columns


def test_rank_equity_hotspots_rank_by_risk_ignores_boost():
    ranked = equity.rank_equity_hotspots(_overlay(), rank_by="risk")
    assert ranked.iloc[0]["segment_id"] == "c"  # highest raw risk 0.9


def test_rank_equity_hotspots_only_disadvantaged_filter():
    ranked = equity.rank_equity_hotspots(_overlay(), only_disadvantaged=True)
    assert set(ranked["segment_id"]) == {"a", "d"}


def test_rank_equity_hotspots_min_svi_filter():
    ranked = equity.rank_equity_hotspots(_overlay(), min_svi=0.75)
    assert set(ranked["segment_id"]) == {"a", "d"}  # svi 0.9, 0.95


def test_rank_equity_hotspots_min_risk_and_top_n():
    ranked = equity.rank_equity_hotspots(_overlay(), min_risk=0.5, top_n=2)
    assert len(ranked) == 2
    assert "d" not in set(ranked["segment_id"])  # risk 0.2 filtered out


def test_rank_equity_hotspots_handles_nan_without_crash():
    overlay = pd.DataFrame(
        {"segment_id": ["x"], "risk": [None], "svi_percentile": [None], "disadvantaged": [False]}
    )
    ranked = equity.rank_equity_hotspots(overlay, min_risk=0.0)
    assert ranked.iloc[0]["equity_priority"] == 0.0


# --- EquityOverlay accessor ---------------------------------------------------


def _overlay_frame() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "tract_geoid": ["06037920100", "06037920200", "06099000100", "06037920300"],
            "risk": [0.8, 0.8, 0.9, 0.2],
            "svi_percentile": [0.9, 0.1, None, 0.95],
            "disadvantaged": [True, False, False, True],
            "center_lat": [34.0, 34.1, 36.0, 34.2],
            "center_lon": [-118.2, -118.3, -119.0, -118.4],
            "fullname": ["Main St", "1st Ave", "Rural Rd", "Elm St"],
        }
    )


def _write_overlay(tmp_path) -> Path:
    path = tmp_path / "segment_equity.parquet"
    _overlay_frame().to_parquet(path, index=False)
    return path


def test_equity_overlay_hotspots_ranked_json_safe(tmp_path):
    overlay = equity.EquityOverlay.from_parquet(_write_overlay(tmp_path))
    records = overlay.hotspots()
    assert records[0]["segment_id"] == "a"  # boosted (disadvantaged, high SVI)
    # JSON-safe: the unknown-SVI row 'c' carries None, not NaN.
    c = next(r for r in records if r["segment_id"] == "c")
    assert c["svi_percentile"] is None
    import json

    json.dumps(records)  # must not raise


def test_equity_overlay_bbox_filter(tmp_path):
    overlay = equity.EquityOverlay.from_parquet(_write_overlay(tmp_path))
    records = overlay.hotspots(bbox=(34.0, 34.25, -118.5, -118.1))
    assert set(r["segment_id"] for r in records) == {"a", "b", "d"}  # excludes rural 'c'


def test_equity_overlay_only_disadvantaged(tmp_path):
    overlay = equity.EquityOverlay.from_parquet(_write_overlay(tmp_path))
    records = overlay.hotspots(only_disadvantaged=True)
    assert set(r["segment_id"] for r in records) == {"a", "d"}


def test_equity_overlay_missing_file_is_empty(tmp_path):
    overlay = equity.EquityOverlay.from_parquet(tmp_path / "nope.parquet")
    assert len(overlay) == 0
    assert overlay.hotspots() == []


def test_load_equity_overlay_honors_env(tmp_path, monkeypatch):
    path = _write_overlay(tmp_path)
    monkeypatch.setenv(equity.EQUITY_OVERLAY_PATH_ENV, str(path))
    assert len(equity.load_equity_overlay()) == 4


def test_equity_overlay_hotspots_nullable_disadvantaged(tmp_path):
    # A nullable boolean column with pd.NA (unfilled left-join) must not 500.
    frame = pd.DataFrame(
        {
            "segment_id": ["a", "b"],
            "risk": [0.8, 0.7],
            "svi_percentile": [0.9, 0.8],
            "disadvantaged": pd.array([True, pd.NA], dtype="boolean"),
            "center_lat": [34.0, 34.1],
            "center_lon": [-118.2, -118.3],
        }
    )
    path = tmp_path / "o.parquet"
    frame.to_parquet(path, index=False)
    overlay = equity.EquityOverlay.from_parquet(path)
    records = overlay.hotspots()  # must not raise
    assert len(records) == 2
    # NA disadvantaged is treated as not-disadvantaged by the filter.
    only = overlay.hotspots(only_disadvantaged=True)
    assert {r["segment_id"] for r in only} == {"a"}


def test_json_scalar_coerces_arrays_and_scalars():
    import numpy as np

    assert equity._json_scalar(np.int64(3)) == 3
    assert equity._json_scalar(np.array([5])) == 5  # size 1 -> item()
    assert equity._json_scalar(np.array([1.0, 2.0])) == [1.0, 2.0]  # size>1 -> tolist()
    assert equity._json_scalar(None) is None
    assert equity._json_scalar("x") == "x"


# --- Equity disparity summary -------------------------------------------------


def _disparity_overlay() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c", "d"],
            "tract_geoid": ["06037920100", "06037920200", "06037920300", "06059000100"],
            "disadvantaged": [True, True, False, False],
            "crashes": [10.0, 6.0, 2.0, 0.0],
            "risk": [0.8, 0.6, 0.4, 0.2],
            "svi_percentile": [0.9, 0.8, 0.3, 0.1],
        }
    )


def test_equity_disparity_metrics():
    result = equity.equity_disparity(_disparity_overlay())
    assert result["segments"] == 4
    assert result["disadvantaged_segments"] == 2
    assert result["crashes_total"] == 18.0
    assert result["crashes_in_disadvantaged"] == 16.0
    assert result["crash_share_disadvantaged"] == pytest.approx(16 / 18, abs=1e-4)
    # burden per segment: disadvantaged 16/2=8, other 2/2=1 -> ratio 8.
    assert result["crash_disparity_ratio"] == pytest.approx(8.0)
    # mean risk: disadvantaged 0.7, other 0.3 -> ratio ~2.333.
    assert result["risk_disparity_ratio"] == pytest.approx(0.7 / 0.3, abs=1e-3)
    assert result["high_svi_segments"] == 2  # svi 0.9, 0.8 >= 0.75


def test_equity_disparity_empty():
    result = equity.equity_disparity(_disparity_overlay().iloc[0:0])
    assert result["segments"] == 0
    assert result["crash_disparity_ratio"] is None
    assert result["weighted_burden"]["burden_ratio"] is None


def test_svi_weighted_crashes():
    assert equity.svi_weighted_crashes(10, 0.9) == 9.0
    assert equity.svi_weighted_crashes(5, None) == 0.0  # unknown SVI weights 0
    assert equity.svi_weighted_crashes(4, -1.0) == 0.0  # out-of-range clamped
    assert equity.svi_weighted_crashes("bad", 0.5) == 0.0


def test_crash_burden_index_concentration():
    frame = pd.DataFrame({"crashes": [10.0, 2.0], "svi_percentile": [0.9, 0.1]})
    burden = equity.crash_burden_index(frame)
    # crash-weighted SVI = (10*0.9 + 2*0.1)/12 = 9.2/12 = 0.7667.
    assert burden["svi_weighted_crashes"] == pytest.approx(9.2)
    assert burden["crash_weighted_svi"] == pytest.approx(0.7667, abs=1e-4)
    assert burden["segment_mean_svi"] == pytest.approx(0.5)
    # crashes over-concentrate in the vulnerable tract -> ratio > 1.
    assert burden["burden_ratio"] == pytest.approx(0.7667 / 0.5, abs=1e-3)


def test_crash_burden_index_excludes_unknown_svi():
    frame = pd.DataFrame({"crashes": [10.0, 5.0], "svi_percentile": [0.8, None]})
    burden = equity.crash_burden_index(frame)
    assert burden["crashes_with_known_svi"] == 10.0  # the unknown-SVI crash excluded
    assert burden["crash_weighted_svi"] == pytest.approx(0.8)


def test_crash_burden_index_empty():
    burden = equity.crash_burden_index(pd.DataFrame({"crashes": [], "svi_percentile": []}))
    assert burden["burden_ratio"] is None
    assert burden["crashes_with_known_svi"] == 0.0


def test_equity_disparity_includes_weighted_burden():
    result = equity.equity_disparity(_disparity_overlay())
    burden = result["weighted_burden"]
    assert "burden_ratio" in burden
    assert burden["svi_weighted_crashes"] > 0


def test_equity_disparity_all_disadvantaged_ratio_none():
    frame = _disparity_overlay()
    frame["disadvantaged"] = True  # no comparison group
    result = equity.equity_disparity(frame)
    assert result["crash_disparity_ratio"] is None  # burden_other == 0


def test_overlay_summary_scopes_by_geoid_prefix(tmp_path):
    path = tmp_path / "o.parquet"
    _disparity_overlay().to_parquet(path, index=False)
    overlay = equity.EquityOverlay.from_parquet(path)

    la = overlay.summary(geoid="06037")  # a, b, c
    assert la["geoid"] == "06037"
    assert la["segments"] == 3
    assert la["crash_disparity_ratio"] == pytest.approx(4.0)  # (16/2)/(2/1)

    national = overlay.summary()  # all 4
    assert national["geoid"] is None
    assert national["segments"] == 4


def test_overlay_summary_unknown_geoid_is_empty(tmp_path):
    path = tmp_path / "o.parquet"
    _disparity_overlay().to_parquet(path, index=False)
    result = equity.EquityOverlay.from_parquet(path).summary(geoid="99")
    assert result["segments"] == 0
    assert result["crash_disparity_ratio"] is None


# --- Tract choropleth aggregation ---------------------------------------------


def _choropleth_overlay() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "segment_id": ["a", "b", "c"],
            "tract_geoid": ["06037920100", "06037920100", "06037920200"],
            "svi_percentile": [0.9, 0.9, 0.2],
            "disadvantaged": [True, True, False],
            "risk": [0.8, 0.6, 0.4],
            "crashes": [5.0, 3.0, 1.0],
            "center_lat": [34.0, 34.02, 34.2],
            "center_lon": [-118.2, -118.22, -118.4],
        }
    )


def test_tract_records_aggregates_by_tract(tmp_path):
    path = tmp_path / "o.parquet"
    _choropleth_overlay().to_parquet(path, index=False)
    records = equity.EquityOverlay.from_parquet(path).tract_records()
    by_tract = {r["tract_geoid"]: r for r in records}
    assert set(by_tract) == {"06037920100", "06037920200"}

    a = by_tract["06037920100"]  # two segments a, b
    assert a["segment_count"] == 2
    assert a["crashes"] == 8.0  # 5 + 3
    assert a["mean_risk"] == pytest.approx(0.7)  # (0.8 + 0.6)/2
    assert a["svi_percentile"] == 0.9
    assert a["svi_category"] == "very_high"
    assert a["disadvantaged"] is True
    assert a["center_lat"] == pytest.approx(34.01)  # centroid mean

    b = by_tract["06037920200"]
    assert b["segment_count"] == 1
    assert b["disadvantaged"] is False


def test_tract_records_bbox_filter(tmp_path):
    path = tmp_path / "o.parquet"
    _choropleth_overlay().to_parquet(path, index=False)
    records = equity.EquityOverlay.from_parquet(path).tract_records(
        bbox=(33.9, 34.1, -118.3, -118.1)
    )
    assert {r["tract_geoid"] for r in records} == {"06037920100"}  # excludes far tract


def test_tract_records_empty(tmp_path):
    assert equity.EquityOverlay.from_parquet(tmp_path / "nope.parquet").tract_records() == []
