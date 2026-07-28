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
