from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_equity_index as bei
import common


def test_default_source_vintages_are_tract_aligned():
    # SVI and CEJST are joined on the raw GEOID, so they must share a census-tract
    # vintage. CEJST 2.0 is 2010-tract; SVI 2020 is 2010-tract (SVI 2022 is
    # 2020-tract). If you bump SVI to a 2020-tract vintage, add a 2010->2020
    # crosswalk before merging or re-tracted areas will be silently mislabeled.
    assert common.CDC_SVI_YEAR == 2020
    assert common.CEJST_VERSION == "2.0"


def test_normalize_geoid():
    assert bei.normalize_geoid("06037920100") == "06037920100"  # 11-digit string
    assert bei.normalize_geoid(1073001100) == "01073001100"  # int, leading zero dropped
    assert bei.normalize_geoid("1073001100.0") == "01073001100"  # float-formatted
    assert bei.normalize_geoid("06037") is None  # county, wrong length
    assert bei.normalize_geoid("not-a-geoid") is None
    assert bei.normalize_geoid(None) is None


def test_to_bool():
    assert bei._to_bool(True) is True
    assert bei._to_bool("True") is True
    assert bei._to_bool("1") is True
    assert bei._to_bool("False") is False
    assert bei._to_bool("") is False
    assert bei._to_bool(None) is False
    assert bei._to_bool(float("nan")) is False


def test_parse_svi_normalizes_and_nulls_sentinel():
    frame = pd.DataFrame(
        {
            "FIPS": ["06037920100", "06037920200", "1073001100"],
            "RPL_THEMES": ["0.8542", "-999", "0.4100"],
            "STATE": ["CA", "CA", "AL"],
        }
    )
    out = bei.parse_svi(frame)
    by_geoid = dict(zip(out["tract_geoid"], out["svi_percentile"]))
    assert by_geoid["06037920100"] == pytest.approx(0.8542)
    assert pd.isna(by_geoid["06037920200"])  # -999 sentinel -> null
    assert by_geoid["01073001100"] == pytest.approx(0.41)  # leading zero restored


def test_parse_svi_missing_column_raises():
    with pytest.raises(ValueError, match="RPL_THEMES"):
        bei.parse_svi(pd.DataFrame({"FIPS": ["06037920100"]}))


def test_parse_cejst_reads_flag():
    frame = pd.DataFrame(
        {
            "Census tract 2010 ID": ["06037920100", "1073001100"],
            "Identified as disadvantaged": ["True", "False"],
        }
    )
    out = bei.parse_cejst(frame)
    by_geoid = dict(zip(out["tract_geoid"], out["disadvantaged"]))
    assert by_geoid["06037920100"] is True or by_geoid["06037920100"] == True  # noqa: E712
    assert by_geoid["01073001100"] == False  # noqa: E712


def test_build_equity_table_outer_join():
    svi = pd.DataFrame({"tract_geoid": ["A", "B"], "svi_percentile": [0.9, 0.5]})
    cejst = pd.DataFrame({"tract_geoid": ["B", "C"], "disadvantaged": [True, True]})
    table = bei.build_equity_table(svi, cejst)
    assert list(table.columns) == ["tract_geoid", "svi_percentile", "disadvantaged"]
    assert list(table["tract_geoid"]) == ["A", "B", "C"]  # sorted
    rows = {r["tract_geoid"]: r for r in table.to_dict("records")}
    # A: SVI only -> not disadvantaged.
    assert rows["A"]["disadvantaged"] is False or rows["A"]["disadvantaged"] == False  # noqa: E712
    # C: CEJST only -> disadvantaged, no SVI.
    assert rows["C"]["disadvantaged"] == True  # noqa: E712
    assert pd.isna(rows["C"]["svi_percentile"])
    assert rows["B"]["svi_percentile"] == pytest.approx(0.5)


def test_build_equity_table_end_to_end_from_raw():
    svi_raw = pd.DataFrame(
        {"FIPS": ["06037920100", "06037920200"], "RPL_THEMES": ["0.80", "-999"]}
    )
    cejst_raw = pd.DataFrame(
        {
            "Census tract 2010 ID": ["06037920100", "06037920300"],
            "Identified as disadvantaged": ["True", "True"],
        }
    )
    table = bei.build_equity_table(bei.parse_svi(svi_raw), bei.parse_cejst(cejst_raw))
    assert set(table["tract_geoid"]) == {"06037920100", "06037920200", "06037920300"}
    assert int(table["disadvantaged"].sum()) == 2
