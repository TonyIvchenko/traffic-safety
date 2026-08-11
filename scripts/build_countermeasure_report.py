"""Recommend a countermeasure and its benefit-cost for the top High Injury
Network segments.

For each segment: infer its crash-type profile (crash_typing) from roadway
attributes, find the applicable FHWA countermeasures (countermeasures), and pick
the treatment with the best benefit-cost. Writes a ranked CSV plus a GeoJSON of
point features for mapping.

Roadway context is taken from MTFCC (rur_urb / func_sys are crash-level, not on
the HIN segment, so urban is assumed when unknown) — a screening-level input.

    python scripts/build_countermeasure_report.py --top-n 100 --analysis-years 5

Output: data/reports/countermeasures.csv, data/reports/countermeasures.geojson
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from countermeasures import (
    annualize_crashes,
    applicable_countermeasures,
    countermeasure_benefit_cost,
)
from crash_typing import crash_type_profile

from common import ACCIDENTS_CLEAN_PATH, HIGH_INJURY_NETWORK_PATH, ensure_dirs

REPORTS_DIR = REPO_DIR / "data" / "reports"
DEFAULT_ANALYSIS_YEARS = 5

REPORT_COLUMNS = [
    "segment_id", "fullname", "mtfcc", "length_km", "center_lat", "center_lon",
    "fatal_crashes", "hin_rank", "applicable_count",
    "recommended_id", "recommended_name", "match_score", "cmf", "crash_reduction",
    "expected_annual_crashes", "annual_crashes_reduced", "treatment_cost",
    "present_value_benefit", "net_benefit", "benefit_cost_ratio",
]


def _bcr_key(record: dict) -> float:
    return record.get("benefit_cost_ratio") or 0.0


def recommend_for_segment(
    segment: dict,
    *,
    analysis_years: int,
    top_n: int = 5,
    min_score: float = 0.2,
    vru_share: float = 0.0,
    catalog=None,
) -> dict | None:
    """Best-benefit-cost countermeasure for one segment, or None if none apply."""
    attrs = {
        "mtfcc": segment.get("mtfcc"),
        "rur_urb": segment.get("rur_urb"),
        "func_sys": segment.get("func_sys"),
    }
    profile = crash_type_profile(
        mtfcc=attrs["mtfcc"], rur_urb=attrs["rur_urb"], func_sys=attrs["func_sys"],
        vru_share=vru_share,
    )
    recs = applicable_countermeasures(
        attrs, profile, catalog=catalog, top_n=top_n, min_score=min_score
    )
    if not recs:
        return None

    expected = annualize_crashes(segment.get("fatal_crashes", 0.0), analysis_years)
    scored = [
        {**rec, **countermeasure_benefit_cost(
            rec, expected_annual_crashes=expected, length_km=segment.get("length_km")
        )}
        for rec in recs
    ]
    best = max(scored, key=_bcr_key)
    return {
        "segment_id": segment.get("segment_id"),
        "fullname": segment.get("fullname"),
        "mtfcc": segment.get("mtfcc"),
        "length_km": segment.get("length_km"),
        "center_lat": segment.get("center_lat"),
        "center_lon": segment.get("center_lon"),
        "fatal_crashes": segment.get("fatal_crashes"),
        "hin_rank": segment.get("hin_rank"),
        "applicable_count": len(recs),
        "recommended_id": best.get("id"),
        "recommended_name": best.get("name"),
        "match_score": best.get("match_score"),
        "cmf": best.get("cmf"),
        "crash_reduction": best.get("crash_reduction"),
        "expected_annual_crashes": best.get("expected_annual_crashes"),
        "annual_crashes_reduced": best.get("annual_crashes_reduced"),
        "treatment_cost": best.get("treatment_cost"),
        "present_value_benefit": best.get("present_value_benefit"),
        "net_benefit": best.get("net_benefit"),
        "benefit_cost_ratio": best.get("benefit_cost_ratio"),
    }


def build_countermeasure_report(
    segments: pd.DataFrame, *, analysis_years: int, top_n: int = 5, catalog=None
) -> pd.DataFrame:
    """One recommended-treatment row per segment that has an applicable countermeasure."""
    rows = []
    for segment in segments.to_dict("records"):
        recommendation = recommend_for_segment(
            segment, analysis_years=analysis_years, top_n=top_n, catalog=catalog
        )
        if recommendation is not None:
            rows.append(recommendation)
    frame = pd.DataFrame(rows, columns=REPORT_COLUMNS)
    if len(frame):
        frame = frame.sort_values("benefit_cost_ratio", ascending=False, na_position="last")
    return frame.reset_index(drop=True)


def _safe_float(value):
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def countermeasures_geojson(report: pd.DataFrame) -> dict:
    """Point features at segment centroids carrying the recommendation."""
    features = []
    for row in report.to_dict("records"):
        lat = _safe_float(row.get("center_lat"))
        lon = _safe_float(row.get("center_lon"))
        if lat is None or lon is None:
            continue
        properties = {
            key: (None if (isinstance(value, float) and value != value) else value)
            for key, value in row.items()
            if key not in ("center_lat", "center_lon")
        }
        features.append(
            {
                "type": "Feature",
                "geometry": {"type": "Point", "coordinates": [lon, lat]},
                "properties": properties,
            }
        )
    return {"type": "FeatureCollection", "count": len(features), "features": features}


def analysis_span_years(default: int = DEFAULT_ANALYSIS_YEARS) -> int:
    if not ACCIDENTS_CLEAN_PATH.exists():
        return default
    years = pd.read_csv(ACCIDENTS_CLEAN_PATH, usecols=["year"])["year"].dropna()
    if len(years) == 0:
        return default
    return int(years.max()) - int(years.min()) + 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--top-n", type=int, default=100, help="HIN segments to analyze")
    parser.add_argument("--recs-per-segment", type=int, default=5)
    parser.add_argument("--analysis-years", type=int, default=None)
    parser.add_argument("--output-dir", type=Path, default=REPORTS_DIR)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    network = pd.read_parquet(HIGH_INJURY_NETWORK_PATH)
    hin = network[network["hin"].astype(bool)] if "hin" in network.columns else network
    sort_col = "hin_rank" if "hin_rank" in hin.columns else "fatal_crashes"
    ascending = sort_col == "hin_rank"
    top = hin.sort_values(sort_col, ascending=ascending).head(args.top_n)

    analysis_years = args.analysis_years or analysis_span_years()
    report = build_countermeasure_report(
        top, analysis_years=analysis_years, top_n=args.recs_per_segment
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = args.output_dir / "countermeasures.csv"
    geojson_path = args.output_dir / "countermeasures.geojson"
    report.to_csv(csv_path, index=False)
    payload = countermeasures_geojson(report)
    payload["generated_at_utc"] = datetime.now(timezone.utc).isoformat()
    geojson_path.write_text(json.dumps(payload), encoding="utf-8")

    print(f"recommendations={len(report)} analysis_years={analysis_years}")
    print(f"wrote {csv_path}")
    print(f"wrote {geojson_path}")


if __name__ == "__main__":
    main()
