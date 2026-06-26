"""Backtest the cell risk model: calibration + ranking metrics on a holdout year.

Reuses train_model's data-loading/split helpers to rebuild a labeled eval set,
scores it with the saved bundle, and writes data/model_eval_report.json (surfaced
by the API at /v1/model/report and /v1/meta).

    python scripts/evaluate_model.py --eval-years 2024
"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import calibration
from common import MODEL_BUNDLE_PATH, ensure_dirs
from train_model import (
    build_split,
    load_inputs,
    prepare_climatology_lookup,
    weather_defaults_from_climatology,
)

MODEL_EVAL_REPORT_PATH = REPO_DIR / "data" / "model_eval_report.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-years", nargs="+", type=int, default=[2024])
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--bins", type=int, default=10)
    parser.add_argument("--seed", type=int, default=0)
    return parser.parse_args()


def stratify_by_risk_band(y_true: np.ndarray, y_prob: np.ndarray, quantiles) -> dict:
    if not (isinstance(quantiles, (list, tuple)) and len(quantiles) >= 3):
        return {}
    low, mid, high = (float(value) for value in quantiles[:3])
    bands = {
        "low": y_prob < low,
        "moderate": (y_prob >= low) & (y_prob < mid),
        "high": (y_prob >= mid) & (y_prob < high),
        "extreme": y_prob >= high,
    }
    out = {}
    for name, mask in bands.items():
        count = int(mask.sum())
        out[name] = {
            "count": count,
            "observed_positive_rate": float(y_true[mask].mean()) if count else None,
            "mean_predicted": float(y_prob[mask].mean()) if count else None,
        }
    return out


def main() -> None:
    args = parse_args()
    ensure_dirs()
    if not MODEL_BUNDLE_PATH.exists():
        raise FileNotFoundError(f"missing {MODEL_BUNDLE_PATH}; run train_model.py first")

    bundle = joblib.load(MODEL_BUNDLE_PATH)
    model = bundle["model"]

    events, candidate_cells, climatology = load_inputs()
    climatology_lookup = prepare_climatology_lookup(climatology)
    weather_defaults = weather_defaults_from_climatology(climatology)
    x_eval, y_eval = build_split(
        events=events,
        candidate_cells=candidate_cells,
        climatology_lookup=climatology_lookup,
        weather_defaults=weather_defaults,
        years=args.eval_years,
        negative_ratio=args.negative_ratio,
        rng=np.random.default_rng(args.seed),
    )

    y_prob = model.predict_proba(x_eval)[:, 1]
    metrics = calibration.summarize(y_eval, y_prob, n_bins=args.bins)
    metrics["roc_auc"] = float(roc_auc_score(y_eval, y_prob))
    metrics["average_precision"] = float(average_precision_score(y_eval, y_prob))

    report = {
        "model_version": str(bundle.get("model_version", "missing")),
        "eval_years": list(args.eval_years),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": metrics,
        "by_risk_band": stratify_by_risk_band(
            np.asarray(y_eval), np.asarray(y_prob), bundle.get("risk_quantiles")
        ),
    }
    MODEL_EVAL_REPORT_PATH.write_text(json.dumps(report, indent=2), encoding="utf-8")
    print(
        f"roc_auc={metrics['roc_auc']:.4f} ap={metrics['average_precision']:.4f} "
        f"brier={metrics['brier_score']:.4f} ece={metrics['ece']:.4f}"
    )
    print(f"wrote {MODEL_EVAL_REPORT_PATH}")


if __name__ == "__main__":
    main()
