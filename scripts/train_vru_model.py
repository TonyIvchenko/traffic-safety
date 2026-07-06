"""Train the vulnerable-road-user (pedestrian/cyclist) risk model.

Reuses train_model.py's split/weather/bundle machinery, but points the events
and candidate cells at the VRU dataset (build_vru_dataset.py) while sharing the
main model's weather-station mapping and climatology (VRU cells are a subset of
the main candidate cells). Produces models/traffic_safety_vru.joblib with the
same feature schema, so it can be served as an alternate layer.

    python scripts/train_vru_model.py --train-years 2018 2019 2020 2021 2022 --eval-years 2023
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import joblib
import numpy as np
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_DIR = Path(__file__).resolve().parents[1]
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from common import (
    CELL_WEATHER_STATIONS_PATH,
    NEGATIVE_RATIO,
    RANDOM_SEED,
    VRU_ACCIDENTS_CLEAN_PATH,
    VRU_CANDIDATE_CELLS_PATH,
    VRU_MODEL_BUNDLE_PATH,
    WEATHER_CLIMATOLOGY_PATH,
    ensure_dirs,
)
from train_model import (
    build_bundle,
    build_split,
    load_inputs,
    prepare_climatology_lookup,
    weather_defaults_from_climatology,
)

VRU_MODEL_VERSION = "vru-0.1.0"


def finalize_vru_bundle(bundle: dict, metrics: dict) -> dict:
    """Tag a bundle as the VRU layer so serving can distinguish it."""
    bundle["model_version"] = VRU_MODEL_VERSION
    bundle["layer"] = "vru"
    bundle["target"] = "pedestrian_or_cyclist_crash"
    bundle["metrics"] = metrics
    return bundle


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", nargs="+", type=int,
                        default=[2018, 2019, 2020, 2021, 2022])
    parser.add_argument("--eval-years", nargs="+", type=int, default=[2023])
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)

    events, candidate_cells, climatology = load_inputs(
        accidents_path=VRU_ACCIDENTS_CLEAN_PATH,
        candidate_cells_path=VRU_CANDIDATE_CELLS_PATH,
        cell_weather_stations_path=CELL_WEATHER_STATIONS_PATH,
        climatology_path=WEATHER_CLIMATOLOGY_PATH,
    )
    climatology_lookup = prepare_climatology_lookup(climatology)
    weather_defaults = weather_defaults_from_climatology(climatology)

    x_train, y_train = build_split(
        events=events,
        candidate_cells=candidate_cells,
        climatology_lookup=climatology_lookup,
        weather_defaults=weather_defaults,
        years=args.train_years,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )
    x_eval, y_eval = build_split(
        events=events,
        candidate_cells=candidate_cells,
        climatology_lookup=climatology_lookup,
        weather_defaults=weather_defaults,
        years=args.eval_years,
        negative_ratio=args.negative_ratio,
        rng=rng,
    )

    model = HistGradientBoostingClassifier(
        max_depth=6,
        learning_rate=0.06,
        max_iter=220,
        min_samples_leaf=40,
        random_state=RANDOM_SEED,
    )
    model.fit(x_train, y_train)
    eval_prob = model.predict_proba(x_eval)[:, 1]
    metrics = {
        "val_roc_auc": float(roc_auc_score(y_eval, eval_prob)),
        "val_average_precision": float(average_precision_score(y_eval, eval_prob)),
    }
    print(metrics)

    bundle = build_bundle(
        model=model,
        events=events,
        candidate_cells=candidate_cells,
        climatology=climatology,
        weather_defaults=weather_defaults,
        metrics=metrics,
        train_years=args.train_years,
        eval_years=args.eval_years,
    )
    bundle = finalize_vru_bundle(bundle, metrics)
    joblib.dump(bundle, VRU_MODEL_BUNDLE_PATH)
    print(f"wrote {VRU_MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
