from __future__ import annotations

import argparse
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_support import (
    WEATHER_FEATURE_NAMES,
    build_feature_matrix,
    lookup_weather_climatology,
)

from common import (
    ACCIDENTS_CLEAN_PATH,
    CANDIDATE_CELLS_PATH,
    CELL_WEATHER_STATIONS_PATH,
    DEFAULT_EVAL_YEAR,
    DEFAULT_TRAIN_YEARS,
    H3_RESOLUTION,
    MODEL_BUNDLE_PATH,
    NEGATIVE_RATIO,
    RANDOM_SEED,
    WEATHER_CLIMATOLOGY_PATH,
    WEATHER_HOURLY_DIR,
    current_month,
    ensure_dirs,
)


WEATHER_EXACT_KEYS = ["station_index", "month", "day", "hour"]
WEATHER_CLIMATOLOGY_KEYS = ["station_index", "month", "hour_of_week"]


@dataclass
class FeatureContext:
    lat_by_cell: dict[str, float]
    lon_by_cell: dict[str, float]
    total_by_cell: dict[str, int]
    hour_by_cell: dict[tuple[str, int], int]


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required_paths = [
        ACCIDENTS_CLEAN_PATH,
        CANDIDATE_CELLS_PATH,
        CELL_WEATHER_STATIONS_PATH,
        WEATHER_CLIMATOLOGY_PATH,
    ]
    for path in required_paths:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    events = pd.read_csv(ACCIDENTS_CLEAN_PATH)
    candidate_cells = pd.read_csv(CANDIDATE_CELLS_PATH)
    cell_station_map = pd.read_csv(CELL_WEATHER_STATIONS_PATH)
    climatology = pd.read_csv(WEATHER_CLIMATOLOGY_PATH)

    candidate_cells = candidate_cells.merge(
        cell_station_map[["cell_id", "station_index"]],
        on="cell_id",
        how="left",
        validate="one_to_one",
    )
    if candidate_cells["station_index"].isna().any():
        missing = int(candidate_cells["station_index"].isna().sum())
        raise ValueError(f"missing station_index for {missing} candidate cells")

    candidate_cells["station_index"] = candidate_cells["station_index"].astype(np.int16)
    climatology["station_index"] = climatology["station_index"].astype(np.int16)
    climatology["month"] = climatology["month"].astype(np.int8)
    climatology["hour_of_week"] = climatology["hour_of_week"].astype(np.int16)
    for column in WEATHER_FEATURE_NAMES:
        climatology[column] = climatology[column].astype(np.float32)
    return events, candidate_cells, climatology


def build_context(history: pd.DataFrame, candidate_cells: pd.DataFrame) -> FeatureContext:
    total_by_cell = history.groupby("cell_id").size().astype(int).to_dict()
    hour_by_cell = (
        history.groupby(["cell_id", "hour_of_week"]).size().astype(int).to_dict()
    )
    return FeatureContext(
        lat_by_cell=candidate_cells.set_index("cell_id")["center_lat"].to_dict(),
        lon_by_cell=candidate_cells.set_index("cell_id")["center_lon"].to_dict(),
        total_by_cell=total_by_cell,
        hour_by_cell=hour_by_cell,
    )


@lru_cache(maxsize=16)
def load_year_weather(year: int) -> pd.DataFrame:
    year_dir = WEATHER_HOURLY_DIR / str(year)
    if not year_dir.exists():
        return pd.DataFrame(columns=WEATHER_EXACT_KEYS + WEATHER_FEATURE_NAMES)

    frames = []
    usecols = WEATHER_EXACT_KEYS + WEATHER_FEATURE_NAMES
    for path in sorted(year_dir.glob("*.csv.gz")):
        frame = pd.read_csv(path, usecols=usecols)
        if not frame.empty:
            frames.append(frame)

    if not frames:
        return pd.DataFrame(columns=WEATHER_EXACT_KEYS + WEATHER_FEATURE_NAMES)

    weather = pd.concat(frames, ignore_index=True)
    weather["station_index"] = weather["station_index"].astype(np.int16)
    weather["month"] = weather["month"].astype(np.int8)
    weather["day"] = weather["day"].astype(np.int8)
    weather["hour"] = weather["hour"].astype(np.int8)
    for column in WEATHER_FEATURE_NAMES:
        weather[column] = weather[column].astype(np.float32)
    return weather


def weather_defaults_from_climatology(climatology: pd.DataFrame) -> np.ndarray:
    defaults = climatology[WEATHER_FEATURE_NAMES].mean().to_numpy(dtype=np.float32)
    defaults = np.nan_to_num(defaults, nan=0.0, posinf=0.0, neginf=0.0)
    return defaults.astype(np.float32)


def prepare_climatology_lookup(climatology: pd.DataFrame) -> pd.DataFrame:
    renamed = climatology[WEATHER_CLIMATOLOGY_KEYS + WEATHER_FEATURE_NAMES].rename(
        columns={column: f"clim_{column}" for column in WEATHER_FEATURE_NAMES}
    )
    return renamed


def attach_weather_features(
    rows: pd.DataFrame,
    exact_weather: pd.DataFrame,
    climatology_lookup: pd.DataFrame,
    weather_defaults: np.ndarray,
) -> tuple[pd.DataFrame, float]:
    exact_lookup = exact_weather.rename(
        columns={column: f"exact_{column}" for column in WEATHER_FEATURE_NAMES}
    )
    merged = rows.merge(
        exact_lookup,
        on=WEATHER_EXACT_KEYS,
        how="left",
    )
    merged = merged.merge(
        climatology_lookup,
        on=WEATHER_CLIMATOLOGY_KEYS,
        how="left",
    )

    exact_hit = np.zeros(len(merged), dtype=bool)
    if len(merged):
        exact_hit = merged[f"exact_{WEATHER_FEATURE_NAMES[0]}"].notna().to_numpy()

    for feature_idx, column in enumerate(WEATHER_FEATURE_NAMES):
        merged[column] = (
            merged.get(f"exact_{column}")
            .fillna(merged.get(f"clim_{column}"))
            .fillna(float(weather_defaults[feature_idx]))
            .astype(np.float32)
        )

    helper_columns = [
        *[f"exact_{column}" for column in WEATHER_FEATURE_NAMES],
        *[f"clim_{column}" for column in WEATHER_FEATURE_NAMES],
    ]
    merged = merged.drop(columns=helper_columns, errors="ignore")
    return merged, float(exact_hit.mean()) if len(exact_hit) else 0.0


def build_examples(rows: pd.DataFrame, context: FeatureContext) -> np.ndarray:
    cells = rows["cell_id"].to_numpy(dtype=object)
    hour_of_week = rows["hour_of_week"].to_numpy(dtype=np.int16)
    months = rows["month"].to_numpy(dtype=np.int8)

    lats = np.array([context.lat_by_cell.get(cell, 0.0) for cell in cells], dtype=np.float32)
    lons = np.array([context.lon_by_cell.get(cell, 0.0) for cell in cells], dtype=np.float32)
    totals = np.array([context.total_by_cell.get(cell, 0) for cell in cells], dtype=np.float32)
    same_hour = np.array(
        [
            context.hour_by_cell.get((cell, int(frame_idx)), 0)
            for cell, frame_idx in zip(cells, hour_of_week, strict=False)
        ],
        dtype=np.float32,
    )

    return build_feature_matrix(
        latitudes=lats,
        longitudes=lons,
        hour_of_week=hour_of_week,
        months=months,
        totals=totals,
        same_hour=same_hour,
        temp_c=rows["temp_c"].to_numpy(dtype=np.float32),
        dewpoint_c=rows["dewpoint_c"].to_numpy(dtype=np.float32),
        relative_humidity_pct=rows["relative_humidity_pct"].to_numpy(dtype=np.float32),
        wind_speed_mps=rows["wind_speed_mps"].to_numpy(dtype=np.float32),
        wet_hour=rows["wet_hour"].to_numpy(dtype=np.float32),
    )


def sample_negatives(
    year: int,
    count: int,
    candidate_cells: pd.DataFrame,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled_idx = rng.integers(0, len(candidate_cells), size=count)
    sampled_cells = candidate_cells.iloc[sampled_idx].reset_index(drop=True)

    start = np.datetime64(f"{year}-01-01T00:00:00")
    end = np.datetime64(f"{year + 1}-01-01T00:00:00")
    total_hours = int((end - start) / np.timedelta64(1, "h"))
    offsets = rng.integers(0, total_hours, size=count)
    timestamps = pd.to_datetime(
        offsets,
        unit="h",
        origin=pd.Timestamp(f"{year}-01-01 00:00:00"),
    )

    return pd.DataFrame(
        {
            "cell_id": sampled_cells["cell_id"].to_numpy(dtype=object),
            "station_index": sampled_cells["station_index"].to_numpy(dtype=np.int16),
            "month": timestamps.month.to_numpy(dtype=np.int8),
            "day": timestamps.day.to_numpy(dtype=np.int8),
            "hour": timestamps.hour.to_numpy(dtype=np.int8),
            "hour_of_week": (
                timestamps.dayofweek.to_numpy(dtype=np.int16) * 24
                + timestamps.hour.to_numpy(dtype=np.int16)
            ),
        }
    )


def build_split(
    events: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    climatology_lookup: pd.DataFrame,
    weather_defaults: np.ndarray,
    years: list[int],
    negative_ratio: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []
    cell_station_lookup = candidate_cells[["cell_id", "station_index"]]

    for year in years:
        positives = events.loc[
            events["year"] == year,
            ["cell_id", "month", "day", "hour", "hour_of_week"],
        ].merge(
            cell_station_lookup,
            on="cell_id",
            how="left",
            validate="many_to_one",
        )
        history = events.loc[events["year"] < year]
        context = build_context(history, candidate_cells)
        exact_weather = load_year_weather(year)

        positives, pos_exact_hit_rate = attach_weather_features(
            positives,
            exact_weather=exact_weather,
            climatology_lookup=climatology_lookup,
            weather_defaults=weather_defaults,
        )
        x_pos = build_examples(positives, context)
        y_pos = np.ones(len(x_pos), dtype=np.int8)

        neg_count = len(x_pos) * negative_ratio
        negatives = sample_negatives(
            year=year,
            count=neg_count,
            candidate_cells=candidate_cells,
            rng=rng,
        )
        negatives, neg_exact_hit_rate = attach_weather_features(
            negatives,
            exact_weather=exact_weather,
            climatology_lookup=climatology_lookup,
            weather_defaults=weather_defaults,
        )
        x_neg = build_examples(negatives, context)
        y_neg = np.zeros(len(x_neg), dtype=np.int8)

        feature_chunks.extend([x_pos, x_neg])
        label_chunks.extend([y_pos, y_neg])
        print(
            f"year={year} positives={len(x_pos)} negatives={len(x_neg)} "
            f"weather_exact_pos={pos_exact_hit_rate:.3f} weather_exact_neg={neg_exact_hit_rate:.3f}"
        )

    return np.vstack(feature_chunks), np.concatenate(label_chunks)


def build_weather_cube(
    climatology: pd.DataFrame,
    weather_defaults: np.ndarray,
) -> np.ndarray:
    station_count = int(climatology["station_index"].max()) + 1
    feature_count = len(WEATHER_FEATURE_NAMES)
    cube = np.full((station_count, 12, 24 * 7, feature_count), np.nan, dtype=np.float32)

    station_idx = climatology["station_index"].to_numpy(dtype=np.int32)
    month_idx = climatology["month"].to_numpy(dtype=np.int32) - 1
    hour_idx = climatology["hour_of_week"].to_numpy(dtype=np.int32)
    for feature_idx, column in enumerate(WEATHER_FEATURE_NAMES):
        cube[station_idx, month_idx, hour_idx, feature_idx] = climatology[column].to_numpy(
            dtype=np.float32
        )

    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # Stations with sparse climatology produce all-NaN slices; the means are
        # simply unused there (the next fallback fills them), so the resulting
        # "Mean of empty slice" RuntimeWarnings are expected noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        station_month_means = np.nanmean(cube, axis=2, keepdims=True)
        cube = np.where(np.isnan(cube), station_month_means, cube)
        station_means = np.nanmean(cube, axis=(1, 2), keepdims=True)
        cube = np.where(np.isnan(cube), station_means, cube)

    cube = np.where(
        np.isnan(cube),
        weather_defaults.reshape(1, 1, 1, feature_count),
        cube,
    )
    return cube.astype(np.float32)


def estimate_risk_quantiles(
    model: HistGradientBoostingClassifier,
    candidate_cells: pd.DataFrame,
    context: FeatureContext,
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
    month: int,
) -> list[float]:
    candidate_ids = candidate_cells["cell_id"].to_numpy(dtype=object)
    candidate_lats = candidate_cells["center_lat"].to_numpy(dtype=np.float32)
    candidate_lons = candidate_cells["center_lon"].to_numpy(dtype=np.float32)
    candidate_station_indices = candidate_cells["station_index"].to_numpy(dtype=np.int16)
    cell_total_counts = np.array(
        [context.total_by_cell.get(cell, 0) for cell in candidate_ids],
        dtype=np.float32,
    )
    predictions = []
    for frame_idx in range(24 * 7):
        same_hour = np.array(
            [context.hour_by_cell.get((cell, frame_idx), 0) for cell in candidate_ids],
            dtype=np.float32,
        )
        month_values = np.full(len(candidate_ids), month, dtype=np.int8)
        frame_values = np.full(len(candidate_ids), frame_idx, dtype=np.int16)
        weather = lookup_weather_climatology(
            weather_cube=weather_cube,
            station_indices=candidate_station_indices,
            months=month_values,
            hour_of_week=frame_values,
            weather_defaults=weather_defaults,
        )
        features = build_feature_matrix(
            latitudes=candidate_lats,
            longitudes=candidate_lons,
            hour_of_week=frame_values,
            months=month_values,
            totals=cell_total_counts,
            same_hour=same_hour,
            temp_c=weather[:, 0],
            dewpoint_c=weather[:, 1],
            relative_humidity_pct=weather[:, 2],
            wind_speed_mps=weather[:, 3],
            wet_hour=weather[:, 4],
        )
        predictions.append(model.predict_proba(features)[:, 1].astype(np.float32))
    all_predictions = np.concatenate(predictions)
    return np.quantile(all_predictions, [0.50, 0.80, 0.95]).astype(float).tolist()


def build_bundle(
    model: HistGradientBoostingClassifier,
    events: pd.DataFrame,
    candidate_cells: pd.DataFrame,
    climatology: pd.DataFrame,
    weather_defaults: np.ndarray,
    metrics: dict[str, float],
    train_years: list[int],
    eval_years: list[int],
) -> dict[str, object]:
    full_context = build_context(events, candidate_cells)
    candidate_ids = candidate_cells["cell_id"].tolist()
    cell_total_counts = np.array(
        [full_context.total_by_cell.get(cell, 0) for cell in candidate_ids],
        dtype=np.float32,
    )
    cell_hour_counts = np.zeros((len(candidate_ids), 24 * 7), dtype=np.float32)
    for idx, cell in enumerate(candidate_ids):
        for frame_idx in range(24 * 7):
            cell_hour_counts[idx, frame_idx] = full_context.hour_by_cell.get(
                (cell, frame_idx),
                0,
            )

    weather_cube = build_weather_cube(climatology, weather_defaults)
    quantiles = estimate_risk_quantiles(
        model=model,
        candidate_cells=candidate_cells,
        context=full_context,
        weather_cube=weather_cube,
        weather_defaults=weather_defaults,
        month=current_month(),
    )

    return {
        "model": model,
        "model_version": "0.2.0",
        "resolution": H3_RESOLUTION,
        "train_years": train_years,
        "eval_years": eval_years,
        "negative_ratio": NEGATIVE_RATIO,
        "feature_names": [
            "lat",
            "lon",
            "hour_sin",
            "hour_cos",
            "dow_sin",
            "dow_cos",
            "month_sin",
            "month_cos",
            "log_total_events",
            "log_same_hour_events",
            "same_hour_share",
            *WEATHER_FEATURE_NAMES,
        ],
        "weather_feature_names": WEATHER_FEATURE_NAMES,
        "candidate_cells": candidate_ids,
        "candidate_lats": candidate_cells["center_lat"].to_numpy(dtype=np.float32),
        "candidate_lons": candidate_cells["center_lon"].to_numpy(dtype=np.float32),
        "candidate_station_indices": candidate_cells["station_index"].to_numpy(
            dtype=np.int16
        ),
        "cell_total_counts": cell_total_counts,
        "cell_hour_counts": cell_hour_counts,
        "weather_climatology": weather_cube,
        "weather_defaults": weather_defaults.astype(np.float32),
        "metrics": metrics,
        "risk_quantiles": quantiles,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--train-years",
        nargs="+",
        type=int,
        default=DEFAULT_TRAIN_YEARS,
    )
    parser.add_argument(
        "--eval-years",
        nargs="+",
        type=int,
        default=[DEFAULT_EVAL_YEAR],
    )
    parser.add_argument("--negative-ratio", type=int, default=NEGATIVE_RATIO)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    events, candidate_cells, climatology = load_inputs()
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
    joblib.dump(bundle, MODEL_BUNDLE_PATH)
    print(f"wrote {MODEL_BUNDLE_PATH}")


if __name__ == "__main__":
    main()
