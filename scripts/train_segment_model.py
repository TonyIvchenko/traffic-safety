from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import average_precision_score, roc_auc_score
from sklearn.neighbors import BallTree

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from model_support import lookup_weather_climatology
from segment_model_support import (
    DEFAULT_MTFCC_VALUES,
    DEFAULT_RTTYP_VALUES,
    build_segment_feature_matrix,
    build_static_segment_frame,
)

from common import (
    ACTIVE_ROAD_SEGMENTS_PATH,
    REPRESENTATIVE_STATIONS_PATH,
    ROAD_SEGMENTS_PATH,
    RANDOM_SEED,
    SEGMENT_EVENTS_PATH,
    SEGMENT_MODEL_BUNDLE_PATH,
    WEATHER_CLIMATOLOGY_PATH,
    ensure_dirs,
)


WEATHER_COLUMNS = ["temp_c", "relative_humidity_pct", "wind_speed_mps", "wet_hour"]


@dataclass
class SegmentContext:
    total_by_idx: dict[int, int]
    hour_by_idx: dict[tuple[int, int], int]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--train-years", nargs="+", type=int, default=[2020, 2021, 2022])
    parser.add_argument("--eval-years", nargs="+", type=int, default=[2023])
    parser.add_argument("--negative-ratio", type=int, default=1)
    parser.add_argument("--max-positives-per-year", type=int, default=250000)
    return parser.parse_args()


def load_inputs() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    required = [
        ROAD_SEGMENTS_PATH,
        SEGMENT_EVENTS_PATH,
        REPRESENTATIVE_STATIONS_PATH,
        WEATHER_CLIMATOLOGY_PATH,
    ]
    for path in required:
        if not path.exists():
            raise FileNotFoundError(f"missing {path}")

    roads = pd.read_parquet(ROAD_SEGMENTS_PATH).sort_values("segment_id").reset_index(drop=True)
    events = pd.read_parquet(SEGMENT_EVENTS_PATH)
    representative = pd.read_csv(REPRESENTATIVE_STATIONS_PATH)
    climatology = pd.read_csv(WEATHER_CLIMATOLOGY_PATH)
    return roads, events, representative, climatology


def build_segment_station_indices(
    roads: pd.DataFrame,
    representative: pd.DataFrame,
) -> np.ndarray:
    rep_coords = np.radians(representative[["LAT", "LON"]].to_numpy(dtype=np.float64))
    tree = BallTree(rep_coords, metric="haversine")
    road_coords = np.radians(roads[["center_lat", "center_lon"]].to_numpy(dtype=np.float64))
    _, idx = tree.query(road_coords, k=1)
    return representative.iloc[idx[:, 0]]["station_index"].to_numpy(dtype=np.int16)


def build_weather_cube(climatology: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    station_count = int(climatology["station_index"].max()) + 1
    cube = np.full((station_count, 12, 168, len(WEATHER_COLUMNS)), np.nan, dtype=np.float32)
    station_idx = climatology["station_index"].to_numpy(dtype=np.int32)
    month_idx = climatology["month"].to_numpy(dtype=np.int32) - 1
    hour_idx = climatology["hour_of_week"].to_numpy(dtype=np.int32)
    for feature_idx, column in enumerate(WEATHER_COLUMNS):
        cube[station_idx, month_idx, hour_idx, feature_idx] = climatology[column].to_numpy(
            dtype=np.float32
        )

    defaults = climatology[WEATHER_COLUMNS].mean().to_numpy(dtype=np.float32)
    defaults = np.nan_to_num(defaults, nan=0.0, posinf=0.0, neginf=0.0)
    with np.errstate(invalid="ignore"), warnings.catch_warnings():
        # Stations with sparse climatology produce all-NaN slices; those means
        # are unused (the next fallback fills them), so the resulting "Mean of
        # empty slice" RuntimeWarnings are expected noise.
        warnings.simplefilter("ignore", category=RuntimeWarning)
        station_month_means = np.nanmean(cube, axis=2, keepdims=True)
        cube = np.where(np.isnan(cube), station_month_means, cube)
        station_means = np.nanmean(cube, axis=(1, 2), keepdims=True)
        cube = np.where(np.isnan(cube), station_means, cube)
    cube = np.where(np.isnan(cube), defaults.reshape(1, 1, 1, len(WEATHER_COLUMNS)), cube)
    return cube.astype(np.float32), defaults.astype(np.float32)


def build_context(history: pd.DataFrame) -> SegmentContext:
    total_by_idx = history.groupby("segment_idx").size().astype(int).to_dict()
    hour_by_idx = history.groupby(["segment_idx", "hour_of_week"]).size().astype(int).to_dict()
    return SegmentContext(total_by_idx=total_by_idx, hour_by_idx=hour_by_idx)


def attach_positive_weather(
    positives: pd.DataFrame,
    station_indices: np.ndarray,
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
) -> pd.DataFrame:
    weather = lookup_weather_climatology(
        weather_cube=weather_cube,
        station_indices=station_indices[positives["segment_idx"].to_numpy(dtype=np.int32)],
        months=positives["month"].to_numpy(dtype=np.int16),
        hour_of_week=positives["hour_of_week"].to_numpy(dtype=np.int16),
        weather_defaults=weather_defaults,
    )
    for feature_idx, column in enumerate(WEATHER_COLUMNS):
        event_values = positives[column].to_numpy(dtype=np.float32)
        fallback = weather[:, feature_idx]
        mask = np.isnan(event_values)
        event_values[mask] = fallback[mask]
        positives[column] = event_values.astype(np.float32)
    return positives


def sample_negatives(
    roads: pd.DataFrame,
    count: int,
    year: int,
    rng: np.random.Generator,
) -> pd.DataFrame:
    sampled_idx = rng.integers(0, len(roads), size=count)
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
            "segment_idx": sampled_idx.astype(np.int32),
            "month": timestamps.month.to_numpy(dtype=np.int8),
            "hour_of_week": (
                timestamps.dayofweek.to_numpy(dtype=np.int16) * 24
                + timestamps.hour.to_numpy(dtype=np.int16)
            ),
        }
    )


def build_design_matrix(
    rows: pd.DataFrame,
    static_features: np.ndarray,
    context: SegmentContext,
) -> np.ndarray:
    segment_idx = rows["segment_idx"].to_numpy(dtype=np.int32)
    totals = np.array([context.total_by_idx.get(int(idx), 0) for idx in segment_idx], dtype=np.float32)
    same_hour = np.array(
        [
            context.hour_by_idx.get((int(idx), int(hour_of_week)), 0)
            for idx, hour_of_week in zip(
                segment_idx,
                rows["hour_of_week"].to_numpy(dtype=np.int16),
                strict=False,
            )
        ],
        dtype=np.float32,
    )
    return build_segment_feature_matrix(
        static_features=static_features[segment_idx],
        hour_of_week=rows["hour_of_week"].to_numpy(dtype=np.int16),
        months=rows["month"].to_numpy(dtype=np.int8),
        totals=totals,
        same_hour=same_hour,
        temp_c=rows["temp_c"].to_numpy(dtype=np.float32),
        relative_humidity_pct=rows["relative_humidity_pct"].to_numpy(dtype=np.float32),
        wind_speed_mps=rows["wind_speed_mps"].to_numpy(dtype=np.float32),
        wet_hour=rows["wet_hour"].to_numpy(dtype=np.float32),
    )


def build_split(
    events: pd.DataFrame,
    roads: pd.DataFrame,
    static_features: np.ndarray,
    station_indices: np.ndarray,
    weather_cube: np.ndarray,
    weather_defaults: np.ndarray,
    years: list[int],
    negative_ratio: int,
    max_positives_per_year: int,
    rng: np.random.Generator,
) -> tuple[np.ndarray, np.ndarray]:
    feature_chunks: list[np.ndarray] = []
    label_chunks: list[np.ndarray] = []

    for year in years:
        positives = events.loc[
            events["year"] == year,
            ["segment_idx", "month", "hour_of_week", *WEATHER_COLUMNS],
        ].copy()
        if len(positives) > max_positives_per_year:
            sample_idx = rng.choice(
                len(positives),
                size=max_positives_per_year,
                replace=False,
            )
            positives = positives.iloc[sample_idx].reset_index(drop=True)
        history = events.loc[events["year"] < year, ["segment_idx", "hour_of_week"]]
        context = build_context(history)
        positives = attach_positive_weather(
            positives,
            station_indices=station_indices,
            weather_cube=weather_cube,
            weather_defaults=weather_defaults,
        )
        x_pos = build_design_matrix(positives, static_features=static_features, context=context)
        y_pos = np.ones(len(x_pos), dtype=np.int8)

        negatives = sample_negatives(
            roads=roads,
            count=len(x_pos) * negative_ratio,
            year=year,
            rng=rng,
        )
        weather = lookup_weather_climatology(
            weather_cube=weather_cube,
            station_indices=station_indices[negatives["segment_idx"].to_numpy(dtype=np.int32)],
            months=negatives["month"].to_numpy(dtype=np.int16),
            hour_of_week=negatives["hour_of_week"].to_numpy(dtype=np.int16),
            weather_defaults=weather_defaults,
        )
        for feature_idx, column in enumerate(WEATHER_COLUMNS):
            negatives[column] = weather[:, feature_idx]
        x_neg = build_design_matrix(negatives, static_features=static_features, context=context)
        y_neg = np.zeros(len(x_neg), dtype=np.int8)

        feature_chunks.extend([x_pos, x_neg])
        label_chunks.extend([y_pos, y_neg])
        print(f"year={year} positives={len(x_pos)} negatives={len(x_neg)}")

    return np.vstack(feature_chunks), np.concatenate(label_chunks)


def main() -> None:
    args = parse_args()
    ensure_dirs()
    rng = np.random.default_rng(RANDOM_SEED)
    roads, events, representative, climatology = load_inputs()

    segment_index = {segment_id: idx for idx, segment_id in enumerate(roads["segment_id"].tolist())}
    events = events.loc[events["segment_id"].isin(segment_index)].copy()
    events["segment_idx"] = events["segment_id"].map(segment_index).astype(np.int32)

    static_frame, static_feature_names = build_static_segment_frame(
        roads,
        rttyp_values=DEFAULT_RTTYP_VALUES,
        mtfcc_values=DEFAULT_MTFCC_VALUES,
    )
    static_features = static_frame.to_numpy(dtype=np.float32)
    station_indices = build_segment_station_indices(roads, representative)
    weather_cube, weather_defaults = build_weather_cube(climatology)

    x_train, y_train = build_split(
        events=events,
        roads=roads,
        static_features=static_features,
        station_indices=station_indices,
        weather_cube=weather_cube,
        weather_defaults=weather_defaults,
        years=args.train_years,
        negative_ratio=args.negative_ratio,
        max_positives_per_year=args.max_positives_per_year,
        rng=rng,
    )
    x_eval, y_eval = build_split(
        events=events,
        roads=roads,
        static_features=static_features,
        station_indices=station_indices,
        weather_cube=weather_cube,
        weather_defaults=weather_defaults,
        years=args.eval_years,
        negative_ratio=args.negative_ratio,
        max_positives_per_year=args.max_positives_per_year,
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

    full_context = build_context(events[["segment_idx", "hour_of_week"]])
    segment_total_counts = np.array(
        [full_context.total_by_idx.get(idx, 0) for idx in range(len(roads))],
        dtype=np.float32,
    )
    segment_hour_counts = np.zeros((len(roads), 168), dtype=np.float32)
    for idx in range(len(roads)):
        for hour_of_week in range(168):
            segment_hour_counts[idx, hour_of_week] = full_context.hour_by_idx.get(
                (idx, hour_of_week),
                0,
            )

    bundle = {
        "model": model,
        "model_version": "segments-0.1.0",
        "segment_ids": roads["segment_id"].tolist(),
        "static_feature_names": static_feature_names,
        "rttyp_values": DEFAULT_RTTYP_VALUES,
        "mtfcc_values": DEFAULT_MTFCC_VALUES,
        "segment_station_indices": station_indices.astype(np.int16),
        "segment_total_counts": segment_total_counts,
        "segment_hour_counts": segment_hour_counts,
        "weather_climatology": weather_cube,
        "weather_defaults": weather_defaults,
        "metrics": metrics,
    }
    joblib.dump(bundle, SEGMENT_MODEL_BUNDLE_PATH)
    active_roads = roads.loc[segment_total_counts > 0].copy()
    active_roads.insert(0, "segment_idx", np.flatnonzero(segment_total_counts > 0).astype(np.int32))
    active_roads.to_parquet(ACTIVE_ROAD_SEGMENTS_PATH, index=False)
    print(f"wrote {SEGMENT_MODEL_BUNDLE_PATH}")
    print(f"wrote {ACTIVE_ROAD_SEGMENTS_PATH}")


if __name__ == "__main__":
    main()
