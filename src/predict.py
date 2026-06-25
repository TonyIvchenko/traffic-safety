"""Model-bundle loading and per-point traffic-safety risk prediction.

These scorers were extracted from ``main.py`` so the public ``/v1`` API layer can
reuse them without importing the full web application (which would create a
circular import). ``main.py`` re-imports the public names from here, so its
behaviour is unchanged.
"""

from __future__ import annotations

from pathlib import Path
import sys

import h3
import joblib
import numpy as np

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from live_weather import fetch_live_weather
from model_support import build_feature_matrix, lookup_weather_climatology

MODEL_PATH = REPO_DIR / "models" / "traffic_safety.joblib"


def _load_joblib_bundle(path: Path) -> dict[str, object]:
    if not path.exists():
        return {}
    loaded = joblib.load(path)
    return loaded if isinstance(loaded, dict) else {}


MODEL_BUNDLE = _load_joblib_bundle(MODEL_PATH)
MODEL_VERSION = str(MODEL_BUNDLE.get("model_version", "missing"))
CELL_INDEX = {
    str(cell): idx for idx, cell in enumerate(MODEL_BUNDLE.get("candidate_cells", []))
}


def _risk_level(probability: float) -> str:
    quantiles = MODEL_BUNDLE.get("risk_quantiles")
    if (
        isinstance(quantiles, list)
        and len(quantiles) >= 3
        and all(isinstance(value, (int, float)) for value in quantiles[:3])
    ):
        low_cut, mid_cut, high_cut = (float(value) for value in quantiles[:3])
        if probability < low_cut:
            return "low"
        if probability < mid_cut:
            return "moderate"
        if probability < high_cut:
            return "high"
        return "extreme"
    if probability < 0.10:
        return "low"
    if probability < 0.25:
        return "moderate"
    if probability < 0.45:
        return "high"
    return "extreme"


def _bundle_array(key: str, dtype: np.dtype | type, default: object | None = None) -> np.ndarray:
    value = MODEL_BUNDLE.get(key, default)
    if value is None:
        raise RuntimeError(f"traffic safety bundle is missing '{key}'")
    return np.asarray(value, dtype=dtype)


def _coverage_log_max() -> float:
    if "cell_total_counts" not in MODEL_BUNDLE:
        return 0.0
    counts = _bundle_array("cell_total_counts", np.float32)
    return float(np.log1p(float(counts.max()))) if counts.size else 0.0


_COVERAGE_LOG_MAX = _coverage_log_max()


def coverage_confidence(historical_cell_events: float) -> float:
    """A 0-1 confidence proxy from how much crash history backs a cell.

    Mirrors the overlay confidence grid: log1p(events) normalized by the busiest
    cell in the bundle. Sparse/out-of-coverage cells return ~0.
    """
    if _COVERAGE_LOG_MAX <= 0.0:
        return 0.0
    return float(min(1.0, np.log1p(max(0.0, float(historical_cell_events))) / _COVERAGE_LOG_MAX))


# Feature-row column groups used for marginal-attribution explanations. Indices
# match build_feature_matrix / _prediction_feature_row.
_EXPLAIN_GROUPS_16 = {
    "crash_history": (8, 9, 10),
    "time_of_day": (2, 3),
    "day_of_week": (4, 5),
    "season": (6, 7),
    "temperature": (11, 12, 13),
    "wind": (14,),
    "precipitation": (15,),
}
_EXPLAIN_GROUPS_11 = {
    "crash_history": (8, 9, 10),
    "time_of_day": (2, 3),
    "day_of_week": (4, 5),
    "season": (6, 7),
}


def _history_baseline() -> tuple[float, float, float]:
    if "cell_total_counts" not in MODEL_BUNDLE or "cell_hour_counts" not in MODEL_BUNDLE:
        return (0.0, 0.0, 0.0)
    totals = _bundle_array("cell_total_counts", np.float32)
    hours = _bundle_array("cell_hour_counts", np.float32)
    if not totals.size or not hours.size:
        return (0.0, 0.0, 0.0)
    mean_total = float(totals.mean())
    return (
        float(np.log1p(totals).mean()),
        float(np.log1p(hours).mean()),
        float(hours.mean() / max(mean_total, 1.0)),
    )


_HISTORY_BASELINE = _history_baseline()


def _default_weather_for_index(idx: int, hour_of_week: int, month: int) -> np.ndarray:
    if "weather_climatology" not in MODEL_BUNDLE or "candidate_station_indices" not in MODEL_BUNDLE:
        return np.zeros(5, dtype=np.float32)

    weather = lookup_weather_climatology(
        weather_cube=_bundle_array("weather_climatology", np.float32),
        station_indices=int(_bundle_array("candidate_station_indices", np.int16)[idx]),
        months=int(month),
        hour_of_week=int(hour_of_week),
        weather_defaults=_bundle_array("weather_defaults", np.float32, default=np.zeros(5)),
    )
    return weather[0]


def _prediction_feature_row(
    *,
    model: object,
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
    prior_total: float,
    prior_same_hour: float,
    temp_c: float,
    dewpoint_c: float,
    relative_humidity_pct: float,
    wind_speed_mps: float,
    wet_hour: float,
) -> np.ndarray:
    feature_count = int(getattr(model, "n_features_in_", 16))
    if feature_count <= 11:
        hour_angle = 2.0 * np.pi * float(hour) / 24.0
        dow_angle = 2.0 * np.pi * float(day_of_week - 1) / 7.0
        month_angle = 2.0 * np.pi * float(month) / 12.0
        return np.array(
            [
                [
                    float(lat),
                    float(lon),
                    float(np.sin(hour_angle)),
                    float(np.cos(hour_angle)),
                    float(np.sin(dow_angle)),
                    float(np.cos(dow_angle)),
                    float(np.sin(month_angle)),
                    float(np.cos(month_angle)),
                    float(np.log1p(prior_total)),
                    float(np.log1p(prior_same_hour)),
                    float(prior_same_hour / max(prior_total, 1.0)),
                ]
            ],
            dtype=np.float32,
        )

    hour_of_week = (day_of_week - 1) * 24 + hour
    return build_feature_matrix(
        latitudes=np.array([lat], dtype=np.float32),
        longitudes=np.array([lon], dtype=np.float32),
        hour_of_week=np.array([hour_of_week], dtype=np.int16),
        months=np.array([month], dtype=np.int8),
        totals=np.array([prior_total], dtype=np.float32),
        same_hour=np.array([prior_same_hour], dtype=np.float32),
        temp_c=np.array([temp_c], dtype=np.float32),
        dewpoint_c=np.array([dewpoint_c], dtype=np.float32),
        relative_humidity_pct=np.array([relative_humidity_pct], dtype=np.float32),
        wind_speed_mps=np.array([wind_speed_mps], dtype=np.float32),
        wet_hour=np.array([wet_hour], dtype=np.float32),
    )


def _predict_with_weather(
    *,
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
    temp_c: float,
    dewpoint_c: float,
    relative_humidity_pct: float,
    wind_speed_mps: float,
    wet_hour: float,
    weather_source: str,
    weather_summary: str = "",
    provider: str | None = None,
    provider_label: str | None = None,
    timestamp_local: str | None = None,
    forecast_hours: int | None = None,
) -> dict[str, object]:
    if not MODEL_BUNDLE:
        raise RuntimeError(f"traffic safety model is unavailable; expected {MODEL_PATH}")

    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))

    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    cell_id = h3.latlng_to_cell(float(lat), float(lon), resolution)
    idx = CELL_INDEX.get(cell_id)
    if idx is None:
        return {
            "model_version": MODEL_VERSION,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "local_day_of_week": day_of_week,
            "local_hour": hour,
            "month": month,
            "historical_cell_events": 0,
            "historical_same_hour_events": 0,
            "in_coverage": False,
            "confidence": 0.0,
            "risk_score": 0.0,
            "risk_level": "low",
            "weather_source": weather_source,
            "weather": {
                "temp_c": float(temp_c),
                "dewpoint_c": float(dewpoint_c),
                "relative_humidity_pct": float(relative_humidity_pct),
                "wind_speed_mps": float(wind_speed_mps),
                "wet_hour": float(wet_hour),
                "summary": weather_summary,
            },
            "live_provider": provider,
            "live_provider_label": provider_label,
            "target_timestamp_local": timestamp_local,
            "forecast_hours": forecast_hours,
        }

    model = MODEL_BUNDLE["model"]
    candidate_lats = _bundle_array("candidate_lats", np.float32)
    candidate_lons = _bundle_array("candidate_lons", np.float32)
    cell_total_counts = _bundle_array("cell_total_counts", np.float32)
    cell_hour_counts = _bundle_array("cell_hour_counts", np.float32)

    hour_of_week = (day_of_week - 1) * 24 + hour
    prior_total = float(cell_total_counts[idx])
    prior_same_hour = float(cell_hour_counts[idx, hour_of_week])
    features = _prediction_feature_row(
        model=model,
        lat=float(candidate_lats[idx]),
        lon=float(candidate_lons[idx]),
        day_of_week=day_of_week,
        hour=hour,
        month=month,
        prior_total=prior_total,
        prior_same_hour=prior_same_hour,
        temp_c=float(temp_c),
        dewpoint_c=float(dewpoint_c),
        relative_humidity_pct=float(relative_humidity_pct),
        wind_speed_mps=float(wind_speed_mps),
        wet_hour=float(wet_hour),
    )
    probability = float(model.predict_proba(features)[0, 1])
    probability = max(0.0, min(1.0, probability))
    return {
        "model_version": MODEL_VERSION,
        "cell_id": cell_id,
        "lat": float(lat),
        "lon": float(lon),
        "local_day_of_week": day_of_week,
        "local_hour": hour,
        "month": month,
        "historical_cell_events": int(prior_total),
        "historical_same_hour_events": int(prior_same_hour),
        "in_coverage": True,
        "confidence": coverage_confidence(prior_total),
        "risk_score": probability,
        "risk_level": _risk_level(probability),
        "weather_source": weather_source,
        "weather": {
            "temp_c": float(temp_c),
            "dewpoint_c": float(dewpoint_c),
            "relative_humidity_pct": float(relative_humidity_pct),
            "wind_speed_mps": float(wind_speed_mps),
            "wet_hour": float(wet_hour),
            "summary": weather_summary,
        },
        "live_provider": provider,
        "live_provider_label": provider_label,
        "target_timestamp_local": timestamp_local,
        "forecast_hours": forecast_hours,
    }


def predict_traffic_safety(
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
) -> dict[str, object]:
    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))
    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    cell_id = h3.latlng_to_cell(float(lat), float(lon), resolution)
    idx = CELL_INDEX.get(cell_id)

    hour_of_week = (day_of_week - 1) * 24 + hour
    if idx is None:
        default_weather = np.zeros(5, dtype=np.float32)
    else:
        default_weather = _default_weather_for_index(idx=idx, hour_of_week=hour_of_week, month=month)
    return _predict_with_weather(
        lat=float(lat),
        lon=float(lon),
        day_of_week=day_of_week,
        hour=hour,
        month=month,
        temp_c=float(default_weather[0]),
        dewpoint_c=float(default_weather[1]),
        relative_humidity_pct=float(default_weather[2]),
        wind_speed_mps=float(default_weather[3]),
        wet_hour=float(default_weather[4]),
        weather_source="climatology",
        weather_summary="station climatology",
    )


def predict_traffic_safety_live(
    lat: float,
    lon: float,
    forecast_hours: int = 0,
    provider: str = "auto",
) -> dict[str, object]:
    snapshot = fetch_live_weather(
        lat=float(lat),
        lon=float(lon),
        forecast_hours=int(forecast_hours),
        provider=provider,
    )
    timestamp_local = snapshot.timestamp_local
    return _predict_with_weather(
        lat=float(lat),
        lon=float(lon),
        day_of_week=timestamp_local.weekday() + 1,
        hour=timestamp_local.hour,
        month=timestamp_local.month,
        temp_c=snapshot.temp_c,
        dewpoint_c=snapshot.dewpoint_c,
        relative_humidity_pct=snapshot.relative_humidity_pct,
        wind_speed_mps=snapshot.wind_speed_mps,
        wet_hour=snapshot.wet_hour,
        weather_source=f"live_{snapshot.observed_or_forecast}",
        weather_summary=snapshot.summary,
        provider=snapshot.provider,
        provider_label=snapshot.provider_label,
        timestamp_local=timestamp_local.isoformat(),
        forecast_hours=int(snapshot.forecast_hours),
    )


def explain_prediction(
    *,
    lat: float,
    lon: float,
    day_of_week: int,
    hour: int,
    month: int,
    temp_c: float,
    dewpoint_c: float,
    relative_humidity_pct: float,
    wind_speed_mps: float,
    wet_hour: float,
) -> dict[str, object]:
    """Explain a point prediction via marginal attribution.

    Each feature group is neutralized to a reference value (climatology for
    weather; the bundle-average for crash history; the cyclic mean for
    time/day/season) and the resulting drop in modeled risk is that group's
    contribution. Reuses the live model — no extra training or SHAP dependency.
    """
    empty: dict[str, object] = {"baseline_risk": 0.0, "risk_score": 0.0, "factors": []}
    if not MODEL_BUNDLE:
        return empty

    day_of_week = max(1, min(7, int(day_of_week)))
    hour = max(0, min(23, int(hour)))
    month = max(1, min(12, int(month)))
    resolution = int(MODEL_BUNDLE.get("resolution", 5))
    idx = CELL_INDEX.get(h3.latlng_to_cell(float(lat), float(lon), resolution))
    if idx is None:
        return empty

    model = MODEL_BUNDLE["model"]
    candidate_lats = _bundle_array("candidate_lats", np.float32)
    candidate_lons = _bundle_array("candidate_lons", np.float32)
    cell_total_counts = _bundle_array("cell_total_counts", np.float32)
    cell_hour_counts = _bundle_array("cell_hour_counts", np.float32)

    hour_of_week = (day_of_week - 1) * 24 + hour
    prior_total = float(cell_total_counts[idx])
    prior_same_hour = float(cell_hour_counts[idx, hour_of_week])
    row = _prediction_feature_row(
        model=model,
        lat=float(candidate_lats[idx]),
        lon=float(candidate_lons[idx]),
        day_of_week=day_of_week,
        hour=hour,
        month=month,
        prior_total=prior_total,
        prior_same_hour=prior_same_hour,
        temp_c=float(temp_c),
        dewpoint_c=float(dewpoint_c),
        relative_humidity_pct=float(relative_humidity_pct),
        wind_speed_mps=float(wind_speed_mps),
        wet_hour=float(wet_hour),
    )
    feature_count = int(row.shape[1])
    groups = _EXPLAIN_GROUPS_16 if feature_count >= 16 else _EXPLAIN_GROUPS_11

    clim = _default_weather_for_index(idx=idx, hour_of_week=hour_of_week, month=month)
    neutral = {
        2: 0.0, 3: 0.0, 4: 0.0, 5: 0.0, 6: 0.0, 7: 0.0,
        8: _HISTORY_BASELINE[0], 9: _HISTORY_BASELINE[1], 10: _HISTORY_BASELINE[2],
    }
    if feature_count >= 16:
        neutral.update({
            11: float(clim[0]),
            12: float(clim[1]),
            13: float(clim[2]) / 100.0,
            14: float(clim[3]),
            15: float(clim[4]),
        })

    def _proba(features: np.ndarray) -> float:
        return float(min(1.0, max(0.0, model.predict_proba(features)[0, 1])))

    actual_risk = _proba(row)
    baseline_row = row.copy()
    for indices in groups.values():
        for column in indices:
            baseline_row[0, column] = neutral[column]
    baseline_risk = _proba(baseline_row)

    factors: list[dict[str, object]] = []
    for name, indices in groups.items():
        ablated = row.copy()
        for column in indices:
            ablated[0, column] = neutral[column]
        contribution = actual_risk - _proba(ablated)
        factors.append(
            {
                "factor": name,
                "contribution": round(contribution, 4),
                "direction": (
                    "increases" if contribution > 0.001
                    else "decreases" if contribution < -0.001
                    else "neutral"
                ),
            }
        )
    factors.sort(key=lambda factor: abs(factor["contribution"]), reverse=True)
    return {
        "baseline_risk": round(baseline_risk, 4),
        "risk_score": round(actual_risk, 4),
        "factors": factors,
    }


def explain_for_result(result: dict) -> dict[str, object]:
    """Explain a prediction dict produced by the predictors above."""
    weather = result.get("weather", {}) or {}
    return explain_prediction(
        lat=float(result["lat"]),
        lon=float(result["lon"]),
        day_of_week=int(result["local_day_of_week"]),
        hour=int(result["local_hour"]),
        month=int(result["month"]),
        temp_c=float(weather.get("temp_c", 0.0)),
        dewpoint_c=float(weather.get("dewpoint_c", 0.0)),
        relative_humidity_pct=float(weather.get("relative_humidity_pct", 0.0)),
        wind_speed_mps=float(weather.get("wind_speed_mps", 0.0)),
        wet_hour=float(weather.get("wet_hour", 0.0)),
    )
