"""Vulnerable-road-user (pedestrian & cyclist) risk layer.

Serves the VRU model (``models/traffic_safety_vru.joblib``, trained by
``scripts/train_vru_model.py``) as a first-class layer alongside the all-crash
model. The VRU bundle is self-contained and shares the main model's 16-feature
schema, so scoring reuses :func:`predict._prediction_feature_row` against the VRU
bundle's own candidate cells and weather climatology.

Climatological only (how risky a place/time is for VRUs on a typical day); a point
outside the VRU candidate cells, or a missing bundle, degrades to an
out-of-coverage reading (risk 0), so the API always answers.

The bundle path is ``TRAFFIC_SAFETY_VRU_MODEL_PATH`` or
``models/traffic_safety_vru.joblib``.
"""

from __future__ import annotations

import functools
import os
from pathlib import Path

import h3
import numpy as np

from model_support import lookup_weather_climatology
import predict

REPO_DIR = Path(__file__).resolve().parents[1]
VRU_MODEL_PATH = Path(
    os.environ.get("TRAFFIC_SAFETY_VRU_MODEL_PATH") or REPO_DIR / "models" / "traffic_safety_vru.joblib"
)
WEEKLY_FRAMES = 168


class VRULayer:
    """Loaded VRU model with climatological point/weekly scoring."""

    def __init__(self, bundle: dict | None) -> None:
        self.bundle = bundle or {}
        self.model = self.bundle.get("model")
        self.model_version = str(self.bundle.get("model_version", "missing"))
        self.target = str(self.bundle.get("target", "pedestrian_or_cyclist_crash"))
        self.resolution = int(self.bundle.get("resolution", 5))
        self.cell_index = {
            str(cell): idx for idx, cell in enumerate(self.bundle.get("candidate_cells", []) or [])
        }
        counts = self.bundle.get("cell_total_counts")
        self._coverage_log_max = (
            float(np.log1p(float(np.max(counts)))) if counts is not None and len(counts) else 0.0
        )

    @property
    def ready(self) -> bool:
        return self.model is not None and bool(self.cell_index)

    def _array(self, key: str, dtype) -> np.ndarray:
        return np.asarray(self.bundle[key], dtype=dtype)

    def _climatological_weather(self, idx: int, hour_of_week: int, month: int) -> np.ndarray:
        if "weather_climatology" not in self.bundle or "candidate_station_indices" not in self.bundle:
            return np.zeros(5, dtype=np.float32)
        weather = lookup_weather_climatology(
            weather_cube=self._array("weather_climatology", np.float32),
            station_indices=int(self._array("candidate_station_indices", np.int16)[idx]),
            months=int(month),
            hour_of_week=int(hour_of_week),
            weather_defaults=self._array("weather_defaults", np.float32),
        )
        return weather[0]

    def _base(self, lat, lon, dow, hour, month, *, cell_id, in_coverage, score, confidence, prior_total, prior_same):
        return {
            "layer": "vru",
            "target": self.target,
            "model_version": self.model_version,
            "cell_id": cell_id,
            "lat": float(lat),
            "lon": float(lon),
            "local_day_of_week": dow,
            "local_hour": hour,
            "month": month,
            "historical_cell_events": int(prior_total),
            "historical_same_hour_events": int(prior_same),
            "in_coverage": in_coverage,
            "confidence": round(float(confidence), 4),
            "risk_score": round(float(score), 6),
            "risk_level": predict.risk_level_for(score, self.bundle.get("risk_quantiles")),
            "weather_source": "climatology",
        }

    def _probability(self, idx: int, dow: int, hour: int, month: int) -> tuple[float, float, float]:
        totals = self._array("cell_total_counts", np.float32)
        hours = self._array("cell_hour_counts", np.float32)
        lats = self._array("candidate_lats", np.float32)
        lons = self._array("candidate_lons", np.float32)
        hour_of_week = (dow - 1) * 24 + hour
        prior_total = float(totals[idx])
        prior_same = float(hours[idx, hour_of_week])
        weather = self._climatological_weather(idx, hour_of_week, month)
        features = predict._prediction_feature_row(
            model=self.model,
            lat=float(lats[idx]),
            lon=float(lons[idx]),
            day_of_week=dow,
            hour=hour,
            month=month,
            prior_total=prior_total,
            prior_same_hour=prior_same,
            temp_c=float(weather[0]),
            dewpoint_c=float(weather[1]),
            relative_humidity_pct=float(weather[2]),
            wind_speed_mps=float(weather[3]),
            wet_hour=float(weather[4]),
        )
        probability = float(self.model.predict_proba(features)[0, 1])
        return max(0.0, min(1.0, probability)), prior_total, prior_same

    def _cell(self, lat: float, lon: float):
        try:
            cell_id = h3.latlng_to_cell(float(lat), float(lon), self.resolution)
        except (ValueError, TypeError):
            return None, None
        return cell_id, self.cell_index.get(cell_id)

    def score_point(self, lat, lon, *, day_of_week: int, hour: int, month: int) -> dict:
        dow = max(1, min(7, int(day_of_week)))
        hour = max(0, min(23, int(hour)))
        month = max(1, min(12, int(month)))
        cell_id, idx = self._cell(lat, lon) if self.ready else (None, None)
        if idx is None:
            return self._base(
                lat, lon, dow, hour, month, cell_id=cell_id, in_coverage=False,
                score=0.0, confidence=0.0, prior_total=0, prior_same=0,
            )
        probability, prior_total, prior_same = self._probability(idx, dow, hour, month)
        confidence = predict.coverage_confidence_for(prior_total, self._coverage_log_max)
        return self._base(
            lat, lon, dow, hour, month, cell_id=cell_id, in_coverage=True,
            score=probability, confidence=confidence, prior_total=prior_total, prior_same=prior_same,
        )

    def weekly_profile(self, lat, lon, *, month: int = 1) -> list[float]:
        """VRU risk for every hour-of-week (0-167) at a point; zeros if uncovered."""
        month = max(1, min(12, int(month)))
        cell_id, idx = self._cell(lat, lon) if self.ready else (None, None)
        if idx is None:
            return [0.0] * WEEKLY_FRAMES
        profile = []
        for frame in range(WEEKLY_FRAMES):
            dow = frame // 24 + 1
            hour = frame % 24
            probability, _, _ = self._probability(idx, dow, hour, month)
            profile.append(round(probability, 6))
        return profile


@functools.lru_cache(maxsize=2)
def _load_layer_cached(path_str: str) -> VRULayer:
    return VRULayer(predict._load_joblib_bundle(Path(path_str)))


def load_vru_layer(path=None) -> VRULayer:
    """Process-wide VRU layer; ``TRAFFIC_SAFETY_VRU_MODEL_PATH`` overrides the path."""
    resolved = Path(path) if path is not None else VRU_MODEL_PATH
    return _load_layer_cached(str(resolved))
