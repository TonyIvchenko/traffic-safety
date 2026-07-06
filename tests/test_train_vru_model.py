from __future__ import annotations

from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import train_vru_model as tvm


def test_finalize_vru_bundle_tags_layer():
    bundle = {"model": object(), "model_version": "0.2.0", "candidate_cells": []}
    metrics = {"val_roc_auc": 0.8, "val_average_precision": 0.3}
    result = tvm.finalize_vru_bundle(bundle, metrics)

    assert result["layer"] == "vru"
    assert result["target"] == "pedestrian_or_cyclist_crash"
    assert result["model_version"] == tvm.VRU_MODEL_VERSION
    assert result["metrics"] == metrics


def test_reuses_shared_weather_but_vru_events():
    # The VRU trainer must read VRU events/cells but the shared weather mapping.
    from common import (
        CELL_WEATHER_STATIONS_PATH,
        VRU_ACCIDENTS_CLEAN_PATH,
        VRU_CANDIDATE_CELLS_PATH,
        WEATHER_CLIMATOLOGY_PATH,
    )

    assert "vru" in VRU_ACCIDENTS_CLEAN_PATH.name
    assert "vru" in VRU_CANDIDATE_CELLS_PATH.name
    assert "vru" not in CELL_WEATHER_STATIONS_PATH.name
    assert "vru" not in WEATHER_CLIMATOLOGY_PATH.name
