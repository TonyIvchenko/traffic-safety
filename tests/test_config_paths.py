"""Artifact paths (model bundle, tiles, segment model) are environment-configurable
so a deployment can mount them anywhere (e.g. a Docker volume at a custom path).
"""

from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
SRC = REPO / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def _fresh(name: str, relpath: str):
    spec = importlib.util.spec_from_file_location(name, SRC / relpath)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_model_path_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_MODEL_PATH", str(tmp_path / "nope.joblib"))
    mod = _fresh("predict_env_override", "predict.py")
    assert Path(mod.MODEL_PATH).name == "nope.joblib"
    assert mod.MODEL_BUNDLE == {}  # missing file -> empty bundle (degrades, no crash)


def test_model_path_default(monkeypatch):
    monkeypatch.delenv("TRAFFIC_SAFETY_MODEL_PATH", raising=False)
    mod = _fresh("predict_env_default", "predict.py")
    assert Path(mod.MODEL_PATH) == REPO / "models" / "traffic_safety.joblib"


def test_tiles_and_segment_env_override(tmp_path, monkeypatch):
    monkeypatch.setenv("TRAFFIC_SAFETY_TILES_DIR", str(tmp_path / "tiles_x"))
    monkeypatch.setenv("TRAFFIC_SAFETY_SEGMENT_MODEL_PATH", str(tmp_path / "seg.joblib"))
    mod = _fresh("main_env_override", "main.py")
    assert Path(mod.TILES_DIR).name == "tiles_x"
    assert Path(mod.SEGMENT_MODEL_PATH).name == "seg.joblib"
    # A missing overlay at the configured tiles dir degrades to an all-zero cube.
    assert not np.asarray(mod.OVERLAY["risk"]).any()
