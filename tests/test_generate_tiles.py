from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pytest

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import generate_tiles as gt


def test_normalize_probs_scales_to_unit_range():
    probs = np.array([0.0, 1.0, 2.0, 3.0, 4.0], dtype=np.float32)
    normalized = gt.normalize_probs(probs)

    assert normalized.dtype == np.float32
    assert normalized.shape == probs.shape
    assert normalized.min() == pytest.approx(0.0)
    # The top of the distribution saturates at 1.0.
    assert normalized.max() == pytest.approx(1.0)
    assert np.all((normalized >= 0.0) & (normalized <= 1.0))


def test_normalize_probs_all_zero_returns_zeros():
    normalized = gt.normalize_probs(np.zeros(5, dtype=np.float32))
    assert np.all(normalized == 0.0)
    assert normalized.dtype == np.float32


def test_paint_frame_marks_cell_near_a_point():
    lats = np.array([39.5], dtype=np.float32)
    lons = np.array([-98.35], dtype=np.float32)
    risk = np.array([1.0], dtype=np.float32)
    conf = np.array([1.0], dtype=np.float32)

    risk_grid, conf_grid = gt.paint_frame(lats, lons, risk, conf)

    assert risk_grid.shape == (gt.OVERLAY_HEIGHT, gt.OVERLAY_WIDTH)
    assert conf_grid.shape == (gt.OVERLAY_HEIGHT, gt.OVERLAY_WIDTH)
    assert risk_grid.dtype == np.float32
    assert risk_grid.max() > 0.0
    assert risk_grid.max() <= 1.0 + 1e-6


def test_paint_frame_empty_when_no_points():
    empty = np.array([], dtype=np.float32)
    risk_grid, conf_grid = gt.paint_frame(empty, empty, empty, empty)
    assert risk_grid.max() == 0.0
    assert conf_grid.max() == 0.0
