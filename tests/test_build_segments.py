from __future__ import annotations

from pathlib import Path
from types import SimpleNamespace
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_segments


def test_grid_bucket_floors_to_quarter_degree_cells():
    assert build_segments.grid_bucket(0.0) == 0
    assert build_segments.grid_bucket(0.25) == 1
    assert build_segments.grid_bucket(0.3) == 1
    assert build_segments.grid_bucket(0.5) == 2
    assert build_segments.grid_bucket(-0.1) == -1


def test_iter_shape_parts_single_part():
    shape = SimpleNamespace(points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], parts=[0])
    polylines = build_segments.iter_shape_parts(shape)
    assert len(polylines) == 1
    assert polylines[0] == [(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)]


def test_iter_shape_parts_splits_multiple_parts():
    shape = SimpleNamespace(
        points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0), (3.0, 3.0)],
        parts=[0, 2],
    )
    polylines = build_segments.iter_shape_parts(shape)
    assert polylines == [
        [(0.0, 0.0), (1.0, 1.0)],
        [(2.0, 2.0), (3.0, 3.0)],
    ]


def test_iter_shape_parts_drops_degenerate_parts():
    # Second part has only one point and must be skipped.
    shape = SimpleNamespace(points=[(0.0, 0.0), (1.0, 1.0), (2.0, 2.0)], parts=[0, 2])
    polylines = build_segments.iter_shape_parts(shape)
    assert polylines == [[(0.0, 0.0), (1.0, 1.0)]]
