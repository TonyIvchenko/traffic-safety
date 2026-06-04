from __future__ import annotations

from pathlib import Path
import sys

import h3
import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_dataset


def test_normalize_day_of_week_maps_fars_to_monday_first():
    # FARS encodes Sunday=1..Saturday=7; the model uses Monday=1..Sunday=7.
    assert build_dataset.normalize_day_of_week(1) == 7  # Sunday
    assert build_dataset.normalize_day_of_week(2) == 1  # Monday
    assert build_dataset.normalize_day_of_week(3) == 2  # Tuesday
    assert build_dataset.normalize_day_of_week(7) == 6  # Saturday


def test_normalize_day_of_week_is_within_range_for_all_inputs():
    assert sorted(build_dataset.normalize_day_of_week(d) for d in range(1, 8)) == [
        1, 2, 3, 4, 5, 6, 7
    ]


def test_build_candidate_cells_includes_neighbors_and_counts():
    cell_id = h3.latlng_to_cell(34.05, -118.25, build_dataset.H3_RESOLUTION)
    events = pd.DataFrame(
        {
            "cell_id": [cell_id, cell_id, cell_id],
            "hour_of_week": [10, 10, 35],
        }
    )

    candidate_cells = build_dataset.build_candidate_cells(events)

    assert cell_id in set(candidate_cells["cell_id"])
    # grid_disk(k=1) pulls in the six neighbouring cells too.
    assert len(candidate_cells) > 1
    by_cell = candidate_cells.set_index("cell_id")
    assert int(by_cell.loc[cell_id, "event_count"]) == 3
    # Neighbour cells with no events must still be present with a zero count.
    neighbor_counts = candidate_cells.loc[
        candidate_cells["cell_id"] != cell_id, "event_count"
    ]
    assert (neighbor_counts == 0).all()
    assert candidate_cells["center_lat"].notna().all()
    assert candidate_cells["center_lon"].notna().all()
