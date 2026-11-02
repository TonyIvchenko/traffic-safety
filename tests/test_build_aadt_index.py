from __future__ import annotations

import importlib.util
from pathlib import Path
import sys

import pandas as pd

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))


def _load_builder():
    module_path = REPO_DIR / "scripts" / "build_aadt_index.py"
    spec = importlib.util.spec_from_file_location("build_aadt_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def test_build_records_maps_hpms_f_system_and_lengths():
    frame = pd.DataFrame(
        [
            {"route_id": "I-5", "LAT": 47.6, "LON": -122.33, "AADT": 200000, "F_SYSTEM": 1, "length_miles": 1.0, "state": "WA", "year": 2022},
            {"route_id": "Local St", "LAT": 47.61, "LON": -122.34, "AADT": 3000, "F_SYSTEM": 7, "length_km": 0.8},
            {"route_id": "bad", "LAT": None, "LON": -122.0, "AADT": 5000, "F_SYSTEM": 3},  # dropped
            {"route_id": "neg", "LAT": 47.0, "LON": -122.0, "AADT": -1, "F_SYSTEM": 3},     # dropped
        ]
    )

    records = BUILDER.build_records(frame)

    assert len(records) == 2  # the two valid rows
    interstate = next(r for r in records if r["name"] == "I-5")
    assert interstate["functional_class"] == "interstate"  # F_SYSTEM 1
    assert interstate["aadt"] == 200000
    # length_miles was converted to km.
    assert abs(interstate["length_km"] - 1.0 / BUILDER.MILES_PER_KM) < 1e-6

    local = next(r for r in records if r["name"] == "Local St")
    assert local["functional_class"] == "local"  # F_SYSTEM 7
    assert local["length_km"] == 0.8


def test_builder_output_is_loadable_by_exposure(tmp_path):
    import exposure

    csv_path = tmp_path / "hpms.csv"
    pd.DataFrame(
        [{"name": "Test Fwy", "lat": 34.0, "lon": -118.0, "aadt": 180000, "f_system": 1, "length_miles": 1.0}]
    ).to_csv(csv_path, index=False)
    out_path = tmp_path / "aadt.parquet"

    frame = pd.read_csv(csv_path)
    records = BUILDER.build_records(frame)
    pd.DataFrame(records).to_parquet(out_path, index=False)

    store = exposure.AADTStore.from_parquet(out_path)
    hit = store.lookup(34.001, -118.001, max_km=3.0)
    assert hit["basis"] == "hpms_segment"
    assert hit["aadt"] == 180000
