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
    module_path = REPO_DIR / "scripts" / "build_npmrds_index.py"
    spec = importlib.util.spec_from_file_location("build_npmrds_index", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


BUILDER = _load_builder()


def test_build_records_normalizes_columns_and_drops_invalid():
    frame = pd.DataFrame(
        [
            {"TMC": "1", "Latitude": 34.0, "Longitude": -118.0, "reference_speed": 65, "peak_speed": 20, "offpeak_speed": 60},
            {"TMC": "2", "Latitude": None, "Longitude": -118.0, "reference_speed": 60},  # dropped
            {"TMC": "3", "Latitude": 34.1, "Longitude": -118.1, "reference_speed": 0, "peak_speed": 10},  # dropped: ref 0
        ]
    )
    records = BUILDER.build_records(frame)
    assert len(records) == 1
    assert records[0]["reference_speed_mph"] == 65
    assert records[0]["peak_speed_mph"] == 20


def test_builder_output_is_loadable(tmp_path):
    import traffic_speed

    csv_path = tmp_path / "npmrds.csv"
    pd.DataFrame(
        [{"id": "s1", "lat": 34.0, "lon": -118.0, "reference_speed_mph": 65, "peak_speed_mph": 18}]
    ).to_csv(csv_path, index=False)
    out_path = tmp_path / "npmrds.parquet"

    pd.DataFrame(BUILDER.build_records(pd.read_csv(csv_path))).to_parquet(out_path, index=False)

    store = traffic_speed.TrafficSpeedStore.from_parquet(out_path)
    hit = store.lookup(34.001, -118.001, period="peak", max_km=3.0)
    assert hit["basis"] == "npmrds_segment"
    assert hit["speed_risk_index"] == traffic_speed.HEAVY_CONGESTION_RISK  # 18/65 ~ 0.28
