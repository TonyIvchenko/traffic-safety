from __future__ import annotations

from pathlib import Path
import sys

import numpy as np
import pandas as pd

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import segment_model_support as sms


def _roads() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "center_lat": [34.0, 41.0],
            "center_lon": [-118.0, -87.0],
            "length_km": [0.5, 0.3],
            "fullname": ["Main St", ""],
            "rttyp": ["I", "C"],
            "mtfcc": ["S1100", "S1400"],
        }
    )


def test_build_static_segment_frame_columns_and_one_hot():
    frame, columns = sms.build_static_segment_frame(_roads())

    assert columns == frame.columns.tolist()
    expected = (
        ["center_lat", "center_lon", "length_km", "named_road"]
        + [f"rttyp_{value}" for value in sms.DEFAULT_RTTYP_VALUES]
        + [f"mtfcc_{value}" for value in sms.DEFAULT_MTFCC_VALUES]
    )
    assert columns == expected

    # named_road flags whether a road has a non-empty name.
    assert frame["named_road"].tolist() == [1.0, 0.0]
    # Interstate route type / MTFCC one-hots line up with the first row only.
    assert frame["rttyp_I"].tolist() == [1.0, 0.0]
    assert frame["rttyp_C"].tolist() == [0.0, 1.0]
    assert frame["mtfcc_S1100"].tolist() == [1.0, 0.0]
    assert frame["mtfcc_S1200"].tolist() == [0.0, 0.0]
    assert all(frame[col].dtype == np.float32 for col in frame.columns)


def test_build_static_segment_frame_handles_missing_attributes():
    roads = pd.DataFrame(
        {
            "center_lat": [10.0],
            "center_lon": [20.0],
            "length_km": [1.0],
            "fullname": [None],
            "rttyp": [None],
            "mtfcc": [None],
        }
    )
    frame, _ = sms.build_static_segment_frame(roads)
    assert frame["named_road"].tolist() == [0.0]
    assert frame["rttyp_I"].tolist() == [0.0]
    assert frame["mtfcc_S1100"].tolist() == [0.0]


def test_build_segment_feature_matrix_shape_and_temporal_decomposition():
    static_frame, columns = sms.build_static_segment_frame(_roads())
    static = static_frame.to_numpy(dtype=np.float32)

    features = sms.build_segment_feature_matrix(
        static_features=static,
        hour_of_week=np.array([17, 100], dtype=np.int16),
        months=np.array([9, 1], dtype=np.int8),
        totals=np.array([10.0, 0.0], dtype=np.float32),
        same_hour=np.array([2.0, 0.0], dtype=np.float32),
        temp_c=np.array([20.0, -5.0], dtype=np.float32),
        relative_humidity_pct=np.array([50.0, 80.0], dtype=np.float32),
        wind_speed_mps=np.array([3.0, 7.0], dtype=np.float32),
        wet_hour=np.array([0.0, 1.0], dtype=np.float32),
    )

    # static columns + 13 temporal feature columns.
    assert features.shape == (2, len(columns) + 13)
    assert features.dtype == np.float32
    assert np.isfinite(features).all()
    # The static block is preserved unchanged at the front of each row.
    assert np.allclose(features[:, : len(columns)], static)
