"""High Injury Network (HIN) construction.

A HIN is the small share of roadway where a large share of severe crashes
concentrate — the core artifact behind Vision Zero plans and the data-driven
analysis that SS4A/HSIP grant applications require. Segments are ranked by
severity-weighted crash intensity (weighted crashes per km) and selected until
they account for a target share of the severity-weighted total; the headline
result is "X% of the network carries Y% of severe crashes".

Severity weighting comes from crash_costs.severity_weight (a fatal crash = 1.0).
"""

from __future__ import annotations

from pathlib import Path
import sys

import numpy as np

SRC_DIR = Path(__file__).resolve().parent
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crash_costs import severity_weight

DEFAULT_SEVERITY_SHARE = 0.5  # capture 50% of severity-weighted crashes
MIN_SEGMENT_KM = 0.01  # floor so near-zero-length segments cannot blow up intensity


def severity_weighted_crashes(counts_by_severity: dict, costs: dict | None = None) -> float:
    """Weighted crash count where one fatal (K) crash counts as 1.0."""
    return float(
        sum(
            severity_weight(level, costs) * float(count)
            for level, count in counts_by_severity.items()
        )
    )


def build_hin(
    frame,
    *,
    weighted_col: str = "weighted_crashes",
    length_col: str = "length_km",
    target_share: float = DEFAULT_SEVERITY_SHARE,
):
    """Rank segments and flag the High Injury Network.

    Returns a copy sorted by intensity (descending) with added columns:
    ``hin_intensity`` (weighted crashes per km), ``hin_rank``,
    ``hin_cumulative_share``, and ``hin`` (bool). The segment that crosses the
    target share is included, so the selection always reaches ``target_share``.
    """
    result = frame.copy()
    lengths = result[length_col].astype(float).clip(lower=MIN_SEGMENT_KM)
    weighted = result[weighted_col].astype(float)
    result["hin_intensity"] = weighted / lengths
    # mergesort keeps input order stable among equal-intensity segments.
    result = result.sort_values("hin_intensity", ascending=False, kind="mergesort").reset_index(
        drop=True
    )
    result["hin_rank"] = np.arange(1, len(result) + 1, dtype=int)

    total_weighted = float(result[weighted_col].astype(float).sum())
    if total_weighted <= 0.0:
        result["hin_cumulative_share"] = 0.0
        result["hin"] = False
        return result

    cumulative = result[weighted_col].astype(float).cumsum()
    result["hin_cumulative_share"] = cumulative / total_weighted
    prior_share = result["hin_cumulative_share"].shift(1, fill_value=0.0)
    result["hin"] = prior_share < float(target_share)
    return result


def hin_summary(
    hin_frame,
    *,
    weighted_col: str = "weighted_crashes",
    length_col: str = "length_km",
) -> dict:
    """Headline stats: what share of network length carries what share of crashes."""
    selected = hin_frame[hin_frame["hin"]]
    network_length = float(hin_frame[length_col].astype(float).sum())
    network_weighted = float(hin_frame[weighted_col].astype(float).sum())
    hin_length = float(selected[length_col].astype(float).sum())
    hin_weighted = float(selected[weighted_col].astype(float).sum())
    return {
        "hin_segments": int(len(selected)),
        "network_segments": int(len(hin_frame)),
        "hin_length_km": round(hin_length, 3),
        "network_length_km": round(network_length, 3),
        "length_share": round(hin_length / network_length, 4) if network_length else 0.0,
        "weighted_crashes_captured": round(hin_weighted, 3),
        "network_weighted_crashes": round(network_weighted, 3),
        "weighted_crash_share": round(hin_weighted / network_weighted, 4)
        if network_weighted
        else 0.0,
    }
