"""Build the High Injury Network by joining FARS fatal crashes to road segments.

FARS is the defensible severity source: every FARS crash is a fatal (KABCO "K")
crash. The US-Accidents "Severity" carried in segment_events is *traffic impact*,
not injury severity, so it is deliberately not used here.

Each fatal crash is matched to the nearest segment of the modeled (active)
primary/secondary road network within --max-match-km; crashes with no road that
close are excluded (our network is primary/secondary roads only, so a loose
radius would falsely snap local-road crashes onto highways). Segments are then
ranked into a High Injury Network by severity-weighted crashes per km.

    python scripts/build_hin.py --max-match-km 0.25 --target-share 0.5

Output: data/processed/safety/high_injury_network.parquet
"""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from sklearn.neighbors import BallTree

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from crash_costs import severity_weight
from hin import build_hin, hin_summary
from segment_support import coords_from_json, point_to_polyline_distance_km

from common import (
    ACCIDENTS_CLEAN_PATH,
    ACTIVE_ROAD_SEGMENTS_PATH,
    HIGH_INJURY_NETWORK_PATH,
    SEGMENT_GEOID_PATH,
    SEGMENT_MATCH_CANDIDATES,
    ensure_dirs,
)

DEFAULT_MAX_MATCH_KM = 0.25

SEGMENT_COLUMNS = [
    "segment_id",
    "state_fips",
    "linearid",
    "fullname",
    "rttyp",
    "mtfcc",
    "length_km",
    "center_lat",
    "center_lon",
    "coords_json",
]


def state_fips_from_code(state_code) -> str:
    """FARS STATE is a numeric FIPS; segments use the zero-padded 2-digit string."""
    return f"{int(state_code):02d}"


def build_state_indexes(segments: pd.DataFrame):
    """Decoded polylines plus a per-state BallTree over segment centroids."""
    coords = [coords_from_json(payload) for payload in segments["coords_json"].tolist()]
    state_indices: dict[str, np.ndarray] = {}
    state_trees: dict[str, BallTree] = {}
    for state_fips, state_frame in segments.groupby("state_fips", sort=False):
        state_indices[str(state_fips)] = state_frame.index.to_numpy(dtype=np.int64)
        state_trees[str(state_fips)] = BallTree(
            np.radians(state_frame[["center_lat", "center_lon"]].to_numpy(dtype=np.float64)),
            metric="haversine",
        )
    return coords, state_indices, state_trees


def match_crashes(
    crashes: pd.DataFrame,
    coords: list,
    state_indices: dict,
    state_trees: dict,
    max_match_km: float = DEFAULT_MAX_MATCH_KM,
) -> np.ndarray:
    """Nearest segment position per crash row, or -1 when none is close enough."""
    crashes = crashes.reset_index(drop=True)
    matched = np.full(len(crashes), -1, dtype=np.int64)
    for state_fips, chunk in crashes.groupby("state_fips", sort=False):
        tree = state_trees.get(str(state_fips))
        indices = state_indices.get(str(state_fips))
        if tree is None or indices is None or len(indices) == 0:
            continue
        neighbors = min(SEGMENT_MATCH_CANDIDATES, len(indices))
        points = np.radians(chunk[["lat", "lon"]].to_numpy(dtype=np.float64))
        _, candidates = tree.query(points, k=neighbors)
        for row, candidate_row in zip(chunk.index.to_numpy(), candidates, strict=False):
            lat = float(crashes.at[row, "lat"])
            lon = float(crashes.at[row, "lon"])
            best_segment, best_distance = -1, float("inf")
            for position in np.atleast_1d(candidate_row):
                segment = int(indices[int(position)])
                distance = point_to_polyline_distance_km(lat, lon, coords[segment])
                if distance < best_distance:
                    best_distance, best_segment = distance, segment
            if best_segment >= 0 and best_distance <= max_match_km:
                matched[row] = best_segment
    return matched


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--max-match-km", type=float, default=DEFAULT_MAX_MATCH_KM)
    parser.add_argument("--target-share", type=float, default=0.5)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    ensure_dirs()

    segments = pd.read_parquet(ACTIVE_ROAD_SEGMENTS_PATH, columns=SEGMENT_COLUMNS)
    segments = segments.reset_index(drop=True)
    crashes = pd.read_csv(ACCIDENTS_CLEAN_PATH, usecols=["lat", "lon", "state_code", "fatals"])
    crashes["state_fips"] = crashes["state_code"].map(state_fips_from_code)
    print(f"segments={len(segments)} fatal_crashes={len(crashes)}")

    coords, state_indices, state_trees = build_state_indexes(segments)
    matched = match_crashes(crashes, coords, state_indices, state_trees, args.max_match_km)
    matched_count = int((matched >= 0).sum())
    print(
        f"matched {matched_count}/{len(crashes)} fatal crashes to a segment "
        f"within {args.max_match_km} km ({matched_count / max(len(crashes), 1):.1%})"
    )

    counts = pd.Series(matched[matched >= 0]).value_counts()
    segments["fatal_crashes"] = segments.index.map(counts).fillna(0.0).astype(float)
    # Every FARS crash is a fatal (K) crash, so the severity weight is 1.0 each.
    segments["weighted_crashes"] = segments["fatal_crashes"] * severity_weight("K")

    network = build_hin(
        segments.drop(columns=["coords_json"]),
        weighted_col="weighted_crashes",
        length_col="length_km",
        target_share=args.target_share,
    )
    if SEGMENT_GEOID_PATH.exists():
        geoids = pd.read_csv(SEGMENT_GEOID_PATH, dtype=str)
        network = network.merge(geoids, on="segment_id", how="left")
    else:
        print(f"note: {SEGMENT_GEOID_PATH.name} missing; run build_geo_lookup.py for county/tract")

    network.to_parquet(HIGH_INJURY_NETWORK_PATH, index=False)
    summary = hin_summary(network)
    print(
        f"HIN: {summary['hin_segments']} segments, {summary['hin_length_km']} km "
        f"({summary['length_share']:.1%} of network) carry "
        f"{summary['weighted_crash_share']:.1%} of severity-weighted crashes"
    )
    print(f"wrote {HIGH_INJURY_NETWORK_PATH}")


if __name__ == "__main__":
    main()
