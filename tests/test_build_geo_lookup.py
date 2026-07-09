from __future__ import annotations

from pathlib import Path
import sys

import pandas as pd
from shapely.geometry import box

REPO_DIR = Path(__file__).resolve().parents[1]
for _path in (REPO_DIR / "scripts", REPO_DIR / "src", REPO_DIR):
    if str(_path) not in sys.path:
        sys.path.insert(0, str(_path))

import build_geo_lookup as bgl
import geo_lookup


def test_enrich_with_geoids_assigns_county_and_tract():
    # Two adjacent boxes in (lon, lat) space serve as both county and tract areas.
    counties = geo_lookup.GeoIndex(
        [box(-119.0, 34.0, -118.0, 35.0), box(-118.0, 34.0, -117.0, 35.0)],
        ["06001", "06002"],
    )
    tracts = geo_lookup.GeoIndex(
        [box(-119.0, 34.0, -118.0, 35.0), box(-118.0, 34.0, -117.0, 35.0)],
        ["06001000100", "06002000200"],
    )
    frame = pd.DataFrame(
        {
            "segment_id": ["a", "b", "ocean"],
            "center_lat": [34.5, 34.5, 0.0],
            "center_lon": [-118.5, -117.5, 0.0],
        }
    )

    enriched = bgl.enrich_with_geoids(
        frame, lat_col="center_lat", lon_col="center_lon",
        county_index=counties, tract_index=tracts,
    )

    assert list(enriched["county_geoid"]) == ["06001", "06002", None]
    assert list(enriched["tract_geoid"]) == ["06001000100", "06002000200", None]
    # Original frame is untouched.
    assert "county_geoid" not in frame.columns


def test_enrich_preserves_input_columns():
    counties = geo_lookup.GeoIndex([box(-119.0, 34.0, -118.0, 35.0)], ["06001"])
    frame = pd.DataFrame({"cell_id": ["x"], "center_lat": [34.5], "center_lon": [-118.5], "extra": [7]})
    enriched = bgl.enrich_with_geoids(
        frame, lat_col="center_lat", lon_col="center_lon",
        county_index=counties, tract_index=counties,
    )
    assert enriched.loc[0, "extra"] == 7
    assert enriched.loc[0, "county_geoid"] == "06001"
