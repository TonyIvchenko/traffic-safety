from __future__ import annotations

from pathlib import Path
import sys

import shapefile
from shapely.geometry import box

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import geo_lookup


# Two adjacent 1x1 boxes in (lon, lat) space.
_WEST = box(-119.0, 34.0, -118.0, 35.0)
_EAST = box(-118.0, 34.0, -117.0, 35.0)


def test_geoindex_lookup_by_containment():
    index = geo_lookup.GeoIndex([_WEST, _EAST], ["06001", "06002"])
    assert len(index) == 2
    assert index.lookup(lat=34.5, lon=-118.5) == "06001"  # inside the west box
    assert index.lookup(lat=34.5, lon=-117.5) == "06002"  # inside the east box
    assert index.lookup(lat=0.0, lon=0.0) is None  # outside both


def test_empty_index_returns_none():
    assert geo_lookup.GeoIndex([], []).lookup(34.0, -118.0) is None


def test_from_shapefiles_reads_geoid_field(tmp_path):
    base = tmp_path / "counties"
    writer = shapefile.Writer(str(base))
    writer.field("GEOID", "C", size=5)
    writer.poly([[[-119, 34], [-118, 34], [-118, 35], [-119, 35], [-119, 34]]])
    writer.record("06001")
    writer.poly([[[-118, 34], [-117, 34], [-117, 35], [-118, 35], [-118, 34]]])
    writer.record("06002")
    writer.close()

    index = geo_lookup.GeoIndex.from_shapefiles([base])
    assert len(index) == 2
    assert index.lookup(lat=34.5, lon=-118.5) == "06001"
    assert index.lookup(lat=34.5, lon=-117.5) == "06002"


def test_from_shapefiles_rejects_missing_geoid_field(tmp_path):
    base = tmp_path / "noid"
    writer = shapefile.Writer(str(base))
    writer.field("NAME", "C", size=10)
    writer.poly([[[-119, 34], [-118, 34], [-118, 35], [-119, 35], [-119, 34]]])
    writer.record("somewhere")
    writer.close()

    import pytest

    with pytest.raises(ValueError, match="GEOID"):
        geo_lookup.GeoIndex.from_shapefiles([base])
