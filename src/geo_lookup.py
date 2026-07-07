"""Point-in-polygon lookup of county / census-tract GEOIDs from TIGER boundaries.

Backs the geographic enrichment used by the grant, equity, and countermeasure
features: given a lat/lon, return the containing county (5-digit GEOID) or census
tract (11-digit GEOID). A shapely STRtree gives fast bbox candidate lookup; exact
`covers` containment resolves the winner.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path
import sys

import shapefile
from shapely.geometry import Point, shape
from shapely.strtree import STRtree

SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))

from scripts.common import STATE_FIPS, tiger_county_path, tiger_tract_path


class GeoIndex:
    """Spatial index over boundary polygons keyed by GEOID."""

    def __init__(self, geoms, geoids):
        self._geoms = list(geoms)
        self._geoids = list(geoids)
        self._tree = STRtree(self._geoms) if self._geoms else None

    def __len__(self) -> int:
        return len(self._geoms)

    @classmethod
    def from_shapefiles(cls, paths, geoid_field: str = "GEOID") -> "GeoIndex":
        geoms: list = []
        geoids: list[str] = []
        for path in paths:
            reader = shapefile.Reader(str(path))
            field_names = [field[0] for field in reader.fields[1:]]
            if geoid_field not in field_names:
                raise ValueError(f"{path} has no '{geoid_field}' field (has {field_names})")
            geoid_col = field_names.index(geoid_field)
            for shape_record in reader.iterShapeRecords():
                geoms.append(shape(shape_record.shape.__geo_interface__))
                geoids.append(str(shape_record.record[geoid_col]))
        return cls(geoms, geoids)

    def lookup(self, lat: float, lon: float) -> str | None:
        if self._tree is None:
            return None
        point = Point(float(lon), float(lat))
        for index in self._tree.query(point):
            geom = self._geoms[int(index)]
            if geom.covers(point):  # covers() keeps boundary points assigned
                return self._geoids[int(index)]
        return None


@lru_cache(maxsize=2)
def load_county_index(year: int = 2024) -> GeoIndex:
    return GeoIndex.from_shapefiles([tiger_county_path(year)])


@lru_cache(maxsize=2)
def load_tract_index(year: int = 2024) -> GeoIndex:
    paths = [
        tiger_tract_path(state_fips, year)
        for state_fips in STATE_FIPS
        if tiger_tract_path(state_fips, year).exists()
    ]
    return GeoIndex.from_shapefiles(paths)


def county_of(lat: float, lon: float, year: int = 2024) -> str | None:
    return load_county_index(year).lookup(lat, lon)


def tract_of(lat: float, lon: float, year: int = 2024) -> str | None:
    return load_tract_index(year).lookup(lat, lon)
