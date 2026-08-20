"""Catalog of metro regions for the Road Risk Advisory.

Each region is a named bounding box (``[min_lat, max_lat, min_lon, max_lon]``)
used to aggregate segment/grid risk into a regional advisory. Approximate metro
extents — a screening geography, not official MSA boundaries.
"""

from __future__ import annotations

REGIONS = (
    {"id": "los_angeles", "name": "Los Angeles, CA", "bbox": [33.60, 34.35, -118.90, -117.50]},
    {"id": "san_diego", "name": "San Diego, CA", "bbox": [32.50, 33.30, -117.40, -116.80]},
    {"id": "bay_area", "name": "San Francisco Bay Area, CA", "bbox": [37.20, 38.10, -122.60, -121.60]},
    {"id": "sacramento", "name": "Sacramento, CA", "bbox": [38.30, 38.90, -121.70, -121.10]},
    {"id": "seattle", "name": "Seattle, WA", "bbox": [47.20, 47.90, -122.55, -121.90]},
    {"id": "portland", "name": "Portland, OR", "bbox": [45.30, 45.75, -123.00, -122.35]},
    {"id": "phoenix", "name": "Phoenix, AZ", "bbox": [33.15, 33.85, -112.45, -111.55]},
    {"id": "las_vegas", "name": "Las Vegas, NV", "bbox": [35.90, 36.40, -115.45, -114.90]},
    {"id": "denver", "name": "Denver, CO", "bbox": [39.50, 40.00, -105.25, -104.60]},
    {"id": "dallas_fort_worth", "name": "Dallas-Fort Worth, TX", "bbox": [32.55, 33.15, -97.55, -96.45]},
    {"id": "houston", "name": "Houston, TX", "bbox": [29.45, 30.15, -95.85, -95.00]},
    {"id": "austin", "name": "Austin, TX", "bbox": [30.10, 30.55, -98.00, -97.50]},
    {"id": "san_antonio", "name": "San Antonio, TX", "bbox": [29.20, 29.70, -98.75, -98.20]},
    {"id": "chicago", "name": "Chicago, IL", "bbox": [41.60, 42.10, -88.05, -87.45]},
    {"id": "minneapolis", "name": "Minneapolis-St. Paul, MN", "bbox": [44.75, 45.15, -93.55, -92.95]},
    {"id": "atlanta", "name": "Atlanta, GA", "bbox": [33.55, 34.05, -84.65, -84.15]},
    {"id": "miami", "name": "Miami, FL", "bbox": [25.55, 26.35, -80.55, -80.05]},
    {"id": "tampa", "name": "Tampa-St. Petersburg, FL", "bbox": [27.75, 28.20, -82.75, -82.30]},
    {"id": "new_york", "name": "New York, NY", "bbox": [40.40, 41.05, -74.30, -73.60]},
    {"id": "philadelphia", "name": "Philadelphia, PA", "bbox": [39.80, 40.20, -75.45, -74.90]},
    {"id": "washington_dc", "name": "Washington, DC", "bbox": [38.70, 39.10, -77.35, -76.80]},
    {"id": "boston", "name": "Boston, MA", "bbox": [42.20, 42.55, -71.25, -70.85]},
    {"id": "detroit", "name": "Detroit, MI", "bbox": [42.15, 42.60, -83.45, -82.85]},
)


def _contains(bbox, lat: float, lon: float) -> bool:
    min_lat, max_lat, min_lon, max_lon = bbox
    return min_lat <= lat <= max_lat and min_lon <= lon <= max_lon


def _area(bbox) -> float:
    min_lat, max_lat, min_lon, max_lon = bbox
    return (max_lat - min_lat) * (max_lon - min_lon)


def list_regions() -> list[dict]:
    return [dict(region) for region in REGIONS]


def get_region(region_id) -> dict | None:
    key = str(region_id).strip().lower()
    for region in REGIONS:
        if region["id"] == key:
            return dict(region)
    return None


def region_bbox(region: dict) -> tuple[float, float, float, float]:
    min_lat, max_lat, min_lon, max_lon = region["bbox"]
    return (float(min_lat), float(max_lat), float(min_lon), float(max_lon))


def region_for_point(lat, lon) -> dict | None:
    """The smallest-area region containing the point, or None if none does."""
    try:
        latitude, longitude = float(lat), float(lon)
    except (TypeError, ValueError):
        return None
    candidates = [region for region in REGIONS if _contains(region["bbox"], latitude, longitude)]
    if not candidates:
        return None
    return dict(min(candidates, key=lambda region: _area(region["bbox"])))
