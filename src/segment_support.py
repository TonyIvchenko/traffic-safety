from __future__ import annotations

import json
import math
from typing import Iterable


EARTH_RADIUS_KM = 6371.0088


def haversine_km(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    lat1_rad = math.radians(lat1)
    lon1_rad = math.radians(lon1)
    lat2_rad = math.radians(lat2)
    lon2_rad = math.radians(lon2)
    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad
    a = (
        math.sin(dlat / 2.0) ** 2
        + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(dlon / 2.0) ** 2
    )
    return 2.0 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))


def polyline_length_km(points: list[tuple[float, float]]) -> float:
    total = 0.0
    for idx in range(1, len(points)):
        lon1, lat1 = points[idx - 1]
        lon2, lat2 = points[idx]
        total += haversine_km(lat1, lon1, lat2, lon2)
    return total


def interpolate_point(
    a: tuple[float, float],
    b: tuple[float, float],
    ratio: float,
) -> tuple[float, float]:
    lon1, lat1 = a
    lon2, lat2 = b
    return (
        lon1 + (lon2 - lon1) * ratio,
        lat1 + (lat2 - lat1) * ratio,
    )


def densify_polyline(
    points: list[tuple[float, float]],
    max_edge_km: float,
) -> list[tuple[float, float]]:
    if len(points) < 2:
        return points[:]

    dense = [points[0]]
    for idx in range(1, len(points)):
        start = points[idx - 1]
        end = points[idx]
        edge_km = haversine_km(start[1], start[0], end[1], end[0])
        if edge_km <= max_edge_km:
            dense.append(end)
            continue
        pieces = max(2, math.ceil(edge_km / max_edge_km))
        for step in range(1, pieces):
            dense.append(interpolate_point(start, end, step / pieces))
        dense.append(end)
    return dense


def segmentize_polyline(
    points: list[tuple[float, float]],
    max_length_km: float,
) -> list[list[tuple[float, float]]]:
    dense = densify_polyline(points, max_edge_km=max_length_km)
    if len(dense) < 2:
        return []

    segments: list[list[tuple[float, float]]] = []
    current = [dense[0]]
    current_length = 0.0
    for idx in range(1, len(dense)):
        prev = dense[idx - 1]
        point = dense[idx]
        edge_km = haversine_km(prev[1], prev[0], point[1], point[0])
        if current_length + edge_km > max_length_km and len(current) > 1:
            segments.append(current)
            current = [current[-1], point]
            current_length = edge_km
        else:
            current.append(point)
            current_length += edge_km
    if len(current) > 1:
        segments.append(current)
    return segments


def polyline_bounds(points: list[tuple[float, float]]) -> tuple[float, float, float, float]:
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return min(lats), max(lats), min(lons), max(lons)


def polyline_centroid(points: list[tuple[float, float]]) -> tuple[float, float]:
    lons = [point[0] for point in points]
    lats = [point[1] for point in points]
    return sum(lats) / len(lats), sum(lons) / len(lons)


def to_local_xy_km(
    lat: float,
    lon: float,
    ref_lat: float,
    ref_lon: float,
) -> tuple[float, float]:
    km_per_deg_lat = 111.32
    km_per_deg_lon = 111.32 * math.cos(math.radians(ref_lat))
    x = (lon - ref_lon) * km_per_deg_lon
    y = (lat - ref_lat) * km_per_deg_lat
    return x, y


def point_to_polyline_distance_km(
    lat: float,
    lon: float,
    polyline: list[tuple[float, float]],
) -> float:
    if len(polyline) < 2:
        return float("inf")

    best = float("inf")
    for idx in range(1, len(polyline)):
        lon1, lat1 = polyline[idx - 1]
        lon2, lat2 = polyline[idx]
        ax, ay = to_local_xy_km(lat1, lon1, lat, lon)
        bx, by = to_local_xy_km(lat2, lon2, lat, lon)
        abx = bx - ax
        aby = by - ay
        denom = abx * abx + aby * aby
        if denom <= 1e-12:
            dist = math.hypot(ax, ay)
        else:
            t = max(0.0, min(1.0, -((ax * abx) + (ay * aby)) / denom))
            px = ax + t * abx
            py = ay + t * aby
            dist = math.hypot(px, py)
        best = min(best, dist)
    return best


def coords_to_json(points: Iterable[tuple[float, float]]) -> str:
    return json.dumps([[float(lon), float(lat)] for lon, lat in points], separators=(",", ":"))


def coords_from_json(payload: str) -> list[tuple[float, float]]:
    return [(float(lon), float(lat)) for lon, lat in json.loads(payload)]
