from __future__ import annotations

import base64
import json
import math
from functools import lru_cache
from pathlib import Path
import sqlite3

import numpy as np


SRC_DIR = Path(__file__).resolve().parent
REPO_DIR = SRC_DIR.parent
TILES_DIR = REPO_DIR / "tiles"
ROAD_TILE_DB_PATH = TILES_DIR / "road_tile_index.sqlite3"
ROAD_TILE_META_PATH = TILES_DIR / "road_tiles_meta.json"
ROAD_TILE_FORECAST_PATH = TILES_DIR / "segment_forecast_uint8.npy"
ROAD_TILE_BASELINE_PATH = TILES_DIR / "segment_baseline_uint8.npy"
ROAD_RASTER_TILE_DB_PATH = TILES_DIR / "road_raster_tiles.sqlite3"

ROAD_TILE_COORD_SCALE = 8.0
DEFAULT_TILE_ZOOM_MIN = 4
DEFAULT_TILE_ZOOM_MAX = 11


def road_kind_from_attrs(rttyp: str | None, mtfcc: str | None) -> int:
    rttyp_value = (rttyp or "").strip().upper()
    mtfcc_value = (mtfcc or "").strip().upper()
    if rttyp_value == "I" or mtfcc_value == "S1100":
        return 2
    if rttyp_value in {"U", "S"} or mtfcc_value == "S1200":
        return 1
    return 0


def mercator_world_pixel(lat: float, lon: float, zoom: int) -> tuple[float, float]:
    lat_clamped = max(min(float(lat), 85.05112878), -85.05112878)
    siny = math.sin(math.radians(lat_clamped))
    scale = 256.0 * (2**int(zoom))
    x = (float(lon) + 180.0) / 360.0 * scale
    y = (
        0.5
        - math.log((1.0 + siny) / max(1.0 - siny, 1e-12)) / (4.0 * math.pi)
    ) * scale
    return x, y


def encode_relative_path(points: list[tuple[float, float]]) -> bytes:
    packed = np.empty((len(points), 2), dtype=np.int16)
    for idx, (x_coord, y_coord) in enumerate(points):
        packed[idx, 0] = int(round(float(x_coord) * ROAD_TILE_COORD_SCALE))
        packed[idx, 1] = int(round(float(y_coord) * ROAD_TILE_COORD_SCALE))
    return packed.tobytes()


def _points_close(
    first: tuple[float, float],
    second: tuple[float, float],
    tolerance: float = 1e-6,
) -> bool:
    return (
        abs(float(first[0]) - float(second[0])) <= tolerance
        and abs(float(first[1]) - float(second[1])) <= tolerance
    )


def _clip_segment_to_rect(
    x0: float,
    y0: float,
    x1: float,
    y1: float,
    min_x: float,
    min_y: float,
    max_x: float,
    max_y: float,
) -> tuple[tuple[float, float], tuple[float, float]] | None:
    dx = float(x1) - float(x0)
    dy = float(y1) - float(y0)
    t0 = 0.0
    t1 = 1.0

    for p_value, q_value in (
        (-dx, float(x0) - min_x),
        (dx, max_x - float(x0)),
        (-dy, float(y0) - min_y),
        (dy, max_y - float(y0)),
    ):
        if abs(p_value) <= 1e-12:
            if q_value < 0.0:
                return None
            continue
        ratio = q_value / p_value
        if p_value < 0.0:
            if ratio > t1:
                return None
            t0 = max(t0, ratio)
        else:
            if ratio < t0:
                return None
            t1 = min(t1, ratio)

    return (
        (float(x0) + t0 * dx, float(y0) + t0 * dy),
        (float(x0) + t1 * dx, float(y0) + t1 * dy),
    )


def _clip_polyline_to_tile(
    world_points: list[tuple[float, float]],
    tile_x: int,
    tile_y: int,
) -> list[list[tuple[float, float]]]:
    min_x = float(tile_x * 256)
    min_y = float(tile_y * 256)
    max_x = min_x + 256.0
    max_y = min_y + 256.0

    paths: list[list[tuple[float, float]]] = []
    current_path: list[tuple[float, float]] = []

    for idx in range(1, len(world_points)):
        start = world_points[idx - 1]
        end = world_points[idx]
        clipped = _clip_segment_to_rect(
            start[0],
            start[1],
            end[0],
            end[1],
            min_x,
            min_y,
            max_x,
            max_y,
        )
        if clipped is None:
            if len(current_path) >= 2:
                paths.append(current_path)
            current_path = []
            continue

        clipped_start, clipped_end = clipped
        if not current_path:
            current_path = [clipped_start]
        elif not _points_close(current_path[-1], clipped_start):
            if len(current_path) >= 2:
                paths.append(current_path)
            current_path = [clipped_start]

        if not _points_close(current_path[-1], clipped_end):
            current_path.append(clipped_end)

    if len(current_path) >= 2:
        paths.append(current_path)

    return paths


def iter_tile_paths(
    coords: list[list[float] | tuple[float, float]],
    zoom: int,
) -> list[tuple[int, int, bytes]]:
    if len(coords) < 2:
        return []

    world_points = [
        mercator_world_pixel(lat=float(lat), lon=float(lon), zoom=zoom)
        for lon, lat in coords
    ]
    xs = [point[0] for point in world_points]
    ys = [point[1] for point in world_points]
    tile_x_min = int(math.floor(min(xs) / 256.0))
    tile_x_max = int(math.floor(max(xs) / 256.0))
    tile_y_min = int(math.floor(min(ys) / 256.0))
    tile_y_max = int(math.floor(max(ys) / 256.0))
    tile_count = 2**int(zoom)

    output: list[tuple[int, int, bytes]] = []
    for tile_x in range(max(0, tile_x_min), min(tile_count - 1, tile_x_max) + 1):
        for tile_y in range(max(0, tile_y_min), min(tile_count - 1, tile_y_max) + 1):
            origin_x = float(tile_x * 256)
            origin_y = float(tile_y * 256)
            for clipped_path in _clip_polyline_to_tile(world_points, tile_x, tile_y):
                output.append(
                    (
                        tile_x,
                        tile_y,
                        encode_relative_path(
                            [
                                (x_coord - origin_x, y_coord - origin_y)
                                for x_coord, y_coord in clipped_path
                            ]
                        ),
                    )
                )
    return output


def load_road_tile_meta() -> dict[str, object]:
    if not ROAD_TILE_META_PATH.exists():
        return {}
    return json.loads(ROAD_TILE_META_PATH.read_text(encoding="utf-8"))


def road_tile_assets_ready() -> bool:
    return (
        ROAD_TILE_DB_PATH.exists()
        and ROAD_TILE_FORECAST_PATH.exists()
        and ROAD_TILE_BASELINE_PATH.exists()
        and ROAD_TILE_META_PATH.exists()
    )


def raster_tile_assets_ready() -> bool:
    return ROAD_RASTER_TILE_DB_PATH.exists() and ROAD_TILE_META_PATH.exists()


@lru_cache(maxsize=2)
def _load_forecast_scores(path_str: str, mtime_ns: int) -> np.ndarray:
    del mtime_ns
    return np.load(path_str, mmap_mode="r")


@lru_cache(maxsize=2048)
def _load_cached_tile_payload(
    db_path_str: str,
    db_mtime_ns: int,
    forecast_path_str: str,
    forecast_mtime_ns: int,
    revision: str,
    z: int,
    x: int,
    y: int,
) -> dict[str, object]:
    del db_mtime_ns, revision
    scores = _load_forecast_scores(forecast_path_str, forecast_mtime_ns)
    entries: list[dict[str, object]] = []
    db_uri = f"file:{db_path_str}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as connection:
        cursor = connection.execute(
            """
            SELECT segment_idx, road_kind, path
            FROM tile_entries
            WHERE z = ? AND x = ? AND y = ?
            """,
            (int(z), int(x), int(y)),
        )
        for segment_idx, road_kind, path_blob in cursor:
            risk_row = np.asarray(scores[int(segment_idx)], dtype=np.uint8)
            entries.append(
                {
                    "s": int(segment_idx),
                    "k": int(road_kind),
                    "p": base64.b64encode(path_blob).decode("ascii"),
                    "r": base64.b64encode(risk_row.tobytes()).decode("ascii"),
                }
            )
    return {
        "z": int(z),
        "x": int(x),
        "y": int(y),
        "count": len(entries),
        "entries": entries,
    }


def load_tile_payload(z: int, x: int, y: int) -> dict[str, object] | None:
    if not road_tile_assets_ready():
        return None

    meta = load_road_tile_meta()
    if not meta:
        return None

    db_mtime_ns = ROAD_TILE_DB_PATH.stat().st_mtime_ns
    forecast_mtime_ns = ROAD_TILE_FORECAST_PATH.stat().st_mtime_ns
    revision = str(meta.get("run_id") or meta.get("generated_at_utc") or "")
    return _load_cached_tile_payload(
        str(ROAD_TILE_DB_PATH),
        db_mtime_ns,
        str(ROAD_TILE_FORECAST_PATH),
        forecast_mtime_ns,
        revision,
        int(z),
        int(x),
        int(y),
    )


@lru_cache(maxsize=32768)
def _load_cached_raster_tile(
    db_path_str: str,
    db_mtime_ns: int,
    revision: str,
    frame_idx: int,
    z: int,
    x: int,
    y: int,
) -> bytes | None:
    del db_mtime_ns, revision
    db_uri = f"file:{db_path_str}?mode=ro"
    with sqlite3.connect(db_uri, uri=True) as connection:
        row = connection.execute(
            """
            SELECT png
            FROM raster_tiles
            WHERE frame_idx = ? AND z = ? AND x = ? AND y = ?
            """,
            (int(frame_idx), int(z), int(x), int(y)),
        ).fetchone()
    return None if row is None else bytes(row[0])


def load_raster_tile_png(frame_idx: int, z: int, x: int, y: int) -> bytes | None:
    if not raster_tile_assets_ready():
        return None
    meta = load_road_tile_meta()
    revision = str(meta.get("run_id") or meta.get("generated_at_utc") or "")
    db_mtime_ns = ROAD_RASTER_TILE_DB_PATH.stat().st_mtime_ns
    return _load_cached_raster_tile(
        str(ROAD_RASTER_TILE_DB_PATH),
        db_mtime_ns,
        revision,
        int(frame_idx),
        int(z),
        int(x),
        int(y),
    )
