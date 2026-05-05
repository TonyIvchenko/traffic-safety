from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
import sqlite3
import sys

import pyarrow.parquet as pq

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

from road_tiles import (
    DEFAULT_TILE_ZOOM_MAX,
    DEFAULT_TILE_ZOOM_MIN,
    ROAD_TILE_DB_PATH,
    iter_tile_paths,
    road_kind_from_attrs,
)
from scripts.common import ROAD_SEGMENTS_PATH, ensure_dirs


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--zoom-min", type=int, default=DEFAULT_TILE_ZOOM_MIN)
    parser.add_argument("--zoom-max", type=int, default=DEFAULT_TILE_ZOOM_MAX)
    parser.add_argument("--batch-size", type=int, default=20000)
    parser.add_argument("--insert-batch", type=int, default=25000)
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def create_db(path: Path) -> sqlite3.Connection:
    connection = sqlite3.connect(path)
    connection.execute("PRAGMA journal_mode=OFF")
    connection.execute("PRAGMA synchronous=OFF")
    connection.execute("PRAGMA temp_store=MEMORY")
    connection.execute("PRAGMA locking_mode=EXCLUSIVE")
    connection.executescript(
        """
        CREATE TABLE tile_entries (
            z INTEGER NOT NULL,
            x INTEGER NOT NULL,
            y INTEGER NOT NULL,
            segment_idx INTEGER NOT NULL,
            road_kind INTEGER NOT NULL,
            path BLOB NOT NULL
        );
        CREATE TABLE meta (
            key TEXT PRIMARY KEY,
            value TEXT NOT NULL
        );
        """
    )
    return connection


def write_meta(connection: sqlite3.Connection, values: dict[str, object]) -> None:
    connection.executemany(
        "INSERT OR REPLACE INTO meta(key, value) VALUES(?, ?)",
        [(str(key), json.dumps(value)) for key, value in values.items()],
    )


def main() -> None:
    args = parse_args()
    ensure_dirs()

    if ROAD_TILE_DB_PATH.exists() and not args.force:
        print(f"tile index already exists at {ROAD_TILE_DB_PATH}")
        return

    tmp_path = ROAD_TILE_DB_PATH.with_suffix(".tmp.sqlite3")
    if tmp_path.exists():
        tmp_path.unlink()

    connection = create_db(tmp_path)
    parquet_file = pq.ParquetFile(ROAD_SEGMENTS_PATH)

    total_segments = 0
    total_tile_rows = 0
    insert_rows: list[tuple[int, int, int, int, int, bytes]] = []

    try:
        for batch in parquet_file.iter_batches(
            batch_size=max(1, int(args.batch_size)),
            columns=["coords_json", "rttyp", "mtfcc"],
        ):
            rows = batch.to_pydict()
            batch_size = len(rows["coords_json"])
            for offset in range(batch_size):
                coords_json = rows["coords_json"][offset] or "[]"
                try:
                    coords = json.loads(coords_json)
                except json.JSONDecodeError:
                    total_segments += 1
                    continue
                if len(coords) < 2:
                    total_segments += 1
                    continue

                road_kind = road_kind_from_attrs(
                    rows["rttyp"][offset],
                    rows["mtfcc"][offset],
                )
                for zoom in range(int(args.zoom_min), int(args.zoom_max) + 1):
                    for tile_x, tile_y, path_blob in iter_tile_paths(coords, zoom):
                        insert_rows.append(
                            (
                                int(zoom),
                                int(tile_x),
                                int(tile_y),
                                int(total_segments),
                                int(road_kind),
                                sqlite3.Binary(path_blob),
                            )
                        )
                        total_tile_rows += 1
                total_segments += 1

                if len(insert_rows) >= int(args.insert_batch):
                    connection.executemany(
                        """
                        INSERT INTO tile_entries(z, x, y, segment_idx, road_kind, path)
                        VALUES(?, ?, ?, ?, ?, ?)
                        """,
                        insert_rows,
                    )
                    insert_rows.clear()

            print(
                f"processed_segments={total_segments} inserted_rows={total_tile_rows}",
                flush=True,
            )

        if insert_rows:
            connection.executemany(
                """
                INSERT INTO tile_entries(z, x, y, segment_idx, road_kind, path)
                VALUES(?, ?, ?, ?, ?, ?)
                """,
                insert_rows,
            )
            insert_rows.clear()

        connection.commit()
        connection.execute(
            "CREATE INDEX idx_tile_entries_lookup ON tile_entries(z, x, y)"
        )
        connection.commit()

        zoom_counts = dict(
            connection.execute(
                "SELECT z, COUNT(DISTINCT x || ':' || y) FROM tile_entries GROUP BY z"
            ).fetchall()
        )
        write_meta(
            connection,
            {
                "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "zoom_min": int(args.zoom_min),
                "zoom_max": int(args.zoom_max),
                "segment_count": int(total_segments),
                "tile_entry_count": int(total_tile_rows),
                "tile_count_by_zoom": {str(key): int(value) for key, value in zoom_counts.items()},
            },
        )
        connection.commit()
    finally:
        connection.close()

    if ROAD_TILE_DB_PATH.exists():
        ROAD_TILE_DB_PATH.unlink()
    tmp_path.replace(ROAD_TILE_DB_PATH)
    print(f"wrote {ROAD_TILE_DB_PATH}")


if __name__ == "__main__":
    main()
