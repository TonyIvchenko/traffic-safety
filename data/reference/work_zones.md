# Work zones (WZDx)

`work_zones.json` is a curated, illustrative **WZDx** (Work Zone Data Exchange)
feature collection — USDOT's open spec for publishing active construction /
maintenance zones as GeoJSON `RoadEvent`s. Work zones raise crash risk (lane
shifts, narrowed lanes, sudden stops), so this is a dynamic hazard layer on top
of the base risk model.

## Shape

Each feature is a GeoJSON `LineString` with WZDx v4 properties:

| Field | Meaning |
| --- | --- |
| `core_details.road_names` | Affected road(s) |
| `core_details.direction` | Travel direction |
| `core_details.description` | Human-readable summary |
| `vehicle_impact` | `all-lanes-closed` / `some-lanes-closed` / `alternating-one-way` / `all-lanes-open-shift-*` / `all-lanes-open` / `unknown` |
| `start_date`, `end_date` | ISO-8601 activity window (used to filter active events) |

The runtime also accepts an older **flat** layout (fields directly on
`properties`), reads `Point`/`LineString`/`MultiLineString` geometries, and
derives a representative midpoint per event.

## Production data

WZDx coverage is uneven and published per agency. Pull a live feed with
`scripts/download_wzdx.py --url <feed-url>` and point `TRAFFIC_SAFETY_WZDX_PATH`
at it (<https://www.transportation.gov/wzdx> lists feeds). A "no work zone"
answer means "none in this feed", not "none on the ground".

The sample values (geometry, impact, dates) are illustrative, not real; the
dates span a broad window so events read as active during development, except
`wz-sea-i5-006-expired`, which is intentionally past its `end_date`.
