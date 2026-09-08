# Critical Facilities Reference

Points of safety for the emergency / evacuation endpoints: hospitals (some trauma
centers), fire stations, and emergency shelters. The committed
`critical_facilities.json` is a **curated illustrative sample** with approximate
coordinates — for production, replace it with an authoritative feed
(HIFLD Hospitals / Fire Stations, OpenStreetMap `amenity=hospital|fire_station|
shelter`) via `scripts/download_facilities.py`.

## Schema

Top level:

| Field | Meaning |
|---|---|
| `schema_version` | dataset schema version |
| `source` / `note` | provenance and screening disclaimer |
| `kinds` | the facility kinds present |
| `facilities` | list of facility records |

Each facility:

| Field | Type | Meaning |
|---|---|---|
| `id` | string | stable identifier |
| `name` | string | facility name |
| `kind` | string | `hospital`, `fire_station`, or `emergency_shelter` |
| `lat`, `lon` | float | approximate coordinates (WGS84) |
| `state` | string | 2-letter state (optional) |
| `capacity` | int \| null | hospital beds or shelter capacity where known |
| `trauma_center` | bool | true for designated trauma centers |

The runtime loads this file (or a processed parquet) through the facilities
store; see `TRAFFIC_SAFETY_FACILITIES_PATH`.
