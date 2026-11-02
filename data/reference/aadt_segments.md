# AADT segments (traffic exposure)

`aadt_segments.json` is a curated, illustrative sample of road segments with
approximate **AADT** (annual average daily traffic), used to normalize crash risk
by **exposure** — crashes per million vehicle-miles traveled (VMT = AADT ×
length), the FHWA-standard denominator. Without exposure, busy roads look
dangerous simply because more vehicles pass; per-VMT normalization separates
"how much traffic" from "how dangerous per unit of travel".

## Fields

| Field | Meaning |
| --- | --- |
| `id` | Stable record id |
| `name` | Human-readable road/segment name |
| `state` | USPS state code |
| `lat`, `lon` | Approximate segment midpoint (screening only) |
| `aadt` | Annual average daily traffic (vehicles/day) |
| `functional_class` | `interstate` / `freeway_expressway` / `principal_arterial` / `minor_arterial` / `collector` / `local` |
| `length_km` | Sampled segment length |
| `year` | AADT reference year |

## Production data

Replace this sample with the authoritative **FHWA HPMS** extract (functional
system + AADT): <https://www.fhwa.dot.gov/policyinformation/hpms/>. Run
`scripts/build_aadt_index.py` to convert an HPMS-style CSV into
`data/processed/aadt_segments.parquet`, which the runtime prefers over this
reference file. Point `TRAFFIC_SAFETY_AADT_PATH` at an alternate index to
override both.

The values here are illustrative and not measured; use them only for screening
and local development.
