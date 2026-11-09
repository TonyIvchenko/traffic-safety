# NPMRDS speed segments (traffic-speed risk)

`npmrds_segments.json` is a curated, illustrative sample of road segments with
observed vs free-flow speeds, used to derive **speed-based risk features**. Speed
is a safety signal at both extremes: heavy congestion drives rear-end / sideswipe
crashes, and travel well above free-flow raises crash severity.

## Fields

| Field | Meaning |
| --- | --- |
| `id` / `name` | Segment identifier / road name |
| `lat`, `lon` | Approximate segment midpoint (screening only) |
| `reference_speed_mph` | Free-flow (~85th-percentile) speed |
| `peak_speed_mph` | Typical average speed in the AM/PM peak |
| `offpeak_speed_mph` | Typical average speed off-peak |
| `road_class` | Coarse class label |
| `year` | Data year |

The congestion ratio is `observed / reference`; the screening `speed_risk_index`
is U-shaped — elevated for heavy congestion (ratio < 0.5), mild for moderate
congestion (< 0.75), and elevated again for travel well above free-flow
(> 1.15) — otherwise 1.0.

## Production data

Replace with **NPMRDS** (<https://ops.fhwa.dot.gov/perf_measurement/npmrds/>, via
the RITIS Massive Data Downloader): aggregate a travel-time export into reference
+ peak/offpeak average speeds per segment, then run
`scripts/build_npmrds_index.py` to write `data/processed/npmrds_segments.parquet`
(the runtime prefers it). `TRAFFIC_SAFETY_NPMRDS_PATH` overrides the path.

Values here are illustrative, not measured; use for screening and development
only.
