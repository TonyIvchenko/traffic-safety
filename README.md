# Traffic Safety

US-wide traffic incident risk modeling and map overlay design.

This app is self-contained. You can run it directly with:

```bash
python src/main.py
```

## Current Build

The current standalone app includes:

- an offline nationwide baseline trained from `FARS` fatal-crash data
- hourly NOAA `ISD-Lite` weather joins for training
- a weekly overlay generated from model output plus station climatology fallback
- a manual predictor tab that uses climatology when live weather is not requested
- a live predictor tab and `/api/live-risk` endpoint that can use:
  - `NWS` first, with no API key
  - `OpenWeather` if `TRAFFIC_SAFETY_ENABLE_OPENWEATHER=1` and `OPENWEATHER_API_KEY` is set
  - `Tomorrow.io` if `TRAFFIC_SAFETY_ENABLE_TOMORROW_IO=1` and `TOMORROW_IO_API_KEY` is set

If you want the interactive map basemap, set `GMAPS_API_KEY` first:

```bash
export GMAPS_API_KEY=your_google_maps_key
python src/main.py
```

The goal here is not a one-state demo. The goal is a nationwide system that can:

- train offline on historical incident, weather, and road-context data
- score risk online for the next hour or next few hours
- render a map overlay that changes by time of day, day of week, weather, and live conditions

## Offline Pipeline

To rebuild the full offline stack inside the shared `playground` conda env:

```bash
conda run -n playground python scripts/build_dataset.py
conda run -n playground python scripts/download_weather.py
conda run -n playground python scripts/train_model.py
conda run -n playground python scripts/generate_tiles.py
```

That pipeline writes:

- processed incidents under `data/processed`
- processed NOAA weather under `data/processed/weather`
- the trained bundle at `models/traffic_safety.joblib`
- the overlay tiles at `tiles`

## Geographic Enrichment & Costs

Grant, equity, and countermeasure analysis need each road segment and H3 cell
tagged with its county and census tract, plus a way to value crashes in dollars.

```bash
conda run -n playground python scripts/download_geographies.py   # TIGER county + tract shapefiles
conda run -n playground python scripts/build_geo_lookup.py       # centroid -> county/tract GEOID
```

That writes county/tract lookups to `data/processed/geo/`:

- `segment_geoid.csv.gz` — `segment_id, county_geoid, tract_geoid`
- `cell_geoid.csv.gz` — `cell_id, county_geoid, tract_geoid`

Supporting modules:

- `src/geo_lookup.py` — `county_of(lat, lon)` / `tract_of(lat, lon)` via a shapely
  point-in-polygon index over the TIGER boundaries (requires `shapely`).
- `src/crash_costs.py` — FHWA KABCO comprehensive crash costs (2016 USD,
  `FHWA-SA-17-071`) plus `expected_annual_cost(...)` and `benefit_cost(...)` for
  HSIP/SS4A benefit-cost analysis.

## Live Provider Flags

Feature flags are environment-variable based:

```bash
export TRAFFIC_SAFETY_ENABLE_NWS=1
export TRAFFIC_SAFETY_ENABLE_OPENWEATHER=0
export TRAFFIC_SAFETY_ENABLE_TOMORROW_IO=0
export TRAFFIC_SAFETY_LIVE_PROVIDERS=nws,openweather,tomorrow
```

Optional paid providers:

```bash
export TRAFFIC_SAFETY_ENABLE_OPENWEATHER=1
export OPENWEATHER_API_KEY=...

export TRAFFIC_SAFETY_ENABLE_TOMORROW_IO=1
export TOMORROW_IO_API_KEY=...
```

## Public API (`/v1`)

A versioned, openly accessible REST API exposes the model outputs for external
consumers. Interactive docs are served at `/v1/docs` (Swagger) and `/v1/redoc`;
the schema is at `/v1/openapi.json`. There is **no authentication** — access is
controlled only by a per-client-IP rate limiter.

Endpoints:

| Method & path | Purpose |
|---|---|
| `GET /v1/health` | API liveness + model readiness |
| `GET /v1/meta` | Discovery: coverage bbox, frames, risk bands, providers, limits, units |
| `GET /v1/risk/point` | Risk at one point — `mode=climatology` (default) or `mode=live` |
| `GET /v1/risk/point/weekly` | Full 168-hour risk curve for a point + safest/riskiest hour |
| `POST /v1/risk/route` | Score a drive (waypoints or GeoJSON LineString) — per-step risk, route index, riskiest stretch |
| `GET /v1/risk/area` | Scored road segments in a bounding box |

The geospatial endpoints (`/v1/risk/route`, `/v1/risk/area`) accept
`?format=geojson` and return a GeoJSON `FeatureCollection` (`application/geo+json`)
that drops straight into Leaflet/Mapbox/QGIS. Risk responses use SI units
(`celsius`, `m/s`, `km`) and a `risk_score` probability in `[0, 1]`.

Examples:

```bash
# Climatological point risk (Fri 5pm, September, downtown LA)
curl "http://127.0.0.1:8080/v1/risk/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9"

# Live point risk (current NWS conditions)
curl "http://127.0.0.1:8080/v1/risk/point?lat=34.0522&lon=-118.2437&mode=live"

# When is this spot safest during the week?
curl "http://127.0.0.1:8080/v1/risk/point/weekly?lat=34.0522&lon=-118.2437&month=9"

# Score a drive (climatological), as GeoJSON
curl -X POST "http://127.0.0.1:8080/v1/risk/route?format=geojson" \
  -H 'Content-Type: application/json' \
  -d '{"waypoints": [[-118.2437,34.0522],[-118.40,34.02],[-118.49,34.02]],
       "mode":"climatology","day_of_week":5,"hour":17,"month":9,"sample_spacing_km":3.0}'

# Scored segments in a bounding box
curl "http://127.0.0.1:8080/v1/risk/area?min_lat=33.9&max_lat=34.2&min_lon=-118.5&max_lon=-118.1"
```

Configuration (all optional):

```bash
# Rate limiting (in-process, per client IP; limits are per server instance)
export TRAFFIC_SAFETY_RATE_LIMIT_PER_MIN=120   # default 120; set 0 or ENABLED=0 to disable
export TRAFFIC_SAFETY_RATE_LIMIT_BURST=120     # default = per-minute rate
export TRAFFIC_SAFETY_RATE_LIMIT_ENABLED=1

# CORS for browser clients (comma-separated origins; default "*")
export TRAFFIC_SAFETY_CORS_ORIGINS="*"
```

Successful `/v1` responses carry `X-RateLimit-Limit` / `X-RateLimit-Remaining`;
throttled requests return `429` with `Retry-After`.

## Federal Safety-Grant Analysis (SS4A / HSIP)

Federal safety programs — USDOT's **Safe Streets and Roads for All (SS4A)** and
the FHWA **Highway Safety Improvement Program (HSIP)** — require a data-driven
safety analysis: a High Injury Network (HIN), systemic risk screening, a crash
summary, and benefit-cost justification. This service generates that analysis for
any jurisdiction (state, county, or census tract) directly from FARS fatal
crashes and the TIGER road network.

### Build the datasets (offline)

Requires the [geographic enrichment](#geographic-enrichment--costs) above (TIGER
county boundaries).

```bash
conda run -n playground python scripts/build_hin.py            # HIN            -> data/processed/safety/high_injury_network.parquet
conda run -n playground python scripts/build_grant_dataset.py  # per-county     -> data/reports/grant/<GEOID>.json
conda run -n playground python scripts/build_grant_index.py    # rollup index   -> data/reports/grant/index.parquet
```

- `build_hin.py` ranks segments by severity-weighted fatal crashes per km and
  flags the smallest set carrying the target share (`--target-share`, default
  50%) of the weighted total.
- `build_grant_dataset.py` aggregates the HIN, systemic risk, crash summary, and
  benefit-cost per county (`--counties 06037` to scope, `--crash-reduction` /
  `--treatment-cost-per-km` to tune the benefit-cost).
- `build_grant_index.py` flattens every county report into one Parquet table for
  fast all-jurisdiction serving.

The serving directory is configurable via `TRAFFIC_SAFETY_GRANT_DIR` (default
`data/reports/grant`), so a refreshed dataset can be dropped in without a restart.

### Serve the analysis (`/v1/grants`)

| Method & path | Purpose |
|---|---|
| `GET /v1/grants/summary?geoid=` | Headline crash + HIN + benefit-cost summary for a jurisdiction |
| `GET /v1/grants/hin?geoid=` (or `?min_lat=&max_lat=&min_lon=&max_lon=`) | HIN corridors — JSON or `?format=geojson` |
| `GET /v1/grants/report?geoid=&format=json\|html` | Full analysis — JSON, or a self-contained printable HTML deliverable |

```bash
# County headline summary
curl "http://127.0.0.1:8080/v1/grants/summary?geoid=06037"

# High Injury Network corridors as GeoJSON (drops into QGIS/Leaflet)
curl "http://127.0.0.1:8080/v1/grants/hin?geoid=06037&format=geojson"

# Download the full grant report as a standalone HTML document
curl "http://127.0.0.1:8080/v1/grants/report?geoid=06037&format=html" -o safety-analysis-06037.html
```

GEOIDs are 2-digit (state), 5-digit (county), or 11-digit (census tract);
`/v1/meta` reports how many jurisdictions are loaded.

### Methodology & disclaimers

- **Severity basis:** FARS fatal crashes only (KABCO "K"). The US-Accidents
  `Severity` field is traffic impact, not injury severity, and is deliberately
  not used for the HIN.
- **Crash costs:** FHWA comprehensive KABCO costs (2016 USD, `FHWA-SA-17-071`).
- **Benefit-cost** currently applies a **placeholder** crash-reduction factor and
  a nominal per-km treatment cost; corridor-specific Crash Modification Factors
  (from the countermeasures work) are intended to replace the placeholder.
- This is **decision-support screening, not an official government determination
  or a substitute for an engineering study.** Validate corridor rankings on the
  ground before programming funds.

## Equity & Justice40 Overlay

Surfaces road segments that are both **high-risk and underserved** by joining the
risk model to two federal equity datasets at census-tract level: the CDC/ATSDR
**Social Vulnerability Index (SVI)** and the **Justice40 / CEJST** "disadvantaged
community" designation. This supports Justice40 reporting (the goal that 40% of
benefits flow to disadvantaged communities) and equitable project prioritization.

### Build the datasets (offline)

```bash
conda run -n playground python scripts/download_equity_data.py   # CDC SVI + CEJST CSVs -> data/raw/equity/
conda run -n playground python scripts/build_equity_index.py     # per-tract index      -> data/processed/equity/tract_equity.csv.gz
conda run -n playground python scripts/build_equity_overlay.py   # per-segment overlay  -> data/processed/equity/segment_equity.parquet
```

- `download_equity_data.py` fetches the source CSVs (`--svi-url` / `--cejst-url`
  override the defaults, since these federal endpoints move).
- `build_equity_index.py` joins SVI (`RPL_THEMES` percentile) and CEJST
  (`Identified as disadvantaged`) per tract.
- `build_equity_overlay.py` attaches each segment's tract equity, forecast risk,
  and crash count.

Serving paths are configurable via `TRAFFIC_SAFETY_EQUITY_PATH` (tract index) and
`TRAFFIC_SAFETY_EQUITY_OVERLAY_PATH` (segment overlay).

### Serve the analysis (`/v1/equity`)

| Method & path | Purpose |
|---|---|
| `GET /v1/equity/point?lat=&lon=` | Tract SVI percentile + band + Justice40 flag at a location |
| `GET /v1/equity/hotspots` | High-risk segments in disadvantaged / high-SVI tracts (`only_disadvantaged`, `min_svi`, bbox; JSON or `?format=geojson`) |
| `GET /v1/equity/summary?geoid=` | Crash/risk disparity for a jurisdiction (disparity ratios, SVI-weighted crash burden) |
| `GET /v1/equity/choropleth` | Tract-level equity map as GeoJSON (SVI / Justice40 / risk) |

```bash
# Equity at a point
curl "http://127.0.0.1:8080/v1/equity/point?lat=34.0522&lon=-118.2437"

# Equity-prioritized hotspots (dangerous AND underserved) as GeoJSON
curl "http://127.0.0.1:8080/v1/equity/hotspots?only_disadvantaged=true&format=geojson"

# County disparity: are crashes/risk disproportionately in disadvantaged tracts?
curl "http://127.0.0.1:8080/v1/equity/summary?geoid=06037"
```

### Provenance & caveats

- **Sources:** CDC/ATSDR SVI and CEJST (Justice40). SVI and CEJST are paired on a
  common **2010 census-tract** vintage; road segments are tagged from TIGER 2024
  (**2020 tracts**), so the overlay join covers tracts unchanged between 2010 and
  2020 and leaves re-tracted segments with unknown equity (a 2010→2020 crosswalk
  would close the gap).
- The crash-burden metric weights crashes by SVI; it is **not** a true per-capita
  rate (the index does not carry tract population).
- Equity flags describe **communities, not individuals**, and are decision-support
  inputs — not a substitute for community engagement.

## Countermeasure Recommendations (FHWA CMFs)

For each high-risk corridor, recommends proven safety countermeasures with a
benefit-cost estimate. It infers the corridor's likely crash types from its
roadway attributes, matches applicable FHWA Proven Safety Countermeasures, and
values the crashes each would avoid (via its Crash Modification Factor) against
the treatment cost.

The curated CMF table is committed at
[data/reference/countermeasures.json](data/reference/countermeasures.json)
(schema in `countermeasures.md`).

### Build the report (offline)

```bash
conda run -n playground python scripts/build_countermeasure_report.py --top-n 100 --analysis-years 5
```

Writes `data/reports/countermeasures.csv` and `.geojson` — one row per HIN
segment with its best-benefit-cost treatment. The serving store reads the HIN
parquet, configurable via `TRAFFIC_SAFETY_CM_SEGMENTS_PATH`.

### Serve the analysis (`/v1/countermeasures`)

| Method & path | Purpose |
|---|---|
| `GET /v1/countermeasures/segment?segment_id=` | Ranked treatments for a segment, each with CMF + benefit-cost |
| `GET /v1/countermeasures/hotspots` | Top crash-risk segments each with a recommended treatment (bbox, `min_fatal_crashes`; JSON or `?format=geojson`) |

```bash
# Ranked treatments for one HIN segment
curl "http://127.0.0.1:8080/v1/countermeasures/segment?segment_id=<id>"

# Hotspots in a bbox, each with a recommended treatment, as GeoJSON
curl "http://127.0.0.1:8080/v1/countermeasures/hotspots?min_lat=33.9&max_lat=34.2&min_lon=-118.5&max_lon=-118.1&format=geojson"
```

The corridor-specific CMFs also drive the grant benefit-cost by default
(`scripts/build_grant_dataset.py --benefit-cost-method cmf`, with `flat` as a
fallback).

### Methodology & disclaimers

- **CMFs:** a CMF multiplies expected crashes (CMF < 1 reduces them); the crash
  reduction factor is `CRF = 1 − CMF`. Stacked treatments combine with a
  conservative diminishing-returns rule, not naive multiplication.
- **Benefit-cost** values avoided crashes at the FHWA comprehensive fatal-crash
  cost, discounted over the treatment service life.
- Each recommendation carries a `cmf_confidence` (from the CMF Clearinghouse star
  rating). Roadway context is inferred from MTFCC (urban assumed when unknown).
- These are **representative screening values**, not project-specific CMFs, and
  **not a substitute for an engineering study** before programming funds.

## Road Risk Advisory ("AQI for driving")

A public, five-level advisory that answers "how risky is driving here, right
now, compared to normal?" The level is a location's risk **percentile within its
own 168-hour weekly climatology**, so it shifts with time and place — a wet
Saturday 2 a.m. reads high, a quiet Wednesday noon reads low — instead of
saturating at "extreme" everywhere dense (which absolute crash-risk does, since
urban crash density is high around the clock).

| Level | Color | Percentile of the local week |
|---|---|---|
| Low | `#1a9850` | below 50th |
| Moderate | `#a6d96a` | 50th–70th |
| Elevated | `#fdae61` | 70th–85th |
| High | `#f46d43` | 85th–95th |
| Extreme | `#d7191c` | 95th and above |

The weekly reference comes from the raw model (the display overlay is normalised
and too saturated to discriminate); it is scored in one batched pass per point
and cached. Regions are ~two dozen metro bounding boxes (see `/v1/meta`).

### Serve the analysis (`/v1/advisory`)

| Method & path | Purpose |
|---|---|
| `GET /v1/advisory/point` | Advisory for a point (`mode=climatology\|live`, `day_of_week`/`hour`/`month`, or live `forecast_hours`/`provider`; `compare=true` adds now-vs-normal) |
| `GET /v1/advisory/region` | Advisory for a metro (`region=<id>` or `lat`+`lon`; same modes; live samples a grid) |
| `GET /v1/advisory/national` | All metros ranked by how elevated each is vs its own normal (`min_level`, JSON or `?format=geojson&geometry=point\|bbox`) |

```bash
# Point advisory now vs a normal Friday evening (live, with comparison)
curl "http://127.0.0.1:8080/v1/advisory/point?lat=34.0522&lon=-118.2437&mode=live&compare=true"

# A metro's climatological advisory for Friday 5pm
curl "http://127.0.0.1:8080/v1/advisory/region?region=los_angeles&day_of_week=5&hour=17"

# Nationwide snapshot as GeoJSON (metro centroids coloured by level)
curl "http://127.0.0.1:8080/v1/advisory/national?day_of_week=5&hour=17&format=geojson"
```

Each response carries a context-aware `message` and honest `caveats`; the scale,
percentile bands, and region catalog are discoverable under `meta.advisory` at
`/v1/meta`.

### Precompute a snapshot (offline)

```bash
conda run -n playground python scripts/build_advisory_snapshot.py --month 1 --day-of-week 5 --hour 17
```

Writes each region's weekly reference profile plus a ranked national snapshot
(JSON + GeoJSON) to `data/advisory/` — precomputing lets the API warm its cache
instead of paying the first-request cost.

### Methodology & caveats

- The level is **relative** — a percentile within the location's own typical
  week — not an absolute crash probability. `risk_score` (the raw 0–1 model
  value) is reported alongside for reference.
- Out-of-coverage or flat-climatology locations have no relative signal and fall
  back to an **absolute** scale (`advisory.basis == "absolute"`).
- Live mode depends on third-party weather providers; a partial provider outage
  is tolerated (the reading reflects the points that scored) and flagged.
- Regions are approximate metro bounding boxes for screening, **not** official
  MSA boundaries. The advisory is guidance, not a guarantee of safety.

## Recommended Shape

The most practical nationwide design is a two-layer system:

- open-data baseline model for full-US coverage
- optional commercial enrichment layer for better short-horizon accuracy

The baseline model should be the system of record. That keeps us deployable even if we do not buy live traffic data on day one.

## Recommended Training Stack

- historical incidents: `US-Accidents` as the broad non-fatal incident corpus
- severe/fatal calibration: `NHTSA FARS`
- historical weather: `NOAA NCEI` hourly observations
- road geometry and topology: `TIGER/Line` plus optional `OpenStreetMap`
- traffic exposure: `FHWA HPMS` AADT and road class
- work zones and restrictions: `WZDx` where available
- optional historical traffic: `NPMRDS` for public agencies or `HERE` / `TomTom` / `INRIX`

## Spatial And Temporal Unit

Use `H3` as the core modeling unit, not raw crash points and not custom raster cells.

- spatial unit: H3 cells over the contiguous US, with Alaska/Hawaii handled separately
- temporal unit: hourly
- prediction target: `P(any reportable incident in the next hour in this cell)`
- secondary target: expected incident count in the next hour

This gives us a clean nationwide index for training, online feature generation, and map serving.

## Model Layout

Train two models:

1. `baseline_risk_model`
   Uses hour-of-week, month, holidays, road density, road class, historical incident intensity, and weather.

2. `live_adjustment_model`
   Uses live forecast weather, active incidents, work zones, and optional speed anomaly features from commercial traffic feeds.

The serving path should be:

- precompute baseline hourly risk surfaces
- apply online adjustments every 5 to 15 minutes
- publish tiles for the current timestamp and a small forecast horizon

## Overlay Strategy

We should not try to render every road segment individually first.

Start with:

- H3-based risk overlay converted to vector or raster tiles
- time slider or time-aware endpoint for `hour_of_week` and forecast timestamp

If we later buy segment-level feeds, we can add a road-segment overlay on top for highways and arterial roads.

## Why This Shape

This avoids the main nationwide trap: there is no clean, open, public, full-US, all-severity, fully geocoded crash census with strong real-time support. A pure official-data-only stack is not enough for a serious nationwide online risk layer.

The open stack gets us full-US coverage. The commercial layer improves freshness and short-horizon accuracy.

## Near-Term Build Plan

1. Normalize all historical data into `parquet` partitions keyed by `year/month`.
2. Build an H3 feature table with one row per `cell_id,timestamp_hour`.
3. Train a calibrated baseline model.
4. Add a weather-only online scorer.
5. Add optional live traffic connectors behind provider-specific adapters.
6. Serve an overlay tile endpoint.

## Development

Install the runtime and test dependencies, then run the test suite:

```bash
pip install -r requirements.txt -r requirements-dev.txt
pytest
```

## Files In This Folder

- [main.py](/Users/toxa/git/traffic-safety/src/main.py)
- [ARCHITECTURE.md](/Users/toxa/git/traffic-safety/ARCHITECTURE.md)
- [DATA_SOURCES.md](/Users/toxa/git/traffic-safety/DATA_SOURCES.md)
- [source_catalog.json](/Users/toxa/git/traffic-safety/source_catalog.json)
