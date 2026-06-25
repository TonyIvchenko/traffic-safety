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
