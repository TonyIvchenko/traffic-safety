# Traffic Safety Architecture

## Problem

Predict incident probability on US roads as a function of:

- time of day
- day of week
- season
- weather
- road context
- optional live traffic conditions

and expose that prediction as a map overlay.

## Core Assumption

Nationwide public data is good enough for a strong baseline, but not good enough for the best short-horizon live model.

That leads to a layered design:

- `baseline`: fully open, fully reproducible, nationwide
- `realtime`: vendor-optional live enrichment

## Canonical Data Model

Use a single canonical table:

`cell_hour_features`

Columns:

- `cell_id`
- `timestamp_hour_utc`
- `timezone_name`
- `local_hour`
- `day_of_week`
- `month`
- `is_holiday`
- `incident_count_prev_1h`
- `incident_count_prev_24h`
- `incident_count_prev_7d_same_hour`
- `fatal_count_prev_365d`
- `road_km_total`
- `road_km_interstate`
- `road_km_arterial`
- `intersection_density`
- `aadt_proxy`
- `weather_temperature`
- `weather_precipitation`
- `weather_snow`
- `weather_visibility`
- `weather_wind`
- `weather_severity_bucket`
- `work_zone_active`
- `speed_anomaly`
- `active_incident_upstream`
- `target_any_incident_next_1h`
- `target_incident_count_next_1h`

This table should be partitioned by `year/month`.

## Spatial Strategy

Use `H3` as the canonical spatial index.

Reasons:

- easy nationwide tiling
- easy point-to-cell joins
- stable aggregation unit for offline and online scoring
- much simpler than inventing custom grid logic

Recommended starting resolution:

- `res 8` for nationwide baseline
- optional `res 9` for metros if compute/storage allows

## Offline Training Pipeline

### 1. Ingest

Ingest these raw streams into source-specific staging tables:

- incidents
- severe/fatal incidents
- weather observations
- road geometry
- road exposure
- work zones
- optional historical speed/traffic feeds

### 2. Normalize

Normalize all timestamps to UTC and keep:

- `utc timestamp`
- `local timestamp`
- `timezone`

Normalize all geospatial records into:

- `lat`
- `lon`
- `cell_id`

### 3. Feature Build

Build lagged features and structural features:

- history in the last hour, day, and week
- weather bins and forecast deltas
- static road exposure
- neighborhood spillover from adjacent cells

### 4. Train

Recommended training order:

1. calibrated gradient-boosted trees for `P(any incident next hour)`
2. Poisson or negative-binomial model for count estimate
3. isotonic or Platt calibration for probability output

### 5. Evaluate

Do not random-split.

Use:

- temporal holdout
- geographic holdout
- severe-weather holdout
- metro and rural slices

Primary metrics:

- Brier score
- PR-AUC
- top-k hotspot recall
- calibration error

## Online Inference

### Open-Data Path

Inputs:

- current local hour and weekday
- forecast weather
- recent weather observations
- recent incident counts from our own accumulated feed archive
- active work zones where available

Cadence:

- refresh weather every 15 minutes
- refresh overlays every 15 minutes

### Enriched Path

If we license live traffic data, add:

- live incident feed
- live speed or travel time anomaly
- lane closures and congestion indicators

Cadence:

- refresh every 5 minutes

## Serving Pattern

Use a two-stage serving path:

1. baseline cube or feature cache
2. online adjustment and tile publish

Recommended tile outputs:

- raster PNG tiles for quick integration with the current map stack
- later MVT tiles if we want richer client interactions

Recommended API shape:

- `GET /health`
- `GET /traffic-safety/metadata`
- `GET /traffic-safety/tiles/{frame}/{z}/{x}/{y}.png`
- `GET /traffic-safety/score?lat=...&lon=...&timestamp=...`

## Data Retention

Keep three tiers:

- raw source snapshots
- normalized parquet tables
- model-ready feature tables

For live vendor feeds, confirm contract terms before storing long-lived raw data for retraining.

## Recommended Rollout

1. Ship open-data baseline overlay first.
2. Add NOAA/NWS-based online weather adjustment.
3. Add work zone layer.
4. Add commercial live traffic only after the baseline is stable enough to measure uplift.
