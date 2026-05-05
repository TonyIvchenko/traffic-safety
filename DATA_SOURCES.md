# Traffic Safety Data Sources

This file separates `offline training` sources from `online inference` sources.

## Best Open Baseline

If we want a nationwide baseline without buying data, the strongest stack is:

- incidents: `US-Accidents`
- severe calibration: `FARS`
- weather history: `NOAA NCEI`
- weather forecast and alerts: `NWS API`
- road geometry: `TIGER/Line`
- optional extra road attributes: `OpenStreetMap`
- exposure: `FHWA HPMS`
- work zones: `WZDx`

## Offline Training Sources

### 1. US-Accidents

- coverage: nationwide, community-curated, non-fatal and fatal events mixed
- use: primary historical incident corpus for the baseline model
- strengths: large, timestamped, geocoded, weather/context fields
- caveats: not an official crash census; licensing and source provenance need review before production use
- source: https://smoosavi.org/datasets/us_accidents

### 2. NHTSA FARS

- coverage: nationwide fatal crashes only
- use: severity calibration, fatal-risk overlay, model validation for severe outcomes
- strengths: official US fatal crash census
- caveats: only fatal crashes, so not enough for general incident probability by itself
- source: https://www.nhtsa.gov/research-data/fatality-analysis-reporting-system-fars

### 3. NHTSA CRSS

- coverage: nationwide sampled crash data
- use: optional calibration and bias checks
- strengths: official source for broader crash patterns
- caveats: sample, not full census, so weak as a map-training backbone
- source: https://www.nhtsa.gov/crash-data-systems/crash-report-sampling-system

### 4. State And City Open Crash Portals

- coverage: fragmented, but sometimes excellent
- use: override or improve local labels where public high-quality crash data exists
- strengths: often more detailed than federal public files
- caveats: no clean nationwide uniform download path
- source: https://www.data.gov/

### 5. NOAA NCEI Hourly Weather

- coverage: nationwide station observations
- use: historical weather joins for training
- strengths: official, long time range, reproducible
- caveats: station-based, so we need interpolation or nearest-station joins
- source: https://www.ncei.noaa.gov/
- implemented here: `NOAA ISD-Lite` nearest representative-station hourly join plus station climatology fallback

### 6. FHWA HPMS

- coverage: nationwide highway and road performance inventory
- use: AADT, functional class, exposure proxies
- strengths: official federal source for road extent and usage statistics
- caveats: not a drop-in live feed and not perfect for every local road segment
- source: https://www.fhwa.dot.gov/policyinformation/hpms/

### 7. TIGER/Line

- coverage: nationwide road geometry
- use: core open geometry for map joins and overlay generation
- strengths: official nationwide geometry source
- caveats: fewer traffic attributes than commercial road networks
- source: https://www.census.gov/geographies/mapping-files/time-series/geo/tiger-line-file.html

### 8. OpenStreetMap

- coverage: nationwide and global
- use: extra road features like lane counts, turn restrictions, and maxspeed where present
- strengths: richer road attributes than TIGER in many places
- caveats: inconsistent completeness by region, ODbL license implications
- source: https://www.openstreetmap.org/

### 9. WZDx

- coverage: growing but uneven work zone feed coverage
- use: work zone features for training and inference
- strengths: official public exchange format and feed registry
- caveats: coverage is not universal and historical archives are not guaranteed unless we keep them ourselves
- source: https://www.transportation.gov/wzdx

### 10. NPMRDS

- coverage: National Highway System
- use: historical speed and travel time features, especially for public-sector deployments
- strengths: national, structured, 5-minute travel time and speed data
- caveats: not a general public open dataset; access is aimed at federal, state, and regional agencies
- source: https://ops.fhwa.dot.gov/publications/fhwahop20028/index.htm

## Online Inference Sources

### Free Or Open

#### NWS API

- use: forecast weather, alerts, severe-weather context
- best for: open nationwide online scoring
- caveats: weather, not traffic
- source: https://www.weather.gov/documentation/services-web-api
- implemented here: yes, as the default live weather adapter for `/api/live-risk`

#### State Or Regional 511 Feeds

- use: current incidents, closures, restrictions
- best for: supplementing open-data live incident coverage
- caveats: patchwork, inconsistent schemas, no clean nationwide contract

#### WZDx Feeds

- use: active work zones and lane restrictions
- best for: live work zone features
- caveats: useful but incomplete
- source: https://www.transportation.gov/wzdx

### Commercial / Subscription

#### HERE

- use: live incidents, flow, road events, and historical traffic products
- best for: strong nationwide live enrichment
- caveats: enterprise licensing and retention terms matter
- source: https://developer.here.com/

#### TomTom

- use: live incidents, live flow, and traffic statistics
- best for: strong map-ready live traffic signals
- caveats: commercial API pricing and usage terms
- source: https://developer.tomtom.com/

#### INRIX

- use: real-time and historical traffic speeds, travel times, and incident intelligence
- best for: strongest short-horizon congestion and anomaly features
- caveats: typically enterprise or agency procurement
- source: https://inrix.com/

#### Mapbox Traffic

- use: live and typical traffic conditions
- best for: traffic speed anomaly features if we already use Mapbox
- caveats: not the cleanest standalone incident history source
- source: https://www.mapbox.com/

#### Weather Vendors

- use: higher-SLA forecast and nowcast data
- candidates: `Tomorrow.io`, `AccuWeather`, `Meteomatics`, `OpenWeather`
- best for: production-grade live weather if NWS alone is not enough
- implemented here: optional `Tomorrow.io` and `OpenWeather` adapters behind feature flags

## Recommendation

### Open-data baseline

- `US-Accidents + FARS + NOAA + HPMS + TIGER + WZDx`

This is the best fully open nationwide path.

### Production-grade live system

- open-data baseline
- plus `HERE` or `TomTom` or `INRIX`
- plus stronger forecast weather if we need SLA or denser forecast variables

That gives us a realistic nationwide service instead of a research-only overlay.
