"""Catalog of the open datasets this service produces.

A registry describing the model/analysis outputs a consumer can retrieve — the
High Injury Network, equity overlay, grant reports, countermeasures, the risk
overlay, advisory + emergency snapshots, and critical facilities — with the
serving endpoints, supported formats, upstream sources (ids from
``source_catalog.json``), and licensing/provenance notes.

Pure and static: this module has no I/O; :mod:`api_v1` exposes it at
``/v1/datasets`` and the download endpoint keys off ``supports_format``.
"""

from __future__ import annotations

FORMAT_MEDIA_TYPES = {
    "json": "application/json",
    "geojson": "application/geo+json",
    "csv": "text/csv",
    "parquet": "application/vnd.apache.parquet",
}

DATASETS = (
    {
        "id": "high_injury_network",
        "title": "High Injury Network",
        "category": "safety",
        "description": "Road corridors carrying a disproportionate share of fatal and serious-injury crashes.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/grants/hin"],
        "sources": ["fars", "tiger_line"],
        "license": "Derived from public federal data (FARS crashes, Census TIGER roads); verify upstream terms before redistribution.",
        "update_cadence": "annual",
    },
    {
        "id": "equity_overlay",
        "title": "Equity & Justice40 Overlay",
        "category": "equity",
        "description": "Road segments joined to tract social vulnerability (CDC/ATSDR SVI) and Justice40 disadvantaged status (CEJST).",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/equity/hotspots", "/v1/equity/choropleth"],
        "sources": ["tiger_line"],
        "license": "Derived from public data (CDC/ATSDR SVI, CEJST, Census TIGER); SVI and CEJST share 2010 tract boundaries.",
        "update_cadence": "with each source release",
    },
    {
        "id": "grant_reports",
        "title": "Federal Safety-Grant Analysis",
        "category": "safety",
        "description": "Per-jurisdiction SS4A/HSIP safety analysis: crash summary, HIN, systemic risk, and benefit-cost.",
        "formats": ["json", "html"],
        "endpoints": ["/v1/grants/report", "/v1/grants/summary"],
        "sources": ["fars", "tiger_line"],
        "license": "Derived from public federal data (FARS, Census TIGER).",
        "update_cadence": "annual",
    },
    {
        "id": "countermeasures",
        "title": "Countermeasure Recommendations",
        "category": "safety",
        "description": "FHWA Proven Safety Countermeasures matched to HIN segments with Crash Modification Factor benefit-cost.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/countermeasures/segment", "/v1/countermeasures/hotspots"],
        "sources": ["fars", "tiger_line"],
        "license": "CMFs are representative screening values (FHWA CMF Clearinghouse); crash/road data are public.",
        "update_cadence": "annual",
    },
    {
        "id": "risk_overlay",
        "title": "Climatological Risk Overlay",
        "category": "risk",
        "description": "Nationwide weekly (168-hour) crash-risk grid from the model.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/heatmap"],
        "sources": ["us_accidents", "fars", "noaa_ncei", "tiger_line"],
        "license": "Model output derived from public incident and weather data.",
        "update_cadence": "per model release",
    },
    {
        "id": "advisory_snapshot",
        "title": "Road Risk Advisory Snapshot",
        "category": "risk",
        "description": "National relative advisory ('AQI for driving') per metro region for an hour-of-week.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/advisory/national"],
        "sources": ["us_accidents", "fars", "noaa_ncei"],
        "license": "Model output derived from public incident and weather data.",
        "update_cadence": "per model release",
    },
    {
        "id": "critical_facilities",
        "title": "Critical Facilities",
        "category": "emergency",
        "description": "Hospitals, fire stations, and emergency shelters used by the evacuation endpoints.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/emergency/nearest-facility"],
        "sources": ["openstreetmap"],
        "license": "Curated illustrative sample; replace with an authoritative feed (HIFLD, OpenStreetMap) for operational use.",
        "update_cadence": "as sourced",
    },
    {
        "id": "emergency_readiness",
        "title": "Evacuation Readiness Snapshot",
        "category": "emergency",
        "description": "Per-region evacuation readiness: typical conditions, facility coverage, and a readiness rating.",
        "formats": ["json", "geojson"],
        "endpoints": ["/v1/emergency/readiness"],
        "sources": ["us_accidents", "fars", "noaa_ncei", "openstreetmap"],
        "license": "Model output plus curated facility coverage.",
        "update_cadence": "per model release",
    },
)


def list_datasets() -> list[dict]:
    """The full catalog as a list of copies (safe for callers to mutate)."""
    return [dict(dataset) for dataset in DATASETS]


def dataset_ids() -> list[str]:
    return [dataset["id"] for dataset in DATASETS]


def get_dataset(dataset_id) -> dict | None:
    """A dataset by id (case-insensitive), or None if unknown."""
    key = str(dataset_id).strip().lower()
    for dataset in DATASETS:
        if dataset["id"] == key:
            return dict(dataset)
    return None


def supports_format(dataset_id, fmt) -> bool:
    dataset = get_dataset(dataset_id)
    if dataset is None:
        return False
    return str(fmt).strip().lower() in dataset["formats"]


def catalog() -> dict:
    """A discovery view: dataset count, the format vocabulary, and the datasets."""
    return {
        "count": len(DATASETS),
        "formats": sorted({fmt for dataset in DATASETS for fmt in dataset["formats"]}),
        "datasets": list_datasets(),
    }
