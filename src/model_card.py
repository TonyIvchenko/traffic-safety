"""Model card for the road-risk model (a transparency / trust artifact).

Builds a structured "Model Card" in the sense of Mitchell et al. (2019), "Model
Cards for Model Reporting" — model details, intended use, factors, metrics,
training/evaluation data, quantitative analyses, ethical considerations, and
caveats. It composes facts the service already holds (the model bundle's
metadata + metrics, the calibration/backtest report, the risk thresholds, and
the documented caveats) into one honest, machine-readable document.

Pure and degrade-not-crash: every field tolerates missing/malformed inputs and
falls back to ``None`` / an empty list rather than raising, so the card renders
even when the model bundle or report is absent.
"""

from __future__ import annotations

from datetime import datetime, timezone

CARD_SCHEMA_VERSION = "1.0"
CARD_FRAMEWORK = "model_cards_mitchell_2019"
CITATION = (
    "Mitchell et al. (2019), Model Cards for Model Reporting, FAT* '19. "
    "doi:10.1145/3287560.3287596"
)

# Editorial sections. These describe the service's intended use and ethical
# posture; they match the stance documented across the README and /v1/meta.
PRIMARY_USES = [
    "Prioritizing where to target road-safety investments and countermeasures",
    "Situational awareness and public advisories (relative 'AQI for driving')",
    "Federal safety-grant analysis (SS4A / HSIP) and benefit-cost screening",
    "Research and planning on crash-risk patterns across place and time",
]
PRIMARY_USERS = [
    "State DOTs, MPOs, counties, and city traffic-safety offices",
    "Emergency managers and evacuation planners",
    "Researchers and civic / public-interest developers",
]
OUT_OF_SCOPE_USES = [
    "Predicting a specific crash for a specific vehicle, driver, or moment",
    "Individual driver surveillance, scoring, insurance pricing, or liability",
    "Law-enforcement targeting of individuals",
    "Use outside the United States coverage area",
]
ETHICAL_CONSIDERATIONS = [
    "Crash data under-reports some crash types, areas, and populations; the "
    "model can inherit that reporting bias.",
    "Climatological risk reflects historical patterns, which can encode past "
    "inequities in road design and investment; the SVI / Justice40 equity "
    "overlay is provided so the distribution of risk and of recommended "
    "spending can be examined, not hidden.",
    "Outputs describe risk for an area and time, not fault or blame for any "
    "individual, and are not validated for enforcement or punitive use.",
    "Live-weather scoring is optional and off by default; the base model is "
    "climatological and must not be read as a real-time collision warning.",
]
MODEL_CAVEATS = [
    "Representative Crash Modification Factors are used for countermeasure "
    "benefit-cost, not project-specific CMFs.",
    "Several shipped reference datasets (facilities, AADT, NPMRDS, work zones) "
    "are curated illustrative samples; build the production indexes from the "
    "upstream sources before operational use.",
    "Rate limits and usage metering are enforced per server instance.",
    "Emergency distances are straight-line, not routed.",
    "Validate the US-Accidents dataset license before any production training.",
]


def _clean_str(value: object) -> str | None:
    text = str(value).strip() if value is not None else ""
    return text or None


def _clean_list(values: object) -> list:
    if not isinstance(values, (list, tuple)):
        return []
    return [v for v in values if v is not None]


def _finite(value: object) -> float | None:
    try:
        number = float(value)  # type: ignore[arg-type]
    except (TypeError, ValueError):
        return None
    if number != number or number in (float("inf"), float("-inf")):
        return None
    return number


def _year_span(years: object) -> dict | None:
    cleaned = []
    for year in _clean_list(years):
        value = _finite(year)
        if value is not None:
            cleaned.append(int(value))
    if not cleaned:
        return None
    return {"start": min(cleaned), "end": max(cleaned), "years": sorted(set(cleaned))}


def _performance(metrics: object, report: object) -> dict:
    """Pull headline discrimination metrics from the bundle or the report."""
    merged: dict = {}
    for source in (metrics, (report or {}).get("metrics") if isinstance(report, dict) else None):
        if isinstance(source, dict):
            for key, value in source.items():
                number = _finite(value)
                if number is not None:
                    merged[str(key)] = number
    out: dict = {}
    for label, keys in (
        ("roc_auc", ("val_roc_auc", "roc_auc", "auc")),
        ("average_precision", ("val_average_precision", "average_precision", "pr_auc")),
        ("brier_score", ("val_brier", "brier_score", "brier")),
    ):
        for key in keys:
            if key in merged:
                out[label] = round(merged[key], 4)
                break
    return out


def build_model_card(
    *,
    service_name: str = "Road Risk Monitor",
    model_version: str,
    model_ready: bool,
    metadata: dict | None = None,
    metrics: dict | None = None,
    report: dict | None = None,
    risk_thresholds: dict | None = None,
    coverage: dict | None = None,
    frame_count: int | None = None,
    caveats: list | None = None,
    data_sources: dict | None = None,
    license_name: str = "MIT",
) -> dict:
    metadata = metadata or {}
    report = report if isinstance(report, dict) else {}
    data_sources = data_sources or {}

    features = _clean_list(metadata.get("feature_names"))
    weather_features = _clean_list(metadata.get("weather_feature_names"))
    report_available = bool(report.get("available"))

    performance = _performance(metrics, report)

    thresholds = {}
    if isinstance(risk_thresholds, dict):
        for level in ("moderate", "high", "extreme"):
            value = _finite(risk_thresholds.get(level))
            if value is not None:
                thresholds[level] = round(value, 4)

    return {
        "schema_version": CARD_SCHEMA_VERSION,
        "framework": CARD_FRAMEWORK,
        "framework_citation": CITATION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "model_details": {
            "name": f"{service_name} crash-risk model",
            "version": _clean_str(model_version),
            "ready": bool(model_ready),
            "type": _clean_str(metadata.get("model_type")),
            "objective": "binary_classification_crash_hour_probability",
            "developer": service_name,
            "license": license_name,
            "spatial_unit": {
                "scheme": "H3",
                "resolution": metadata.get("resolution") if _finite(metadata.get("resolution")) is not None else None,
                "cells_modeled": metadata.get("spatial_cells"),
            },
            "temporal_unit": {
                "scheme": "weekly_cycle_hourly",
                "frames": frame_count,
            },
            "features": {
                "count": len(features) or metadata.get("n_features"),
                "names": features,
                "weather_features": weather_features,
            },
            "training_window": _year_span(metadata.get("train_years")),
            "evaluation_window": _year_span(metadata.get("eval_years")),
            "negative_sampling_ratio": metadata.get("negative_sampling_ratio"),
            "geographic_coverage": coverage or {},
        },
        "intended_use": {
            "primary_uses": list(PRIMARY_USES),
            "primary_users": list(PRIMARY_USERS),
            "out_of_scope_uses": list(OUT_OF_SCOPE_USES),
        },
        "factors": {
            "relevant": [
                "Geography (U.S. coverage area; coverage is uneven by region)",
                "Time of week and hour (weekly climatological cycle)",
                "Weather conditions (temperature, humidity, wind, wet hours)",
                "Local crash history and exposure",
            ],
            "evaluation": "Held-out evaluation year(s); see evaluation_window.",
        },
        "metrics": {
            "performance": performance,
            "decision_thresholds": thresholds,
            "notes": (
                "ROC-AUC and average precision are discrimination metrics on the "
                "held-out year. Decision thresholds are the learned risk-score "
                "quantiles that separate the moderate / high / extreme levels."
            ),
        },
        "training_data": {
            "sources": _clean_list(data_sources.get("training")),
            "window": _year_span(metadata.get("train_years")),
            "notes": (
                "Incident history joined to road geometry and historical weather, "
                "with negative sampling of non-crash cell-hours."
            ),
        },
        "evaluation_data": {
            "sources": _clean_list(data_sources.get("training")),
            "window": _year_span(metadata.get("eval_years")),
            "live_inference_sources": _clean_list(data_sources.get("live")),
        },
        "quantitative_analyses": {
            "calibration_report_available": report_available,
            "calibration_report_endpoint": "/v1/model/report",
            "decision_thresholds": thresholds,
        },
        "ethical_considerations": list(ETHICAL_CONSIDERATIONS),
        "caveats_and_recommendations": [
            *[c for c in (_clean_str(x) for x in _clean_list(caveats)) if c],
            *MODEL_CAVEATS,
        ],
    }
