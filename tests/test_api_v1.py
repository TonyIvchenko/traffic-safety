from __future__ import annotations

from datetime import datetime, timezone
import importlib.util
from pathlib import Path
from types import SimpleNamespace
import sys

from fastapi.testclient import TestClient


def load_module():
    module_path = Path(__file__).resolve().parents[1] / "src" / "main.py"
    spec = importlib.util.spec_from_file_location("traffic_safety_main", module_path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


MODULE = load_module()


def test_v1_health():
    client = TestClient(MODULE.api)
    response = client.get("/v1/health")
    assert response.status_code == 200
    payload = response.json()
    assert payload["status"] == "ok"
    assert payload["api_version"] == "1.0"
    assert payload["model_ready"] is True
    assert response.headers["cache-control"] == "no-store"


def test_v1_meta_describes_coverage_and_providers():
    client = TestClient(MODULE.api)
    response = client.get("/v1/meta")
    assert response.status_code == 200
    payload = response.json()
    assert set(payload["coverage"]) == {"lat_min", "lat_max", "lon_min", "lon_max"}
    assert payload["timeline"]["frame_count"] == 168
    assert payload["risk_levels"] == ["low", "moderate", "high", "extreme"]
    assert isinstance(payload["live_providers"], list)
    assert payload["rate_limit"]["scope"] == "per_client_ip"
    assert "nws" in payload["providers_accepted"]


def test_v1_meta_describes_advisory_scale():
    client = TestClient(MODULE.api)
    advisory_meta = client.get("/v1/meta").json()["advisory"]
    assert advisory_meta["enabled"] is True
    assert advisory_meta["basis"] == "relative_to_local_weekly_normal"
    assert [level["name"] for level in advisory_meta["scale"]] == [
        "Low", "Moderate", "Elevated", "High", "Extreme"
    ]
    assert advisory_meta["scale"][-1]["max_percentile"] == 1.0
    assert advisory_meta["region_count"] == len(advisory_meta["regions"])
    assert advisory_meta["region_count"] >= 15
    assert any(r["id"] == "los_angeles" for r in advisory_meta["regions"])
    assert "/v1/advisory/national" in advisory_meta["endpoints"]


def test_v1_point_climatology():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["weather_source"] == "climatology"
    assert 0.0 <= payload["risk_score"] <= 1.0
    assert "weather" in payload
    assert payload["in_coverage"] is True
    assert 0.0 <= payload["confidence"] <= 1.0
    assert set(payload["hazards"]) >= {"ice_risk", "fog_risk", "wet", "labels"}
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_point_explain():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9&explain=true"
    )
    assert response.status_code == 200
    explanation = response.json()["explanation"]
    assert explanation is not None
    assert 0.0 <= explanation["baseline_risk"] <= 1.0
    factors = explanation["factors"]
    assert factors
    assert {"factor", "contribution", "direction"} <= set(factors[0])
    assert any(f["factor"] == "crash_history" for f in factors)
    # Factors are sorted by absolute contribution.
    magnitudes = [abs(f["contribution"]) for f in factors]
    assert magnitudes == sorted(magnitudes, reverse=True)


def test_v1_point_omits_explanation_by_default():
    client = TestClient(MODULE.api)
    response = client.get("/v1/risk/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9")
    assert response.json().get("explanation") is None


def test_v1_point_live_uses_mocked_snapshot(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=22.0,
        dewpoint_c=15.0,
        relative_humidity_pct=63.0,
        wind_speed_mps=4.5,
        wet_hour=0.0,
        summary="Clear",
    )

    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", lambda **kwargs: snapshot)
    client = TestClient(MODULE.api)
    response = client.get("/v1/risk/point?lat=34.0522&lon=-118.2437&mode=live")
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_provider"] == "nws"
    assert payload["weather_source"] == "live_observation"
    assert response.headers["cache-control"] == "no-store"


def test_v1_point_live_rejects_unknown_provider():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/point?lat=34.0522&lon=-118.2437&mode=live&provider=bogus"
    )
    assert response.status_code == 400


def test_v1_point_rejects_bad_mode_and_coords():
    client = TestClient(MODULE.api)
    assert client.get("/v1/risk/point?lat=34&lon=-118&mode=weird").status_code == 422
    assert client.get("/v1/risk/point?lat=120&lon=-118").status_code == 422


def test_v1_advisory_point_climatology():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/advisory/point?lat=34.0522&lon=-118.2437&day_of_week=5&hour=17&month=9"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "climatology"
    assert isinstance(payload["caveats"], list) and payload["caveats"]
    assert any("percentile" in c for c in payload["caveats"])
    adv = payload["advisory"]
    assert adv["level"] in {"Low", "Moderate", "Elevated", "High", "Extreme"}
    assert adv["level_index"] in {1, 2, 3, 4, 5}
    assert adv["color"].startswith("#")
    assert adv["message"].startswith(adv["level"])
    assert payload["risk_score"] == adv["risk_score"]
    assert "compared_to_normal" not in payload  # climatology has no now-vs-normal
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_advisory_point_rejects_bad_mode():
    client = TestClient(MODULE.api)
    assert (
        client.get("/v1/advisory/point?lat=34&lon=-118&mode=weird").status_code == 422
    )


def test_v1_advisory_point_out_of_coverage_is_low():
    client = TestClient(MODULE.api)
    # An ocean point is out of coverage: its all-zero weekly profile carries no
    # relative signal, so the advisory falls back to absolute -> Low (not Moderate).
    payload = client.get("/v1/advisory/point?lat=0&lon=-30").json()
    assert payload["in_coverage"] is False
    assert payload["risk_score"] == 0.0
    assert payload["advisory"]["level"] == "Low"
    assert payload["advisory"]["basis"] == "absolute"


def test_v1_advisory_point_live_folds_hazards_and_compares(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=12.0,
        dewpoint_c=11.0,
        relative_humidity_pct=97.0,
        wind_speed_mps=3.0,
        wet_hour=1.0,  # -> "wet" hazard label -> "wet roads" advisory driver
        summary="Rain",
    )

    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", lambda **kwargs: snapshot)
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/advisory/point?lat=34.0522&lon=-118.2437&mode=live&compare=true"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "live"
    assert payload["weather_source"] == "live_observation"
    assert "wet roads" in payload["advisory"]["drivers"]
    comparison = payload["compared_to_normal"]
    assert comparison["comparison"] in {
        "worse than normal",
        "about normal",
        "better than normal",
    }
    assert 0.0 <= comparison["now"] <= 1.0 and 0.0 <= comparison["normal"] <= 1.0
    assert response.headers["cache-control"] == "no-store"


def test_v1_advisory_point_live_rejects_unknown_provider():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/advisory/point?lat=34.0522&lon=-118.2437&mode=live&provider=bogus"
    )
    assert response.status_code == 400


def test_v1_advisory_region_by_id():
    client = TestClient(MODULE.api)
    response = client.get("/v1/advisory/region?region=los_angeles&day_of_week=5&hour=17")
    assert response.status_code == 200
    payload = response.json()
    assert payload["region_id"] == "los_angeles"
    assert payload["mode"] == "climatology"
    assert payload["resolved_by"] == "id"
    assert payload["frame_idx"] == (5 - 1) * 24 + 17
    assert payload["advisory"]["level"] in {
        "Low", "Moderate", "Elevated", "High", "Extreme"
    }
    # Advisory is relative to the region's own weekly climatology.
    assert payload["advisory"]["basis"] == "relative_to_local_weekly_normal"
    assert 0.0 <= payload["advisory"]["percentile"] <= 1.0
    assert payload["reference_size"] == 168
    # Context-aware message names the region and the time-of-week.
    message = payload["advisory"]["message"]
    assert message.startswith(payload["advisory"]["level"])
    assert "Los Angeles" in message
    assert "Friday evening" in message
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_advisory_region_by_point():
    client = TestClient(MODULE.api)
    response = client.get("/v1/advisory/region?lat=34.0522&lon=-118.2437")
    assert response.status_code == 200
    payload = response.json()
    assert payload["region_id"] == "los_angeles"
    assert payload["resolved_by"] == "point"


def test_v1_advisory_region_requires_selector():
    client = TestClient(MODULE.api)
    assert client.get("/v1/advisory/region").status_code == 422


def test_v1_advisory_region_blank_region_falls_back_to_point():
    client = TestClient(MODULE.api)
    # A blank region param must not shadow valid point resolution.
    response = client.get("/v1/advisory/region?region=&lat=34.0522&lon=-118.2437")
    assert response.status_code == 200
    payload = response.json()
    assert payload["region_id"] == "los_angeles"
    assert payload["resolved_by"] == "point"


def test_v1_advisory_region_unknown_id():
    client = TestClient(MODULE.api)
    assert client.get("/v1/advisory/region?region=atlantis").status_code == 404


def test_v1_advisory_region_point_outside_all_regions():
    client = TestClient(MODULE.api)
    # Gulf of Guinea: not inside any metro bbox.
    assert client.get("/v1/advisory/region?lat=0&lon=0").status_code == 404


def test_v1_advisory_region_live_and_compare(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=20.0,
        dewpoint_c=12.0,
        relative_humidity_pct=60.0,
        wind_speed_mps=4.0,
        wet_hour=0.0,
        summary="Clear",
    )

    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", lambda **kwargs: snapshot)
    client = TestClient(MODULE.api)
    # Deliberately omit day_of_week/hour: the compare baseline must align to the
    # live time-of-week (the snapshot is Fri 2024-09-06 17:00 -> frame 4*24+17).
    response = client.get(
        "/v1/advisory/region?region=los_angeles&mode=live&compare=true"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "live"
    assert payload["status"] == "ok"
    assert payload["sample_count"] == 9  # 3x3 representative grid
    assert payload["advisory"]["level"] in {
        "Low", "Moderate", "Elevated", "High", "Extreme"
    }
    assert payload["advisory"]["basis"] == "relative_to_local_weekly_normal"
    assert payload["compared_to_normal"]["comparison"] in {
        "worse than normal", "about normal", "better than normal"
    }
    # Relative reference frame aligned to now (Fri 17:00), not the default Mon 00:00.
    assert payload["frame_idx"] == (5 - 1) * 24 + 17
    assert payload["frame_label"] == "Fri 17:00"
    assert response.headers["cache-control"] == "no-store"


def test_v1_advisory_region_live_unavailable(monkeypatch):
    from live_weather import LiveWeatherProviderError

    def boom(**kwargs):
        raise LiveWeatherProviderError("provider down")

    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", boom)
    client = TestClient(MODULE.api)
    response = client.get("/v1/advisory/region?region=los_angeles&mode=live")
    assert response.status_code == 503


def test_v1_advisory_national_json():
    client = TestClient(MODULE.api)
    response = client.get("/v1/advisory/national?day_of_week=5&hour=17")
    assert response.status_code == 200
    payload = response.json()
    assert payload["mode"] == "climatology"
    assert payload["frame_idx"] == (5 - 1) * 24 + 17
    assert payload["count"] == len(payload["regions"])
    assert payload["count"] >= 15
    # Ranked by how elevated each metro is versus its own normal (percentile desc).
    percentiles = [r["percentile"] for r in payload["regions"]]
    assert percentiles == sorted(percentiles, reverse=True)
    first = payload["regions"][0]
    assert {"region_id", "region_name", "risk_score", "percentile", "advisory"} <= set(first)
    # The relative scale discriminates: metros no longer all read the same level.
    levels = {r["advisory"]["level"] for r in payload["regions"]}
    assert len(levels) >= 2
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_advisory_national_geojson_point_and_bbox():
    client = TestClient(MODULE.api)
    point = client.get("/v1/advisory/national?format=geojson").json()
    assert point["type"] == "FeatureCollection"
    assert point["features"]
    feat = point["features"][0]
    assert feat["geometry"]["type"] == "Point"
    assert len(feat["geometry"]["coordinates"]) == 2
    assert {"region_id", "level", "color", "risk_score"} <= set(feat["properties"])

    poly = client.get("/v1/advisory/national?format=geojson&geometry=bbox").json()
    poly_feat = poly["features"][0]
    assert poly_feat["geometry"]["type"] == "Polygon"
    ring = poly_feat["geometry"]["coordinates"][0]
    assert len(ring) == 5 and ring[0] == ring[-1]  # closed ring


def test_v1_advisory_national_geojson_media_type():
    client = TestClient(MODULE.api)
    response = client.get("/v1/advisory/national?format=geojson")
    assert response.headers["content-type"].startswith("application/geo+json")


def test_v1_advisory_national_min_level_filter():
    client = TestClient(MODULE.api)
    full = client.get("/v1/advisory/national?min_level=1").json()
    filtered = client.get("/v1/advisory/national?min_level=4").json()
    # Filtering never grows the set, and every survivor meets the threshold.
    assert filtered["count"] <= full["count"]
    assert all(r["advisory"]["level_index"] >= 4 for r in filtered["regions"])


def test_v1_advisory_national_rejects_bad_format_and_geometry():
    client = TestClient(MODULE.api)
    assert client.get("/v1/advisory/national?format=xml").status_code == 422
    assert client.get("/v1/advisory/national?format=geojson&geometry=blob").status_code == 422


def test_v1_nearest_facility():
    client = TestClient(MODULE.api)
    payload = client.get("/v1/emergency/nearest-facility?lat=34.05&lon=-118.24&k=3").json()
    assert payload["count"] <= 3 and payload["count"] >= 1
    facilities = payload["facilities"]
    # Sorted by straight-line distance, each annotated with distance + bearing.
    distances = [f["distance_km"] for f in facilities]
    assert distances == sorted(distances)
    assert all("bearing_deg" in f and "kind" in f for f in facilities)
    assert client.get("/v1/emergency/nearest-facility?lat=34.05&lon=-118.24").headers[
        "cache-control"
    ] == "public, max-age=3600"


def test_v1_nearest_facility_kind_filter():
    client = TestClient(MODULE.api)
    payload = client.get(
        "/v1/emergency/nearest-facility?lat=34.05&lon=-118.24&kind=hospital&k=5"
    ).json()
    assert payload["kind"] == "hospital"
    assert payload["facilities"] and all(f["kind"] == "hospital" for f in payload["facilities"])


def test_v1_nearest_facility_radius_and_bad_kind():
    client = TestClient(MODULE.api)
    # Tight radius around downtown LA returns only nearby facilities.
    near = client.get(
        "/v1/emergency/nearest-facility?lat=34.05&lon=-118.24&max_km=15&k=25"
    ).json()
    assert all(f["distance_km"] <= 15.0 for f in near["facilities"])
    # Unknown kind -> 422.
    assert client.get(
        "/v1/emergency/nearest-facility?lat=34&lon=-118&kind=zoo"
    ).status_code == 422


def test_v1_evacuate_compass_climatology():
    client = TestClient(MODULE.api)
    payload = client.get(
        "/v1/emergency/evacuate?lat=34.05&lon=-118.24&distance_km=20&day_of_week=5&hour=17"
    ).json()
    assert payload["mode"] == "climatology" and payload["target"] == "compass"
    assert payload["count"] == 8
    routes = payload["routes"]
    means = [r["route_risk_score_mean"] for r in routes]
    assert means == sorted(means)  # safest first
    assert routes[0]["recommended"] is True and payload["recommended_index"] == 0
    assert routes[0]["destination"]["compass"] in {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}


def test_v1_evacuate_toward_facilities():
    client = TestClient(MODULE.api)
    payload = client.get(
        "/v1/emergency/evacuate?lat=34.05&lon=-118.24&target=facilities&facility_kind=hospital&facility_count=3"
    ).json()
    assert payload["target"] == "facilities"
    assert 1 <= payload["count"] <= 3
    # Destinations carry facility identity.
    assert all(r["destination"].get("kind") == "hospital" for r in payload["routes"])
    assert payload["routes"][0]["destination"].get("name")


def test_v1_evacuate_geojson():
    client = TestClient(MODULE.api)
    response = client.get("/v1/emergency/evacuate?lat=34.05&lon=-118.24&format=geojson")
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert len(feature["geometry"]["coordinates"]) >= 2
    assert "recommended" in feature["properties"]


def test_v1_evacuate_validation():
    client = TestClient(MODULE.api)
    assert client.get("/v1/emergency/evacuate?lat=34&lon=-118&mode=x").status_code == 422
    assert client.get("/v1/emergency/evacuate?lat=34&lon=-118&target=x").status_code == 422
    assert client.get("/v1/emergency/evacuate?lat=34&lon=-118&rank_by=x").status_code == 422
    assert client.get(
        "/v1/emergency/evacuate?lat=34&lon=-118&target=facilities&facility_kind=zoo"
    ).status_code == 422
    # A bad format is a 422 even when the facilities lookup would otherwise 404.
    assert client.get(
        "/v1/emergency/evacuate?lat=0&lon=0&target=facilities&facility_radius_km=1&format=xml"
    ).status_code == 422


def test_v1_evacuate_live(monkeypatch):
    snapshot = SimpleNamespace(
        provider="nws", provider_label="NWS", observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0, temp_c=18.0, dewpoint_c=12.0, relative_humidity_pct=70.0,
        wind_speed_mps=6.0, wet_hour=0.0, summary="Clear",
    )
    import predict

    monkeypatch.setattr(predict, "fetch_live_weather", lambda **kwargs: snapshot)
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/emergency/evacuate?lat=34.05&lon=-118.24&mode=live&distance_km=10"
    )
    assert response.status_code == 200
    assert response.headers["cache-control"] == "no-store"
    assert response.json()["count"] == 8


def test_v1_meta_includes_model_metrics():
    client = TestClient(MODULE.api)
    payload = client.get("/v1/meta").json()
    assert "model_metrics" in payload
    assert isinstance(payload["model_metrics"], dict)


def test_v1_model_report():
    client = TestClient(MODULE.api)
    response = client.get("/v1/model/report")
    assert response.status_code == 200
    payload = response.json()
    assert isinstance(payload, dict)
    # Either a generated report (has metrics) or the no-report fallback.
    assert "metrics" in payload or "available" in payload


def test_v1_sun_glare_endpoint():
    client = TestClient(MODULE.api)
    # Low morning sun in the ENE, driver heading east -> glare.
    response = client.get(
        "/v1/hazards/sun-glare?lat=34.05&lon=-118.24&bearing=90&datetime=2024-06-21T13:30:00Z"
    )
    assert response.status_code == 200
    payload = response.json()
    assert payload["glare"] is True
    assert payload["window"] == "sunrise"
    assert "sun_elevation" in payload and "sun_azimuth" in payload
    # Facing away -> no glare.
    away = client.get(
        "/v1/hazards/sun-glare?lat=34.05&lon=-118.24&bearing=270&datetime=2024-06-21T13:30:00Z"
    )
    assert away.json()["glare"] is False


def test_v1_route_glare_annotation():
    client = TestClient(MODULE.api)
    body = {
        "waypoints": [[-118.2437, 34.0522], [-118.0, 34.0522]],  # heading roughly east
        "mode": "climatology",
        "day_of_week": 5,
        "hour": 6,
        "month": 6,
        "sample_spacing_km": 3.0,
        "glare_datetime": "2024-06-21T13:30:00Z",
    }
    response = client.post("/v1/risk/route", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["glare_segments"] is not None
    assert payload["steps"][0]["sun_glare"] is not None
    assert "glare" in payload["steps"][0]["sun_glare"]


def test_v1_route_without_glare_datetime_has_no_glare_fields():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route", json=_ROUTE_BODY)
    payload = response.json()
    assert payload["glare_segments"] is None
    assert payload["steps"][0]["sun_glare"] is None


def test_v1_openapi_lists_v1_paths():
    client = TestClient(MODULE.api)
    paths = client.get("/openapi.json").json()["paths"]
    assert "/v1/risk/point" in paths
    assert "/v1/risk/point/weekly" in paths
    assert "/v1/risk/route" in paths
    assert "/v1/risk/route/compare" in paths
    assert "/v1/risk/area" in paths
    assert "/v1/hotspots" in paths
    assert "/v1/heatmap" in paths
    assert "/v1/meta" in paths
    assert "/v1/health" in paths


def test_v1_weekly_profile():
    client = TestClient(MODULE.api)
    response = client.get("/v1/risk/point/weekly?lat=34.0522&lon=-118.2437&month=9")
    assert response.status_code == 200
    payload = response.json()
    assert len(payload["risk_by_hour_of_week"]) == 168
    assert len(payload["frame_labels"]) == 168
    assert 0 <= payload["safest"]["hour_of_week"] < 168
    assert 0 <= payload["riskiest"]["hour_of_week"] < 168
    assert payload["riskiest"]["risk_score"] >= payload["safest"]["risk_score"]
    assert all(0.0 <= v <= 1.0 for v in payload["risk_by_hour_of_week"])


_ROUTE_BODY = {
    "waypoints": [[-118.2437, 34.0522], [-118.40, 34.02], [-118.49, 34.02]],
    "mode": "climatology",
    "day_of_week": 5,
    "hour": 17,
    "month": 9,
    "sample_spacing_km": 3.0,
}


def test_v1_route_climatology():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route", json=_ROUTE_BODY)
    assert response.status_code == 200
    payload = response.json()
    assert payload["sample_count"] >= 2
    assert payload["distance_km"] > 0.0
    assert 0.0 <= payload["route_risk_score_mean"] <= 1.0
    assert payload["route_risk_score_max"] >= payload["route_risk_score_mean"]
    assert payload["riskiest_point"]["risk_score"] == payload["route_risk_score_max"]
    distances = [step["distance_km"] for step in payload["steps"]]
    assert distances == sorted(distances)  # cumulative distance is monotonic
    assert payload["steps"][0]["hazards"] is not None
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_route_geojson_output():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route?format=geojson", json=_ROUTE_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    coords = feature["geometry"]["coordinates"]
    assert len(coords) == len(feature["properties"]["risk_scores"])
    assert all(len(pair) == 2 for pair in coords)  # [lon, lat]


def test_v1_route_rejects_too_long_for_spacing():
    client = TestClient(MODULE.api)
    body = {
        "waypoints": [[-118.0, 34.0], [-110.0, 40.0]],  # ~900 km apart
        "sample_spacing_km": 1.0,
    }
    response = client.post("/v1/risk/route", json=body)
    assert response.status_code == 422


def test_v1_route_rejects_too_few_waypoints():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route", json={"waypoints": [[-118.0, 34.0]]})
    assert response.status_code == 422


def test_v1_route_live_dedupes_weather_by_cell(monkeypatch):
    calls = {"n": 0}
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=22.0,
        dewpoint_c=15.0,
        relative_humidity_pct=63.0,
        wind_speed_mps=4.5,
        wet_hour=0.0,
        summary="Clear",
    )

    import predict

    def fake_fetch(**kwargs):
        calls["n"] += 1
        return snapshot

    monkeypatch.setattr(predict, "fetch_live_weather", fake_fetch)
    client = TestClient(MODULE.api)
    body = {
        "waypoints": [[-118.2437, 34.0522], [-118.2440, 34.0525]],
        "mode": "live",
        "sample_spacing_km": 2.0,
    }
    response = client.post("/v1/risk/route", json=body)
    assert response.status_code == 200
    payload = response.json()
    assert payload["live_provider"] == "nws"
    assert payload["sample_count"] >= 2
    # Both samples fall in one H3 cell, so weather is fetched exactly once.
    assert calls["n"] == 1
    assert response.headers["cache-control"] == "no-store"


_FAKE_AREA = {
    "count": 2,
    "segments": [
        {
            "segment_id": "06:1:0:0",
            "fullname": "Main St",
            "coords_json": "[[-118.25, 34.05], [-118.24, 34.06]]",
            "center_lat": 34.055,
            "center_lon": -118.245,
            "segment_idx": 10,
            "risk_score": 0.71,
            "forecast_hours": 0,
            "target_timestamp_local": "2024-09-06T17:00:00",
            "weather_provider": "climatology",
        },
        {
            "segment_id": "06:2:0:0",
            "fullname": "Second Ave",
            "coords_json": "[[-118.30, 34.00], [-118.29, 34.01]]",
            "center_lat": 34.005,
            "center_lon": -118.295,
            "segment_idx": 11,
            "risk_score": 0.42,
            "forecast_hours": 0,
            "target_timestamp_local": "2024-09-06T17:00:00",
            "weather_provider": "climatology",
        },
    ],
}

_AREA_QUERY = "min_lat=33.9&max_lat=34.2&min_lon=-118.5&max_lon=-118.1"


def test_v1_area_json(monkeypatch):
    import segment_runtime

    monkeypatch.setattr(segment_runtime, "score_segments_in_bbox", lambda **kwargs: _FAKE_AREA)
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/risk/area?{_AREA_QUERY}")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["segments"][0]["risk_score"] == 0.71
    assert response.headers["cache-control"] == "no-store"


def test_v1_area_geojson(monkeypatch):
    import segment_runtime

    monkeypatch.setattr(segment_runtime, "score_segments_in_bbox", lambda **kwargs: _FAKE_AREA)
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/risk/area?{_AREA_QUERY}&format=geojson")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 2
    feature = payload["features"][0]
    assert feature["geometry"]["type"] == "LineString"
    assert feature["geometry"]["coordinates"] == [[-118.25, 34.05], [-118.24, 34.06]]
    assert feature["properties"]["segment_id"] == "06:1:0:0"
    assert feature["properties"]["risk_score"] == 0.71


def test_v1_area_rejects_unknown_provider():
    # Provider is validated before the scorer runs, so no mock is needed.
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/risk/area?{_AREA_QUERY}&provider=bogus")
    assert response.status_code == 400


def test_v1_area_rejects_out_of_range_bbox():
    client = TestClient(MODULE.api)
    response = client.get(
        "/v1/risk/area?min_lat=33.9&max_lat=200.0&min_lon=-118.5&max_lon=-118.1"
    )
    assert response.status_code == 422


_FAKE_HOTSPOTS = {
    "count": 2,
    "rank_by": "delta",
    "segments": [
        {
            "segment_id": "06:1:0:0",
            "fullname": "Main St",
            "coords_json": "[[-118.25, 34.05], [-118.24, 34.06]]",
            "segment_idx": 10,
            "risk_score": 0.82,
            "baseline_score": 0.30,
            "delta": 0.52,
            "weather_provider": "nws",
        },
        {
            "segment_id": "06:2:0:0",
            "fullname": "Second Ave",
            "coords_json": "[[-118.30, 34.00], [-118.29, 34.01]]",
            "segment_idx": 11,
            "risk_score": 0.60,
            "baseline_score": 0.40,
            "delta": 0.20,
            "weather_provider": "nws",
        },
    ],
}


def test_v1_hotspots_json(monkeypatch):
    import segment_runtime

    monkeypatch.setattr(segment_runtime, "rank_hotspots_in_bbox", lambda **kwargs: _FAKE_HOTSPOTS)
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/hotspots?{_AREA_QUERY}&rank_by=delta&top_n=10")
    assert response.status_code == 200
    payload = response.json()
    assert payload["count"] == 2
    assert payload["rank_by"] == "delta"
    assert payload["segments"][0]["delta"] == 0.52
    assert response.headers["cache-control"] == "no-store"


def test_v1_hotspots_geojson(monkeypatch):
    import segment_runtime

    monkeypatch.setattr(segment_runtime, "rank_hotspots_in_bbox", lambda **kwargs: _FAKE_HOTSPOTS)
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/hotspots?{_AREA_QUERY}&format=geojson")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    props = payload["features"][0]["properties"]
    assert props["delta"] == 0.52
    assert props["baseline_score"] == 0.30


def test_v1_hotspots_rejects_bad_rank_by():
    client = TestClient(MODULE.api)
    assert client.get(f"/v1/hotspots?{_AREA_QUERY}&rank_by=nonsense").status_code == 422


def test_v1_hotspots_rejects_unknown_provider():
    client = TestClient(MODULE.api)
    assert client.get(f"/v1/hotspots?{_AREA_QUERY}&provider=bogus").status_code == 400


_COMPARE_BODY = {
    "routes": [
        {"label": "city", "waypoints": [[-118.2437, 34.0522], [-118.40, 34.02]]},
        # Out-of-coverage ocean route scores 0 everywhere -> always safest.
        {"label": "ocean", "waypoints": [[-140.0, 30.0], [-140.1, 30.1]]},
    ],
    "mode": "climatology",
    "day_of_week": 5,
    "hour": 17,
    "month": 9,
    "sample_spacing_km": 5.0,
}


def test_v1_route_compare_recommends_lowest_risk():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route/compare", json=_COMPARE_BODY)
    assert response.status_code == 200
    payload = response.json()
    assert payload["recommended_index"] == 1
    assert payload["recommended_label"] == "ocean"
    by_label = {c["label"]: c for c in payload["candidates"]}
    assert by_label["ocean"]["rank"] == 1 and by_label["ocean"]["recommended"] is True
    assert by_label["city"]["rank"] == 2 and by_label["city"]["recommended"] is False
    assert by_label["city"]["route_risk_score_mean"] > by_label["ocean"]["route_risk_score_mean"]
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_route_compare_geojson():
    client = TestClient(MODULE.api)
    response = client.post("/v1/risk/route/compare?format=geojson", json=_COMPARE_BODY)
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    assert len(payload["features"]) == 2
    assert payload["recommended_index"] == 1
    recommended = [f for f in payload["features"] if f["properties"]["recommended"]]
    assert len(recommended) == 1
    assert recommended[0]["geometry"]["type"] == "LineString"


def test_v1_route_compare_validation():
    client = TestClient(MODULE.api)
    single = {**_COMPARE_BODY, "routes": _COMPARE_BODY["routes"][:1]}
    assert client.post("/v1/risk/route/compare", json=single).status_code == 422
    bad_objective = {**_COMPARE_BODY, "objective": "weird"}
    assert client.post("/v1/risk/route/compare", json=bad_objective).status_code == 422
    six = {**_COMPARE_BODY, "routes": _COMPARE_BODY["routes"] * 3}
    assert client.post("/v1/risk/route/compare", json=six).status_code == 422


def test_v1_route_compare_live_shares_weather_cache(monkeypatch):
    calls = {"n": 0}
    snapshot = SimpleNamespace(
        provider="nws",
        provider_label="NWS",
        observed_or_forecast="observation",
        timestamp_local=datetime(2024, 9, 6, 17, 0, tzinfo=timezone.utc),
        forecast_hours=0,
        temp_c=22.0,
        dewpoint_c=15.0,
        relative_humidity_pct=63.0,
        wind_speed_mps=4.5,
        wet_hour=0.0,
        summary="Clear",
    )

    import predict

    def fake_fetch(**kwargs):
        calls["n"] += 1
        return snapshot

    monkeypatch.setattr(predict, "fetch_live_weather", fake_fetch)
    client = TestClient(MODULE.api)
    body = {
        "routes": [
            {"label": "a", "waypoints": [[-118.2437, 34.0522], [-118.2440, 34.0525]]},
            {"label": "b", "waypoints": [[-118.2437, 34.0522], [-118.2445, 34.0530]]},
        ],
        "mode": "live",
        "sample_spacing_km": 2.0,
    }
    response = client.post("/v1/risk/route/compare", json=body)
    assert response.status_code == 200
    # Both candidates sit in one H3 cell; the shared cache means one fetch total.
    assert calls["n"] == 1
    assert response.headers["cache-control"] == "no-store"


_HEATMAP_QUERY = "min_lat=33.7&max_lat=34.3&min_lon=-118.7&max_lon=-118.0&day_of_week=6&hour=2"


def test_v1_heatmap_samples_grid_within_bbox():
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/heatmap?{_HEATMAP_QUERY}&max_cells=500")
    assert response.status_code == 200
    payload = response.json()
    assert payload["frame_idx"] == (6 - 1) * 24 + 2
    assert payload["cell_count"] == len(payload["cells"])
    for cell in payload["cells"]:
        assert 33.7 <= cell["lat"] <= 34.3
        assert -118.7 <= cell["lon"] <= -118.0
        assert 0.0 <= cell["risk"] <= 1.0
    assert response.headers["cache-control"] == "public, max-age=3600"


def test_v1_heatmap_min_risk_filter_and_cap():
    client = TestClient(MODULE.api)
    low = client.get(f"/v1/heatmap?{_HEATMAP_QUERY}").json()["cell_count"]
    high = client.get(f"/v1/heatmap?{_HEATMAP_QUERY}&min_risk=0.9").json()["cell_count"]
    assert high <= low
    # A CONUS-wide request with a small cap must stay bounded.
    capped = client.get(
        "/v1/heatmap?min_lat=25&max_lat=49&min_lon=-124&max_lon=-67&day_of_week=6&hour=2&max_cells=300"
    ).json()
    assert capped["cell_count"] <= 300


def test_v1_heatmap_geojson():
    client = TestClient(MODULE.api)
    response = client.get(f"/v1/heatmap?{_HEATMAP_QUERY}&format=geojson")
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("application/geo+json")
    payload = response.json()
    assert payload["type"] == "FeatureCollection"
    if payload["features"]:
        assert payload["features"][0]["geometry"]["type"] == "Point"
