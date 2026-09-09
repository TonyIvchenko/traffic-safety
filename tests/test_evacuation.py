from __future__ import annotations

from pathlib import Path
import sys

import pytest

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import evacuation
import geo_math


def test_candidate_destinations_default_compass():
    dests = evacuation.candidate_destinations(34.0, -118.0, distance_km=25.0)
    assert len(dests) == 8
    assert {d["compass"] for d in dests} == {"N", "NE", "E", "SE", "S", "SW", "W", "NW"}
    # Each destination is ~25 km from the origin.
    for d in dests:
        assert geo_math.haversine_km(34.0, -118.0, d["lat"], d["lon"]) == pytest.approx(25.0, abs=0.2)
    # North increases latitude; South decreases it.
    north = next(d for d in dests if d["compass"] == "N")
    south = next(d for d in dests if d["compass"] == "S")
    assert north["lat"] > 34.0 > south["lat"]


def test_candidate_destinations_custom_bearings():
    dests = evacuation.candidate_destinations(0.0, 0.0, distance_km=10.0, bearings=[(90.0, "E")])
    assert len(dests) == 1 and dests[0]["compass"] == "E"
    assert dests[0]["lon"] > 0.0


def _fake_scorer(risk_by_lat):
    # A scorer whose route risk depends on the destination latitude, so ranking
    # is deterministic. points = [(origin_lon, origin_lat), (dest_lon, dest_lat)].
    def score(points):
        dest_lat = points[-1][1]
        mean = risk_by_lat(dest_lat)
        return {
            "distance_km": 25.0,
            "route_risk_score_mean": mean,
            "route_risk_score_max": mean + 0.1,
            "route_risk_level": "high" if mean > 0.5 else "low",
            "high_risk_fraction": 1.0 if mean > 0.5 else 0.0,
            "sample_count": 13,
            "steps": [{"lat": dest_lat, "lon": points[-1][0], "risk_score": mean}],
        }

    return score


def test_rank_evacuation_routes_recommends_safest():
    # Heading north (higher latitude) is safest.
    scorer = _fake_scorer(lambda lat: max(0.0, 1.0 - (lat - 34.0)))
    dests = evacuation.candidate_destinations(34.0, -118.0, distance_km=25.0)
    result = evacuation.rank_evacuation_routes(34.0, -118.0, dests, scorer)
    assert result["count"] == 8
    assert result["recommended_index"] == 0
    assert result["rank_by"] == "route_risk_score_mean"
    routes = result["routes"]
    means = [r["route_risk_score_mean"] for r in routes]
    assert means == sorted(means)  # safest first
    assert routes[0]["recommended"] is True and routes[0]["rank"] == 1
    assert routes[0]["destination"]["compass"] == "N"
    assert all(r["recommended"] is False for r in routes[1:])


def test_rank_by_max():
    scorer = _fake_scorer(lambda lat: 0.5)
    dests = evacuation.candidate_destinations(34.0, -118.0)
    result = evacuation.rank_evacuation_routes(34.0, -118.0, dests, scorer, rank_by="max")
    assert result["rank_by"] == "route_risk_score_max"


def test_include_steps_toggle():
    scorer = _fake_scorer(lambda lat: 0.3)
    dests = evacuation.candidate_destinations(34.0, -118.0, bearings=[(0.0, "N")])
    with_steps = evacuation.rank_evacuation_routes(34.0, -118.0, dests, scorer)
    without = evacuation.rank_evacuation_routes(34.0, -118.0, dests, scorer, include_steps=False)
    assert "steps" in with_steps["routes"][0]
    assert "steps" not in without["routes"][0]


def test_rank_skips_bad_summaries_and_handles_empty():
    def flaky(points):
        return None  # scorer yields nothing usable

    dests = evacuation.candidate_destinations(34.0, -118.0)
    result = evacuation.rank_evacuation_routes(34.0, -118.0, dests, flaky)
    assert result["count"] == 0
    assert result["recommended_index"] is None
    assert result["routes"] == []
    # No destinations at all.
    empty = evacuation.rank_evacuation_routes(34.0, -118.0, [], _fake_scorer(lambda lat: 0.1))
    assert empty["count"] == 0
