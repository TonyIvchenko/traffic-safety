from __future__ import annotations

import json
from pathlib import Path

REFERENCE_PATH = (
    Path(__file__).resolve().parents[1] / "data" / "reference" / "countermeasures.json"
)

VALID_CONTEXTS = {"urban", "suburban", "rural"}
VALID_SETTINGS = {"segment", "intersection", "crossing", "curve"}
VALID_COST_UNITS = {"per_mile", "per_intersection", "per_location"}


def _load() -> dict:
    return json.loads(REFERENCE_PATH.read_text(encoding="utf-8"))


def test_reference_file_committed_and_parses():
    assert REFERENCE_PATH.exists(), "countermeasures.json must be committed (not gitignored)"
    data = _load()
    assert data["schema_version"]
    assert data["source"]
    assert isinstance(data["countermeasures"], list) and len(data["countermeasures"]) >= 10


def test_ids_are_unique():
    ids = [cm["id"] for cm in _load()["countermeasures"]]
    assert len(ids) == len(set(ids))
    assert all(id_ == id_.lower() and " " not in id_ for id_ in ids)


def test_every_entry_matches_schema():
    for cm in _load()["countermeasures"]:
        label = cm.get("id", "<missing id>")
        assert isinstance(cm["id"], str) and cm["id"], label
        assert isinstance(cm["name"], str) and cm["name"], label
        assert isinstance(cm["category"], str) and cm["category"], label

        assert isinstance(cm["cmf"], (int, float)), label
        assert 0.0 < cm["cmf"] <= 1.0, f"{label}: cmf out of (0, 1]"
        assert isinstance(cm["cmf_basis"], str) and cm["cmf_basis"], label
        assert cm["cmf_star_rating"] in {1, 2, 3, 4, 5}, label

        crash_types = cm["applicable_crash_types"]
        assert isinstance(crash_types, list) and crash_types, label
        assert all(isinstance(t, str) and t for t in crash_types), label

        roadway = cm["roadway"]
        assert isinstance(roadway["mtfcc"], list) and roadway["mtfcc"], label
        assert all(str(m).startswith("S") for m in roadway["mtfcc"]), label
        assert set(roadway["context"]) <= VALID_CONTEXTS, f"{label}: bad context"
        assert roadway["context"], label
        assert set(roadway["setting"]) <= VALID_SETTINGS, f"{label}: bad setting"
        assert roadway["setting"], label

        assert isinstance(cm["vru_focused"], bool), label
        assert isinstance(cm["typical_cost_usd"], (int, float)) and cm["typical_cost_usd"] > 0, label
        assert cm["cost_unit"] in VALID_COST_UNITS, label


def test_has_vru_and_roadway_departure_coverage():
    cms = _load()["countermeasures"]
    assert any(cm["vru_focused"] for cm in cms), "expected VRU-focused entries"
    assert any("pedestrian" in cm["applicable_crash_types"] for cm in cms)
    assert any("run_off_road" in cm["applicable_crash_types"] for cm in cms)
    # The VRU staples the plan calls out are present.
    ids = {cm["id"] for cm in cms}
    assert {"rrfb", "leading_pedestrian_interval"} <= ids


def test_cmfs_are_reductions():
    # Every curated entry is a crash-reducing treatment (CMF < 1).
    assert all(cm["cmf"] < 1.0 for cm in _load()["countermeasures"])
