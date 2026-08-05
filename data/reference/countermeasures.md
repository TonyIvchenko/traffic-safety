# Countermeasures reference (`countermeasures.json`)

A curated, committed table of FHWA Proven Safety Countermeasures with
representative Crash Modification Factors (CMFs), used to recommend treatments
for high-risk corridors and estimate their benefit-cost.

> These CMFs are **illustrative selections** for screening, not a substitute for
> choosing a project-specific CMF from the [FHWA CMF Clearinghouse](https://www.cmfclearinghouse.org/).
> A CMF below 1.0 reduces crashes; the crash reduction factor **CRF = 1 − CMF**.

## Top-level fields

| Field | Type | Description |
|---|---|---|
| `schema_version` | string | Schema version of this file. |
| `source` | string | Provenance of the values. |
| `note` | string | Usage caveats. |
| `countermeasures` | array | The countermeasure entries (below). |

## Countermeasure entry

| Field | Type | Description |
|---|---|---|
| `id` | string | Stable unique identifier (snake_case). |
| `name` | string | Human-readable name. |
| `category` | string | Grouping, e.g. `roadway_departure`, `pedestrian_crossing`, `intersection_geometry`. |
| `cmf` | number | Crash Modification Factor in `(0, 1]`; `< 1` reduces crashes. |
| `cmf_basis` | string | What crashes/severity the CMF applies to. |
| `cmf_star_rating` | integer | CMF Clearinghouse quality rating, 1–5. |
| `applicable_crash_types` | string[] | Crash types the treatment addresses (e.g. `run_off_road`, `pedestrian`, `angle`, `total`). |
| `roadway.mtfcc` | string[] | TIGER MTFCC feature classes it applies to (e.g. `S1100`, `S1200`, `S1400`). |
| `roadway.context` | string[] | `urban`, `suburban`, `rural`. |
| `roadway.setting` | string[] | `segment`, `intersection`, `crossing`, `curve`. |
| `vru_focused` | boolean | True if it primarily protects pedestrians/cyclists (VRUs). |
| `typical_cost_usd` | number | Rough planning-level cost. |
| `cost_unit` | string | `per_mile`, `per_intersection`, or `per_location`. |

Consumed by `src/countermeasures.py` (loader + applicability matching) and the
`/v1/countermeasures/*` endpoints. `tests/test_countermeasures_reference.py`
validates every entry against this schema.
