"""Evaluate active risk watches and deliver webhook alerts on breaches.

Meant to run on a schedule (cron/systemd timer), like refresh_segment_tiles.py:

    python scripts/evaluate_watches.py            # evaluate + notify
    python scripts/evaluate_watches.py --dry-run  # evaluate only

Each active watch is scored with the live predictors; the breach decision is
*relative* — the current risk's percentile within the watched location's own
weekly climatology — so a watch fires when conditions are unusually bad for that
place/time, not merely because dense-urban risk is always high in absolute terms.
When the relative level reaches the watch's threshold and the cooldown has
elapsed, the alert is POSTed to the subscriber's webhook signed with the watch's
secret. Poll-channel watches just get their status refreshed for
GET /v1/watches/{id}.
"""

from __future__ import annotations

import argparse
from datetime import datetime, timedelta, timezone
from pathlib import Path
import sys

REPO_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = REPO_DIR / "src"
if str(REPO_DIR) not in sys.path:
    sys.path.insert(0, str(REPO_DIR))
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import advisory
import notify
import risk_eval
import segment_runtime
from predict import (
    MODEL_BUNDLE,
    _risk_level,
    predict_traffic_safety,
    predict_traffic_safety_live,
    weekly_risk_profile,
)
from watch_store import WatchStore, get_default_store, level_at_least

# Map a watch's absolute threshold word to a relative advisory level index
# (Low 1, Moderate 2, Elevated 3, High 4, Extreme 5).
_THRESHOLD_TO_RELATIVE_INDEX = {"low": 1, "moderate": 2, "elevated": 3, "high": 4, "extreme": 5}


def evaluate_point(params: dict) -> dict:
    result = predict_traffic_safety_live(
        lat=float(params["lat"]),
        lon=float(params["lon"]),
        forecast_hours=int(params.get("forecast_hours", 0)),
        provider=params.get("provider", "auto"),
    )
    return {
        "risk_score": float(result["risk_score"]),
        "risk_level": result["risk_level"],
        "ref": {"lat": float(params["lat"]), "lon": float(params["lon"])},
        "detail": {
            "weather_source": result.get("weather_source"),
            "live_provider": result.get("live_provider"),
            "hazards": (result.get("hazards") or {}).get("labels", []),
        },
    }


def evaluate_route(params: dict) -> dict:
    points = [(float(lon), float(lat)) for lon, lat in params["waypoints"]]
    scored = risk_eval.score_route(
        points,
        mode="live",
        predict_point=predict_traffic_safety,
        predict_point_live=predict_traffic_safety_live,
        h3_resolution=int(MODEL_BUNDLE.get("resolution", 5)),
        sample_spacing_km=float(params.get("sample_spacing_km", 2.0)),
        forecast_hours=int(params.get("forecast_hours", 0)),
        provider=params.get("provider", "auto"),
    )
    riskiest = scored["riskiest_point"]
    return {
        "risk_score": float(scored["route_risk_score_max"]),
        "risk_level": scored["route_risk_level"],
        "ref": {"lat": riskiest["lat"], "lon": riskiest["lon"]},
        "detail": {
            "distance_km": scored["distance_km"],
            "high_risk_fraction": scored["high_risk_fraction"],
            "riskiest_point": {
                "lat": scored["riskiest_point"]["lat"],
                "lon": scored["riskiest_point"]["lon"],
            },
        },
    }


def evaluate_area(params: dict) -> dict:
    result = segment_runtime.score_segments_in_bbox(
        min_lat=float(params["min_lat"]),
        max_lat=float(params["max_lat"]),
        min_lon=float(params["min_lon"]),
        max_lon=float(params["max_lon"]),
        forecast_hours=int(params.get("forecast_hours", 0)),
        provider=params.get("provider", "auto"),
        limit=int(params.get("limit", 100)),
    )
    segments = result.get("segments", [])
    top_score = max((float(s["risk_score"]) for s in segments), default=0.0)
    top = max(segments, key=lambda s: float(s["risk_score"]), default=None)
    ref = None
    if top and top.get("center_lat") is not None and top.get("center_lon") is not None:
        ref = {"lat": float(top["center_lat"]), "lon": float(top["center_lon"])}
    return {
        "risk_score": top_score,
        "risk_level": _risk_level(top_score),
        "ref": ref,
        "detail": {
            "segment_count": result.get("count", 0),
            "riskiest_segment": (
                {"segment_id": top["segment_id"], "name": top["fullname"]} if top else None
            ),
        },
    }


EVALUATORS = {"point": evaluate_point, "route": evaluate_route, "area": evaluate_area}


def relative_breach(level_index, threshold_level) -> bool:
    """True when a relative advisory level meets a watch's threshold word."""
    target = _THRESHOLD_TO_RELATIVE_INDEX.get(str(threshold_level).strip().lower(), 99)
    try:
        return int(level_index) >= target
    except (TypeError, ValueError):
        return False


def relative_assessment(risk_score, ref, *, month: int):
    """Relative advisory of a score vs its reference cell's weekly climatology.

    Returns the :func:`advisory.relative_advisory` block, or None when there is no
    usable reference point (the caller then falls back to the absolute level).
    """
    if not isinstance(ref, dict):
        return None
    try:
        profile = weekly_risk_profile(float(ref["lat"]), float(ref["lon"]), int(month))
    except Exception:  # noqa: BLE001 - a bad reference falls back to absolute, never fatal
        return None
    return advisory.relative_advisory(risk_score, list(profile))


def _cooldown_active(watch: dict, now: datetime) -> bool:
    last = watch.get("last_notified_at")
    if not last:
        return False
    return now - datetime.fromisoformat(last) < timedelta(minutes=int(watch["cooldown_minutes"]))


def run_once(store: WatchStore, *, dry_run: bool = False, now: datetime | None = None) -> dict:
    now = now or datetime.now(timezone.utc)
    summary = {"evaluated": 0, "breached": 0, "notified": 0, "errors": 0}
    for watch in store.list_active():
        evaluator = EVALUATORS.get(watch["kind"])
        try:
            outcome = evaluator(watch["params"])
        except Exception as exc:  # one broken watch must not stop the sweep
            print(f"watch {watch['id']} ({watch['kind']}): evaluation failed: {exc}")
            summary["errors"] += 1
            continue

        # Relative breach: how unusual the risk is for this place/time. Falls back
        # to the absolute level when there is no usable local-normal reference.
        relative = relative_assessment(outcome["risk_score"], outcome.get("ref"), month=now.month)
        if relative is not None and relative.get("basis") == "relative_to_local_weekly_normal":
            level = str(relative["level"]).lower()
            breached = relative_breach(relative["level_index"], watch["threshold_level"])
        else:
            level = outcome["risk_level"]
            breached = level_at_least(level, watch["threshold_level"])

        notified = False
        if (
            breached
            and not dry_run
            and watch["channel"] == "webhook"
            and watch["webhook_url"]
            and not _cooldown_active(watch, now)
        ):
            payload = {
                "watch_id": watch["id"],
                "kind": watch["kind"],
                "evaluated_at_utc": now.isoformat(),
                "risk_score": outcome["risk_score"],
                "risk_level": level,
                "percentile": relative.get("percentile") if relative else None,
                "basis": relative.get("basis") if relative else "absolute",
                "relative_descriptor": relative.get("relative_descriptor") if relative else None,
                "threshold_level": watch["threshold_level"],
                "params": watch["params"],
                "detail": {**outcome["detail"], "absolute_risk_level": outcome["risk_level"]},
            }
            delivery = notify.post_webhook(
                watch["webhook_url"], payload, watch["webhook_secret"]
            )
            notified = bool(delivery["delivered"])
            if not notified:
                print(f"watch {watch['id']}: webhook delivery failed: {delivery}")

        store.record_evaluation(
            watch["id"], level=level, breached=breached, notified=notified, now_iso=now.isoformat()
        )
        summary["evaluated"] += 1
        summary["breached"] += int(breached)
        summary["notified"] += int(notified)
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--db", type=Path, default=None, help="override the watch DB path")
    parser.add_argument("--dry-run", action="store_true", help="evaluate without notifying")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    store = WatchStore(args.db) if args.db else get_default_store()
    summary = run_once(store, dry_run=args.dry_run)
    print(
        f"evaluated={summary['evaluated']} breached={summary['breached']} "
        f"notified={summary['notified']} errors={summary['errors']}"
    )


if __name__ == "__main__":
    main()
