"""Evaluate active risk watches and deliver webhook alerts on breaches.

Meant to run on a schedule (cron/systemd timer), like refresh_segment_tiles.py:

    python scripts/evaluate_watches.py            # evaluate + notify
    python scripts/evaluate_watches.py --dry-run  # evaluate only

Each active watch is scored with the live predictors; when the risk level
reaches the watch's threshold and the cooldown has elapsed, the alert is POSTed
to the subscriber's webhook signed with the watch's secret. Poll-channel
watches just get their status columns refreshed for GET /v1/watches/{id}.
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

import notify
import risk_eval
import segment_runtime
from predict import MODEL_BUNDLE, _risk_level, predict_traffic_safety, predict_traffic_safety_live
from watch_store import WatchStore, get_default_store, level_at_least


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
    return {
        "risk_score": float(scored["route_risk_score_max"]),
        "risk_level": scored["route_risk_level"],
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
    return {
        "risk_score": top_score,
        "risk_level": _risk_level(top_score),
        "detail": {
            "segment_count": result.get("count", 0),
            "riskiest_segment": (
                {"segment_id": top["segment_id"], "name": top["fullname"]} if top else None
            ),
        },
    }


EVALUATORS = {"point": evaluate_point, "route": evaluate_route, "area": evaluate_area}


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
                "threshold_level": watch["threshold_level"],
                "params": watch["params"],
                "detail": outcome["detail"],
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
