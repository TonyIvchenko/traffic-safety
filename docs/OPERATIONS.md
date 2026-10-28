# Operations: the watch evaluator

The proactive-alert system has two moving parts:

1. **The API** (`main:api`) — creates and manages watches (`POST /v1/watches`, …)
   and stores them in a SQLite database.
2. **The evaluator** (`scripts/evaluate_watches.py`) — a separate process that
   periodically scores every active watch and delivers alerts on a breach. This
   is the piece you schedule. The API never evaluates watches itself.

Both processes must point at the **same** watch database and share the same
model/tiles artifacts.

## What one sweep does

For each active watch, a sweep:

- scores the watched geometry with the live predictors (a forecast-horizon watch
  scans the next N hours and reports the earliest hour predicted to breach);
- decides a breach **relative** to that place/time's own weekly normal (falling
  back to the absolute level only where there is no usable local normal);
- on a breach, and if the per-watch cooldown has elapsed, delivers the alert over
  the watch's channel (`webhook` or `email`; `poll` watches are only refreshed
  for `GET /v1/watches/{id}`);
- records bookkeeping (`last_evaluated_at`, `last_level`, `last_breach_at`,
  `last_notified_at`) back onto the watch.

One broken watch (bad geometry, a provider outage for a whole region) is counted
as an error and skipped — it never aborts the sweep.

Each sweep prints one summary line to stdout (captured by journald under
systemd):

```
2026-10-28T11:02:03+00:00 evaluate_watches evaluated=42 breached=3 notified=2 errors=1
```

## Running it

### One-shot (cron / systemd timer) — recommended

```bash
python scripts/evaluate_watches.py              # evaluate + notify, then exit
python scripts/evaluate_watches.py --dry-run    # evaluate only, deliver nothing
python scripts/evaluate_watches.py --db /data/watches.sqlite3
```

Run it on a fixed cadence (e.g. every 5 minutes). The cadence is your alert
latency floor, so match it to how fresh you need alerts to be versus live-weather
provider load.

A cron example (every 5 minutes):

```cron
*/5 * * * * cd /opt/traffic-safety && TRAFFIC_SAFETY_WATCH_DB=/data/watches.sqlite3 \
  /opt/traffic-safety/.venv/bin/python scripts/evaluate_watches.py >> /var/log/roadrisk/watches.log 2>&1
```

### Daemon (long-running) — `--interval`

For hosts where a long-running service is preferred over a timer:

```bash
python scripts/evaluate_watches.py --interval 300   # sweep every 300 s
```

It sweeps once immediately, then every `--interval` seconds, and shuts down
cleanly on `SIGINT`/`SIGTERM` after finishing the current sweep.

### systemd

Units are in [`deploy/systemd/`](../deploy/systemd). The timer+oneshot pair is the
default; the daemon unit is an alternative — use one or the other, not both.

```bash
sudo cp deploy/systemd/roadrisk-evaluate-watches.{service,timer} /etc/systemd/system/
sudo systemctl daemon-reload
sudo systemctl enable --now roadrisk-evaluate-watches.timer   # timer mode
# or, instead:
sudo cp deploy/systemd/roadrisk-watch-daemon.service /etc/systemd/system/
sudo systemctl enable --now roadrisk-watch-daemon.service     # daemon mode
```

Adjust `WorkingDirectory`, `User`, the venv path, and the `Environment=` lines to
your host, then `systemctl daemon-reload`.

## Configuration (environment)

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRAFFIC_SAFETY_WATCH_DB` | Watch database path (must match the API) | `data/watches.sqlite3` |
| `TRAFFIC_SAFETY_MODEL_PATH` | Trained model bundle | `models/traffic_safety.joblib` |
| `TRAFFIC_SAFETY_SEGMENT_MODEL_PATH` | Segment model (area/region watches) | `models/traffic_safety_segments.joblib` |
| `TRAFFIC_SAFETY_TILES_DIR` | Risk overlay tiles | `tiles/` |

### Email delivery (SMTP)

`email`-channel watches need an SMTP relay. Without one, the sweep still runs and
logs `smtp_not_configured` for those watches instead of failing.

| Variable | Purpose | Default |
| --- | --- | --- |
| `TRAFFIC_SAFETY_SMTP_HOST` | Relay host (enables email when set) | — (email disabled) |
| `TRAFFIC_SAFETY_SMTP_PORT` | Relay port | `587` |
| `TRAFFIC_SAFETY_SMTP_USERNAME` | Login user (also the default From) | — |
| `TRAFFIC_SAFETY_SMTP_PASSWORD` | Login password | — |
| `TRAFFIC_SAFETY_SMTP_FROM` | From address | username, else `alerts@roadriskmonitor.local` |
| `TRAFFIC_SAFETY_SMTP_STARTTLS` | STARTTLS (`0`/`false`/`no`/`off` disables) | enabled |

`GET /v1/meta` reports `watches.email_configured` so you can confirm the API host
sees the relay. Note the evaluator, not the API, sends email — set the SMTP
variables in the evaluator's environment.

## Monitoring

- **Timer health:** `systemctl list-timers roadrisk-evaluate-watches.timer` and
  `journalctl -u roadrisk-evaluate-watches.service`.
- **Daemon health:** `systemctl status roadrisk-watch-daemon.service`.
- **Per-sweep:** grep the summary line for `errors=` (a persistently non-zero
  count means a provider outage or a bad watch) and `notified=` (deliveries).
- **Delivery failures:** failed webhook/email deliveries print a
  `watch <id>: <channel> delivery failed: …` line and are not counted as
  notified; the watch's `last_notified_at` stays unchanged so the next sweep
  retries (subject to cooldown).

## Failure modes & recovery

| Symptom | Cause | What to do |
| --- | --- | --- |
| `errors=N` every sweep | Live-weather provider outage, or a watch with bad geometry | Check which watch IDs log failures; provider outages clear on their own |
| `smtp_not_configured` | SMTP env not set in the evaluator's environment | Set `TRAFFIC_SAFETY_SMTP_*` and restart |
| No alerts despite high risk | Breach is relative — risk may be normal for that place/time; or cooldown still active | Expected; see `last_level` / `last_notified_at` via `GET /v1/watches/{id}` |
| Watches not found by the evaluator | API and evaluator point at different databases | Set the same `TRAFFIC_SAFETY_WATCH_DB` for both |
| Sweep is slow | Many watches, long horizons, or many regions (each samples a grid) | Lengthen the interval, or shard watches across hosts by DB |
