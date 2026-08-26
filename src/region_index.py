"""Regional Road Risk Advisory index — climatological and live.

Two ways to reduce a metro region to a single representative risk score and its
public advisory (:mod:`advisory`):

* **Climatological** (:func:`region_index`, :func:`index_all_regions`) — sample
  the nationwide weekly risk cube (``OVERLAY["risk"]`` — a
  ``(frames, height, width)`` grid) within the region's bounding box. The
  bbox→cell index math mirrors ``api_v1._sample_risk_grid`` so a region's index
  agrees with the heatmap it summarises.
* **Live** (:func:`live_region_index`) — call an injected point predictor at a
  small grid of representative interior points and aggregate the returned
  scores. Per-point provider/network failures are counted, not fatal, so a
  partial outage still yields a region reading.

:func:`compare_to_normal` expresses a live score relative to its climatological
baseline ("worse/better than normal").

* **Relative** (:func:`cell_relative_advisory`, :func:`region_relative_advisory`)
  — build a location's 168-hour weekly climatology from the *raw model* (an
  injected ``predict(lat, lon, day_of_week, hour, month)``) and express a current
  score as its percentile within that week. The overlay cube is normalised/clipped
  and near-saturated in metros, so it cannot supply a discriminating reference;
  the raw model retains the weekly variation, hence the separate predictor here.

Pure and deterministic (the predictor is injected). Everything degrades to an
empty sample (risk 0.0, "Low") rather than raising on malformed
cubes/coverage/regions/predictions.
"""

from __future__ import annotations

import math

import advisory
import regions as _regions

DEFAULT_MAX_CELLS = 1024
DEFAULT_STATISTIC = "p90"


def _safe_int(value):
    if value is None:
        return None
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _len(obj) -> int:
    try:
        return len(obj)
    except TypeError:
        return 0


def _cov(coverage, key: str, default: float) -> float:
    try:
        return float(coverage.get(key, default))
    except (TypeError, ValueError, AttributeError):
        return default


def _finite(value) -> bool:
    try:
        return math.isfinite(float(value))
    except (TypeError, ValueError):
        return False


def _percentile(sorted_values: list[float], q: float) -> float:
    """Linear-interpolated percentile of an ascending list; q in [0, 1]."""
    if not sorted_values:
        return 0.0
    if len(sorted_values) == 1:
        return sorted_values[0]
    pos = q * (len(sorted_values) - 1)
    lo = int(math.floor(pos))
    hi = int(math.ceil(pos))
    if lo == hi:
        return sorted_values[lo]
    frac = pos - lo
    return sorted_values[lo] * (1.0 - frac) + sorted_values[hi] * frac


def _reduce(values: list[float], statistic) -> float:
    """Reduce sampled risks to one representative score by ``statistic``.

    Supports ``mean``, ``max``, ``min`` and ``pNN`` percentiles (e.g. ``p90``).
    Unknown statistics fall back to the mean. Empty input reduces to 0.0.
    """
    clean = [float(v) for v in values if _finite(v)]
    if not clean:
        return 0.0
    stat = str(statistic).strip().lower()
    if stat == "mean":
        return sum(clean) / len(clean)
    if stat == "max":
        return max(clean)
    if stat == "min":
        return min(clean)
    if stat.startswith("p"):
        try:
            q = float(stat[1:]) / 100.0
        except ValueError:
            q = 0.90
        return _percentile(sorted(clean), max(0.0, min(1.0, q)))
    return sum(clean) / len(clean)


def _cell_bounds(coverage, bbox, height: int, width: int):
    """Row/col index window for a bbox, or None if it falls outside the grid.

    Mirrors ``api_v1._sample_risk_grid``: ceil on the north/left edge and floor on
    the south/right edge so every sampled cell centre stays within the bbox.
    """
    if height == 0 or width == 0:
        return None
    lat_min = _cov(coverage, "lat_min", -90.0)
    lat_max = _cov(coverage, "lat_max", 90.0)
    lon_min = _cov(coverage, "lon_min", -180.0)
    lon_max = _cov(coverage, "lon_max", 180.0)
    lat_span = (lat_max - lat_min) or 1.0
    lon_span = (lon_max - lon_min) or 1.0
    row_denom = (height - 1) or 1
    col_denom = (width - 1) or 1
    try:
        qmin_lat, qmax_lat, qmin_lon, qmax_lon = (float(x) for x in bbox)
    except (TypeError, ValueError):
        return None

    # A bbox with no overlap with the covered extent samples nothing (rather than
    # clamping to a spurious edge cell as the raw heatmap math would).
    if (
        qmax_lat < lat_min
        or qmin_lat > lat_max
        or qmax_lon < lon_min
        or qmin_lon > lon_max
    ):
        return None

    def clip(value, hi):
        return max(0, min(hi, int(value)))

    row_top = clip(math.ceil((lat_max - qmax_lat) / lat_span * row_denom), height - 1)
    row_bottom = clip(math.floor((lat_max - qmin_lat) / lat_span * row_denom), height - 1)
    col_left = clip(math.ceil((qmin_lon - lon_min) / lon_span * col_denom), width - 1)
    col_right = clip(math.floor((qmax_lon - lon_min) / lon_span * col_denom), width - 1)
    if row_top > row_bottom or col_left > col_right:
        return None
    return row_top, row_bottom, col_left, col_right


def sample_region_cells(
    cube,
    coverage,
    bbox,
    *,
    frame_idx=None,
    max_cells: int = DEFAULT_MAX_CELLS,
    min_risk: float = 0.0,
) -> list[float]:
    """Risk values sampled within ``bbox`` (bounded to ~``max_cells`` per frame).

    With ``frame_idx=None`` every frame is sampled (whole-week climatology);
    otherwise a single frame (clamped into range). Non-finite cells and cells
    below ``min_risk`` are dropped. Returns [] on any malformed input.
    """
    frames = _len(cube)
    if frames == 0:
        return []
    if frame_idx is None:
        indices = list(range(frames))
    else:
        fi = _safe_int(frame_idx)
        if fi is None:
            return []
        indices = [max(0, min(frames - 1, fi))]

    first = cube[indices[0]]
    height = _len(first)
    width = _len(first[0]) if height else 0
    bounds = _cell_bounds(coverage, bbox, height, width)
    if bounds is None:
        return []
    row_top, row_bottom, col_left, col_right = bounds

    try:
        cap = max(1, int(max_cells))
    except (TypeError, ValueError):
        cap = DEFAULT_MAX_CELLS
    total = (row_bottom - row_top + 1) * (col_right - col_left + 1)
    stride = max(1, int(math.ceil(math.sqrt(total / cap)))) if total > cap else 1

    try:
        floor = float(min_risk)
    except (TypeError, ValueError):
        floor = 0.0

    values: list[float] = []
    for fi in indices:
        frame = cube[fi]
        for row in range(row_top, row_bottom + 1, stride):
            frame_row = frame[row]
            for col in range(col_left, col_right + 1, stride):
                try:
                    risk = float(frame_row[col])
                except (TypeError, ValueError, IndexError):
                    continue
                if not math.isfinite(risk) or risk < floor:
                    continue
                values.append(risk)
    return values


def region_risk_score(
    cube,
    coverage,
    bbox,
    *,
    frame_idx=None,
    statistic=DEFAULT_STATISTIC,
    max_cells: int = DEFAULT_MAX_CELLS,
    min_risk: float = 0.0,
) -> float:
    """Representative risk score for a bbox under ``statistic`` (default p90)."""
    values = sample_region_cells(
        cube, coverage, bbox, frame_idx=frame_idx, max_cells=max_cells, min_risk=min_risk
    )
    return _reduce(values, statistic)


def region_index(
    cube,
    coverage,
    region,
    *,
    frame_idx=None,
    statistic=DEFAULT_STATISTIC,
    max_cells: int = DEFAULT_MAX_CELLS,
    min_risk: float = 0.0,
    drivers=None,
) -> dict | None:
    """Full advisory index for one region, or None if the region is unknown.

    ``region`` may be a region dict (``{"id","name","bbox"}``) or a region id
    string. The returned dict carries the headline ``risk_score`` (per
    ``statistic``), transparency stats (mean/max/p90), the sample count, and the
    mapped :func:`advisory.advisory` block.
    """
    reg = region if isinstance(region, dict) else _regions.get_region(region)
    if reg is None or "bbox" not in reg:
        return None
    bbox = _regions.region_bbox(reg)
    values = sample_region_cells(
        cube, coverage, bbox, frame_idx=frame_idx, max_cells=max_cells, min_risk=min_risk
    )
    score = _reduce(values, statistic)
    return {
        "region_id": reg.get("id"),
        "region_name": reg.get("name"),
        "bbox": list(bbox),
        "frame_idx": _safe_int(frame_idx),
        "statistic": str(statistic),
        "sample_count": len(values),
        "risk_score": round(score, 4),
        "risk_mean": round(_reduce(values, "mean"), 4),
        "risk_max": round(_reduce(values, "max"), 4),
        "risk_p90": round(_reduce(values, "p90"), 4),
        "advisory": advisory.advisory(score, drivers=drivers),
    }


def index_all_regions(
    cube,
    coverage,
    *,
    region_catalog=None,
    frame_idx=None,
    statistic=DEFAULT_STATISTIC,
    max_cells: int = DEFAULT_MAX_CELLS,
    min_risk: float = 0.0,
) -> list[dict]:
    """Advisory index for every region, sorted by risk score (highest first)."""
    catalog = region_catalog if region_catalog is not None else _regions.list_regions()
    out: list[dict] = []
    for reg in catalog:
        idx = region_index(
            cube,
            coverage,
            reg,
            frame_idx=frame_idx,
            statistic=statistic,
            max_cells=max_cells,
            min_risk=min_risk,
        )
        if idx is not None:
            out.append(idx)
    out.sort(key=lambda r: r["risk_score"], reverse=True)
    return out


# --------------------------------------------------------------------------- #
# Live regional index (injected point predictor)
# --------------------------------------------------------------------------- #

DEFAULT_LIVE_GRID = (3, 3)


def _clamp01(value) -> float:
    try:
        v = float(value)
    except (TypeError, ValueError):
        return 0.0
    if not math.isfinite(v):
        return 0.0
    return 0.0 if v < 0.0 else 1.0 if v > 1.0 else v


def grid_sample_points(bbox, rows: int = 3, cols: int = 3) -> list[dict]:
    """A ``rows`` x ``cols`` grid of interior sample points within ``bbox``.

    Points sit at sub-cell centres (fractions ``(k + 0.5) / n``) so none land on
    the bbox edges. Returns [] for a degenerate/malformed bbox.
    """
    try:
        min_lat, max_lat, min_lon, max_lon = (float(x) for x in bbox)
    except (TypeError, ValueError):
        return []
    try:
        r = max(1, int(rows))
        c = max(1, int(cols))
    except (TypeError, ValueError):
        r, c = 3, 3
    if max_lat < min_lat or max_lon < min_lon:
        return []
    lat_span = max_lat - min_lat
    lon_span = max_lon - min_lon
    points: list[dict] = []
    for i in range(r):
        lat = min_lat + (i + 0.5) / r * lat_span
        for j in range(c):
            lon = min_lon + (j + 0.5) / c * lon_span
            points.append({"lat": round(lat, 6), "lon": round(lon, 6)})
    return points


def _score_of(prediction):
    """Extract a finite [0, 1] risk score from a predictor result, or None."""
    value = prediction.get("risk_score") if isinstance(prediction, dict) else prediction
    try:
        score = float(value)
    except (TypeError, ValueError):
        return None
    if not math.isfinite(score):
        return None
    return _clamp01(score)


def live_region_index(
    predict,
    region,
    *,
    rows: int = 3,
    cols: int = 3,
    statistic=DEFAULT_STATISTIC,
    drivers=None,
) -> dict | None:
    """Live advisory index for a region from an injected point predictor.

    ``predict`` is called as ``predict(lat, lon)`` per representative point and
    must return a dict with a ``risk_score`` (or a bare number). Per-point
    exceptions and non-numeric results are counted in ``failed_points`` rather
    than propagated, so a partial provider outage still produces a reading;
    ``status`` is ``"unavailable"`` (and ``advisory`` None) only when *every*
    point failed. Returns None for an unknown region.
    """
    reg = region if isinstance(region, dict) else _regions.get_region(region)
    if reg is None or "bbox" not in reg:
        return None
    bbox = _regions.region_bbox(reg)
    points = grid_sample_points(bbox, rows=rows, cols=cols)

    scores: list[float] = []
    failed = 0
    for point in points:
        try:
            prediction = predict(point["lat"], point["lon"])
        except Exception:  # noqa: BLE001 - per-point provider/network faults are non-fatal
            failed += 1
            continue
        score = _score_of(prediction)
        if score is None:
            failed += 1
            continue
        scores.append(score)

    available = bool(scores)
    score = _reduce(scores, statistic) if available else 0.0
    return {
        "region_id": reg.get("id"),
        "region_name": reg.get("name"),
        "bbox": list(bbox),
        "mode": "live",
        "status": "ok" if available else "unavailable",
        "statistic": str(statistic),
        "sample_points": len(points),
        "sample_count": len(scores),
        "failed_points": failed,
        "risk_score": round(score, 4),
        "risk_mean": round(_reduce(scores, "mean"), 4),
        "risk_max": round(_reduce(scores, "max"), 4),
        "risk_p90": round(_reduce(scores, "p90"), 4),
        "advisory": advisory.advisory(score, drivers=drivers) if available else None,
    }


def compare_to_normal(now_score, normal_score, *, tolerance: float = 0.05) -> dict:
    """Express a current score relative to its climatological baseline.

    ``comparison`` is "worse than normal" / "better than normal" when the scores
    differ by more than ``tolerance``, else "about normal". ``ratio`` is None
    when the baseline is zero (undefined).
    """
    now = _clamp01(now_score)
    normal = _clamp01(normal_score)
    delta = now - normal
    try:
        band = abs(float(tolerance))
    except (TypeError, ValueError):
        band = 0.05
    if abs(delta) <= band:
        comparison = "about normal"
    elif delta > 0:
        comparison = "worse than normal"
    else:
        comparison = "better than normal"
    ratio = round(now / normal, 3) if normal > 0.0 else None
    return {
        "now": round(now, 4),
        "normal": round(normal, 4),
        "delta": round(delta, 4),
        "ratio": ratio,
        "comparison": comparison,
    }


# --------------------------------------------------------------------------- #
# Relative advisory: raw-model weekly climatology as the reference distribution
# --------------------------------------------------------------------------- #

WEEKLY_FRAMES = 168  # 24 hours x 7 days


def cell_weekly_profile(predict, lat, lon, *, month: int = 1) -> list[float]:
    """The 168 climatological risk scores at a point, one per hour-of-week.

    ``predict`` is an injected point predictor called as
    ``predict(lat, lon, day_of_week, hour, month)`` (day_of_week 1-7, hour 0-23).
    A frame whose prediction fails or is non-numeric contributes 0.0 so the list
    stays index-aligned with the hour-of-week. This raw-model profile — not the
    clipped overlay cube — is the discriminating reference for a relative advisory.
    """
    profile: list[float] = []
    for hour_of_week in range(WEEKLY_FRAMES):
        try:
            result = predict(lat, lon, hour_of_week // 24 + 1, hour_of_week % 24, month)
        except Exception:  # noqa: BLE001 - a single bad frame must not sink the profile
            profile.append(0.0)
            continue
        score = _score_of(result)
        profile.append(score if score is not None else 0.0)
    return profile


def region_weekly_profile(
    predict,
    region,
    *,
    rows: int = 3,
    cols: int = 3,
    month: int = 1,
    statistic=DEFAULT_STATISTIC,
) -> list[float]:
    """A representative region score per hour-of-week from the raw model.

    For each of the 168 frames, ``predict`` is evaluated at the region's
    representative interior grid and reduced by ``statistic`` (default p90).
    ``region`` may be a region dict or id; returns [] for an unknown region.
    """
    reg = region if isinstance(region, dict) else _regions.get_region(region)
    if reg is None or "bbox" not in reg:
        return []
    bbox = _regions.region_bbox(reg)
    points = grid_sample_points(bbox, rows=rows, cols=cols)
    if not points:
        return []
    profile: list[float] = []
    for hour_of_week in range(WEEKLY_FRAMES):
        day_of_week = hour_of_week // 24 + 1
        hour = hour_of_week % 24
        scores: list[float] = []
        for point in points:
            try:
                result = predict(point["lat"], point["lon"], day_of_week, hour, month)
            except Exception:  # noqa: BLE001 - per-point faults are non-fatal
                continue
            score = _score_of(result)
            if score is not None:
                scores.append(score)
        profile.append(_reduce(scores, statistic) if scores else 0.0)
    return profile


def cell_relative_advisory(
    predict,
    lat,
    lon,
    *,
    day_of_week: int,
    hour: int,
    month: int = 1,
    value=None,
    drivers=None,
) -> dict:
    """Relative advisory for a point: current score vs its own weekly climatology.

    ``value`` overrides the compared score (pass the live score in live mode);
    when None the climatological score at ``day_of_week``/``hour`` is used.
    """
    profile = cell_weekly_profile(predict, lat, lon, month=month)
    frame_idx = (day_of_week - 1) * 24 + hour
    if value is None:
        value = profile[frame_idx] if 0 <= frame_idx < len(profile) else 0.0
    result = advisory.relative_advisory(value, profile, drivers=drivers)
    result["frame_idx"] = frame_idx
    result["reference_size"] = len(profile)
    return result


def region_relative_advisory(
    predict,
    region,
    *,
    day_of_week: int,
    hour: int,
    month: int = 1,
    value=None,
    drivers=None,
    rows: int = 3,
    cols: int = 3,
    statistic=DEFAULT_STATISTIC,
) -> dict | None:
    """Relative advisory for a region: current aggregate vs its weekly climatology.

    ``value`` overrides the compared score (pass the live region aggregate in live
    mode); when None the climatological aggregate at ``day_of_week``/``hour`` is
    used. Returns None for an unknown region.
    """
    reg = region if isinstance(region, dict) else _regions.get_region(region)
    if reg is None or "bbox" not in reg:
        return None
    profile = region_weekly_profile(
        predict, reg, rows=rows, cols=cols, month=month, statistic=statistic
    )
    frame_idx = (day_of_week - 1) * 24 + hour
    if value is None:
        value = profile[frame_idx] if 0 <= frame_idx < len(profile) else 0.0
    result = advisory.relative_advisory(value, profile, drivers=drivers)
    result["region_id"] = reg.get("id")
    result["region_name"] = reg.get("name")
    result["frame_idx"] = frame_idx
    result["reference_size"] = len(profile)
    return result
