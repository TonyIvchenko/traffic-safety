"""Regional Road Risk Advisory index from the climatological risk overlay.

Samples the nationwide weekly risk cube (``OVERLAY["risk"]`` — a
``(frames, height, width)`` grid) within a metro region's bounding box, reduces
the sampled cells to a single representative risk score, then maps that score to
a public advisory via :mod:`advisory`.

Pure and deterministic. The cube may be a numpy array or nested Python lists;
the bbox→cell index math mirrors ``api_v1._sample_risk_grid`` so a region's index
agrees with the heatmap it summarises. Everything degrades to an empty sample
(risk 0.0, "Low") rather than raising on malformed cubes/coverage/regions.
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
