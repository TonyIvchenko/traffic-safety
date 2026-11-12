"""Crash-trend and emerging-hotspot analytics.

Answers "where are crashes *rising*?" — the question a safety program asks after
it has mapped where crashes *are*. Given a per-location yearly crash series, it
reports a robust trend and classifies each location into an emerging-hotspot
taxonomy inspired by the Getis-Ord / ESRI framework (new, intensifying,
persistent, diminishing, past, emerging, sporadic), so an agency can separate a
genuinely worsening corridor from one that is merely always bad or bouncing
around.

Two non-parametric statistics do the work, both standard for crash counts (short,
noisy, non-normal series):

* **Mann-Kendall** — a monotonic trend test. Counts how often later years exceed
  earlier ones (``S``), corrects the variance for ties, and reports a two-sided
  p-value and a direction. No distributional assumption, robust to outliers.
* **Theil-Sen** — the median of all pairwise slopes, a robust crashes-per-year
  trend that a single freak year cannot swing.

"Hot" is defined cross-sectionally in :func:`emerging_hotspots`: for each year, a
location is hot if its count is at or above the ``hot_quantile`` of all locations
that year (or above an absolute ``hot_threshold``). The emerging taxonomy then
combines each location's per-year hot flags with its trend.

Pure and defensive: short/degenerate series return an ``insufficient`` trend
rather than raising, and nothing here performs I/O.
"""

from __future__ import annotations

import math
from statistics import median

DEFAULT_ALPHA = 0.05
DEFAULT_RECENT_YEARS = 3
DEFAULT_HOT_QUANTILE = 0.9
MIN_SERIES_LEN = 3  # Mann-Kendall needs at least a few points to mean anything

# Ranking priority per emerging-hotspot category: rising locations first.
_CATEGORY_PRIORITY = {
    "intensifying_hotspot": 7,
    "new_hotspot": 6,
    "emerging": 5,
    "sporadic_hotspot": 4,
    "persistent_hotspot": 3,
    "diminishing_hotspot": 2,
    "past_hotspot": 1,
    "none": 0,
    "insufficient": -1,
}
# The emerging-hotspot categories a classified location can receive, worst-to-best.
CATEGORIES = (
    "intensifying_hotspot",
    "new_hotspot",
    "emerging",
    "sporadic_hotspot",
    "persistent_hotspot",
    "diminishing_hotspot",
    "past_hotspot",
    "none",
)


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _clean_series(values) -> list[float]:
    """Coerce a series to finite floats, dropping values that cannot parse."""
    out: list[float] = []
    for value in values or []:
        number = _finite(value)
        if number is not None:
            out.append(number)
    return out


def _normal_cdf(z: float) -> float:
    return 0.5 * (1.0 + math.erf(z / math.sqrt(2.0)))


def mann_kendall(values, *, alpha: float = DEFAULT_ALPHA) -> dict | None:
    """Mann-Kendall monotonic trend test over an ordered series.

    Returns ``s``, tie-corrected variance, ``z``, two-sided ``p_value``, and a
    ``trend`` label ("increasing"/"decreasing"/"no trend") at ``alpha``. None when
    fewer than ``MIN_SERIES_LEN`` finite points are supplied.
    """
    series = _clean_series(values)
    n = len(series)
    if n < MIN_SERIES_LEN:
        return None

    s = 0
    for i in range(n - 1):
        for j in range(i + 1, n):
            diff = series[j] - series[i]
            s += (diff > 0) - (diff < 0)  # sign(+1/0/-1)

    # Tie-corrected variance of S.
    counts: dict[float, int] = {}
    for value in series:
        counts[value] = counts.get(value, 0) + 1
    tie_term = sum(t * (t - 1) * (2 * t + 5) for t in counts.values())
    variance = (n * (n - 1) * (2 * n + 5) - tie_term) / 18.0

    if variance <= 0:
        z = 0.0
    elif s > 0:
        z = (s - 1) / math.sqrt(variance)
    elif s < 0:
        z = (s + 1) / math.sqrt(variance)
    else:
        z = 0.0

    p_value = 2.0 * (1.0 - _normal_cdf(abs(z)))
    p_value = min(1.0, max(0.0, p_value))
    significant = p_value <= float(alpha)
    if significant and s > 0:
        trend = "increasing"
    elif significant and s < 0:
        trend = "decreasing"
    else:
        trend = "no trend"

    return {
        "s": int(s),
        "variance": round(variance, 4),
        "z": round(z, 4),
        "p_value": round(p_value, 4),
        "trend": trend,
        "significant": bool(significant),
        "n": n,
    }


def theil_sen(values, times=None) -> dict | None:
    """Theil-Sen robust slope (and intercept) of a series over time.

    ``slope`` is the median of all pairwise slopes — crashes per unit time (per
    year when ``times`` is omitted, so time is 0,1,2,…). None if fewer than two
    finite points, or all time values coincide.
    """
    series = _clean_series(values)
    n = len(series)
    if n < 2:
        return None
    if times is None:
        time_points = [float(i) for i in range(n)]
    else:
        time_points = _clean_series(times)
        if len(time_points) != n:
            return None

    slopes = []
    for i in range(n - 1):
        for j in range(i + 1, n):
            dt = time_points[j] - time_points[i]
            if dt != 0:
                slopes.append((series[j] - series[i]) / dt)
    if not slopes:
        return None
    slope = median(slopes)
    intercept = median(series[i] - slope * time_points[i] for i in range(n))
    return {"slope": round(slope, 6), "intercept": round(intercept, 6)}


def _recent_baseline(series: list[float], recent_years: int) -> dict:
    """Split a series into a recent window and the earlier baseline, with means
    and the change between them."""
    n = len(series)
    window = max(1, min(int(recent_years), n))
    recent = series[-window:]
    baseline = series[:-window]
    recent_mean = sum(recent) / len(recent)
    baseline_mean = sum(baseline) / len(baseline) if baseline else None
    change = recent_mean - baseline_mean if baseline_mean is not None else None
    percent_change = (
        round(100.0 * change / baseline_mean, 2)
        if (change is not None and baseline_mean)
        else None
    )
    return {
        "recent_years": window,
        "recent_mean": round(recent_mean, 4),
        "baseline_mean": round(baseline_mean, 4) if baseline_mean is not None else None,
        "recent_vs_baseline_change": round(change, 4) if change is not None else None,
        "recent_vs_baseline_percent": percent_change,
    }


def _yoy(series: list[float]) -> dict:
    """Year-over-year changes and the latest one."""
    changes = [round(series[i] - series[i - 1], 4) for i in range(1, len(series))]
    latest = changes[-1] if changes else None
    prev = series[-2] if len(series) >= 2 else None
    latest_pct = round(100.0 * latest / prev, 2) if (latest is not None and prev) else None
    return {
        "yoy_changes": changes,
        "latest_yoy_change": latest,
        "latest_yoy_percent": latest_pct,
    }


def classify(hot_flags, trend: str, *, recent_window: int = DEFAULT_RECENT_YEARS) -> str:
    """Emerging-hotspot category from per-period hot flags + a trend direction.

    ``hot_flags`` is a boolean list aligned to the series (hot in that period).
    The taxonomy is a simplified Getis-Ord emerging-hotspot classification.
    """
    flags = [bool(f) for f in hot_flags]
    n = len(flags)
    if n == 0:
        return "insufficient"
    recent_hot = flags[-1]
    ever_hot = any(flags)
    hot_count = sum(flags)
    window = max(1, min(int(recent_window), n))
    earlier = flags[:-window]
    hot_earlier = any(earlier)
    mostly_hot = hot_count >= math.ceil(n / 2)

    increasing = trend == "increasing"
    decreasing = trend == "decreasing"

    if not ever_hot:
        # Never crossed the hot threshold; only a rising path is noteworthy.
        return "emerging" if increasing else "none"
    if recent_hot and increasing and mostly_hot:
        return "intensifying_hotspot"
    if recent_hot and not hot_earlier:
        # Hot now but not before the recent window -> newly emerged.
        return "new_hotspot"
    if recent_hot and increasing:
        return "intensifying_hotspot"
    if recent_hot and decreasing:
        return "diminishing_hotspot"
    if recent_hot and mostly_hot:
        return "persistent_hotspot"
    if recent_hot:
        return "sporadic_hotspot"
    # Not hot in the most recent period.
    if increasing:
        return "emerging"
    return "past_hotspot"


def series_trend(
    series,
    *,
    years=None,
    recent_years: int = DEFAULT_RECENT_YEARS,
    alpha: float = DEFAULT_ALPHA,
    hot_threshold=None,
    hot_flags=None,
) -> dict:
    """Full trend summary for one location's crash series.

    Combines Mann-Kendall, Theil-Sen, recent-vs-baseline and year-over-year
    change, and (when a ``hot_threshold`` or precomputed ``hot_flags`` is given)
    an emerging-hotspot category. Returns ``status='insufficient'`` for a series
    shorter than ``MIN_SERIES_LEN``.
    """
    clean = _clean_series(series)
    n = len(clean)
    if n < MIN_SERIES_LEN:
        return {"status": "insufficient", "n": n, "note": f"need at least {MIN_SERIES_LEN} finite points"}

    mk = mann_kendall(clean, alpha=alpha)
    slope = theil_sen(clean, times=years)
    trend = mk["trend"] if mk else "no trend"

    result = {
        "status": "ok",
        "n": n,
        "total": round(sum(clean), 4),
        "mean": round(sum(clean) / n, 4),
        "first": round(clean[0], 4),
        "last": round(clean[-1], 4),
        "mann_kendall": mk,
        "trend": trend,
        "trend_per_year": slope["slope"] if slope else None,
        **_recent_baseline(clean, recent_years),
        **_yoy(clean),
    }

    flags = None
    if hot_flags is not None:
        flags = [bool(f) for f in hot_flags][:n]
    elif hot_threshold is not None:
        threshold = _finite(hot_threshold)
        if threshold is not None:
            flags = [value >= threshold and value > 0 for value in clean]
    if flags is not None:
        result["hot_flags"] = flags
        result["recent_hot"] = bool(flags[-1]) if flags else False
        result["category"] = classify(flags, trend, recent_window=recent_years)
    return result


def _period_hot_thresholds(columns: list[list[float]], hot_quantile: float):
    """Per-period cutoff: the ``hot_quantile`` of the positive counts that period."""
    thresholds = []
    q = min(1.0, max(0.0, _finite(hot_quantile) if _finite(hot_quantile) is not None else DEFAULT_HOT_QUANTILE))
    for column in columns:
        positive = sorted(v for v in column if v > 0)
        if not positive:
            thresholds.append(None)
            continue
        # Nearest-rank quantile over the positive counts that period.
        rank = max(0, min(len(positive) - 1, math.ceil(q * len(positive)) - 1))
        thresholds.append(positive[rank])
    return thresholds


def emerging_hotspots(
    locations,
    *,
    recent_years: int = DEFAULT_RECENT_YEARS,
    hot_quantile: float = DEFAULT_HOT_QUANTILE,
    hot_threshold=None,
    alpha: float = DEFAULT_ALPHA,
    top_n: int = 50,
    rising_only: bool = False,
) -> dict:
    """Classify and rank locations by crash trend, surfacing rising ones first.

    ``locations`` is an iterable of dicts, each with a ``series`` (equal-length
    yearly counts) plus optional ``location_id``/``name``/``lat``/``lon``. "Hot"
    is cross-sectional per year (the ``hot_quantile`` of that year's positive
    counts) unless an absolute ``hot_threshold`` is supplied. Returns the ranked
    locations and a category tally. Locations with a too-short/ragged series are
    reported with ``status='insufficient'`` and excluded from the ranking.
    """
    records = []
    for loc in locations or []:
        if not isinstance(loc, dict):
            continue
        records.append(loc)

    cleaned = [(_clean_series(loc.get("series")), loc) for loc in records]
    valid = [(series, loc) for series, loc in cleaned if len(series) >= MIN_SERIES_LEN]
    # Cross-sectional hot flags require a common period length; use the modal length.
    lengths = [len(series) for series, _ in valid]
    series_len = max(set(lengths), key=lengths.count) if lengths else 0

    thresholds = None
    if hot_threshold is None and series_len:
        columns = [
            [series[p] for series, _ in valid if len(series) == series_len]
            for p in range(series_len)
        ]
        thresholds = _period_hot_thresholds(columns, hot_quantile)

    ranked = []
    insufficient = []
    for series, loc in cleaned:
        meta = {key: loc.get(key) for key in ("location_id", "name", "lat", "lon") if loc.get(key) is not None}
        if len(series) < MIN_SERIES_LEN:
            insufficient.append({**meta, "status": "insufficient", "n": len(series)})
            continue

        if hot_threshold is not None:
            threshold = _finite(hot_threshold)
            flags = [v >= threshold and v > 0 for v in series] if threshold is not None else None
        elif thresholds is not None and len(series) == series_len:
            flags = [
                thresholds[p] is not None and series[p] >= thresholds[p]
                for p in range(series_len)
            ]
        else:
            flags = None

        trend = series_trend(
            series, recent_years=recent_years, alpha=alpha, hot_flags=flags
        )
        category = trend.get("category", "none")
        trend.update(meta)
        trend["category"] = category
        trend["emergence_priority"] = _CATEGORY_PRIORITY.get(category, 0)
        ranked.append(trend)

    if rising_only:
        ranked = [r for r in ranked if r["category"] in ("new_hotspot", "intensifying_hotspot", "emerging")]

    ranked.sort(
        key=lambda r: (
            r.get("emergence_priority", 0),
            r.get("trend_per_year") or 0.0,
            r.get("recent_mean") or 0.0,
        ),
        reverse=True,
    )
    limited = ranked[: int(top_n)] if top_n else ranked

    tally: dict[str, int] = {}
    for record in ranked:
        tally[record["category"]] = tally.get(record["category"], 0) + 1

    return {
        "status": "ok",
        "method": "mann_kendall_emerging_hotspot",
        "location_count": len(records),
        "evaluated_count": len(ranked),
        "insufficient_count": len(insufficient),
        "series_length": series_len,
        "hot_basis": "absolute_threshold" if hot_threshold is not None else "cross_sectional_quantile",
        "category_counts": tally,
        "hotspots": limited,
        "insufficient": insufficient,
    }
