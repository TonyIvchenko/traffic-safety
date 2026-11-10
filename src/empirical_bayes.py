"""Empirical-Bayes (EB) before/after safety evaluation.

The standard way to measure whether a countermeasure actually reduced crashes
(AASHTO Highway Safety Manual, Hauer). A naive before/after comparison is biased
by *regression to the mean*: sites are treated because they had a bad spell, and
would have improved anyway. EB corrects for this by blending each site's observed
before-period crashes with a Safety Performance Function (SPF) prediction, then
projecting the EB-expected crashes into the after period to compare against what
was actually observed.

Per treated site the caller supplies:
  * ``observed_before`` / ``observed_after`` — recorded crash counts,
  * ``predicted_before`` / ``predicted_after`` — SPF-predicted crashes for each
    period (these carry traffic-volume and period-length changes), and
  * ``overdispersion`` (``k``) — the SPF's overdispersion parameter.

The result is the safety-effectiveness index ``theta`` (equivalently the crash
modification factor, CMF), its standard error and 95% CI, and the percent change
in crashes. Pure and defensive: invalid sites are skipped, degenerate aggregates
return an ``insufficient`` result rather than dividing by zero.
"""

from __future__ import annotations

import math

Z_95 = 1.959963984540054  # standard normal 97.5th percentile


def _finite(value):
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _nonneg(value):
    number = _finite(value)
    return number if number is not None and number >= 0 else None


def _pos(value):
    number = _finite(value)
    return number if number is not None and number > 0 else None


def eb_weight(predicted_before: float, overdispersion: float) -> float:
    """EB weight on the SPF prediction: ``1 / (1 + k * predicted_before)``.

    1.0 (rely wholly on the SPF) when the overdispersion is non-positive/unknown,
    since RTM cannot then be separated from the observed count.
    """
    predicted = _nonneg(predicted_before)
    k = _pos(overdispersion)
    if predicted is None or k is None:
        return 1.0
    return 1.0 / (1.0 + k * predicted)


def evaluate_site(
    observed_before,
    observed_after,
    predicted_before,
    predicted_after,
    overdispersion,
) -> dict | None:
    """EB expected-after crashes (+ variance) for one treated site, or None if invalid."""
    obs_before = _nonneg(observed_before)
    obs_after = _nonneg(observed_after)
    pred_before = _pos(predicted_before)
    pred_after = _nonneg(predicted_after)
    k = _pos(overdispersion)
    if obs_before is None or obs_after is None or pred_before is None or pred_after is None or k is None:
        return None

    weight = 1.0 / (1.0 + k * pred_before)
    expected_before = weight * pred_before + (1.0 - weight) * obs_before
    ratio = pred_after / pred_before  # carries AADT / period-length change
    expected_after = expected_before * ratio
    variance_after = expected_before * ratio * ratio * (1.0 - weight)
    return {
        "weight": weight,
        "expected_before": expected_before,
        "predicted_ratio": ratio,
        "observed_after": obs_after,
        "expected_after": expected_after,
        "variance_expected_after": variance_after,
    }


def before_after(sites) -> dict:
    """Aggregate EB before/after evaluation over treated sites.

    ``sites`` is an iterable of dicts with ``observed_before``/``observed_after``/
    ``predicted_before``/``predicted_after``/``overdispersion`` (and an optional
    ``site_id``). Returns the per-site EB breakdown plus the aggregate CMF and its
    uncertainty.
    """
    rows: list[dict] = []
    for site in sites or []:
        evaluated = evaluate_site(
            site.get("observed_before"),
            site.get("observed_after"),
            site.get("predicted_before"),
            site.get("predicted_after"),
            site.get("overdispersion"),
        )
        if evaluated is None:
            continue
        rows.append({"site_id": site.get("site_id"), **_rounded(evaluated)})

    total_observed_after = sum(r["observed_after"] for r in rows)
    total_expected_after = sum(r["expected_after"] for r in rows)
    total_variance = sum(r["variance_expected_after"] for r in rows)

    if not rows or total_expected_after <= 0.0:
        return {
            "status": "insufficient",
            "site_count": len(rows),
            "sites": rows,
            "observed_after_total": round(total_observed_after, 4),
            "expected_after_total": round(total_expected_after, 4),
        }

    adjustment = total_variance / (total_expected_after ** 2)
    theta = (total_observed_after / total_expected_after) / (1.0 + adjustment)
    inv_observed = (1.0 / total_observed_after) if total_observed_after > 0 else 0.0
    var_theta = theta * theta * (inv_observed + adjustment) / ((1.0 + adjustment) ** 2)
    stderr = math.sqrt(var_theta) if var_theta > 0 else 0.0
    ci_low = theta - Z_95 * stderr
    ci_high = theta + Z_95 * stderr
    significant = ci_high < 1.0 or ci_low > 1.0

    return {
        "status": "ok",
        "site_count": len(rows),
        "observed_after_total": round(total_observed_after, 4),
        "expected_after_total": round(total_expected_after, 4),
        "cmf": round(theta, 4),
        "cmf_stderr": round(stderr, 4),
        "cmf_ci_95": [round(max(0.0, ci_low), 4), round(ci_high, 4)],
        "percent_change": round(100.0 * (theta - 1.0), 2),
        "percent_reduction": round(100.0 * (1.0 - theta), 2),
        "significant_95": bool(significant),
        "sites": rows,
    }


def _rounded(site: dict) -> dict:
    return {key: round(float(value), 6) for key, value in site.items()}
