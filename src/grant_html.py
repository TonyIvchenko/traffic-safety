"""Render a grant safety-analysis report (grant_report.assemble_grant_report)
into a single self-contained HTML document.

The output is a downloadable SS4A/HSIP deliverable: everything — styling and the
crashes-by-year chart (inline SVG) — is embedded, so the file needs no network,
no external CSS/JS/fonts, and prints cleanly to PDF. The module is pure (no I/O,
no clock); the caller supplies the report dict. Every value flows through one
HTML-escaping choke point (``_html_table`` / ``_esc``) so untrusted names in the
data cannot inject markup.
"""

from __future__ import annotations

import html

CSS = """
:root { color-scheme: light; }
* { box-sizing: border-box; }
body {
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif;
  color: #1a2230; background: #f5f6f8; margin: 0; line-height: 1.45;
}
.report { max-width: 960px; margin: 0 auto; padding: 32px 24px 64px; background: #fff; }
header.report-head { border-bottom: 3px solid #0b3d91; padding-bottom: 16px; margin-bottom: 24px; }
h1 { font-size: 1.6rem; margin: 0 0 4px; color: #0b3d91; }
h2 { font-size: 1.2rem; margin: 32px 0 12px; color: #0b3d91; }
h3 { font-size: 1rem; margin: 20px 0 8px; }
.jurisdiction { font-size: 1.05rem; font-weight: 600; margin: 4px 0; }
.vintage { color: #566; font-size: 0.85rem; margin: 2px 0; }
.stat-grid { display: grid; grid-template-columns: repeat(auto-fit, minmax(150px, 1fr)); gap: 12px; margin: 12px 0; }
.stat { background: #eef2fb; border: 1px solid #d6e0f5; border-radius: 8px; padding: 12px 14px; }
.stat .value { font-size: 1.5rem; font-weight: 700; color: #0b3d91; }
.stat .label { font-size: 0.78rem; text-transform: uppercase; letter-spacing: 0.04em; color: #566; }
table { border-collapse: collapse; width: 100%; margin: 8px 0 4px; font-size: 0.9rem; }
th, td { text-align: left; padding: 7px 10px; border-bottom: 1px solid #e2e6ee; }
th { background: #0b3d91; color: #fff; font-weight: 600; }
tbody tr:nth-child(even) { background: #f7f9fd; }
td.num, th.num { text-align: right; font-variant-numeric: tabular-nums; }
td.empty { text-align: center; color: #889; font-style: italic; }
dl.methodology { margin: 8px 0; }
dl.methodology dt { font-weight: 600; margin-top: 8px; }
dl.methodology dd { margin: 2px 0 0; color: #334; }
figure { margin: 12px 0; }
figcaption { font-size: 0.8rem; color: #566; }
footer.disclaimer { margin-top: 40px; padding-top: 16px; border-top: 1px solid #e2e6ee;
  font-size: 0.78rem; color: #778; }
@media print { body { background: #fff; } .report { max-width: none; } th { -webkit-print-color-adjust: exact; } }
"""

_LEVEL_LABEL = {"state": "State", "county": "County", "tract": "Census tract", "area": "Area"}


def _esc(value) -> str:
    if value is None:
        return ""
    return html.escape(str(value))


def _int(value) -> str:
    try:
        return f"{int(round(float(value))):,}"
    except (TypeError, ValueError):
        return "0"


def _num(value, digits: int = 2) -> str:
    try:
        return f"{float(value):,.{digits}f}"
    except (TypeError, ValueError):
        return "0"


def _pct(value) -> str:
    try:
        return f"{float(value) * 100:.1f}%"
    except (TypeError, ValueError):
        return "0.0%"


def _route_label(row: dict) -> str:
    parts = [str(row.get(key)) for key in ("rttyp", "mtfcc") if row.get(key) not in (None, "")]
    return " / ".join(parts) if parts else "—"


def _html_table(headers, rows, *, num_cols=(), empty="No data") -> str:
    """Build a table; every header and cell is escaped here (the choke point)."""
    num_cols = set(num_cols)
    head = "".join(
        f'<th class="num">{_esc(header)}</th>' if index in num_cols else f"<th>{_esc(header)}</th>"
        for index, header in enumerate(headers)
    )
    if not rows:
        body = f'<tr><td class="empty" colspan="{len(headers)}">{_esc(empty)}</td></tr>'
    else:
        body = ""
        for row in rows:
            cells = "".join(
                f'<td class="num">{_esc(cell)}</td>' if index in num_cols else f"<td>{_esc(cell)}</td>"
                for index, cell in enumerate(row)
            )
            body += f"<tr>{cells}</tr>"
    return f"<table><thead><tr>{head}</tr></thead><tbody>{body}</tbody></table>"


def bar_chart_svg(pairs, *, width: int = 520, height: int = 200, pad: int = 32, title: str = "") -> str:
    """Inline SVG vertical bar chart from ``(label, value)`` pairs.

    Returns a valid ``<svg>`` for any input, including an empty list (renders a
    'No data' placeholder rather than dividing by zero).
    """
    pairs = [(str(label), float(value)) for label, value in pairs]
    if not pairs:
        return (
            f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title or "chart")}">'
            f'<text x="{width // 2}" y="{height // 2}" text-anchor="middle" fill="#889">No data</text></svg>'
        )
    max_value = max((value for _, value in pairs), default=0.0) or 1.0
    plot_w = width - 2 * pad
    plot_h = height - 2 * pad
    count = len(pairs)
    slot = plot_w / count
    bar_w = max(4.0, slot * 0.62)
    baseline = height - pad
    bars = []
    for index, (label, value) in enumerate(pairs):
        bar_h = plot_h * (value / max_value)
        x = pad + slot * index + (slot - bar_w) / 2
        y = baseline - bar_h
        bars.append(
            f'<rect x="{x:.1f}" y="{y:.1f}" width="{bar_w:.1f}" height="{bar_h:.1f}" fill="#0b3d91" rx="2"/>'
            f'<text x="{x + bar_w / 2:.1f}" y="{y - 4:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#334">{_esc(_int(value))}</text>'
            f'<text x="{x + bar_w / 2:.1f}" y="{baseline + 14:.1f}" text-anchor="middle" '
            f'font-size="11" fill="#566">{_esc(label)}</text>'
        )
    axis = f'<line x1="{pad}" y1="{baseline:.1f}" x2="{width - pad}" y2="{baseline:.1f}" stroke="#ccd3e0"/>'
    return (
        f'<svg viewBox="0 0 {width} {height}" role="img" aria-label="{_esc(title or "chart")}">'
        f"{axis}{''.join(bars)}</svg>"
    )


def _header_section(report: dict) -> str:
    jur = report.get("jurisdiction", {})
    level = _LEVEL_LABEL.get(str(jur.get("level")), "Area")
    vintage = report.get("data_vintage", {}) or {}
    years = vintage.get("fars_years") or []
    year_span = f"{years[0]}–{years[-1]}" if len(years) >= 2 else (str(years[0]) if years else "—")
    generated = report.get("generated_at_utc")
    meta_bits = [f"Crash years {year_span}"]
    if vintage.get("crash_source"):
        meta_bits.append(str(vintage["crash_source"]))
    if generated:
        meta_bits.append(f"Generated {generated}")
    return (
        '<header class="report-head">'
        "<h1>Roadway Safety Analysis</h1>"
        f'<p class="jurisdiction">{_esc(jur.get("name", "—"))}</p>'
        f'<p class="vintage">{_esc(level)} · GEOID {_esc(jur.get("geoid", "—"))}</p>'
        f'<p class="vintage">{_esc(" · ".join(meta_bits))}</p>'
        "</header>"
    )


def _summary_section(report: dict) -> str:
    summary = report.get("crash_summary", {}) or {}
    by_mode = summary.get("by_mode", {}) or {}
    cards = [
        ("Fatal crashes", _int(summary.get("total_fatal_crashes", 0))),
        ("Fatalities", _int(summary.get("total_fatalities", 0))),
    ]
    if "pedestrian" in by_mode:
        cards.append(("Pedestrians involved", _int(by_mode["pedestrian"])))
    if "cyclist" in by_mode:
        cards.append(("Cyclists involved", _int(by_mode["cyclist"])))
    stat_html = "".join(
        f'<div class="stat"><div class="value">{_esc(value)}</div>'
        f'<div class="label">{_esc(label)}</div></div>'
        for label, value in cards
    )
    by_year = summary.get("by_year", {}) or {}
    pairs = [(str(year), by_year[year]) for year in sorted(by_year, key=lambda y: int(y))]
    chart = bar_chart_svg(pairs, title="Fatal crashes by year")
    return (
        '<section id="summary"><h2>Crash summary</h2>'
        f'<div class="stat-grid">{stat_html}</div>'
        f'<figure>{chart}<figcaption>Fatal crashes by year</figcaption></figure>'
        "</section>"
    )


def _hin_section(report: dict) -> str:
    hin = report.get("high_injury_network", {}) or {}
    corridors = report.get("hin_corridors", []) or []
    if hin:
        lede = (
            f"The High Injury Network is <strong>{_esc(_int(hin.get('hin_segments', 0)))}</strong> "
            f"segments — {_esc(_pct(hin.get('length_share', 0)))} of the analyzed network length — "
            f"carrying {_esc(_pct(hin.get('weighted_crash_share', 0)))} of severity-weighted crashes."
        )
    else:
        lede = "No High Injury Network was computed for this jurisdiction."
    rows = [
        [
            corridor.get("hin_rank", ""),
            corridor.get("fullname") or "Unnamed road",
            _route_label(corridor),
            _num(corridor.get("length_km", 0), 2),
            _int(corridor.get("fatal_crashes", 0)),
            _num(corridor.get("hin_intensity", 0), 2),
        ]
        for corridor in corridors
    ]
    table = _html_table(
        ["Rank", "Road", "Route / class", "Length (km)", "Fatal crashes", "Crashes / km"],
        rows,
        num_cols=(0, 3, 4, 5),
        empty="No High Injury Network corridors.",
    )
    return (
        '<section id="hin"><h2>High Injury Network</h2>'
        f"<p>{lede}</p><h3>Top corridors</h3>{table}</section>"
    )


def _systemic_section(report: dict) -> str:
    locations = report.get("systemic_locations", []) or []
    rows = [
        [
            location.get("fullname") or "Unnamed road",
            _route_label(location),
            _num(location.get("length_km", 0), 2),
            _num(location.get("systemic_score", 0), 3),
            _num(location.get("systemic_rate", 0), 3),
            _num(location.get("systemic_expected_crashes", 0), 2),
        ]
        for location in locations
    ]
    table = _html_table(
        ["Road", "Route / class", "Length (km)", "Systemic score", "Rate / km", "Expected crashes"],
        rows,
        num_cols=(2, 3, 4, 5),
        empty="No systemic risk locations.",
    )
    return (
        '<section id="systemic"><h2>Systemic risk — history-poor locations</h2>'
        "<p>Segments with no recorded fatal crash whose roadway class carries elevated systemic risk.</p>"
        f"{table}</section>"
    )


def _benefit_cost_section(report: dict) -> str:
    bc = report.get("benefit_cost")
    if not bc:
        return ""
    rows = [
        ["Annual crash-cost reduction", f"${_num(bc.get('annual_benefit', 0), 0)}"],
        ["Present-value benefit", f"${_num(bc.get('present_value_benefit', 0), 0)}"],
        ["Treatment cost", f"${_num(bc.get('treatment_cost', 0), 0)}"],
        ["Net benefit", f"${_num(bc.get('net_benefit', 0), 0)}"],
        ["Benefit-cost ratio", _num(bc.get("benefit_cost_ratio", 0), 2)],
    ]
    table = _html_table(["Measure", "Value"], rows, num_cols=(1,))
    basis = bc.get("cost_basis", "")
    caption = f"<figcaption>{_esc(basis)}</figcaption>" if basis else ""
    return f'<section id="benefit-cost"><h2>Benefit-cost</h2>{table}{caption}</section>'


def _methodology_section(report: dict) -> str:
    methodology = report.get("methodology", {}) or {}
    if not methodology:
        return ""
    items = "".join(
        f"<dt>{_esc(str(key).replace('_', ' ').title())}</dt><dd>{_esc(value)}</dd>"
        for key, value in methodology.items()
    )
    return f'<section id="methodology"><h2>Methodology</h2><dl class="methodology">{items}</dl></section>'


def _sources_section(report: dict) -> str:
    sources = report.get("data_sources", []) or []
    rows = [[src.get("name", ""), src.get("publisher", ""), src.get("use", "")] for src in sources]
    table = _html_table(["Dataset", "Publisher", "Use"], rows, empty="No data sources listed.")
    return f'<section id="sources"><h2>Data sources</h2>{table}</section>'


def render_report(report: dict) -> str:
    """Render a full, self-contained HTML document for a grant report dict."""
    jur = report.get("jurisdiction", {})
    title = f"Roadway Safety Analysis — {jur.get('name', jur.get('geoid', 'Area'))}"
    body = "".join(
        section
        for section in (
            _header_section(report),
            _summary_section(report),
            _hin_section(report),
            _systemic_section(report),
            _benefit_cost_section(report),
            _methodology_section(report),
            _sources_section(report),
        )
    )
    disclaimer = (
        '<footer class="disclaimer">Generated analysis for planning support. Crash locations '
        "and counts derive from published federal data; treat corridor rankings as screening "
        "guidance, not a substitute for engineering study. Not an official government document.</footer>"
    )
    return (
        "<!doctype html>\n"
        '<html lang="en"><head><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        f"<title>{_esc(title)}</title>"
        f"<style>{CSS}</style></head>"
        f'<body><main class="report">{body}{disclaimer}</main></body></html>'
    )
