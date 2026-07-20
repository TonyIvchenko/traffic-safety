from __future__ import annotations

from pathlib import Path
import re
import sys

SRC_DIR = Path(__file__).resolve().parents[1] / "src"
if str(SRC_DIR) not in sys.path:
    sys.path.insert(0, str(SRC_DIR))

import grant_html as gh


def _report(**overrides) -> dict:
    report = {
        "jurisdiction": {"geoid": "06037", "name": "Los Angeles County", "level": "county"},
        "generated_at_utc": "2026-07-14T00:00:00+00:00",
        "data_vintage": {"fars_years": [2018, 2024], "crash_source": "FARS fatal crashes"},
        "crash_summary": {
            "total_fatal_crashes": 42,
            "total_fatalities": 45,
            "years": [2022, 2023],
            "by_year": {2022: 20, 2023: 22},
            "by_mode": {"pedestrian": 10, "cyclist": 3},
        },
        "high_injury_network": {
            "hin_segments": 2,
            "network_segments": 4,
            "hin_length_km": 2.0,
            "network_length_km": 5.0,
            "length_share": 0.4,
            "weighted_crashes_captured": 8.0,
            "network_weighted_crashes": 9.0,
            "weighted_crash_share": 0.889,
        },
        "hin_corridors": [
            {
                "segment_id": "a", "fullname": "Main St", "rttyp": "U", "mtfcc": "S1200",
                "length_km": 1.0, "fatal_crashes": 5.0, "hin_intensity": 5.0, "hin_rank": 1,
                "center_lat": 34.0, "center_lon": -118.2,
            },
        ],
        "systemic_locations": [
            {
                "segment_id": "c", "fullname": "New Rd", "rttyp": "I", "mtfcc": "S1100",
                "length_km": 2.0, "systemic_score": 0.95, "systemic_rate": 0.6,
                "systemic_expected_crashes": 1.2, "center_lat": 34.2, "center_lon": -118.4,
            },
        ],
        "methodology": {"high_injury_network": "Ranked by weighted crashes per km."},
        "data_sources": [{"name": "FARS", "publisher": "NHTSA", "use": "fatal crash counts"}],
    }
    report.update(overrides)
    return report


def test_render_report_is_full_self_contained_document():
    out = gh.render_report(_report())
    assert out.startswith("<!doctype html>")
    assert out.rstrip().endswith("</html>")
    assert "<style>" in out  # CSS is inlined
    # Self-contained: no external assets or network references.
    assert "http://" not in out and "https://" not in out
    assert "<link" not in out
    assert 'src=' not in out


def test_render_report_shows_jurisdiction_and_title():
    out = gh.render_report(_report())
    assert "<title>Roadway Safety Analysis — Los Angeles County</title>" in out
    assert "Los Angeles County" in out
    assert "GEOID 06037" in out


def test_render_report_escapes_untrusted_names():
    out = gh.render_report(
        _report(jurisdiction={"geoid": "06037", "name": "<script>alert(1)</script>", "level": "county"})
    )
    # The injected markup must not appear unescaped anywhere in the document.
    assert "<script>" not in out
    assert "&lt;script&gt;alert(1)&lt;/script&gt;" in out


def test_render_report_escapes_table_cells():
    report = _report()
    report["hin_corridors"][0]["fullname"] = "A & B <b>Rd</b>"
    out = gh.render_report(report)
    assert "<b>Rd</b>" not in out
    assert "A &amp; B &lt;b&gt;Rd&lt;/b&gt;" in out


def test_render_report_includes_corridor_and_systemic_rows():
    out = gh.render_report(_report())
    assert "Main St" in out  # HIN corridor
    assert "New Rd" in out  # systemic location
    assert "U / S1200" in out  # route/class label


def test_render_report_omits_benefit_cost_when_absent():
    assert "Benefit-cost" not in gh.render_report(_report())


def test_render_report_includes_benefit_cost_when_present():
    out = gh.render_report(
        _report(
            benefit_cost={
                "annual_benefit": 100000.0, "present_value_benefit": 1400000.0,
                "treatment_cost": 500000.0, "net_benefit": 900000.0,
                "benefit_cost_ratio": 2.8, "cost_basis": "FHWA-SA-17-071",
            }
        )
    )
    assert "Benefit-cost" in out
    assert "2.80" in out
    assert "FHWA-SA-17-071" in out


def test_render_report_handles_empty_sections():
    sparse = {
        "jurisdiction": {"geoid": "48", "name": "Texas", "level": "state"},
        "crash_summary": {},
        "high_injury_network": {},
        "hin_corridors": [],
        "systemic_locations": [],
        "methodology": {},
        "data_sources": [],
    }
    out = gh.render_report(sparse)  # must not raise
    assert out.startswith("<!doctype html>")
    assert "No High Injury Network corridors." in out
    assert "No systemic risk locations." in out
    assert "No High Injury Network was computed" in out


def test_bar_chart_svg_scales_tallest_bar_to_full_height():
    svg = gh.bar_chart_svg([("2022", 10), ("2023", 20)], width=520, height=200, pad=32)
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    heights = [float(h) for h in re.findall(r'<rect[^>]*height="([\d.]+)"', svg)]
    assert len(heights) == 2
    plot_h = 200 - 2 * 32
    # Tallest value (20) fills the plot area; half value is half height.
    assert heights[1] == plot_h
    assert abs(heights[0] - plot_h / 2) < 0.5


def test_bar_chart_svg_empty_is_valid_no_divide_by_zero():
    svg = gh.bar_chart_svg([])
    assert svg.startswith("<svg") and svg.endswith("</svg>")
    assert "No data" in svg
    assert "<rect" not in svg


def test_bar_chart_svg_all_zero_does_not_crash():
    svg = gh.bar_chart_svg([("2022", 0), ("2023", 0)])
    assert svg.startswith("<svg")
    heights = [float(h) for h in re.findall(r'<rect[^>]*height="([\d.]+)"', svg)]
    assert heights == [0.0, 0.0]


def test_bar_chart_svg_skips_non_numeric_values():
    svg = gh.bar_chart_svg([("2022", 10), ("2023", None), ("2024", "x"), ("2025", 20)])
    heights = re.findall(r'<rect[^>]*height="([\d.]+)"', svg)
    assert len(heights) == 2  # only the two numeric bars survive


def test_render_report_tolerates_malformed_sections():
    # A dropped-in report where every section is the wrong container type (or a
    # list holds non-dict rows) must degrade, never raise -> the /grants/report
    # HTML path can never 500 on operator data.
    malformed = {
        "jurisdiction": [1, 2, 3],
        "data_vintage": "nope",
        "crash_summary": [1, 2],
        "high_injury_network": "bad",
        "hin_corridors": {"not": "a list"},
        "systemic_locations": [None, "x", {"fullname": "OK Rd"}],
        "data_sources": [42, {"name": "FARS", "publisher": "NHTSA", "use": "crashes"}],
        "methodology": ["nope"],
        "benefit_cost": [1, 2, 3],
    }
    out = gh.render_report(malformed)
    assert out.startswith("<!doctype html>")
    assert out.rstrip().endswith("</html>")
    assert "OK Rd" in out  # the one well-formed systemic row survives
    assert "FARS" in out  # the one well-formed data source survives
    assert "Benefit-cost" not in out  # non-dict benefit_cost -> section omitted


def test_render_report_tolerates_bad_by_year_values_and_keys():
    report = {
        "jurisdiction": {"geoid": "06037", "name": "LA", "level": "county"},
        "crash_summary": {"by_year": {"2022": None, "2023": "many", "bad-key": 3, "2021": 5}},
    }
    out = gh.render_report(report)  # must not raise (bad values/keys tolerated)
    assert out.startswith("<!doctype html>")
    assert out.rstrip().endswith("</html>")


def test_render_report_tolerates_non_dict_top_level():
    assert gh.render_report(None).startswith("<!doctype html>")
    assert gh.render_report([1, 2, 3]).startswith("<!doctype html>")
