/* Public impact dashboard page.

   Fetches the screening rollup from /v1/impact/summary and renders the three
   public-good headline numbers: crashes averted, investment targeted, and the
   equity split. Dependency-free; talks only to the public /v1 API. */
(() => {
  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const num = (value) => (typeof value === "number" && isFinite(value) ? value : null);

  const fmtCount = (value) => {
    const n = num(value);
    if (n === null) return "–";
    // A fraction of a crash reads better rounded to one decimal.
    return n >= 10 ? Math.round(n).toLocaleString() : n.toFixed(1);
  };

  const fmtMoney = (value) => {
    const n = num(value);
    if (n === null) return "–";
    const abs = Math.abs(n);
    if (abs >= 1e9) return `$${(n / 1e9).toFixed(1)}B`;
    if (abs >= 1e6) return `$${(n / 1e6).toFixed(1)}M`;
    if (abs >= 1e3) return `$${(n / 1e3).toFixed(0)}K`;
    return `$${Math.round(n).toLocaleString()}`;
  };

  const fmtPct = (value) => {
    const n = num(value);
    return n === null ? "–" : `${Math.round(n * 100)}%`;
  };

  const fmtRatio = (value) => {
    const n = num(value);
    return n === null ? "–" : `${n.toLocaleString(undefined, { maximumFractionDigits: 1 })}:1`;
  };

  const bootstrap = async () => {
    const root = document.getElementById("impact-shell");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    let cfg = {};
    try {
      cfg = JSON.parse(root.dataset.config || "{}");
    } catch (err) {
      console.error("impact: bad config", err);
    }
    const api = (cfg.api_base || "/v1").replace(/\/$/, "");

    const statusNode = root.querySelector("#impact-status");
    const setStatus = (msg, isError) => {
      statusNode.textContent = msg || "";
      statusNode.classList.toggle("error", Boolean(isError));
    };

    const setText = (id, text) => {
      const node = root.querySelector(id);
      if (node) node.textContent = text;
    };

    setStatus("Loading impact summary…");
    let data;
    try {
      const resp = await fetch(`${api}/impact/summary`, { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      data = await resp.json();
    } catch (err) {
      setStatus("Could not load the impact summary. Please try again later.", true);
      console.error("impact: fetch failed", err);
      return;
    }
    setStatus("");

    const prevention = data.prevention || {};
    const equity = data.equity || null;
    const coverage = data.coverage || {};

    setText("#impact-headline", data.headline || "");

    // Crashes averted.
    setText("#impact-crashes", `${fmtCount(prevention.annual_crashes_averted)}/yr`);
    setText(
      "#impact-crashes-sub",
      `${fmtCount(prevention.crashes_averted_service_life)} over a ${
        prevention.service_life_years ?? "–"
      }-year service life, across ${prevention.segments_with_treatment ?? 0} treated segments.`
    );

    // Investment targeted.
    setText("#impact-investment", fmtMoney(prevention.investment_required));
    const bcr = prevention.benefit_cost_ratio;
    setText(
      "#impact-investment-sub",
      `${fmtMoney(prevention.present_value_benefit)} in avoided-crash benefit — a ${fmtRatio(
        bcr
      )} benefit-cost.`
    );

    // Equity.
    if (equity) {
      setText("#impact-equity", fmtPct(equity.disadvantaged_crash_share));
      setText(
        "#impact-equity-sub",
        `of crashes are in disadvantaged communities; ~${fmtCount(
          equity.estimated_disadvantaged_crashes_averted_annual
        )}/yr of the avertable crashes fall there.`
      );
    } else {
      setText("#impact-equity", "–");
      setText("#impact-equity-sub", "No equity overlay loaded for this deployment.");
    }

    // Detail breakdown.
    const rows = [
      ["Annual crashes averted", `${fmtCount(prevention.annual_crashes_averted)}/yr`],
      ["Crashes averted over service life", fmtCount(prevention.crashes_averted_service_life)],
      ["Investment required", fmtMoney(prevention.investment_required)],
      ["Present-value benefit", fmtMoney(prevention.present_value_benefit)],
      ["Benefit-cost ratio", fmtRatio(prevention.benefit_cost_ratio)],
    ];
    if (equity) {
      rows.push(["Disadvantaged crash share", fmtPct(equity.disadvantaged_crash_share)]);
      if (num(equity.crash_disparity_ratio) !== null) {
        rows.push(["Crash disparity ratio", fmtRatio(equity.crash_disparity_ratio)]);
      }
    }
    const breakdown = root.querySelector("#impact-breakdown");
    if (breakdown) {
      breakdown.innerHTML = rows
        .map(([label, value]) => `<div><dt>${esc(label)}</dt><dd>${esc(value)}</dd></div>`)
        .join("");
    }

    // Coverage footer.
    const coverageParts = [
      ["HIN segments", coverage.hin_segments],
      ["Rolled up", coverage.segments_rolled_up],
      ["Equity segments", coverage.equity_segments],
      ["Grant jurisdictions", coverage.grant_jurisdictions],
    ].filter(([, v]) => num(v) !== null);
    const coverageNode = root.querySelector("#impact-coverage");
    if (coverageNode) {
      coverageNode.innerHTML = coverageParts
        .map(([label, v]) => `<span>${esc(label)}: <strong>${esc(v.toLocaleString())}</strong></span>`)
        .join("");
    }

    // Caveats.
    const caveatList = root.querySelector("#impact-caveat-list");
    if (caveatList && Array.isArray(data.caveats)) {
      caveatList.innerHTML = data.caveats.map((c) => `<li>${esc(c)}</li>`).join("");
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
