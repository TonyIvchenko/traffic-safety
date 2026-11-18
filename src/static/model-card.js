/* Model card page.

   Fetches /v1/model/card and renders it as a readable transparency page.
   Dependency-free; the card is composed live from the deployed model, so this
   page always reflects what is actually serving requests. Degrades gracefully
   when the trained model is not loaded (editorial sections still render). */
(() => {
  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const num = (value) => (typeof value === "number" && isFinite(value) ? value : null);

  const dash = "–";

  const fmtInt = (value) => {
    const n = num(value);
    return n === null ? "—" : Math.round(n).toLocaleString();
  };

  const fmtNum = (value) => {
    const n = num(value);
    return n === null ? "—" : n.toLocaleString(undefined, { maximumFractionDigits: 4 });
  };

  const fmtWindow = (window) => {
    if (!window) return "—";
    const start = num(window.start);
    const end = num(window.end);
    if (start === null) return "—";
    return end === null || end === start ? String(start) : `${start}${dash}${end}`;
  };

  const dlRows = (rows) =>
    rows
      .filter(([, v]) => v !== null && v !== undefined && v !== "")
      .map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`)
      .join("");

  const listItems = (values) =>
    (Array.isArray(values) ? values : []).map((v) => `<li>${esc(v)}</li>`).join("");

  const bootstrap = async () => {
    const root = document.getElementById("card-shell");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    let cfg = {};
    try {
      cfg = JSON.parse(root.dataset.config || "{}");
    } catch (err) {
      console.error("model-card: bad config", err);
    }
    const api = (cfg.api_base || "/v1").replace(/\/$/, "");

    const statusNode = root.querySelector("#card-status");
    const setStatus = (msg, isError) => {
      if (!statusNode) return;
      statusNode.textContent = msg || "";
      statusNode.classList.toggle("error", Boolean(isError));
    };
    const fill = (id, htmlText) => {
      const node = root.querySelector(id);
      if (node) node.innerHTML = htmlText;
    };
    const setText = (id, text) => {
      const node = root.querySelector(id);
      if (node) node.textContent = text;
    };

    setStatus("Loading model card…");
    let card;
    try {
      const resp = await fetch(`${api}/model/card`, { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      card = await resp.json();
    } catch (err) {
      setStatus("Could not load the model card. Please try again later.", true);
      console.error("model-card: fetch failed", err);
      return;
    }
    setStatus("");

    const details = card.model_details || {};
    const spatial = details.spatial_unit || {};
    const temporal = details.temporal_unit || {};
    const features = details.features || {};
    const metrics = card.metrics || {};
    const perf = metrics.performance || {};
    const use = card.intended_use || {};
    const training = card.training_data || {};
    const evaluation = card.evaluation_data || {};
    const qa = card.quantitative_analyses || {};

    // Subtitle.
    const version = details.version ? `v${esc(details.version)}` : "";
    const ready = details.ready ? "" : " — trained model not loaded on this instance";
    setText(
      "#card-subtitle",
      `${details.name || "Crash-risk model"} ${details.version ? version : ""}${ready}`
    );

    // Model details.
    const cov = details.geographic_coverage || {};
    const covText =
      num(cov.lat_min) !== null && num(cov.lon_min) !== null
        ? `lat ${fmtNum(cov.lat_min)}${dash}${fmtNum(cov.lat_max)}, lon ${fmtNum(cov.lon_min)}${dash}${fmtNum(cov.lon_max)}`
        : null;
    const featCount = num(features.count);
    const weatherCount = Array.isArray(features.weather_features) ? features.weather_features.length : 0;
    const featText =
      featCount === null
        ? null
        : `${fmtInt(featCount)}${weatherCount ? ` (incl. ${weatherCount} weather)` : ""}`;
    const spatialText =
      num(spatial.resolution) !== null
        ? `H3 resolution ${fmtInt(spatial.resolution)}${
            num(spatial.cells_modeled) !== null ? `, ${fmtInt(spatial.cells_modeled)} cells` : ""
          }`
        : null;
    const temporalText =
      num(temporal.frames) !== null ? `${fmtInt(temporal.frames)} hourly frames (weekly cycle)` : null;

    fill(
      "#card-details",
      dlRows([
        ["Type", details.type ? esc(details.type) : null],
        ["Objective", details.objective ? esc(details.objective) : null],
        ["Version", details.version ? esc(details.version) : null],
        ["Spatial unit", spatialText],
        ["Temporal unit", temporalText],
        ["Features", featText],
        ["Training window", fmtWindow(details.training_window)],
        ["Evaluation window", fmtWindow(details.evaluation_window)],
        [
          "Negative sampling",
          num(details.negative_sampling_ratio) !== null
            ? `${fmtInt(details.negative_sampling_ratio)}:1 non-crash:crash`
            : null,
        ],
        ["License", details.license ? esc(details.license) : null],
        ["Coverage", covText],
      ])
    );

    // Performance.
    const metricDefs = [
      ["ROC-AUC", perf.roc_auc],
      ["Average precision", perf.average_precision],
      ["Brier score", perf.brier_score],
    ].filter(([, v]) => num(v) !== null);
    if (metricDefs.length) {
      fill(
        "#card-metrics",
        metricDefs
          .map(
            ([label, value]) =>
              `<div class="card-metric"><div class="card-metric-label">${esc(label)}</div>` +
              `<div class="card-metric-value">${fmtNum(value)}</div></div>`
          )
          .join("")
      );
    } else {
      fill(
        "#card-metrics",
        `<p class="card-fineprint">Performance metrics are reported when the trained model is loaded on this instance.</p>`
      );
    }

    const thresholds = metrics.decision_thresholds || {};
    const thresholdPills = ["moderate", "high", "extreme"]
      .filter((level) => num(thresholds[level]) !== null)
      .map(
        (level) =>
          `<span class="card-threshold">${esc(level)} <strong>&ge; ${fmtNum(thresholds[level])}</strong></span>`
      )
      .join("");
    fill(
      "#card-thresholds",
      thresholdPills ? `<span class="card-threshold-label">Risk-level cutoffs:</span> ${thresholdPills}` : ""
    );
    setText("#card-metrics-note", metrics.notes || "");

    // Intended use.
    fill("#card-uses", listItems(use.primary_uses));
    fill("#card-users", listItems(use.primary_users));
    fill("#card-out-of-scope", listItems(use.out_of_scope_uses));

    // Data.
    fill(
      "#card-data",
      dlRows([
        ["Training window", fmtWindow(training.window)],
        ["Evaluation window", fmtWindow(evaluation.window)],
        [
          "Calibration report",
          qa.calibration_report_available
            ? `available at <a href="${esc(api)}/model/report">${esc(qa.calibration_report_endpoint || "/v1/model/report")}</a>`
            : "not published on this instance",
        ],
      ])
    );
    const renderSource = (s) => {
      const id = esc(s.id || "source");
      const label = s.url ? `<a href="${esc(s.url)}">${id}</a>` : id;
      const bits = [s.category, s.coverage].filter(Boolean).map(esc).join(", ");
      return `<li>${label}${bits ? ` <span class="card-muted">(${bits})</span>` : ""}</li>`;
    };
    const sources = Array.isArray(training.sources) ? training.sources : [];
    const liveSources = Array.isArray(evaluation.live_inference_sources)
      ? evaluation.live_inference_sources
      : [];
    let sourcesHtml = "";
    if (sources.length) {
      sourcesHtml += `<li><strong>Training &amp; history:</strong></li>` + sources.map(renderSource).join("");
    }
    if (liveSources.length) {
      sourcesHtml += `<li><strong>Live inference:</strong></li>` + liveSources.map(renderSource).join("");
    }
    fill("#card-sources", sourcesHtml);

    // Ethics + caveats.
    fill("#card-ethics", listItems(card.ethical_considerations));
    fill("#card-caveats", listItems(card.caveats_and_recommendations));

    // Footer.
    const generated = card.generated_at ? String(card.generated_at).slice(0, 10) : "";
    const parts = [];
    parts.push(`Rendered live from <a href="${esc(api)}/model/card">${esc(api)}/model/card</a>`);
    if (generated) parts.push(`generated ${esc(generated)}`);
    if (card.framework_citation) parts.push(esc(card.framework_citation));
    fill("#card-footer", parts.join(" &middot; "));
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
