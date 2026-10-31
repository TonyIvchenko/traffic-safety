/* Citizen-facing Road Risk Advisory & Alerts page.

   Shows how unusual driving risk is right now for each metro (relative to that
   metro's own typical week), lets a visitor drill into a live regional advisory,
   and lets them subscribe to alerts by creating a watch. Dependency-free; talks
   only to the public /v1 API. */
(() => {
  const LEVEL_COLORS = {
    Low: "#1a9850",
    Moderate: "#a6d96a",
    Elevated: "#fdae61",
    High: "#f46d43",
    Extreme: "#d7191c",
  };
  // Approximate CONUS extent used to project metro centroids into the SVG.
  const BBOX = { west: -125.0, east: -66.5, south: 24.0, north: 49.5 };
  const VIEW = { w: 960, h: 600, pad: 20 };

  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const project = (lon, lat) => {
    const x =
      VIEW.pad +
      ((lon - BBOX.west) / (BBOX.east - BBOX.west)) * (VIEW.w - 2 * VIEW.pad);
    const y =
      VIEW.pad +
      ((BBOX.north - lat) / (BBOX.north - BBOX.south)) * (VIEW.h - 2 * VIEW.pad);
    return [x, y];
  };

  const centroid = (bbox) => {
    // bbox is [min_lat, max_lat, min_lon, max_lon].
    const [minLat, maxLat, minLon, maxLon] = bbox;
    return { lat: (minLat + maxLat) / 2, lon: (minLon + maxLon) / 2 };
  };

  const colorFor = (advisory) =>
    (advisory && advisory.color) || LEVEL_COLORS[advisory && advisory.level] || "#8899aa";

  const bootstrap = () => {
    const root = document.getElementById("alerts-shell");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    let cfg = {};
    try {
      cfg = JSON.parse(root.dataset.config || "{}");
    } catch (err) {
      console.error("alerts: bad config", err);
    }
    const api = (cfg.api_base || "/v1").replace(/\/$/, "");

    const svg = root.querySelector("#alerts-map");
    const rankList = root.querySelector("#alerts-ranklist");
    const detail = root.querySelector("#alerts-detail");
    const subscribe = root.querySelector("#alerts-subscribe");
    const frameLabel = root.querySelector("#alerts-frame");
    const statusNode = root.querySelector("#alerts-status");
    const locateBtn = root.querySelector("#alerts-locate");
    const form = root.querySelector("#watch-form");
    const watchResult = root.querySelector("#watch-result");
    const channelSelect = form.querySelector('[name="channel"]');
    const destField = form.querySelector("#watch-dest-field");
    const destInput = form.querySelector('[name="destination"]');
    const destLabel = form.querySelector("#watch-dest-label");
    const regionNameNode = form.querySelector("#watch-region-name");

    const state = { regions: [], byId: new Map(), selected: null };

    const setStatus = (msg) => {
      statusNode.textContent = msg || "";
    };

    const nowFrame = () => {
      const d = new Date();
      return {
        day_of_week: ((d.getDay() + 6) % 7) + 1, // JS Sun=0 -> Mon=1..Sun=7
        hour: d.getHours(),
        month: d.getMonth() + 1,
      };
    };

    const renderMap = () => {
      const dots = state.regions
        .map((r) => {
          const c = centroid(r.bbox);
          const [x, y] = project(c.lon, c.lat);
          const color = colorFor(r.advisory);
          const sel = state.selected === r.region_id ? " selected" : "";
          return `<circle class="metro-dot${sel}" data-id="${esc(r.region_id)}"
            cx="${x.toFixed(1)}" cy="${y.toFixed(1)}" r="8" fill="${esc(color)}">
            <title>${esc(r.region_name)} — ${esc((r.advisory || {}).level || "")}</title>
          </circle>`;
        })
        .join("");
      svg.innerHTML =
        `<rect class="state-outline" x="1" y="1" width="${VIEW.w - 2}" height="${VIEW.h - 2}" rx="18"/>` +
        dots;
      svg.querySelectorAll(".metro-dot").forEach((node) => {
        node.addEventListener("click", () => selectRegion(node.dataset.id));
      });
    };

    const renderRankList = () => {
      rankList.innerHTML = state.regions
        .map((r) => {
          const adv = r.advisory || {};
          const sel = state.selected === r.region_id ? " class=\"selected\"" : "";
          const pct =
            adv.percentile != null ? ` · ${Math.round(adv.percentile * 100)}th pct` : "";
          return `<li data-id="${esc(r.region_id)}"${sel}>
            <span class="rank-dot" style="background:${esc(colorFor(adv))}"></span>
            <span class="rank-name">${esc(r.region_name)}</span>
            <span class="rank-level">${esc(adv.level || "")}${pct}</span>
          </li>`;
        })
        .join("");
      rankList.querySelectorAll("li").forEach((node) => {
        node.addEventListener("click", () => selectRegion(node.dataset.id));
      });
    };

    const renderDetail = (region, live) => {
      const adv = (live && live.advisory) || region.advisory || {};
      const color = colorFor(adv);
      const mode = live ? "live conditions" : "typical-week estimate";
      let compareHtml = "";
      if (live && live.compared_to_normal) {
        compareHtml = `<div class="detail-compare">Right now: ${esc(
          live.compared_to_normal.comparison
        )}</div>`;
      }
      const caveats = (live && live.caveats) || region.caveats || [];
      const caveatHtml = caveats.length
        ? `<ul class="caveats">${caveats.map((c) => `<li>${esc(c)}</li>`).join("")}</ul>`
        : "";
      detail.innerHTML = `
        <div class="detail-head">
          <span class="level-chip" style="background:${esc(color)}">${esc(adv.level || "—")}</span>
          <h3>${esc(region.region_name)}</h3>
        </div>
        <p class="detail-message">${esc(adv.message || "No advisory available.")}</p>
        ${compareHtml}
        <div class="detail-sub">Basis: ${esc(adv.basis || "—")} · ${esc(mode)}</div>
        ${caveatHtml}`;
      detail.hidden = false;
    };

    const openSubscribe = (region) => {
      regionNameNode.textContent = region.region_name;
      form.dataset.regionId = region.region_id;
      watchResult.innerHTML = "";
      subscribe.hidden = false;
    };

    const selectRegion = async (regionId) => {
      const region = state.byId.get(regionId);
      if (!region) return;
      state.selected = regionId;
      renderMap();
      renderRankList();
      openSubscribe(region);
      renderDetail(region, null); // show the climatology estimate immediately
      setStatus(`Loading live conditions for ${region.region_name}…`);
      try {
        const resp = await fetch(
          `${api}/advisory/region?region=${encodeURIComponent(regionId)}&mode=live&compare=true`,
          { headers: { Accept: "application/json" } }
        );
        if (resp.ok) {
          renderDetail(region, await resp.json());
          setStatus("");
        } else {
          // 503 = no live weather for the metro; the climatology estimate stands.
          setStatus("Live weather is unavailable for this metro; showing the typical-week estimate.");
        }
      } catch (err) {
        setStatus("Could not reach the live advisory; showing the typical-week estimate.");
      }
    };

    const loadNational = async () => {
      const { day_of_week, hour, month } = nowFrame();
      setStatus("Loading the national advisory…");
      try {
        const resp = await fetch(
          `${api}/advisory/national?day_of_week=${day_of_week}&hour=${hour}&month=${month}`,
          { headers: { Accept: "application/json" } }
        );
        if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
        const data = await resp.json();
        state.regions = Array.isArray(data.regions) ? data.regions : [];
        state.byId = new Map(state.regions.map((r) => [r.region_id, r]));
        if (frameLabel && data.frame_label) {
          frameLabel.textContent = `Reference frame: ${data.frame_label} (relative to each metro's normal week)`;
        }
        renderMap();
        renderRankList();
        setStatus(
          state.regions.length
            ? "Pick a metro on the map or list for a live reading and to set an alert."
            : "No metro advisories are available right now."
        );
      } catch (err) {
        setStatus("Could not load the national advisory. Please try again shortly.");
      }
    };

    const pickNearestMetro = (lat, lon) => {
      let best = null;
      let bestD = Infinity;
      for (const r of state.regions) {
        const [minLat, maxLat, minLon, maxLon] = r.bbox;
        if (lat >= minLat && lat <= maxLat && lon >= minLon && lon <= maxLon) return r;
        const c = centroid(r.bbox);
        const d = (c.lat - lat) ** 2 + (c.lon - lon) ** 2;
        if (d < bestD) {
          bestD = d;
          best = r;
        }
      }
      return best;
    };

    if (locateBtn) {
      locateBtn.addEventListener("click", () => {
        if (!navigator.geolocation) {
          setStatus("Location is not available in this browser.");
          return;
        }
        setStatus("Finding your location…");
        navigator.geolocation.getCurrentPosition(
          (pos) => {
            const metro = pickNearestMetro(pos.coords.latitude, pos.coords.longitude);
            if (metro) selectRegion(metro.region_id);
            else setStatus("No covered metro is near your location.");
          },
          () => setStatus("Could not get your location.")
        );
      });
    }

    const syncDestField = () => {
      const channel = channelSelect.value;
      if (channel === "email") {
        destField.hidden = false;
        destLabel.textContent = "Your email address";
        destInput.type = "email";
        destInput.placeholder = "you@example.com";
      } else if (channel === "webhook") {
        destField.hidden = false;
        destLabel.textContent = "Webhook URL";
        destInput.type = "url";
        destInput.placeholder = "https://example.com/hook";
      } else {
        destField.hidden = true; // poll: no destination
      }
    };
    channelSelect.addEventListener("change", syncDestField);
    syncDestField();

    form.addEventListener("submit", async (event) => {
      event.preventDefault();
      const regionId = form.dataset.regionId;
      if (!regionId) return;
      const channel = channelSelect.value;
      const body = {
        kind: "region",
        params: {
          region_id: regionId,
          horizon_hours: Number(form.querySelector('[name="horizon_hours"]').value || 0),
        },
        threshold_level: form.querySelector('[name="threshold_level"]').value,
        channel,
      };
      if (channel === "email") body.email = destInput.value.trim();
      if (channel === "webhook") body.webhook_url = destInput.value.trim();

      const button = form.querySelector('button[type="submit"]');
      button.disabled = true;
      watchResult.innerHTML = "";
      try {
        const resp = await fetch(`${api}/watches`, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify(body),
        });
        const data = await resp.json();
        if (resp.ok) {
          const secretLine = data.webhook_secret
            ? `<div>Webhook signing secret: <code>${esc(data.webhook_secret)}</code></div>`
            : "";
          watchResult.innerHTML = `<div class="ok">
            <div><strong>Alert created.</strong> Save these — the token is shown only once and is the only way to manage or delete this alert.</div>
            <div>Watch ID: <code>${esc(data.id)}</code></div>
            <div>Token: <code>${esc(data.token)}</code></div>
            ${secretLine}
            <div>Check status: <code>GET /v1/watches/${esc(data.id)}?token=…</code> · Delete: <code>DELETE</code> the same URL.</div>
          </div>`;
        } else {
          const msg = typeof data.detail === "string" ? data.detail : JSON.stringify(data.detail);
          watchResult.innerHTML = `<div class="err">Could not create the alert: ${esc(msg)}</div>`;
        }
      } catch (err) {
        watchResult.innerHTML = `<div class="err">Could not reach the API. Please try again.</div>`;
      } finally {
        button.disabled = false;
      }
    });

    loadNational();
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
