(() => {
  const escapeHtml = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const utcLabel = (value) => {
    if (!value) return "n/a";
    const date = new Date(value);
    if (Number.isNaN(date.getTime())) return String(value);
    return date.toISOString().replace(".000Z", " UTC").replace("T", " ");
  };

  const lerp = (a, b, t) => Math.round(a + (b - a) * t);
  const clamp = (value, min, max) => Math.max(min, Math.min(max, value));
  const mix = (a, b, t) => a + (b - a) * t;

  const pointSegmentDistance = (px, py, ax, ay, bx, by) => {
    const dx = bx - ax;
    const dy = by - ay;
    const lengthSq = dx * dx + dy * dy;
    if (lengthSq <= 1e-6) {
      return Math.hypot(px - ax, py - ay);
    }
    const t = Math.max(0, Math.min(1, ((px - ax) * dx + (py - ay) * dy) / lengthSq));
    const projX = ax + dx * t;
    const projY = ay + dy * t;
    return Math.hypot(px - projX, py - projY);
  };

  const estimatePathLength = (path) => {
    let total = 0;
    for (let idx = 1; idx < path.length; idx += 1) {
      const [ax, ay] = path[idx - 1];
      const [bx, by] = path[idx];
      total += Math.hypot(bx - ax, by - ay);
    }
    return total;
  };

  const bootstrap = async () => {
    const root = document.getElementById("risk-map-shell");
    if (!root || root.dataset.ready === "1") return [];
    root.dataset.ready = "1";

    let cfg = {};
    try {
      cfg = JSON.parse(root.dataset.config || "{}");
    } catch (err) {
      console.error("Failed to parse map config", err);
    }

    const slider = root.querySelector("#risk-time-slider");
    const playBtn = root.querySelector("#risk-play");
    const progressNode = root.querySelector("#risk-time-progress");
    const markerNode = root.querySelector("#risk-now-marker");
    const mapNode = root.querySelector("#risk-map");
    const statusNode = root.querySelector("#risk-map-status");
    const timelineTicksNode = root.querySelector("#risk-timeline-ticks");
    const timelinePhasesNode = root.querySelector("#risk-timeline-phases");
    const timelineTrackNode = root.querySelector("#risk-timeline-track");
    const frameLabelNode = root.querySelector("#risk-frame-label");
    const frameChipNode = root.querySelector("#ops-frame-chip");
    const updatedChipNode = root.querySelector("#ops-updated-chip");
    const freshnessNode = root.querySelector("#ops-freshness-text");
    const providerNode = root.querySelector("#ops-provider-text");
    const layerRiskToggle = root.querySelector("#layer-risk");
    const layerWeatherToggle = root.querySelector("#layer-weather");
    const weatherModeSelect = root.querySelector("#weather-mode");
    const layerRoadsToggle = root.querySelector("#layer-roads");
    const zoomHintNode = root.querySelector("#risk-zoom-hint");

    const updateStatus = (text, isError = false) => {
      if (!text) {
        statusNode.textContent = "";
        statusNode.classList.remove("show", "error");
        return;
      }
      statusNode.textContent = text;
      statusNode.classList.add("show");
      statusNode.classList.toggle("error", Boolean(isError));
    };

    if (cfg.road_mode) {
      try {
        const response = await fetch("/segment-tiles/meta", { cache: "no-store" });
        if (!response.ok) {
          throw new Error(`Tile metadata request failed: ${response.status}`);
        }
        const meta = await response.json();
        cfg.frames = Array.isArray(meta.frame_labels) && meta.frame_labels.length ? meta.frame_labels : cfg.frames;
        cfg.tile_zoom_min = Number(meta.tile_zoom_min || cfg.tile_zoom_min || 4);
        cfg.tile_zoom_max = Number(meta.tile_zoom_max || cfg.tile_zoom_max || 11);
        cfg.raster_zoom_min = Number(meta.raster_zoom_min || cfg.raster_zoom_min || 4);
        cfg.raster_zoom_max = Number(meta.raster_zoom_max || cfg.raster_zoom_max || 8);
        cfg.vector_zoom_min = Number(meta.vector_zoom_min || cfg.vector_zoom_min || 9);
        cfg.zoom_min = cfg.tile_zoom_min;
        cfg.road_tile_revision = meta.run_id || cfg.road_tile_revision || "";
        cfg.generated_at_utc = meta.generated_at_utc || cfg.generated_at_utc || "";
        cfg.generated_at_label = meta.generated_at_utc || cfg.generated_at_label || "";
        cfg.forecast_start_utc = meta.forecast_start_utc || cfg.forecast_start_utc || "";
        cfg.forecast_end_utc = meta.forecast_end_utc || cfg.forecast_end_utc || "";
        cfg.provider_label = String(meta.provider || cfg.provider_label || "nws").toUpperCase();
      } catch (err) {
        updateStatus(err.message || "Failed to load road tile metadata.", true);
      }
    }

    const frames = Array.isArray(cfg.frames) && cfg.frames.length > 0 ? cfg.frames : ["+0h"];
    const frameCount = frames.length;
    const maxFrameIdx = Math.max(1, frameCount - 1);
    slider.step = "0.001";
    slider.max = String(Math.max(0, frameCount - 1));
    slider.value = String(Math.max(0, Math.min(frameCount - 1, Number(cfg.default_frame_idx || 0))));
    let currentFrameValue = Math.min(maxFrameIdx, Math.max(0, Number(slider.value) || 0));

    const renderFreshness = () => {
      const updated = utcLabel(cfg.generated_at_label || cfg.generated_at_utc || "");
      updatedChipNode.textContent = updated;
      freshnessNode.textContent = updated;
      providerNode.textContent = `${cfg.provider_label || "NWS"} blended weather`;
    };

    if (layerWeatherToggle && !cfg.weather_overlay_ready) {
      layerWeatherToggle.checked = false;
      layerWeatherToggle.disabled = true;
      layerWeatherToggle.title = "Weather overlay will appear after the forecast refresh writes weather assets.";
      if (weatherModeSelect) weatherModeSelect.disabled = true;
    }

    const renderTimelineScaffold = () => {
      const timeline = cfg.timeline || {
        step_pct: 100.0 / Math.max(1, frameCount - 1),
        ticks: [0, 6, 12, 18, Math.max(0, frameCount - 1)]
          .filter((value, idx, arr) => arr.indexOf(value) === idx)
          .map((value) => ({ label: `+${value}h`, frame_idx: value })),
        phases: [{ kind: "live", label: "Forecast next 24 hours", count: Math.max(1, frameCount) }],
      };

      const stepPct = Number(timeline.step_pct || 1);
      timelineTrackNode.style.setProperty("--frame-step", `${stepPct}%`);

      const ticks = Array.isArray(timeline.ticks) ? timeline.ticks : [];
      timelineTicksNode.innerHTML = ticks
        .map((tick) => {
          const frameIdx = Number(tick.frame_idx || 0);
          const left = (frameIdx / maxFrameIdx) * 100.0;
          return `<div class="year-tick" data-frame-index="${frameIdx}" style="left:${left.toFixed(6)}%"><span>${escapeHtml(tick.label || "")}</span></div>`;
        })
        .join("");

      timelinePhasesNode.innerHTML = "";
      timelinePhasesNode.style.background = "var(--track-live)";
    };

    const getFrameBlend = () => {
      const value = clamp(Number(currentFrameValue) || 0, 0, maxFrameIdx);
      const baseFrameIdx = Math.floor(value);
      const nextFrameIdx = Math.min(maxFrameIdx, baseFrameIdx + 1);
      const mixValue = nextFrameIdx === baseFrameIdx ? 0 : value - baseFrameIdx;
      return { value, baseFrameIdx, nextFrameIdx, mixValue };
    };

    const formatFrameLabel = (value) => {
      const rounded = Math.round(value);
      if (Math.abs(value - rounded) < 0.001) {
        return frames[rounded] || `+${rounded}h`;
      }
      return `+${value.toFixed(1)}h`;
    };

    const updateTimeline = () => {
      const { value } = getFrameBlend();
      slider.value = String(value);
      const pct = (value / maxFrameIdx) * 100.0;
      progressNode.style.width = `${pct}%`;
      markerNode.style.left = `${pct}%`;
      const label = formatFrameLabel(value);
      frameLabelNode.textContent = label;
      frameChipNode.textContent = label;

      const ticks = Array.from(root.querySelectorAll(".year-tick"));
      let activeTick = -1;
      ticks.forEach((tick, i) => {
        const startIdx = Number(tick.dataset.frameIndex || "0");
        if (value >= startIdx) activeTick = i;
      });
      ticks.forEach((tick, i) => tick.classList.toggle("active", i === activeTick));
    };

    let timer = null;
    let map = null;
    let infoWindow = null;
    let roadOverlay = null;
    let weatherOverlayData = null;
    let weatherOverlayPromise = null;
    let weatherFieldBase = null;
    let weatherFieldNext = null;
    let weatherFieldFrameKey = "";
    let weatherWindOverlay = [];
    let weatherWindFrameKey = "";
    const activeVectorTiles = new Set();
    const activeRasterTiles = new Set();
    const roadTileCache = new Map();
    const segmentDetailCache = new Map();
    const weatherFrameCache = new Map();
    const roadVectorMinZoom = Number(cfg.vector_zoom_min || 9);
    const roadRasterMaxZoom = Number(cfg.raster_zoom_max || 8);
    const frameDurationMs = 900;
    const vectorRepaintMinIntervalMs = 48;
    const weatherOverlayMinIntervalMs = 48;
    let playRaf = 0;
    let playLastTs = 0;
    let viewportTimelineToken = 0;
    let viewportTimelineTimer = 0;
    let vectorRepaintRaf = 0;
    let vectorRepaintLastTs = 0;
    let weatherOverlayRaf = 0;
    let weatherOverlayLastTs = 0;
    let hoverRaf = 0;
    let pendingHoverLatLng = null;

    const runScheduledVectorRepaint = (timestamp = 0) => {
      const now = Number(timestamp || performance.now());
      if (now - vectorRepaintLastTs < vectorRepaintMinIntervalMs) {
        vectorRepaintRaf = requestAnimationFrame(runScheduledVectorRepaint);
        return;
      }
      vectorRepaintRaf = 0;
      vectorRepaintLastTs = now;
      repaintVectorTiles();
    };

    const scheduleVectorRepaint = () => {
      if (vectorRepaintRaf || !activeVectorTiles.size) return;
      vectorRepaintRaf = requestAnimationFrame(runScheduledVectorRepaint);
    };

    const syncFrameVisuals = () => {
      updateTimeline();
      if (map?.getZoom() > roadRasterMaxZoom) {
        scheduleVectorRepaint();
      } else {
        updateRasterTiles();
      }
      scheduleWeatherOverlayUpdate();
    };

    const stopPlaying = () => {
      if (playRaf) {
        cancelAnimationFrame(playRaf);
        playRaf = 0;
      }
      if (vectorRepaintRaf) {
        cancelAnimationFrame(vectorRepaintRaf);
        vectorRepaintRaf = 0;
      }
      if (weatherOverlayRaf) {
        cancelAnimationFrame(weatherOverlayRaf);
        weatherOverlayRaf = 0;
      }
      playLastTs = 0;
      timer = null;
      playBtn.classList.remove("playing");
      playBtn.setAttribute("aria-label", "Play timeline");
    };

    const playTick = (timestamp) => {
      if (!timer) return;
      if (!playLastTs) {
        playLastTs = timestamp;
      }
      const deltaMs = timestamp - playLastTs;
      playLastTs = timestamp;
      currentFrameValue = clamp(currentFrameValue + deltaMs / frameDurationMs, 0, maxFrameIdx);
      syncFrameVisuals();
      if (currentFrameValue >= maxFrameIdx - 1e-6) {
        currentFrameValue = maxFrameIdx;
        syncFrameVisuals();
        stopPlaying();
        return;
      }
      playRaf = requestAnimationFrame(playTick);
    };

    const setPlaying = (on) => {
      if (on) {
        if (timer) return;
        if (currentFrameValue >= maxFrameIdx - 1e-6) {
          currentFrameValue = 0;
          syncFrameVisuals();
        }
        timer = {};
        playBtn.classList.add("playing");
        playBtn.setAttribute("aria-label", "Pause timeline");
        playRaf = requestAnimationFrame(playTick);
        return;
      }
      stopPlaying();
    };

    const wrapTileX = (x, zoom) => {
      const n = 2 ** zoom;
      return ((x % n) + n) % n;
    };

    const normalizeRoadTileRequest = (coord, zoom) => {
      const sourceZoomMax = Number(cfg.tile_zoom_max || 11);
      const sourceZoom = Math.min(zoom, sourceZoomMax);
      if (zoom <= sourceZoomMax) {
        return {
          sourceZoom,
          sourceX: wrapTileX(coord.x, sourceZoom),
          sourceY: coord.y,
          drawScale: 1,
          drawOffsetX: 0,
          drawOffsetY: 0,
        };
      }

      const delta = zoom - sourceZoomMax;
      const scale = 2 ** delta;
      const sourceX = wrapTileX(Math.floor(coord.x / scale), sourceZoom);
      const maxY = 2 ** sourceZoom - 1;
      const sourceY = Math.max(0, Math.min(maxY, Math.floor(coord.y / scale)));
      const childX = ((coord.x % scale) + scale) % scale;
      const childY = ((coord.y % scale) + scale) % scale;
      return {
        sourceZoom,
        sourceX,
        sourceY,
        drawScale: scale,
        drawOffsetX: childX * 256,
        drawOffsetY: childY * 256,
      };
    };

    const decodeBase64Bytes = (value) => Uint8Array.from(atob(value || ""), (ch) => ch.charCodeAt(0));

    const decodePath = (value) => {
      const bytes = decodeBase64Bytes(value);
      const view = new DataView(bytes.buffer, bytes.byteOffset, bytes.byteLength);
      const points = [];
      for (let offset = 0; offset + 3 < bytes.byteLength; offset += 4) {
        points.push([view.getInt16(offset, true) / 8.0, view.getInt16(offset + 2, true) / 8.0]);
      }
      return points;
    };

    const prepareTilePayload = (payload) => {
      const entries = (payload.entries || []).map((entry) => ({
        segmentIdx: Number(entry.s || 0),
        kind: Number(entry.k || 0),
        path: decodePath(entry.p || ""),
        risk: decodeBase64Bytes(entry.r || ""),
      }));
      const kindFrameSums = [
        new Float32Array(frameCount),
        new Float32Array(frameCount),
        new Float32Array(frameCount),
      ];
      const kindWeights = [0, 0, 0];
      entries.forEach((entry) => {
        const kindIdx = clamp(Number(entry.kind || 0), 0, 2);
        const weight = Math.max(1, estimatePathLength(entry.path) / 24);
        kindWeights[kindIdx] += weight;
        for (let frameIdx = 0; frameIdx < frameCount; frameIdx += 1) {
          kindFrameSums[kindIdx][frameIdx] += (entry.risk[frameIdx] ?? 0) * weight;
        }
      });
      return {
        entries,
        summary: {
          kindWeights,
          kindFrameSums: kindFrameSums.map((values) => Array.from(values)),
        },
      };
    };

    const riskStrokeTriplet = (riskByte) => {
      let t = Math.max(0, Math.min(1, Number(riskByte || 0) / 255));
      t = 1 / (1 + Math.exp(-6 * (t - 0.46)));
      const stops = [
        [0.0, [17, 122, 101]],
        [0.28, [72, 187, 120]],
        [0.52, [242, 201, 76]],
        [0.74, [242, 130, 49]],
        [1.0, [203, 43, 39]],
      ];
      for (let i = 1; i < stops.length; i += 1) {
        if (t <= stops[i][0]) {
          const [startT, startColor] = stops[i - 1];
          const [endT, endColor] = stops[i];
          const localT = (t - startT) / Math.max(0.000001, endT - startT);
          return [
            lerp(startColor[0], endColor[0], localT),
            lerp(startColor[1], endColor[1], localT),
            lerp(startColor[2], endColor[2], localT),
          ];
        }
      }
      return [203, 43, 39];
    };

    const rgbCss = (triplet) => `rgb(${triplet[0]}, ${triplet[1]}, ${triplet[2]})`;

    const riskStrokeColor = (riskByte) => rgbCss(riskStrokeTriplet(riskByte));

    const timelineKindFactors = (zoom) => {
      if (zoom <= 4) return [0.12, 0.34, 1.0];
      if (zoom <= 5) return [0.18, 0.48, 1.0];
      if (zoom <= 6) return [0.28, 0.62, 1.0];
      if (zoom <= 8) return [0.4, 0.76, 1.0];
      return [0.55, 0.86, 1.0];
    };

    const timelineGradientFromValues = (riskValues) => {
      if (!Array.isArray(riskValues) || !riskValues.length) return "var(--track-live)";

      if (riskValues.some((value) => typeof value === "string")) {
        const stops = riskValues.map((value, idx) => {
          const color = typeof value === "string" ? value : riskStrokeColor(value);
          const pos = (idx / Math.max(1, riskValues.length - 1)) * 100.0;
          return `${color} ${pos.toFixed(3)}%`;
        });
        return `linear-gradient(to right, ${stops.join(", ")})`;
      }

      const samples = [];
      const subdivisions = 6;
      for (let idx = 0; idx < riskValues.length - 1; idx += 1) {
        const start = Number(riskValues[idx] ?? 0);
        const end = Number(riskValues[idx + 1] ?? start);
        for (let step = 0; step < subdivisions; step += 1) {
          const t = step / subdivisions;
          const pos = ((idx + t) / Math.max(1, riskValues.length - 1)) * 100.0;
          samples.push(`${riskStrokeColor(mix(start, end, t))} ${pos.toFixed(3)}%`);
        }
      }
      const lastRisk = Number(riskValues[riskValues.length - 1] ?? 0);
      samples.push(`${riskStrokeColor(lastRisk)} 100%`);
      return `linear-gradient(to right, ${samples.join(", ")})`;
    };

    const updateTimelineHeat = (riskValues) => {
      timelinePhasesNode.style.background = timelineGradientFromValues(riskValues);
    };

    const rgbaCss = (triplet, alpha) =>
      `rgba(${triplet[0]}, ${triplet[1]}, ${triplet[2]}, ${clamp(Number(alpha) || 0, 0, 1).toFixed(3)})`;

    const weatherFieldRadiusForZoom = (zoom) => {
      const value = Number(zoom || 5);
      if (value <= 4) return 54;
      if (value <= 5) return 62;
      if (value <= 6) return 72;
      if (value <= 7) return 84;
      if (value <= 8) return 96;
      if (value <= 10) return 112;
      return 124;
    };

    const weatherFieldColorForTemp = (tempC) => {
      const stops = [
        [-20, [28, 91, 215]],
        [-5, [65, 149, 241]],
        [8, [102, 205, 255]],
        [18, [242, 211, 82]],
        [28, [242, 139, 55]],
        [40, [203, 43, 39]],
      ];
      const value = Number(tempC || 0);
      if (value <= stops[0][0]) return stops[0][1];
      for (let idx = 1; idx < stops.length; idx += 1) {
        if (value <= stops[idx][0]) {
          const [startTemp, startColor] = stops[idx - 1];
          const [endTemp, endColor] = stops[idx];
          const t = clamp((value - startTemp) / Math.max(0.000001, endTemp - startTemp), 0, 1);
          return [
            lerp(startColor[0], endColor[0], t),
            lerp(startColor[1], endColor[1], t),
            lerp(startColor[2], endColor[2], t),
          ];
        }
      }
      return stops[stops.length - 1][1];
    };

    const weatherFieldAlpha = (precipPct, wetHour) => {
      const certainty = clamp(Math.max(Number(precipPct || 0) / 100.0, Number(wetHour || 0) * 0.45), 0, 1);
      if (!(certainty > 0.02)) return 0;
      return clamp(Math.pow(certainty, 0.92) * 0.94, 0.06, 0.96);
    };

    const weatherFieldAlphaForTemperature = (tempC) => {
      const value = Number(tempC);
      if (!Number.isFinite(value)) return 0;
      const distanceFromMild = Math.abs(value - 16.0);
      return clamp(0.18 + Math.min(1, distanceFromMild / 24.0) * 0.52, 0.18, 0.7);
    };

    const weatherHeatGradient = (mode) => {
      if (mode === "temperature") {
        return [
          "rgba(0, 0, 0, 0)",
          "rgba(28, 91, 215, 0.18)",
          "rgba(65, 149, 241, 0.34)",
          "rgba(102, 205, 255, 0.5)",
          "rgba(242, 211, 82, 0.64)",
          "rgba(242, 139, 55, 0.8)",
          "rgba(203, 43, 39, 0.92)",
        ];
      }
      return [
        "rgba(0, 0, 0, 0)",
        "rgba(112, 168, 255, 0.08)",
        "rgba(95, 153, 248, 0.2)",
        "rgba(78, 133, 231, 0.36)",
        "rgba(58, 110, 210, 0.56)",
        "rgba(38, 81, 178, 0.76)",
        "rgba(20, 52, 132, 0.92)",
      ];
    };

    const ensureWeatherOverlayData = async () => {
      if (weatherOverlayData) return weatherOverlayData;
      if (weatherOverlayPromise) return weatherOverlayPromise;
      weatherOverlayPromise = fetch('/weather-overlay/meta', { cache: 'no-store' })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Weather overlay request failed: ${response.status}`);
          }
          return response.json();
        })
        .then((payload) => {
          weatherOverlayData = payload;
          return payload;
        })
        .catch((err) => {
          weatherOverlayPromise = null;
          throw err;
        });
      return weatherOverlayPromise;
    };

    const normalizeDegrees = (degrees) => {
      const value = Number(degrees);
      if (!Number.isFinite(value)) return null;
      return ((value % 360) + 360) % 360;
    };

    const interpolateDegrees = (startDeg, endDeg, t) => {
      const start = normalizeDegrees(startDeg);
      const end = normalizeDegrees(endDeg);
      if (start == null && end == null) return null;
      if (start == null) return end;
      if (end == null) return start;
      const delta = ((((end - start) % 360) + 540) % 360) - 180;
      return normalizeDegrees(start + delta * t);
    };

    const windColorForSpeed = (speedMps) => {
      const t = clamp(Number(speedMps || 0) / 18.0, 0, 1);
      const start = [38, 132, 255];
      const mid = [82, 196, 139];
      const end = [242, 139, 55];
      if (t <= 0.55) {
        const local = t / 0.55;
        return rgbCss([
          lerp(start[0], mid[0], local),
          lerp(start[1], mid[1], local),
          lerp(start[2], mid[2], local),
        ]);
      }
      const local = (t - 0.55) / 0.45;
      return rgbCss([
        lerp(mid[0], end[0], local),
        lerp(mid[1], end[1], local),
        lerp(mid[2], end[2], local),
      ]);
    };

    const windScaleForSpeed = (speedMps, zoom) =>
      clamp(3.8 + Number(speedMps || 0) * 0.22 + Math.max(0, (Number(zoom || 5) - 5) * 0.08), 3.8, 9.5);

    const createCanvasOverlay = (drawFn, zIndex = 4) => {
      class CanvasOverlay extends google.maps.OverlayView {
        constructor() {
          super();
          this.div = null;
          this.canvas = null;
          this.visible = false;
          this.opacity = 0;
          this.state = null;
        }

        onAdd() {
          this.div = document.createElement('div');
          this.div.style.position = 'absolute';
          this.div.style.inset = '0';
          this.div.style.pointerEvents = 'none';
          this.div.style.zIndex = String(zIndex);

          this.canvas = document.createElement('canvas');
          this.canvas.style.width = '100%';
          this.canvas.style.height = '100%';
          this.canvas.style.display = 'block';
          this.div.appendChild(this.canvas);

          this.getPanes().overlayLayer.appendChild(this.div);
          this.div.style.display = this.visible ? 'block' : 'none';
          this.div.style.opacity = String(this.opacity);
          if (this.state) {
            this.draw();
          }
        }

        draw() {
          if (!this.div || !this.canvas) return;
          const mapDiv = this.getMap()?.getDiv();
          const projection = this.getProjection();
          if (!mapDiv || !projection) return;
          const width = Math.max(1, mapDiv.clientWidth);
          const height = Math.max(1, mapDiv.clientHeight);
          const dpr = window.devicePixelRatio || 1;
          if (this.canvas.width !== Math.round(width * dpr) || this.canvas.height !== Math.round(height * dpr)) {
            this.canvas.width = Math.round(width * dpr);
            this.canvas.height = Math.round(height * dpr);
          }
          const ctx = this.canvas.getContext('2d');
          ctx.setTransform(dpr, 0, 0, dpr, 0, 0);
          ctx.clearRect(0, 0, width, height);
          this.div.style.display = this.visible ? 'block' : 'none';
          this.div.style.opacity = String(this.opacity);
          if (!this.visible || !this.state) return;
          drawFn(ctx, projection, width, height, this.state);
        }

        onRemove() {
          if (this.div?.parentNode) this.div.parentNode.removeChild(this.div);
          this.div = null;
          this.canvas = null;
        }

        setState(state, opacity = this.opacity, visible = true) {
          this.state = state;
          this.opacity = opacity;
          this.visible = visible;
          this.draw();
        }

        setOpacity(opacity) {
          this.opacity = opacity;
          if (this.div) this.div.style.opacity = String(opacity);
        }

        setVisible(visible) {
          this.visible = visible;
          if (this.div) this.div.style.display = visible ? 'block' : 'none';
        }

        redraw() {
          this.draw();
        }
      }

      const overlay = new CanvasOverlay();
      overlay.setMap(map);
      return overlay;
    };

    const clearWeatherFieldOverlays = () => {
      [weatherFieldBase, weatherFieldNext].forEach((overlay) => {
        if (overlay) overlay.setMap(null);
      });
      weatherFieldFrameKey = '';
    };

    const clearWeatherWindOverlay = () => {
      weatherWindOverlay.forEach((marker) => marker.setMap(null));
      weatherWindFrameKey = '';
    };

    const buildWeatherFieldFrame = (payload, frameIdx, mode) => {
      const cacheKey = `field:${mode}:${frameIdx}`;
      if (weatherFrameCache.has(cacheKey)) return weatherFrameCache.get(cacheKey);
      const stations = Array.isArray(payload?.stations) ? payload.stations : [];
      const points = stations
        .map((station) => {
          const tempValues = Array.isArray(station?.temp_c) ? station.temp_c : [];
          const precipValues = Array.isArray(station?.precip_probability_pct)
            ? station.precip_probability_pct
            : [];
          const wetValues = Array.isArray(station?.wet_hour) ? station.wet_hour : [];
          const tempC = Number(tempValues[frameIdx]);
          const precipPct = Number(precipValues[frameIdx] ?? 0);
          const wetHour = Number(wetValues[frameIdx] ?? 0);
          if (!Number.isFinite(tempC)) return null;
          if (mode === "temperature") {
            const normalized = clamp((tempC + 20.0) / 60.0, 0, 1);
            return {
              location: new google.maps.LatLng(Number(station.lat), Number(station.lon)),
              weight: 0.05 + normalized * 0.95,
            };
          }
          const alpha = weatherFieldAlpha(precipPct, wetHour);
          if (!(alpha > 0)) return null;
          return {
            location: new google.maps.LatLng(Number(station.lat), Number(station.lon)),
            weight: alpha,
          };
        })
        .filter(Boolean);
      weatherFrameCache.set(cacheKey, points);
      return points;
    };

    const ensureWeatherFieldOverlays = () => {
      if (!weatherFieldBase) {
        weatherFieldBase = new google.maps.visualization.HeatmapLayer({
          dissipating: true,
          radius: weatherFieldRadiusForZoom(map?.getZoom() || 5),
          opacity: 0,
        });
      }
      if (!weatherFieldNext) {
        weatherFieldNext = new google.maps.visualization.HeatmapLayer({
          dissipating: true,
          radius: weatherFieldRadiusForZoom(map?.getZoom() || 5),
          opacity: 0,
        });
      }
    };

    const ensureWeatherWindOverlay = () => {
      if (!Array.isArray(weatherWindOverlay)) weatherWindOverlay = [];
    };

    const buildWindOverlayState = (payload) => {
      const stations = Array.isArray(payload?.stations) ? payload.stations : [];
      const { baseFrameIdx, nextFrameIdx, mixValue } = getFrameBlend();
      const arrows = stations
        .map((station) => {
          const speeds = Array.isArray(station?.wind_speed_mps) ? station.wind_speed_mps : [];
          const dirs = Array.isArray(station?.wind_dir_deg) ? station.wind_dir_deg : [];
          const speedStart = Number(speeds[baseFrameIdx] ?? 0);
          const speedEnd = Number(speeds[nextFrameIdx] ?? speedStart);
          const speed = mix(speedStart, speedEnd, mixValue);
          const direction = interpolateDegrees(dirs[baseFrameIdx], dirs[nextFrameIdx], mixValue);
          if (!(speed > 0.5) || direction == null) return null;
          return {
            lat: Number(station.lat),
            lon: Number(station.lon),
            headingDeg: normalizeDegrees(direction + 180),
            scale: windScaleForSpeed(speed, map?.getZoom() || 5),
            alpha: clamp(0.42 + speed / 20.0, 0.42, 0.94),
            color: windColorForSpeed(speed),
          };
        })
        .filter(Boolean);
      return { arrows };
    };

    const updateWeatherOverlay = async () => {
      if (!map || !layerWeatherToggle) return;
      if (!cfg.weather_overlay_ready || !layerWeatherToggle.checked) {
        clearWeatherFieldOverlays();
        clearWeatherWindOverlay();
        return;
      }
      try {
        const payload = await ensureWeatherOverlayData();
        const { baseFrameIdx, nextFrameIdx, mixValue } = getFrameBlend();
        const zoomKey = Math.round(map.getZoom() || 0);
        const weatherMode = weatherModeSelect?.value || "precipitation";

        if (weatherMode === "wind") {
          clearWeatherFieldOverlays();
          const stations = Array.isArray(payload?.stations) ? payload.stations : [];
          const hasDirections = stations.some((station) => Array.isArray(station?.wind_dir_deg));
          if (!hasDirections) {
            updateStatus('Wind arrows will appear after the next forecast refresh writes wind directions.', true);
            clearWeatherWindOverlay();
            return;
          }
          ensureWeatherWindOverlay();
          const frameKey = `${baseFrameIdx}:${nextFrameIdx}:${mixValue.toFixed(3)}:${zoomKey}`;
          if (weatherWindFrameKey !== frameKey) {
            const state = buildWindOverlayState(payload);
            state.arrows.forEach((arrow, idx) => {
              const icon = {
                path: google.maps.SymbolPath.FORWARD_CLOSED_ARROW,
                scale: arrow.scale,
                rotation: arrow.headingDeg,
                fillColor: arrow.color,
                fillOpacity: arrow.alpha,
                strokeColor: arrow.color,
                strokeOpacity: arrow.alpha,
                strokeWeight: 1.2,
              };
              if (!weatherWindOverlay[idx]) {
                weatherWindOverlay[idx] = new google.maps.Marker({
                  clickable: false,
                  zIndex: 4,
                });
              }
              weatherWindOverlay[idx].setPosition({ lat: arrow.lat, lng: arrow.lon });
              weatherWindOverlay[idx].setIcon(icon);
              weatherWindOverlay[idx].setMap(map);
            });
            for (let idx = state.arrows.length; idx < weatherWindOverlay.length; idx += 1) {
              weatherWindOverlay[idx].setMap(null);
            }
            weatherWindFrameKey = frameKey;
          }
        } else {
          clearWeatherWindOverlay();
          ensureWeatherFieldOverlays();
          const frameKey = `${weatherMode}:${baseFrameIdx}:${nextFrameIdx}:${zoomKey}`;
          const radius = weatherFieldRadiusForZoom(zoomKey);
          const gradient = weatherHeatGradient(weatherMode);
          const nextOpacity = nextFrameIdx === baseFrameIdx ? 0 : mixValue;
          weatherFieldBase.setOptions({ map, radius, gradient, opacity: 1 - nextOpacity });
          weatherFieldNext.setOptions({ map: nextFrameIdx === baseFrameIdx ? null : map, radius, gradient, opacity: nextOpacity });
          if (weatherFieldFrameKey !== frameKey) {
            weatherFieldBase.setData(buildWeatherFieldFrame(payload, baseFrameIdx, weatherMode));
            weatherFieldNext.setData(buildWeatherFieldFrame(payload, nextFrameIdx, weatherMode));
            weatherFieldFrameKey = frameKey;
          }
        }

        updateStatus('');
      } catch (err) {
        updateStatus(err.message || 'Failed to load weather overlay.', true);
      }
    };

    const runScheduledWeatherOverlayUpdate = (timestamp = 0) => {
      const now = Number(timestamp || performance.now());
      if (now - weatherOverlayLastTs < weatherOverlayMinIntervalMs) {
        weatherOverlayRaf = requestAnimationFrame(runScheduledWeatherOverlayUpdate);
        return;
      }
      weatherOverlayRaf = 0;
      weatherOverlayLastTs = now;
      updateWeatherOverlay();
    };

    const scheduleWeatherOverlayUpdate = () => {
      if (!map || !layerWeatherToggle?.checked) return;
      if (weatherOverlayRaf) return;
      weatherOverlayRaf = requestAnimationFrame(runScheduledWeatherOverlayUpdate);
    };

    const roadStrokeWeight = (roadKind, zoom) => {
      const zoomBase =
        zoom >= 16 ? 4.2 :
        zoom >= 15 ? 3.8 :
        zoom >= 14 ? 3.3 :
        zoom >= 13 ? 2.8 :
        zoom >= 12 ? 2.3 :
        zoom >= 10 ? 1.9 :
        1.5;
      if (roadKind >= 2) return zoomBase + 1.4;
      if (roadKind >= 1) return zoomBase + 0.7;
      return zoomBase + 0.15;
    };

    const roadStrokeScale = (drawScale) => {
      const scale = Math.max(1, Number(drawScale) || 1);
      if (scale <= 1) return 1;
      return 1 + Math.min(0.9, Math.log2(scale) * 0.18);
    };

    const drawRoadPath = (ctx, entry, tile, strokeStyle, width) => {
      const { drawScale, drawOffsetX, drawOffsetY } = tile.request;
      ctx.strokeStyle = strokeStyle;
      ctx.lineWidth = width;
      ctx.lineCap = "butt";
      ctx.lineJoin = "round";
      ctx.beginPath();
      entry.path.forEach(([x, y], idx) => {
        const px = x * drawScale - drawOffsetX;
        const py = y * drawScale - drawOffsetY;
        if (idx === 0) ctx.moveTo(px, py);
        else ctx.lineTo(px, py);
      });
      ctx.stroke();
    };

    const endpointBucketKey = (kind, worldPoint) =>
      `${kind}:${Math.round(worldPoint[0] / 4)}:${Math.round(worldPoint[1] / 4)}`;

    const buildEntryPaintStates = () => {
      const { baseFrameIdx, nextFrameIdx, mixValue } = getFrameBlend();
      const paintStatesByTile = new Map();
      const endpointBuckets = new Map();
      const allStates = [];

      activeVectorTiles.forEach((tile) => {
        if (!tile?.data?.entries?.length) return;
        const tileStates = [];
        tile.data.entries.forEach((entry) => {
          if (!Array.isArray(entry.path) || entry.path.length < 2) return;
          const riskByte = Math.round(
            mix(
              entry.risk[baseFrameIdx] ?? 0,
              entry.risk[nextFrameIdx] ?? entry.risk[baseFrameIdx] ?? 0,
              mixValue,
            ),
          );
          const baseColor = riskStrokeTriplet(riskByte);
          const { drawScale, drawOffsetX, drawOffsetY } = tile.request;
          const [startRawX, startRawY] = entry.path[0];
          const [endRawX, endRawY] = entry.path[entry.path.length - 1];
          const startLocal = [
            startRawX * drawScale - drawOffsetX,
            startRawY * drawScale - drawOffsetY,
          ];
          const endLocal = [
            endRawX * drawScale - drawOffsetX,
            endRawY * drawScale - drawOffsetY,
          ];
          const startWorld = [
            tile.coord.x * 256 + startLocal[0],
            tile.coord.y * 256 + startLocal[1],
          ];
          const endWorld = [
            tile.coord.x * 256 + endLocal[0],
            tile.coord.y * 256 + endLocal[1],
          ];
          const state = {
            entry,
            tile,
            riskByte,
            baseColor,
            startColor: baseColor,
            endColor: baseColor,
            startLocal,
            endLocal,
            startWorld,
            endWorld,
            startKey: "",
            endKey: "",
          };
          tileStates.push(state);
          allStates.push(state);

          const startKey = endpointBucketKey(entry.kind, startWorld);
          const endKey = endpointBucketKey(entry.kind, endWorld);
          state.startKey = startKey;
          state.endKey = endKey;
          if (!endpointBuckets.has(startKey)) endpointBuckets.set(startKey, []);
          if (!endpointBuckets.has(endKey)) endpointBuckets.set(endKey, []);
          endpointBuckets.get(startKey).push({ state, endpoint: "start" });
          endpointBuckets.get(endKey).push({ state, endpoint: "end" });
        });
        paintStatesByTile.set(tile, tileStates);
      });

      allStates.forEach((state) => {
        const neighborStates = [];
        [state.startKey, state.endKey].forEach((bucketKey) => {
          const items = endpointBuckets.get(bucketKey) || [];
          items.forEach(({ state: otherState }) => {
            if (otherState !== state) {
              neighborStates.push(otherState);
            }
          });
        });
        if (!neighborStates.length) return;
        const avgRisk =
          neighborStates.reduce((sum, otherState) => sum + Number(otherState.riskByte || 0), 0) /
          neighborStates.length;
        const blend = neighborStates.length <= 2 ? 0.5 : 0.35;
        state.riskByte = Math.round(mix(Number(state.riskByte || 0), avgRisk, blend));
        state.baseColor = riskStrokeTriplet(state.riskByte);
        state.startColor = state.baseColor;
        state.endColor = state.baseColor;
      });

      endpointBuckets.forEach((items) => {
        if (items.length < 2) return;
        let avg = [0, 0, 0];
        items.forEach(({ state }) => {
          avg = [
            avg[0] + state.baseColor[0],
            avg[1] + state.baseColor[1],
            avg[2] + state.baseColor[2],
          ];
        });
        avg = avg.map((value) => Math.round(value / items.length));
        items.forEach(({ state, endpoint }) => {
          if (endpoint === "start") state.startColor = avg;
          else state.endColor = avg;
        });
      });

      return paintStatesByTile;
    };

    const rasterTileUrl = (frameIdx, zoom, x, y) =>
      `/segment-raster-tiles/${frameIdx}/${zoom}/${wrapTileX(x, zoom)}/${y}.png?rev=${encodeURIComponent(cfg.road_tile_revision || "current")}`;

    const paintRoadTile = (tile, tileStates = null) => {
      if (!tile || !tile.ctx) return;
      tile.ctx.clearRect(0, 0, 256, 256);
      if (!tile.data || !Array.isArray(tile.data.entries)) return;

      let states = Array.isArray(tileStates) ? tileStates : null;
      if (!states) {
        const { baseFrameIdx, nextFrameIdx, mixValue } = getFrameBlend();
        states = tile.data.entries
          .filter((entry) => Array.isArray(entry.path) && entry.path.length >= 2)
          .map((entry) => {
            const riskByte = Math.round(
              mix(
                entry.risk[baseFrameIdx] ?? 0,
                entry.risk[nextFrameIdx] ?? entry.risk[baseFrameIdx] ?? 0,
                mixValue,
              ),
            );
            const baseColor = riskStrokeTriplet(riskByte);
            const { drawScale, drawOffsetX, drawOffsetY } = tile.request;
            const [startRawX, startRawY] = entry.path[0];
            const [endRawX, endRawY] = entry.path[entry.path.length - 1];
            return {
              entry,
              tile,
              riskByte,
              baseColor,
              startColor: baseColor,
              endColor: baseColor,
              startLocal: [
                startRawX * drawScale - drawOffsetX,
                startRawY * drawScale - drawOffsetY,
              ],
              endLocal: [
                endRawX * drawScale - drawOffsetX,
                endRawY * drawScale - drawOffsetY,
              ],
            };
          });
      }
      states.forEach((state) => {
        const baseWidth = roadStrokeWeight(state.entry.kind, tile.zoom) * roadStrokeScale(tile.request.drawScale);
        const nearFlat =
          Math.abs(state.startColor[0] - state.endColor[0]) +
            Math.abs(state.startColor[1] - state.endColor[1]) +
            Math.abs(state.startColor[2] - state.endColor[2]) <
          12;
        let strokeStyle = rgbCss(state.baseColor);
        if (!nearFlat) {
          const gradient = tile.ctx.createLinearGradient(
            state.startLocal[0],
            state.startLocal[1],
            state.endLocal[0],
            state.endLocal[1],
          );
          gradient.addColorStop(0, rgbCss(state.startColor));
          gradient.addColorStop(1, rgbCss(state.endColor));
          strokeStyle = gradient;
        }
        drawRoadPath(tile.ctx, state.entry, tile, strokeStyle, baseWidth);
      });
    };

    const repaintVectorTiles = () => {
      if (!activeVectorTiles.size) return;
      const paintStatesByTile = buildEntryPaintStates();
      activeVectorTiles.forEach((tile) => paintRoadTile(tile, paintStatesByTile.get(tile)));
    };

    const ensureRasterImageFrame = (img, frameIdx, zoom, x, y) => {
      const targetFrame = String(frameIdx);
      if (img.dataset.frame === targetFrame) return;
      img.dataset.frame = targetFrame;
      img.src = rasterTileUrl(frameIdx, zoom, x, y);
    };

    const updateRasterTile = (tile) => {
      if (!tile) return;
      const { baseFrameIdx, nextFrameIdx, mixValue } = getFrameBlend();
      if (Number(tile.nextImg.dataset.frame) === baseFrameIdx) {
        [tile.baseImg, tile.nextImg] = [tile.nextImg, tile.baseImg];
      }
      ensureRasterImageFrame(tile.baseImg, baseFrameIdx, tile.zoom, tile.coord.x, tile.coord.y);
      ensureRasterImageFrame(tile.nextImg, nextFrameIdx, tile.zoom, tile.coord.x, tile.coord.y);
      const nextOpacity = nextFrameIdx === baseFrameIdx ? 0 : mixValue;
      tile.baseImg.style.opacity = String(1 - nextOpacity);
      tile.nextImg.style.opacity = String(nextOpacity);
    };

    const updateRasterTiles = () => {
      activeRasterTiles.forEach((tile) => updateRasterTile(tile));
    };

    const fetchRoadTileData = async (request) => {
      const cacheKey = `${cfg.road_tile_revision || "current"}:${request.sourceZoom}:${request.sourceX}:${request.sourceY}`;
      const cached = roadTileCache.get(cacheKey);
      if (cached?.data) return cached.data;
      if (cached?.promise) return cached.promise;

      const promise = fetch(`/segment-tiles/${request.sourceZoom}/${request.sourceX}/${request.sourceY}.json?rev=${encodeURIComponent(cfg.road_tile_revision || "current")}`)
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Road tile request failed: ${response.status}`);
          }
          return response.json();
        })
        .then((payload) => prepareTilePayload(payload))
        .then((data) => {
          roadTileCache.set(cacheKey, { data });
          return data;
        })
        .catch((err) => {
          roadTileCache.delete(cacheKey);
          throw err;
        });

      roadTileCache.set(cacheKey, { promise });
      return promise;
    };

    const visibleSourceTileRequests = () => {
      if (!map) return [];
      const bounds = map.getBounds();
      const projection = map.getProjection();
      if (!bounds || !projection) return [];
      const zoom = Math.max(Number(cfg.tile_zoom_min || 4), Math.round(map.getZoom() || 0));
      const northEast = bounds.getNorthEast();
      const southWest = bounds.getSouthWest();
      const northWest = new google.maps.LatLng(northEast.lat(), southWest.lng());
      const southEast = new google.maps.LatLng(southWest.lat(), northEast.lng());
      const northWestPoint = projection.fromLatLngToPoint(northWest);
      const southEastPoint = projection.fromLatLngToPoint(southEast);
      if (!northWestPoint || !southEastPoint) return [];

      const scale = 2 ** zoom;
      const worldToTile = scale / 256.0;
      const minTileX = Math.floor(Math.min(northWestPoint.x, southEastPoint.x) * worldToTile);
      const maxTileX = Math.floor(Math.max(northWestPoint.x, southEastPoint.x) * worldToTile);
      const minTileY = Math.floor(Math.min(northWestPoint.y, southEastPoint.y) * worldToTile);
      const maxTileY = Math.floor(Math.max(northWestPoint.y, southEastPoint.y) * worldToTile);
      const requests = [];
      const seen = new Set();
      for (let tileY = minTileY; tileY <= maxTileY; tileY += 1) {
        for (let tileX = minTileX; tileX <= maxTileX; tileX += 1) {
          const request = normalizeRoadTileRequest({ x: tileX, y: tileY }, zoom);
          const key = `${request.sourceZoom}:${request.sourceX}:${request.sourceY}`;
          if (seen.has(key)) continue;
          seen.add(key);
          requests.push(request);
        }
      }
      return requests.slice(0, 64);
    };

    const refreshTimelineFromViewport = async () => {
      if (!map || !cfg.road_mode || !layerRiskToggle?.checked) {
        updateTimelineHeat(null);
        return;
      }
      const requests = visibleSourceTileRequests();
      if (!requests.length) {
        updateTimelineHeat(null);
        return;
      }
      const token = viewportTimelineToken + 1;
      viewportTimelineToken = token;
      try {
        const zoom = Math.round(map.getZoom() || 0);
        if (zoom <= roadRasterMaxZoom) {
          const response = await fetch("/api/raster-timeline-summary", {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            cache: "no-store",
            body: JSON.stringify({
              z: zoom,
              tiles: requests.map((request) => ({ x: request.sourceX, y: request.sourceY })),
            }),
          });
          if (!response.ok) {
            throw new Error(`Raster summary request failed: ${response.status}`);
          }
          const summary = await response.json();
          if (token !== viewportTimelineToken) return;
          if (Array.isArray(summary.risks)) {
            updateTimelineHeat(summary.risks);
          } else {
            updateTimelineHeat(Array.isArray(summary.colors) ? summary.colors : null);
          }
          return;
        }

        const tiles = await Promise.all(
          requests.map((request) => fetchRoadTileData(request).catch(() => null)),
        );
        if (token !== viewportTimelineToken) return;
        const kindFactors = timelineKindFactors(zoom);
        const frameSums = new Float64Array(frameCount);
        let totalWeight = 0;
        tiles.forEach((tile) => {
          if (!tile?.summary) return;
          tile.summary.kindWeights.forEach((kindWeight, kindIdx) => {
            const zoomWeight = kindFactors[kindIdx] || 0;
            const weightedKind = Number(kindWeight || 0) * zoomWeight;
            if (weightedKind <= 0) return;
            totalWeight += weightedKind;
            const kindSums = tile.summary.kindFrameSums[kindIdx] || [];
            for (let frameIdx = 0; frameIdx < frameCount; frameIdx += 1) {
              frameSums[frameIdx] += Number(kindSums[frameIdx] || 0) * zoomWeight;
            }
          });
        });
        if (totalWeight <= 0) {
          updateTimelineHeat(null);
          return;
        }
        const risks = Array.from(frameSums, (value) => value / totalWeight);
        updateTimelineHeat(risks);
      } catch (err) {
        updateStatus(err.message || "Failed to summarize the visible forecast.", true);
      }
    };

    const scheduleTimelineHeatRefresh = (delay = 120) => {
      if (viewportTimelineTimer) {
        window.clearTimeout(viewportTimelineTimer);
      }
      viewportTimelineTimer = window.setTimeout(() => {
        viewportTimelineTimer = 0;
        refreshTimelineFromViewport();
      }, delay);
    };

    const createRoadTileLayer = () => ({
      alt: "Road Risk",
      name: "Road Risk",
      tileSize: new google.maps.Size(256, 256),
      minZoom: Number(cfg.tile_zoom_min || 4),
      maxZoom: 20,
      getTile: (coord, zoom, ownerDocument) => {
        if (zoom < roadVectorMinZoom) {
          const container = ownerDocument.createElement("div");
          container.className = "road-risk-raster-tile";
          const baseImg = ownerDocument.createElement("img");
          const nextImg = ownerDocument.createElement("img");
          [baseImg, nextImg].forEach((img) => {
            img.width = 256;
            img.height = 256;
            img.alt = "";
            img.decoding = "async";
            img.className = "road-risk-raster-image";
          });
          container.append(baseImg, nextImg);
          const tile = {
            mode: "raster",
            container,
            coord: { x: coord.x, y: coord.y },
            zoom,
            baseImg,
            nextImg,
          };
          container.__roadTile = tile;
          activeRasterTiles.add(tile);
          updateRasterTile(tile);
          return container;
        }

        const canvas = ownerDocument.createElement("canvas");
        canvas.width = 256;
        canvas.height = 256;
        canvas.className = "road-risk-tile";

        const tile = {
          canvas,
          ctx: canvas.getContext("2d"),
          coord: { x: coord.x, y: coord.y },
          zoom,
          request: normalizeRoadTileRequest(coord, zoom),
          data: null,
          mode: "vector",
        };
        canvas.__roadTile = tile;
        activeVectorTiles.add(tile);

        fetchRoadTileData(tile.request)
          .then((data) => {
            tile.data = data;
            scheduleVectorRepaint();
          })
          .catch((err) => {
            updateStatus(err.message || "Failed to load road tiles.", true);
          });

        return canvas;
      },
      releaseTile: (node) => {
        const tile = node?.__roadTile;
        if (!tile) return;
        if (tile.mode === "raster") {
          activeRasterTiles.delete(tile);
          return;
        }
        activeVectorTiles.delete(tile);
      },
    });

    const baseMapStyles = [
      { featureType: "poi", stylers: [{ visibility: "off" }] },
      { featureType: "transit", stylers: [{ visibility: "off" }] },
      { featureType: "administrative.land_parcel", stylers: [{ visibility: "off" }] },
      { featureType: "administrative.neighborhood", stylers: [{ visibility: "off" }] },
      { featureType: "road", elementType: "labels.text.fill", stylers: [{ color: "#555f6b" }] },
      { featureType: "landscape.natural", elementType: "geometry", stylers: [{ color: "#dfe8dd" }] },
      { featureType: "water", elementType: "geometry", stylers: [{ color: "#bfd9eb" }] },
    ];

    const applyMapStyles = () => {
      if (!map) return;
      const showRoads = Boolean(layerRoadsToggle?.checked);
      const styles = showRoads
        ? baseMapStyles
        : [
            ...baseMapStyles,
            { featureType: "road", elementType: "geometry", stylers: [{ visibility: "off" }] },
            { featureType: "road.highway", elementType: "geometry", stylers: [{ visibility: "off" }] },
          ];
      map.setOptions({
        styles,
      });
    };

    const updateZoomHint = () => {
      if (!map) return;
      const showHint = map.getZoom() < roadVectorMinZoom;
      zoomHintNode.classList.toggle("show", showHint);
    };

    const installOverlay = () => {
      if (!map || !cfg.road_mode) return;
      applyMapStyles();
      if (!layerRiskToggle?.checked) {
        map.overlayMapTypes.clear();
        updateTimelineHeat(null);
        return;
      }
      if (map.overlayMapTypes.getLength() === 0) {
        map.overlayMapTypes.push(roadOverlay);
      }
      if (map.getZoom() > roadRasterMaxZoom) {
        scheduleVectorRepaint();
      } else {
        updateRasterTiles();
      }
      updateZoomHint();
      updateStatus("");
      scheduleTimelineHeatRefresh(80);
    };

    const hitTestRoad = (latLng) => {
      if (!map || map.getZoom() < roadVectorMinZoom) return null;
      const projection = map.getProjection();
      if (!projection) return null;

      const zoom = map.getZoom();
      const point = projection.fromLatLngToPoint(latLng);
      if (!point) return null;
      const scale = 2 ** zoom;
      const worldX = point.x * scale;
      const worldY = point.y * scale;
      const tileX = Math.floor(worldX / 256);
      const tileY = Math.floor(worldY / 256);
      const threshold = zoom >= 13 ? 10 : zoom >= 11 ? 12 : 14;

      let best = null;
      activeVectorTiles.forEach((tile) => {
        if (!tile.data?.entries || tile.zoom !== zoom) return;
        if (Math.abs(tile.coord.x - tileX) > 1 || Math.abs(tile.coord.y - tileY) > 1) return;

        const localX = worldX - tile.coord.x * 256;
        const localY = worldY - tile.coord.y * 256;
        const { drawScale, drawOffsetX, drawOffsetY } = tile.request;

        tile.data.entries.forEach((entry) => {
          let distance = Number.POSITIVE_INFINITY;
          for (let idx = 1; idx < entry.path.length; idx += 1) {
            const [axRaw, ayRaw] = entry.path[idx - 1];
            const [bxRaw, byRaw] = entry.path[idx];
            const ax = axRaw * drawScale - drawOffsetX;
            const ay = ayRaw * drawScale - drawOffsetY;
            const bx = bxRaw * drawScale - drawOffsetX;
            const by = byRaw * drawScale - drawOffsetY;
            distance = Math.min(distance, pointSegmentDistance(localX, localY, ax, ay, bx, by));
            if (distance <= threshold / 2) break;
          }
          if (distance <= threshold && (!best || distance < best.distance)) {
            best = { entry, distance };
          }
        });
      });

      return best;
    };

    const fetchSegmentDetail = async (segmentIdx, frameIdx) => {
      const cacheKey = `${segmentIdx}:${frameIdx}`;
      if (segmentDetailCache.has(cacheKey)) return segmentDetailCache.get(cacheKey);
      const promise = fetch(`/api/segment-detail?segment_idx=${encodeURIComponent(segmentIdx)}&frame_idx=${encodeURIComponent(frameIdx)}`, {
        cache: "no-store",
      })
        .then((response) => {
          if (!response.ok) {
            throw new Error(`Segment detail request failed: ${response.status}`);
          }
          return response.json();
        })
        .then((detail) => {
          segmentDetailCache.set(cacheKey, detail);
          return detail;
        })
        .catch((err) => {
          segmentDetailCache.delete(cacheKey);
          throw err;
        });
      segmentDetailCache.set(cacheKey, promise);
      return promise;
    };

    const tooltipHtml = (detail) => {
      const deltaAbs = Math.abs(Number(detail.delta_points || 0)).toFixed(1);
      const deltaTone = Number(detail.delta_points || 0) > 0 ? "up" : Number(detail.delta_points || 0) < 0 ? "down" : "flat";
      return `
        <div class="segment-tooltip">
          <div class="tooltip-road">${escapeHtml(detail.road_name || "Road segment")}</div>
          <div class="tooltip-meta">${escapeHtml(detail.road_class || "")} · ${Number(detail.length_km || 0).toFixed(2)} km</div>
          <div class="tooltip-grid">
            <div class="tooltip-stat">
              <span>Risk index</span>
              <strong>${Number(detail.risk_index || 0).toFixed(1)}</strong>
            </div>
            <div class="tooltip-stat">
              <span>Safety index</span>
              <strong>${Number(detail.safety_index || 0).toFixed(1)}</strong>
            </div>
          </div>
          <div class="tooltip-band tooltip-band-${escapeHtml(String(detail.risk_band || "").toLowerCase())}">${escapeHtml(detail.risk_band || "")}</div>
          <div class="tooltip-delta ${deltaTone}">
            ${escapeHtml(detail.relative_label || "Near normal")}
            <strong>${deltaAbs} pts</strong>
          </div>
          <div class="tooltip-footer">
            <div>${escapeHtml(detail.frame_label || "")} · ${escapeHtml(detail.target_local_label || "")}</div>
            <div>Hist. incidents ${Number(detail.historical_total || 0)}</div>
          </div>
        </div>
      `;
    };

    const showRoadDetail = async (event) => {
      if (!cfg.road_mode || !layerRiskToggle?.checked) return;
      if (map.getZoom() < roadVectorMinZoom) {
        updateStatus("Zoom in to inspect a specific road segment.");
        return;
      }
      const hit = hitTestRoad(event.latLng);
      if (!hit || hit.entry?.segmentIdx == null) return;
      const frameIdx = Math.min(maxFrameIdx, Math.max(0, Math.round(currentFrameValue)));
      try {
        const detail = await fetchSegmentDetail(hit.entry.segmentIdx, frameIdx);
        infoWindow.setContent(tooltipHtml(detail));
        infoWindow.setPosition(event.latLng);
        infoWindow.open({ map });
        updateStatus("");
      } catch (err) {
        updateStatus(err.message || "Failed to load road details.", true);
      }
    };

    const applyHoverCursor = () => {
      hoverRaf = 0;
      if (!map || !cfg.road_mode || !layerRiskToggle?.checked || map.getZoom() < roadVectorMinZoom || !pendingHoverLatLng) {
        mapNode.style.cursor = "";
        return;
      }
      const hit = hitTestRoad(pendingHoverLatLng);
      mapNode.style.cursor = hit && hit.entry?.segmentIdx != null ? "pointer" : "";
    };

    const scheduleHoverCursorUpdate = (latLng) => {
      pendingHoverLatLng = latLng || null;
      if (hoverRaf) return;
      hoverRaf = requestAnimationFrame(applyHoverCursor);
    };

    const initGoogleMap = () => {
      map = new google.maps.Map(mapNode, {
        center: {
          lat: Number(cfg.center_lat || 39.5),
          lng: Number(cfg.center_lon || -98.35),
        },
        zoom: Number(cfg.default_zoom || 5),
        minZoom: Number(cfg.zoom_min || cfg.tile_zoom_min || 4),
        maxZoom: 20,
        mapTypeControl: false,
        streetViewControl: false,
        fullscreenControl: false,
        zoomControl: true,
        zoomControlOptions: { position: google.maps.ControlPosition.RIGHT_BOTTOM },
        rotateControl: false,
        scaleControl: false,
        clickableIcons: false,
        gestureHandling: "greedy",
        styles: layerRoadsToggle?.checked
          ? baseMapStyles
          : [
              ...baseMapStyles,
              { featureType: "road", elementType: "geometry", stylers: [{ visibility: "off" }] },
              { featureType: "road.highway", elementType: "geometry", stylers: [{ visibility: "off" }] },
            ],
      });

      infoWindow = new google.maps.InfoWindow({ maxWidth: 340 });
      if (cfg.road_mode) {
        roadOverlay = createRoadTileLayer();
        installOverlay();
      }

      map.addListener("idle", () => {
        installOverlay();
        scheduleWeatherOverlayUpdate();
      });
      map.addListener("click", (event) => {
        showRoadDetail(event);
      });
      map.addListener("mousemove", (event) => {
        scheduleHoverCursorUpdate(event.latLng);
      });
      map.addListener("mouseout", () => {
        scheduleHoverCursorUpdate(null);
      });
      map.addListener("zoom_changed", () => {
        updateZoomHint();
        scheduleTimelineHeatRefresh(40);
        scheduleHoverCursorUpdate(null);
        scheduleWeatherOverlayUpdate();
      });
      updateZoomHint();
      scheduleWeatherOverlayUpdate();
    };

    const loadGoogleMaps = () => {
      if (!cfg.api_key) {
        updateStatus("GMAPS_API_KEY is required.", true);
        return;
      }
      if (window.google && window.google.maps) {
        initGoogleMap();
        return;
      }
      const callbackName = `gmapsInit_${cfg.service_id}_${Date.now()}`;
      window[callbackName] = () => {
        delete window[callbackName];
        initGoogleMap();
      };
      const script = document.createElement("script");
      script.src = `https://maps.googleapis.com/maps/api/js?key=${cfg.api_key}&callback=${callbackName}&v=weekly&libraries=visualization`;
      script.async = true;
      script.defer = true;
      script.onerror = () => {
        updateStatus("Failed to load Google Maps JavaScript API.", true);
      };
      document.head.appendChild(script);
    };

    slider.addEventListener("input", () => {
      currentFrameValue = clamp(Number(slider.value) || 0, 0, maxFrameIdx);
      syncFrameVisuals();
    });
    playBtn.addEventListener("click", () => setPlaying(!timer));
    layerRiskToggle?.addEventListener("change", () => {
      installOverlay();
    });
    layerWeatherToggle?.addEventListener("change", () => {
      if (!layerWeatherToggle.checked) {
        clearWeatherFieldOverlays();
        clearWeatherWindOverlay();
      }
      scheduleWeatherOverlayUpdate();
    });
    weatherModeSelect?.addEventListener("change", () => {
      clearWeatherFieldOverlays();
      clearWeatherWindOverlay();
      scheduleWeatherOverlayUpdate();
    });
    layerRoadsToggle?.addEventListener("change", () => {
      applyMapStyles();
    });

    renderTimelineScaffold();
    renderFreshness();
    updateTimeline();
    loadGoogleMaps();
    return [];
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap, { once: true });
  } else {
    bootstrap();
  }

  window.bootstrapTrafficSafetyMap = bootstrap;
})();
