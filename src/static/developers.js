/* Developer portal page.

   Reads the public tier table from /v1/meta and lets a developer inspect their
   own key via /v1/keys/me. Dependency-free; talks only to the public /v1 API and
   never stores the key. There is intentionally no key-minting UI here — issuing
   keys is an operator action gated by an admin token. */
(() => {
  const esc = (value) =>
    String(value ?? "")
      .replaceAll("&", "&amp;")
      .replaceAll("<", "&lt;")
      .replaceAll(">", "&gt;")
      .replaceAll('"', "&quot;")
      .replaceAll("'", "&#39;");

  const num = (value) => (typeof value === "number" && isFinite(value) ? value : null);

  const fmtInt = (value) => {
    const n = num(value);
    return n === null ? "–" : Math.round(n).toLocaleString();
  };

  const fmtQuota = (value) => (value === null || value === undefined ? "unlimited" : fmtInt(value));

  const bootstrap = async () => {
    const root = document.getElementById("dev-shell");
    if (!root || root.dataset.ready === "1") return;
    root.dataset.ready = "1";

    let cfg = {};
    try {
      cfg = JSON.parse(root.dataset.config || "{}");
    } catch (err) {
      console.error("developers: bad config", err);
    }
    const api = (cfg.api_base || "/v1").replace(/\/$/, "");

    const statusNode = root.querySelector("#dev-status");
    const authNode = root.querySelector("#dev-auth-note");
    const tiersBody = root.querySelector("#dev-tiers-body");
    const input = root.querySelector("#dev-key-input");
    const button = root.querySelector("#dev-key-check");
    const result = root.querySelector("#dev-key-result");

    const setStatus = (msg, cls) => {
      if (!statusNode) return;
      statusNode.textContent = msg || "";
      statusNode.className = "dev-status" + (cls ? ` ${cls}` : "");
    };

    // --- tier table + feature status from /v1/meta ---
    let meta = null;
    try {
      const resp = await fetch(`${api}/meta`, { headers: { Accept: "application/json" } });
      if (!resp.ok) throw new Error(`HTTP ${resp.status}`);
      meta = await resp.json();
    } catch (err) {
      setStatus("Could not load API metadata. Please try again later.", "error");
      console.error("developers: meta fetch failed", err);
    }

    const keys = (meta && meta.api_keys) || {};
    const tiers = Array.isArray(keys.tiers) ? keys.tiers : [];

    if (tiersBody) {
      tiersBody.innerHTML = tiers
        .map(
          (t) => `<tr>
            <td class="dev-tier-name">${esc(t.label || t.name)}</td>
            <td>${fmtInt(t.rate_per_min)}</td>
            <td>${fmtQuota(t.daily_quota)}</td>
            <td>${t.issuable ? "yes" : "—"}</td>
          </tr>`
        )
        .join("");
    }

    const enabled = Boolean(keys.enabled);
    const requireKey = Boolean(keys.require_key);

    if (meta) {
      if (enabled && requireKey) {
        setStatus("API keys are enabled — a key is required for every /v1 request here.", "enabled");
      } else if (enabled) {
        setStatus("API keys are enabled on this deployment (optional — no key falls back to the anonymous tier).", "enabled");
      } else {
        setStatus("This deployment runs open: no API key required, anonymous IP rate limits apply. The tiers below show what keyed access looks like where it is enabled.", "disabled");
      }
    }

    if (authNode) {
      if (!enabled) {
        authNode.textContent = "Keys are not enabled on this deployment; all access is anonymous.";
      } else if (requireKey) {
        authNode.textContent = "A valid key is required on every /v1 request.";
      } else {
        authNode.textContent = "Keys are optional: requests without one get the anonymous tier.";
      }
    }

    // --- "check your key" tool (only meaningful when keys are enabled) ---
    if (!enabled && input && button && result) {
      input.disabled = true;
      button.disabled = true;
      result.textContent = "Key checking is available on deployments where API keys are enabled.";
    }

    const renderKey = (data) => {
      const limits = data.limits || {};
      const usage = data.usage || {};
      const rows = [
        ["Tier", esc(data.tier)],
        ["Label", esc(data.label || "—")],
        ["Rate limit", `${fmtInt(limits.rate_per_min)} / min`],
        ["Daily quota", fmtQuota(limits.daily_quota)],
        ["Used today", fmtInt(usage.requests_today)],
        [
          "Quota remaining",
          limits.quota_remaining_today === null || limits.quota_remaining_today === undefined
            ? "unlimited"
            : fmtInt(limits.quota_remaining_today),
        ],
        ["Created", esc((data.created_at || "").slice(0, 10) || "—")],
      ];
      result.className = "dev-key-result";
      result.innerHTML =
        "<dl>" +
        rows.map(([k, v]) => `<div><dt>${esc(k)}</dt><dd>${v}</dd></div>`).join("") +
        "</dl>";
    };

    const checkKey = async () => {
      if (!input || input.disabled) return;
      const raw = input.value.trim();
      result.className = "dev-key-result";
      if (!raw) {
        result.textContent = "Enter a key to check.";
        return;
      }
      result.textContent = "Checking…";
      try {
        const resp = await fetch(`${api}/keys/me`, { headers: { "X-API-Key": raw } });
        if (resp.status === 200) {
          renderKey(await resp.json());
          return;
        }
        result.className = "dev-key-result error";
        if (resp.status === 401) {
          result.textContent = "That key was not recognized (it may be revoked).";
        } else if (resp.status === 404) {
          result.textContent = "API keys are not enabled on this deployment.";
        } else {
          result.textContent = `Could not check the key (HTTP ${resp.status}).`;
        }
      } catch (err) {
        result.className = "dev-key-result error";
        result.textContent = "Could not reach the API. Please try again.";
        console.error("developers: keys/me fetch failed", err);
      }
    };

    if (button) button.addEventListener("click", checkKey);
    if (input) {
      input.addEventListener("keydown", (event) => {
        if (event.key === "Enter") {
          event.preventDefault();
          checkKey();
        }
      });
    }
  };

  if (document.readyState === "loading") {
    document.addEventListener("DOMContentLoaded", bootstrap);
  } else {
    bootstrap();
  }
})();
