/* IECNN Explorer — reads only /api/* served by server/server.py.
   Vanilla JS, defensive against missing/malformed data, dark theme only. */
"use strict";

/* ---------------- tiny helpers ---------------- */
const $ = (sel, el = document) => el.querySelector(sel);
const $$ = (sel, el = document) => [...el.querySelectorAll(sel)];
function h(tag, attrs = {}, ...kids) {
  const el = document.createElement(tag);
  for (const [k, v] of Object.entries(attrs)) {
    if (k === "class") el.className = v;
    else if (k.startsWith("on")) el.addEventListener(k.slice(2), v);
    else if (v !== null && v !== undefined) el.setAttribute(k, v);
  }
  for (const kid of kids.flat(9)) {
    if (kid === null || kid === undefined) continue;
    el.append(kid.nodeType ? kid : document.createTextNode(kid));
  }
  return el;
}
async function api(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} → HTTP ${r.status}`);
  return r.json();
}
async function apiSend(path, body) {
  const r = await fetch(path, { method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(body) });
  const d = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(d.error || `${path} → HTTP ${r.status}`);
  return d;
}
const fmt = (x, d = 4) => (x === null || x === undefined ||
  (typeof x === "number" && !isFinite(x))) ? "—" : Number(x).toFixed(d);
const esc = s => String(s ?? "");
const uniq = a => [...new Set(a)];

/* ---- token display ----------------------------------------------------
   256 of the 512 vocab slots are raw byte-fallback tokens. Printable ASCII
   shows as-is; control bytes show as \t \n \r and everything else (high
   bytes 0x80-0xFF, the lone bad merge) as hex <xx>. In checkpoint exports
   the runner-side decode has already replaced invalid bytes with U+FFFD,
   which we can't map back to the original byte — shown as <fffd>. */
const _CTRL_JS = { 9: "\\t", 10: "\\n", 13: "\\r" };
const byteDisp = b => (b in _CTRL_JS) ? _CTRL_JS[b] :
  (b >= 0x20 && b <= 0x7e) ? String.fromCharCode(b) :
  `<${b.toString(16).padStart(2, "0")}>`;
function tokDisp(s) {
  s = String(s ?? "");
  let out = "";
  for (const ch of s) {
    const o = ch.codePointAt(0);
    if (o === 0xfffd) out += "<fffd>";
    else if (o in _CTRL_JS) out += _CTRL_JS[o];
    else if (o < 0x20 || o === 0x7f) out += `<${o.toString(16).padStart(2, "0")}>`;
    else out += ch;
  }
  return out;
}
const tokEscaped = s => tokDisp(s) !== String(s ?? "");
/* display for a vocab id (0..511), using the export's id-ordered labels */
const idDisp = (i, labels) =>
  i < 256 ? byteDisp(i)
  : !Array.isArray(labels) ? `id ${i}`
  : !labels[i] ? `<special ${i}>`
  : tokDisp(labels[i]);

const PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#4db6ac",
                 "#f06292", "#fff176", "#64b5f6", "#aed581"];
const PASS_COLORS = ["#7986cb", "#4fc3f7", "#ffd54f"];   // pass 1/2/3 traces
/* non-color series cue (a11y): dash pattern per series ordinal, so series
   remain distinguishable without relying on hue alone */
const DASHES = [[], [6, 3], [2, 2], [8, 2, 2, 2], [5, 5]];
function rungColor(rung) {
  const r = String(rung || "");
  if (/nolambda/.test(r)) return "#e57373";
  if (/^b0/.test(r)) return "#81c784";
  if (/^e0/.test(r)) return "#4fc3f7";
  if (/^e1/.test(r)) return "#ffb74d";
  if (/^e[2-9]/.test(r)) return "#ba68c8";
  if (/^c[0-9]/.test(r)) return "#4db6ac";
  return "#8a95ad";
}
/* weight in [0,1] -> hot color for the dot-weight heat cells */
const HEAT = w => {
  const t = Math.max(0, Math.min(1, w));
  const r = Math.round(30 + 225 * t), g = Math.round(60 + 60 * t),
        b = Math.round(150 - 90 * t);
  return `rgb(${r},${g},${b})`;
};

/* ---------------- chart.js theme (guarded: page still renders without it) -- */
const HAS_CHART = typeof Chart !== "undefined";
/* honour prefers-reduced-motion everywhere: no chart animations, no replay
   autoplay (replayPanel checks this too), no smooth scrolling */
const REDUCE_MOTION = typeof window !== "undefined" &&
  window.matchMedia && window.matchMedia("(prefers-reduced-motion: reduce)").matches;
if (HAS_CHART) {
  Chart.defaults.color = "#8a95ad";
  Chart.defaults.borderColor = "#26304a";
  Chart.defaults.font.family =
    "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = "#1d2431";
  Chart.defaults.plugins.tooltip.borderColor = "#2a3345";
  if (REDUCE_MOTION) {
    Chart.defaults.animation = false;
    Chart.defaults.transitions = { active: { animation: { duration: 0 } } };
  }
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = "#dce3f2";
}

const AXIS_GRID = { color: "#1c2434", tickColor: "#5b657d" };
const AXIS_TICK = { color: "#8a95ad" };

/* crosshair: thin dashed vertical line at the hovered point */
const crosshairPlugin = {
  id: "iecnnCrosshair",
  afterDatasetsDraw(chart) {
    const act = chart.tooltip && chart.tooltip.getActiveElements &&
                chart.tooltip.getActiveElements();
    if (!act || !act.length) return;
    const x = act[0].element.x, { ctx, chartArea: a } = chart;
    if (!a) return;
    ctx.save();
    ctx.strokeStyle = "#4fc3f799"; ctx.lineWidth = 1;
    ctx.setLineDash([4, 3]);
    ctx.beginPath(); ctx.moveTo(x, a.top); ctx.lineTo(x, a.bottom); ctx.stroke();
    ctx.restore();
  },
};

/* keyboard access for charts (a11y): the canvas is focusable; ←/→ move a
   data cursor that shows the same crosshair + tooltip the mouse gets,
   Enter toggles the next series on/off (what a legend click does),
   Escape clears the cursor. Labels travel in aria-label, not color. */
function wireChartKeys(chart, title) {
  if (!chart || !chart.canvas) return;
  const cv = chart.canvas;
  cv.tabIndex = 0;
  cv.setAttribute("role", "img");
  cv.setAttribute("aria-label", `${title} — series: ` +
    (chart.data.datasets || []).map(d => d.label).filter(Boolean).join("; "));
  let cur = null, toggleIdx = 0;
  const setCursor = () => {
    chart.tooltip.setActiveElements(cur ?
      [{ datasetIndex: cur.di, index: cur.index }] : [], { x: 0, y: 0 });
    chart.update();
  };
  cv.addEventListener("keydown", e => {
    const nDS = chart.data.datasets.length;
    if (e.key === "ArrowRight" || e.key === "ArrowLeft") {
      e.preventDefault();
      const dir = e.key === "ArrowRight" ? 1 : -1;
      if (cur === null) {
        const di = chart.data.datasets.findIndex((d, k) =>
          (d.data || []).length && !chart.getDatasetMeta(k).hidden);
        if (di < 0) return;
        cur = { di, index: dir < 0 ? chart.data.datasets[di].data.length - 1 : 0 };
      } else {
        const n = chart.data.datasets[cur.di].data.length;
        cur.index = Math.min(n - 1, Math.max(0, cur.index + dir));
      }
      setCursor();
    } else if (e.key === "Enter" || e.key === " ") {
      e.preventDefault();                    // legend toggle via keyboard
      for (let tried = 0; tried < nDS; tried++, toggleIdx++) {
        const k = toggleIdx % nDS;
        const meta = chart.getDatasetMeta(k);
        if (chart.data.datasets[k].label) {
          meta.hidden = !meta.hidden;
          toggleIdx++;
          chart.update();
          return;
        }
      }
    } else if (e.key === "Escape") {
      cur = null; setCursor();
    }
  });
}

const HAS_ZOOM = typeof Chart !== "undefined" && typeof Hammer !== "undefined"
  && !!Chart.registry && !!(Chart.registry.getPlugin &&
    Chart.registry.getPlugin("zoom"));
const ZOOM_OPTS = HAS_ZOOM ? {
  zoom: { zoom: { wheel: { enabled: true, mode: "x" },
                  pinch: { enabled: true, mode: "x" }, mode: "x" },
          pan: { enabled: true, mode: "x" },
          limits: { x: { min: "original", max: "original" } } },
} : {};

/* ---------------- chart state (URL hash query + localStorage) -----------
   Canonical spec: server.py chart_state_parse/encode (round-trip tested in
   tests/test_server.py). KEEP THIS MIRROR IN SYNC. */
const CHART_XS = ["steps", "tokens", "wall", "compute"];
const CHART_STORE = "iecnn.chart";
function parseChartState(qs) {
  const st = { s: 0, y: "lin", x: "steps", r: [] };
  for (const pair of String(qs || "").split("&")) {
    const eq = pair.indexOf("=");
    if (eq < 0) continue;
    const k = pair.slice(0, eq), v = pair.slice(eq + 1);
    if (k === "s") {
      const n = parseInt(v, 10);
      if (Number.isInteger(n) && n >= 0 && n <= 5000) st.s = n;
    } else if (k === "y" && (v === "lin" || v === "log")) st.y = v;
    else if (k === "x" && CHART_XS.includes(v)) st.x = v;
    else if (k === "r")
      st.r = v.split(",").filter(t => /^[\w.\-]{1,120}$/.test(t)).slice(0, 64);
  }
  return st;
}
function encodeChartState(st) {
  const s = (Number.isInteger(st.s) && st.s >= 0 && st.s <= 5000) ? st.s : 0;
  const out = [`s=${s}`, `y=${st.y === "log" ? "log" : "lin"}`,
               `x=${CHART_XS.includes(st.x) ? st.x : "steps"}`];
  const r = (st.r || []).filter(t => /^[\w.\-]{1,120}$/.test(t)).slice(0, 64);
  if (r.length) out.push("r=" + r.join(","));
  return out.join("&");
}
/* persist smoothing/log/axis prefs (never the run selection) */
function saveChartPrefs(st) {
  try { localStorage.setItem(CHART_STORE, JSON.stringify(
        { s: st.s, y: st.y, x: st.x })); } catch { /* private mode */ }
}
/* trailing moving average over possibly-null series */
function movingAvg(vals, w) {
  if (!w || w < 2) return vals;
  const out = new Array(vals.length).fill(null);
  const q = [];
  let sum = 0, cnt = 0;
  for (let i = 0; i < vals.length; i++) {
    const v = vals[i];
    q.push(v);
    if (v != null && isFinite(v)) { sum += v; cnt++; }
    if (q.length > w) {
      const old = q.shift();
      if (old != null && isFinite(old)) { sum -= old; cnt--; }
    }
    out[i] = cnt ? sum / cnt : null;
  }
  return out;
}

function chartBox(title, data, type = "line", opts = {}) {
  const box = h("div", { class: "chartbox" },
    h("div", { class: "ct" }, h("h3", {}, title),
      opts.sub ? h("span", { class: "sub" }, opts.sub) : null));
  if (opts.toolbar) {
    const tb = h("div", { class: "charttools" });
    for (const b of opts.toolbar)
      tb.append(h("button", { class: "mini", title: b.title || "",
                              onclick: b.onclick }, b.label));
    box.querySelector(".ct").append(tb);
  }
  if (!HAS_CHART) {
    box.append(h("p", { class: "warn" }, "chart.umd.js failed to load — raw data only."));
    if (data.datasets) box.append(h("pre", {}, JSON.stringify(
      data.datasets.map(d => ({ label: d.label, n: (d.data || []).length })), null, 1)));
    return box;
  }
  const cv = h("canvas", {});
  box.append(cv);
  const axis = extra => ({ grid: AXIS_GRID, border: { color: "#26304a" },
    ticks: { ...AXIS_TICK }, ...extra });
  requestAnimationFrame(() => {
    try {
      const chart = new Chart(cv, { type, data,
        plugins: [crosshairPlugin],
        options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { display: opts.legend !== false,
                    position: opts.legendPos || "top" },
          tooltip: opts.tooltip || undefined,
          ...(HAS_ZOOM ? { zoom: ZOOM_OPTS.zoom } : {}),
        },
        scales: opts.scales || {
          x: axis({ type: opts.xtype || "category",
                    title: { display: !!opts.xlabel, text: opts.xlabel || "",
                             color: "#8a95ad" } }),
          y: axis({ type: opts.ytype || "linear",
                    title: { display: !!opts.ylabel, text: opts.ylabel || "",
                             color: "#8a95ad" } }),
        },
      }});
      box._chart = chart;
      wireChartKeys(chart, title);
      if (HAS_ZOOM) cv.addEventListener("dblclick", () => chart.resetZoom());
      if (opts.after) opts.after(chart);
    } catch (e) { box.append(h("p", { class: "err" }, "chart error: " + e)); }
  });
  return box;
}

/* tiny inline sparkline (no axes) — kept here for quick reuse */
function sparkCanvas(series, color) {
  const wrap = h("div", { class: "spw" }, h("canvas", {}));
  requestAnimationFrame(() => {
    if (!HAS_CHART) return;
    try {
      new Chart($("canvas", wrap), { type: "line", data: {
          labels: series.map((_, i) => i),
          datasets: [{ data: series, borderColor: color, borderWidth: 1.6,
                       pointRadius: 0, tension: .25, fill: false }] },
        options: { responsive: true, animation: false,
          maintainAspectRatio: false,
          plugins: { legend: { display: false }, tooltip: { enabled: true } },
          scales: { x: { display: false }, y: { display: false } } } });
    } catch (e) { /* sparkline is decorative */ }
  });
  return wrap;
}

/* ---------------- shared state ---------------- */
const state = { runsIndex: null, arch: null, sel: null, runData: {}, vizData: {} };

/* ---------------- router ---------------- */
const PAGES = {
  dashboard:    { fn: pageDashboard,    title: "Dashboard" },
  runs:         { fn: pageRuns,         title: "Training runs" },
  compare:      { fn: pageCompare,      title: "Compare" },
  scores:       { fn: pageScores,       title: "Re-score reports" },
  architecture: { fn: pageArchitecture, title: "Architecture" },
  checkpoints:  { fn: pageCheckpoints,  title: "Weights & dots" },
  diagnosis:    { fn: pageDiagnosis,    title: "Diagnosis" },
  vocab:        { fn: pageVocab,        title: "Vocab (UBM)" },
  corpus:       { fn: pageCorpus,       title: "Corpus & split" },
  playground:   { fn: pagePlayground,   title: "Playground" },
};
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const name = PAGES[parts[0]] ? parts[0] : "dashboard";
  $$(".nav").forEach(a => a.classList.toggle("active", a.dataset.page === name));
  $("#pgtitle").textContent = PAGES[name].title;
  const page = $("#page");
  page.innerHTML = "";
  page.setAttribute("aria-busy", "true");
  page.append(h("div", { class: "loading", role: "status" }, "· · ·"));
  Promise.resolve(PAGES[name].fn(page, parts[1])).then(() =>
    page.removeAttribute("aria-busy")).catch(e => {
    page.removeAttribute("aria-busy");
    page.innerHTML = "";
    page.append(h("h1", {}, PAGES[name].title),
      h("div", { class: "errbox" }, "Could not build this page: ",
        h("code", {}, String(e && e.message || e))),
      h("p", { class: "muted" }, "the API may be down or the committed data moved — reload, or check ", h("code", {}, "/api/runs"), "."));
  });
}
window.addEventListener("hashchange", route);

function errbox(root, e) {
  root.append(h("div", { class: "errbox" }, String(e && e.message || e)));
}
function warnbox(root, ...kids) {
  root.append(h("div", { class: "errbox", style: "color:var(--accent2);border-color:#ffb74d66;background:#ffb74d10" }, ...kids));
}
function card(txt, lbl, cls) {
  return h("div", { class: "card" + (cls ? " " + cls : "") },
    h("div", { class: "big" }, String(txt)), h("div", { class: "lbl" }, lbl));
}
/* link to the committed file a number came from (read-only evidence view) */
function rawLink(path) {
  return h("a", { href: `/api/raw?path=${encodeURIComponent(path)}`,
                  target: "_blank", rel: "noopener" }, path);
}
function sourceLine(sources) {
  if (!sources || !sources.length) return null;
  return h("div", { class: "muted srcs" }, "evidence: ",
    ...sources.flatMap((s, i) => {
      const link = /[*]/.test(s) ? h("span", {}, s)
        : rawLink(s);
      return i ? [", ", link] : [link];
    }));
}
/* provenance chips: run · step · tau · vocab sha · wall/compute */
function provChip(label, value, title) {
  return h("span", { class: "hpchip", title: title || "" }, label + " ",
    h("b", {}, String(value)));
}
const shaShort = s => (s ? String(s).slice(0, 12) + "…" : "—");

/* gate status -> pill class */
const GATE_CLS = { PASS: "ok", FAIL: "bad", PENDING: "warn",
                   PARTIAL: "warn", NO_DATA: "" };

function equalComputeTable(table) {
  return h("div", { class: "scrolltb" }, h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "wall s"), h("th", {}, "E0 plain-ensemble bpc (best so far)"),
      h("th", {}, "E1 weights+passes bpc (best so far)"), h("th", {}, "B0 GRU bpc (best so far)"))),
    h("tbody", {}, table.map(r => h("tr", { class:
        (r.e1_bpc != null && r.e0_bpc != null && r.e1_bpc <= r.e0_bpc) ? "best" : "" },
      h("td", { class: "num" }, String(r.wall_s)),
      h("td", { class: "num" }, fmt(r.e0_bpc, 5)),
      h("td", { class: "num" }, fmt(r.e1_bpc, 5)),
      h("td", { class: "num" }, r.b0_bpc != null ? fmt(r.b0_bpc, 5) : "—"))))));
}

/* ======================================================== DASHBOARD ======= */
/* Gates scoreboard G1–G4 — every status and number is computed server-side
   at request time from committed files (results/*.csv, eval/summary/viz
   JSONs). Nothing here is hardcoded. */

function gateDetails(g) {
  const det = h("details", {}, h("summary", {}, "show the numbers"));
  if (g.id === "G2" && Array.isArray(g.table) && g.table.length) {
    det.append(h("p", { class: "muted" },
      "equal-compute table: best bpc_val each run had achieved by each matched wall-time point (lower is better; green = E1 at or ahead of E0 at that wall point)"),
      equalComputeTable(g.table));
  }
  if (g.id === "G3" && Array.isArray(g.vocabs)) {
    det.append(h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "vocab"), h("th", {}, "best bpc_val"),
        h("th", {}, "at step"), h("th", {}, "last-3-eval Δ"), h("th", {}, "state"),
        h("th", {}, "stop kind"))),
      h("tbody", {}, g.vocabs.map(v => h("tr", {},
        h("td", {}, v.vocab),
        h("td", { class: "num" }, fmt(v.best_bpc, 5)),
        h("td", { class: "num" }, String(v.best_step ?? "—")),
        h("td", { class: "num" }, fmt(v.last3_delta, 5)),
        h("td", {}, v.status ?? "—"),
        h("td", {}, v.stop_kind ?? "—"))))));
    if (g.vocabs.some(v => v.note)) det.append(h("p", { class: "muted" },
      g.vocabs.find(v => v.note).note));
  }
  if (g.id === "G4" && Array.isArray(g.exports)) {
    det.append(h("p", { class: "muted" },
      "horse race per export (same 255-position sample): pooled learned consensus vs dumb proxies that need no learned weights. Oracle = some dot held the answer (any pass) — context, not the pass bar."),
      h("table", {},
        h("thead", {}, h("tr", {}, h("th", {}, "export"), h("th", {}, "pooled top-1"),
          h("th", {}, "uniform top-5 mix"), h("th", {}, "plurality vote"),
          h("th", {}, "best single dot"), h("th", {}, "oracle top-1"),
          h("th", {}, "outvote rate"), h("th", {}, "status"))),
        h("tbody", {}, g.exports.map(e => h("tr", { class: e.status === "PASS" ? "best" : "" },
          h("td", {}, e.name),
          h("td", { class: "num" }, `${e.pooled_top1}/${e.sampled}`),
          h("td", { class: "num" }, `${e.uniform_mix_top1}/${e.sampled}`),
          h("td", { class: "num" }, `${e.plurality_top1}/${e.sampled}`),
          h("td", { class: "num" }, `${e.best_single_dot_top1}/${e.sampled}`),
          h("td", { class: "num" }, `${e.oracle_top1}/${e.sampled}`),
          h("td", { class: "num" }, e.outvote_rate != null ? fmt(e.outvote_rate, 3) : "—"),
          h("td", {}, h("span", { class: "pill " + (e.status === "PASS" ? "ok" : "bad") }, e.status)))))));
  }
  if (g.counts) det.append(h("p", { class: "muted" },
    `verified ${g.counts.verified} · unverified ${g.counts.unverified} · retracted (excluded) ${g.counts.retracted}`));
  if (g.note) det.append(h("p", { class: "muted" }, g.note));
  return det;
}

function gateCard(g) {
  const cls = GATE_CLS[g.status] || "";
  const c = h("div", { class: "card gate" },
    h("div", { class: "ghd" },
      h("span", { class: "gid" }, g.id),
      h("span", { class: "pill " + cls }, g.status)),
    h("div", { class: "gt" }, g.title),
    h("ul", { class: "gev" }, (g.evidence || []).map(e => h("li", {}, e))));
  const src = sourceLine(g.sources);
  if (src) c.append(src);
  const det = gateDetails(g);
  if (det.childNodes.length <= 1) det.append(
    h("p", { class: "muted" }, "no extra numbers for this gate yet"));
  c.append(det);
  return c;
}

async function pageDashboard(root) {
  const [gates, runs] = await Promise.all([api("/api/gates"), api("/api/runs")]);
  state.runsIndex = runs;
  root.innerHTML = "";
  root.append(h("h1", {}, "Dashboard"),
    h("p", {}, "Gates scoreboard — every status and number below is computed at request time from committed files (results/ CSVs + eval/summary/viz JSONs), never hardcoded. ",
      "No training happens on this machine — all torch compute runs on GitHub Actions."));
  if (!runs.runs.length) {
    warnbox(root, "No committed training CSVs in results/ yet — expected files look like results/train_<rung>_seed0.csv. ",
      "Dispatch a run from the runner pool: ",
      h("code", {}, "gh workflow run train.yml -f config=e0_plain_ensemble.yaml"),
      " (the workflow commits the CSV back to results/).");
  }

  // headline: E1 vs E0 at equal wall compute (from G2, same numbers as the table)
  const g2 = (gates.gates || []).find(g => g.id === "G2");
  if (g2 && g2.headline) {
    root.append(h("div", { class: "headline " + (GATE_CLS[g2.status] || "") },
      h("span", { class: "pill " + (GATE_CLS[g2.status] || "") }, g2.status),
      " ", h("b", {}, "E1 vs E0 at equal compute — ", g2.headline.replace(/^at equal wall compute[^:]*: /, "")),
      " ",
      h("a", { href: "#/runs/train_e1_weights_passes_eqc_seed0" }, "open curves →")));
  }

  const grid = h("div", { class: "gates" });
  for (const g of gates.gates || []) grid.append(gateCard(g));
  root.append(grid);
  if (gates.computed_at) root.append(h("div", { class: "muted" },
    `gates computed ${gates.computed_at} · git ${esc(gates.git_head || "")}`));

  // all runs table
  const tbl = h("table", {},
    h("thead", {}, h("tr", {}, h("th", {}, "run"), h("th", {}, "seed"),
      h("th", {}, "eval pts"), h("th", {}, "last step"),
      h("th", {}, "best val (sample)"), h("th", {}, "best step"))),
    h("tbody", {}, runs.runs.map(r => h("tr", {},
      h("td", {}, h("a", { href: `#/runs/${r.name}` }, r.name)),
      h("td", { class: "num" }, String(r.seed ?? "—")),
      h("td", { class: "num" }, String(r.points)),
      h("td", { class: "num" }, String(r.last_step ?? "—")),
      h("td", { class: "num" }, fmt(r.best_val)),
      h("td", { class: "num" }, String(r.best_step ?? "—"))))));
  root.append(h("h2", {}, "All runs in results/"), tbl);
}

/* ============================================================= RUNS ======= */
async function pageRuns(root, preselect) {
  if (!state.runsIndex) state.runsIndex = await api("/api/runs");
  const runs = state.runsIndex.runs;

  // ---- chart state: URL hash query wins, then saved prefs, then defaults
  const hashBody = location.hash.replace(/^#\/runs\/?/, "");
  const [deepRun, hashQ] = hashBody.split("?");
  const st = parseChartState(hashQ || "");
  if (!hashQ) {                       // nothing in the URL: saved prefs
    try {
      const saved = JSON.parse(localStorage.getItem(CHART_STORE) || "{}");
      st.s = Number.isInteger(saved.s) ? saved.s : 0;
      if (saved.y === "log") st.y = "log";
      if (CHART_XS.includes(saved.x)) st.x = saved.x;
    } catch { /* fresh browser */ }
  }
  // run selection: r= in the URL wins; deep-linked run joins the default set
  if (st.r.length) state.sel = new Set(st.r.filter(n => runs.some(r => r.name === n)));
  else if (!state.sel) {
    state.sel = new Set(runs.filter(r => /pilot/.test(r.name)).map(r => r.name));
    if (deepRun && runs.some(r => r.name === deepRun)) state.sel.add(deepRun);
  }
  const syncState = () => {           // URL + localStorage, no page rebuild
    st.r = [...state.sel];
    saveChartPrefs(st);
    history.replaceState(null, "", `#/runs?${encodeChartState(st)}`);
  };

  root.innerHTML = "";
  root.append(h("h1", {}, "Training runs"),
    h("p", {}, "Curves come from the committed results/*.csv. Click chips to toggle runs. ",
      "Scroll to zoom a chart's x-axis, drag to pan, double-click to reset. Charts are keyboard-accessible: focus a chart and use ←/→ to move a data cursor, Enter to toggle a series, Esc to clear. Grad-norm columns (gn_*) only exist on runs trained after 2026-11-13 — the chart appears automatically when the selected runs have them."));
  if (!runs.length) {
    warnbox(root, "No committed training CSVs in results/ yet. ",
      "Dispatch one: ",
      h("code", {}, "gh workflow run train.yml -f config=e1_weights_passes_pilot.yaml"),
      " — the workflow commits results/train_<rung>_seed<seed>.csv and this page fills in.");
    return;
  }

  const chips = h("div", { class: "chips" });
  for (const r of runs) {
    const on = state.sel.has(r.name);
    const chip = h("button", { class: "chip" + (on ? " on" : ""), "aria-pressed": String(on), onclick: () => {
      state.sel.has(r.name) ? state.sel.delete(r.name) : state.sel.add(r.name);
      chip.classList.toggle("on");
      chip.setAttribute("aria-pressed", chip.classList.contains("on"));
      syncState();
      draw();
    }}, h("span", { class: "dot", style: `background:${rungColor(r.rung)}` }),
      `${r.rung} · s${r.seed}`, h("span", { class: "n" }, `best ${fmt(r.best_val)}`));
    chips.append(chip);
  }
  root.append(chips);

  // ---- global chart toolbar: smoothing · log-y · x-axis · link · CSV
  const smoothLbl = h("span", { class: "muted" }, st.s >= 2 ? `window ${st.s}` : "off");
  const smooth = h("input", { type: "range", min: "0", max: "50", value: String(st.s),
    "aria-label": "smoothing window (moving average over eval points)" });
  const yBtn = h("button", { class: "mini" + (st.y === "log" ? " on" : ""),
    "aria-pressed": String(st.y === "log"), title: "logarithmic y-axis on the bpc charts" }, "log y");
  const xSel = h("select", { "aria-label": "x-axis mode" },
    ...CHART_XS.map(m => h("option", { value: m, selected: m === st.x ? "" : null },
      m === "steps" ? "steps" : m === "tokens" ? "tokens (step × batch × seq_len)"
      : m === "wall" ? "wall time (elapsed_s)" : "est. compute (wall-normalized)")));
  const copyBtn = h("button", { class: "mini", title: "copy a link to this exact chart state" }, "copy link");
  const csvBtn = h("button", { class: "mini", title: "download the visible series as CSV" }, "export CSV");
  const resetBtn = h("button", { class: "mini", title: "reset zoom on all charts" }, "reset zoom");
  const xNote = h("span", { class: "muted" });
  root.append(h("div", { class: "toolbar" },
    h("label", { class: "muted" }, "smoothing ", smooth, smoothLbl),
    yBtn, h("label", { class: "muted" }, "x-axis ", xSel, xNote),
    copyBtn, csvBtn, resetBtn));

  const plots = h("div", {});
  root.append(plots);
  const registry = [];                        // {chartBox, title, seriesPts}
  let lastReq = 0;

  const applyStateToUi = () => {
    smooth.value = String(st.s);
    smoothLbl.textContent = st.s >= 2 ? `window ${st.s}` : "off";
    yBtn.classList.toggle("on", st.y === "log");
    yBtn.setAttribute("aria-pressed", String(st.y === "log"));
    xSel.value = st.x;
  };

  smooth.addEventListener("input", () => {
    st.s = +smooth.value; smoothLbl.textContent = st.s >= 2 ? `window ${st.s}` : "off";
    saveChartPrefs(st); syncState(); draw();
  });
  yBtn.addEventListener("click", () => {
    st.y = st.y === "log" ? "lin" : "log"; saveChartPrefs(st); syncState(); draw();
  });
  xSel.addEventListener("change", () => {
    st.x = xSel.value; saveChartPrefs(st); syncState(); draw();
  });
  copyBtn.addEventListener("click", async () => {
    const url = `${location.origin}${location.pathname}#/runs?${encodeChartState({ ...st, r: [...state.sel] })}`;
    try { await navigator.clipboard.writeText(url); copyBtn.textContent = "copied ✓"; }
    catch { copyBtn.textContent = "copy failed"; }
    setTimeout(() => { copyBtn.textContent = "copy link"; }, 1600);
  });
  resetBtn.addEventListener("click", () => {
    for (const r of registry) { const c = r.box._chart; if (c) c.resetZoom(); }
  });
  csvBtn.addEventListener("click", () => {
    const rows = ["chart,series,x,y"];
    for (const reg of registry) {
      if (!reg.seriesPts) continue;
      for (const s of reg.seriesPts)
        for (const p of s.pts)
          rows.push(`${JSON.stringify(reg.title)},${JSON.stringify(s.label)},${p.x},${p.y}`);
    }
    const blob = new Blob([rows.join("\n") + "\n"], { type: "text/csv" });
    const a = h("a", { href: URL.createObjectURL(blob),
      download: "iecnn-curves.csv" });
    document.body.append(a); a.click(); a.remove();
  });

  // tokens/step needs batch_size × seq_len: checkpoint config_used.json
  // first, then the rung config from /api/arch
  const [cfgUsed, archIdx] = await Promise.all([
    api("/api/config_used").catch(() => ({})),
    api("/api/arch").catch(() => null),
  ]);
  const tokensPerStep = d => {
    const cu = cfgUsed[d.name || ""];
    if (cu && cu.training && cu.model &&
        cu.training.batch_size && cu.model.seq_len)
      return cu.training.batch_size * cu.model.seq_len;
    const c = archIdx && archIdx.resolved && archIdx.resolved[d.rung];
    if (c && c.config && c.config.training && c.config.model &&
        c.config.training.batch_size && c.config.model.seq_len)
      return c.config.training.batch_size * c.config.model.seq_len;
    return null;
  };

  async function draw() {
    const req = ++lastReq;
    applyStateToUi();
    const names = runs.filter(r => state.sel.has(r.name)).map(r => r.name);
    plots.innerHTML = "";
    registry.length = 0;
    if (!names.length) { plots.append(h("p", { class: "muted" }, "no runs selected")); return; }
    // fetch only runs not already cached
    const missing = names.filter(n => !state.runData[n]);
    if (missing.length) {
      try {
        const got = await Promise.all(missing.map(n => api(`/api/run/${n}`)));
        missing.forEach((n, i) => { state.runData[n] = got[i]; });
      } catch (e) { errbox(plots, e); return; }
    }
    const datasets = names.map(n => state.runData[n]).filter(Boolean);
    if (req !== lastReq) return;             // a newer selection superseded us

    const tpsOf = Object.fromEntries(datasets.map(d => [d.name, tokensPerStep(d)]));
    const tpsKnown = Object.values(tpsOf).every(v => v);
    xNote.textContent =
      st.x === "tokens" && !tpsKnown ? " — batch/seq_len unknown for a selected run; falling back to steps" : "";
    const useX = st.x === "tokens" && !tpsKnown ? "steps" : st.x;

    // x value for point i of run d under the current axis mode
    const normMax = Math.max(1e-9, ...datasets.map(d =>
      Math.max(1e-9, ...(d.series.elapsed_s || []).filter(v => v != null))));
    const xOf = (d, i) => {
      const step = d.series.step[i];
      if (useX === "steps") return step;
      if (useX === "tokens") return step * (tpsOf[d.name] || 0);
      const w = (d.series.elapsed_s || [])[i];
      if (w == null) return null;
      return useX === "wall" ? w : w / normMax;   // est. compute (normalized)
    };
    const points = (d, key) => {
      const ys = movingAvg(d.series[key] || [], st.s);
      const pts = [];
      for (let i = 0; i < ys.length; i++) {
        const x = xOf(d, i), y = ys[i];
        if (x == null || y == null || !isFinite(x) || !isFinite(y)) continue;
        pts.push({ x, y });
      }
      return pts;
    };
    const rawPoints = (d, key) => {
      const pts = [];
      for (let i = 0; i < (d.series[key] || []).length; i++) {
        const x = xOf(d, i), y = d.series[key][i];
        if (x == null || y == null || !isFinite(x)) continue;
        pts.push({ x, y });
      }
      return pts;
    };
    const xlabel = useX === "steps" ? "step" : useX === "tokens"
      ? "tokens (step × batch × seq_len)" : useX === "wall"
      ? "wall time (elapsed_s)" : "est. compute (wall-normalized)";

    const mkBox = (title, seriesSpecs, ylabel, extra = {}) => {
      // seriesSpecs: [{label, pts, rawPts?, color, dash?, hidden?, yAxisID?}]
      const box = h("div", { class: "chartbox" });
      const ct = h("div", { class: "ct" }, h("h3", {}, title),
        extra.sub ? h("span", { class: "sub" }, extra.sub) : null);
      const tb = h("div", { class: "charttools" });
      tb.append(h("button", { class: "mini", title: "download this chart as PNG",
        onclick: () => {
          const c = box._chart; if (!c) return;
          c.canvas.toBlob(b => {
            const a = h("a", { href: URL.createObjectURL(b),
              download: title.replace(/[^\w\-]+/g, "_") + ".png" });
            document.body.append(a); a.click(); a.remove();
          });
        } }, "PNG"));
      if (HAS_ZOOM) tb.append(h("button", { class: "mini", title: "reset zoom (or double-click the chart)",
        onclick: () => { const c = box._chart; if (c) c.resetZoom(); } }, "⟲"));
      ct.append(tb);
      box.append(ct);
      if (!HAS_CHART) {
        box.append(h("p", { class: "warn" }, "chart.umd.js failed to load — raw data only."));
        plots.append(box); return;
      }
      const cv = h("canvas", {});
      box.append(cv);
      // log-y only when every series value is strictly positive
      let yType = extra.yType || "linear";
      if (st.y === "log" && !extra.forceLinear) {
        const all = seriesSpecs.flatMap(s => s.pts.map(p => p.y));
        yType = (all.length && all.every(v => v > 0)) ? "logarithmic"
          : "linear";
        if (yType !== "logarithmic" && st.y === "log" && !extra.noLogNote)
          box.querySelector(".ct").append(h("span", { class: "sub warn" },
            "log y unavailable — series contains zeros"));
      }
      const datasetsCh = seriesSpecs.flatMap(s => {
        const ds = [{ label: s.label, data: s.pts,
          borderColor: s.color, backgroundColor: s.color,
          borderDash: s.dash || [], borderWidth: s.width || 1.8,
          pointRadius: 0, tension: 0.15, spanGaps: true,
          hidden: !!s.hidden, yAxisID: s.yAxisID || "y" }];
        if (s.rawPts) ds.push({ label: s.label + " (raw)", data: s.rawPts,
          borderColor: s.color + "44", borderWidth: 1, pointRadius: 0,
          tension: 0.15, spanGaps: true, hidden: false,
          yAxisID: s.yAxisID || "y" });
        return ds;
      });
      const scales = {
        x: { type: "linear", grid: AXIS_GRID, border: { color: "#26304a" },
             ticks: { ...AXIS_TICK, maxTicksLimit: 10 },
             title: { display: true, text: xlabel, color: "#8a95ad" } },
        y: { type: yType, grid: AXIS_GRID, border: { color: "#26304a" },
             ticks: AXIS_TICK,
             title: { display: true, text: ylabel, color: "#8a95ad" } },
      };
      if (extra.rightAxis) scales.y1 = { position: "right", type: "linear",
        grid: { drawOnChartArea: false }, ...AXIS_GRID,
        title: { display: true, text: extra.rightAxis, color: "#8a95ad" } };
      requestAnimationFrame(() => {
        try {
          const chart = new Chart(cv, { type: "line",
            data: { datasets: datasetsCh },
            plugins: [crosshairPlugin],
            options: { responsive: true, maintainAspectRatio: false,
              interaction: { mode: "nearest", intersect: false },
              plugins: { legend: { display: extra.legend !== false,
                                   labels: { usePointStyle: true, boxWidth: 10 } },
                         tooltip: extra.tooltip || undefined,
                         ...(HAS_ZOOM ? { zoom: ZOOM_OPTS.zoom } : {}) },
              scales } });
          box._chart = chart;
          wireChartKeys(chart, title);
          if (HAS_ZOOM) cv.addEventListener("dblclick", () => chart.resetZoom());
        } catch (e) { box.append(h("p", { class: "err" }, "chart error: " + e)); }
      });
      registry.push({ box, title, seriesPts: seriesSpecs.map(
        s => ({ label: s.label, pts: s.pts })) });
      plots.append(box);
    };

    const N = datasets.length;
    // main: validation vs train bpc
    mkBox("Validation bpc (solid) vs train bpc (dashed)", datasets.flatMap((d, i) => {
      const c = rungColor(d.rung), lbl = `${d.rung}·s${d.seed}`;
      return [
        { label: `val ${lbl}`, pts: points(d, "bpc_val"),
          rawPts: st.s >= 2 ? rawPoints(d, "bpc_val") : null, color: c },
        { label: `train ${lbl}`, pts: points(d, "bpc_train"),
          rawPts: st.s >= 2 ? rawPoints(d, "bpc_train") : null, color: c,
          dash: [4, 4], hidden: N > 2 },
      ];
    }), "bpc");
    // per-pass
    const has = (d, k) => Array.isArray(d.series?.[k]) && d.series[k].some(v => v !== null);
    const pp = datasets.filter(d => has(d, "bpc_p1"));
    if (pp.length) {
      mkBox("Per-pass bpc (later passes should beat earlier ones)", pp.flatMap((d, i) => [
        { label: `p1 ${d.rung}`, pts: points(d, "bpc_p1"), color: "#7986cb",
          dash: DASHES[i % DASHES.length] },
        { label: `p2 ${d.rung}`, pts: points(d, "bpc_p2"), color: "#4fc3f7",
          dash: DASHES[i % DASHES.length] },
        { label: `p3 ${d.rung}`, pts: points(d, "bpc_p3"), color: "#ffd54f",
          dash: DASHES[i % DASHES.length] },
      ]), "bpc", { sub: "pass color · dash pattern per run" });
    }
    // tau0 + weight entropy
    const ie = datasets.filter(d => has(d, "tau0"));
    if (ie.length) {
      mkBox("τ0 (left) and weight entropy (right axis)", ie.flatMap((d, i) => [
        { label: `tau0 ${d.rung}`, pts: points(d, "tau0"), color: rungColor(d.rung),
          dash: DASHES[i % DASHES.length] },
        { label: `w-entropy ${d.rung}`, pts: points(d, "weight_entropy"),
          color: rungColor(d.rung), dash: [4, 4], yAxisID: "y1" },
      ]), "tau0", { rightAxis: "entropy (nats)" });
    }
    // grad norms (only when the columns exist)
    const gn = datasets.filter(d => has(d, "gn_total"));
    if (gn.length) {
      mkBox("Pre-clip gradient norms per component", gn.flatMap((d, i) =>
        ["gn_emb", "gn_regions", "gn_chain", "gn_gate", "gn_messages", "gn_other"]
          .filter(k => has(d, k)).map(k => ({
            label: `${d.rung} ${k.slice(3)}`, pts: points(d, k),
            color: rungColor(d.rung),
            dash: DASHES[i % DASHES.length] }))), "L2 norm",
        { sub: "component color · dash pattern per run" });
    }

    // best-point table for the selection
    const bt = h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "run"), h("th", {}, "points"),
        h("th", {}, "best val (sample)"), h("th", {}, "at step"), h("th", {}, "last step"))),
      h("tbody", {}, datasets.map(d => {
        const vals = d.series.bpc_val.map((v, i) => [v, i]).filter(([v]) => v !== null);
        const b = vals.reduce((a, x) => (x[0] < a[0] ? x : a), [Infinity, -1]);
        return h("tr", {},
          h("td", {}, `${d.rung} · s${d.seed}`),
          h("td", { class: "num" }, String(d.n_points)),
          h("td", { class: "num" }, fmt(b[0])),
          h("td", { class: "num" }, String(d.series.step[b[1]] ?? "—")),
          h("td", { class: "num" }, String(d.series.step[d.series.step.length - 1] ?? "—")));
      })));
    plots.append(h("h2", {}, "Best points"), bt);

    // matched-step table between the first two selected runs
    if (datasets.length >= 2) {
      const A = datasets[0], B = datasets[1];
      const stepsA = new Map(A.series.step.map((s, i) => [s, i]));
      const rows = [];
      B.series.step.forEach((s, j) => {
        const i = stepsA.get(s);
        if (i === undefined || A.series.bpc_val[i] === null || B.series.bpc_val[j] === null) return;
        const va = A.series.bpc_val[i], vb = B.series.bpc_val[j];
        rows.push(h("tr", { class: vb < va ? "best" : "" },
          h("td", { class: "num" }, String(s)),
          h("td", { class: "num" }, fmt(va)),
          h("td", { class: "num" }, fmt(vb)),
          h("td", { class: "num" }, fmt(vb - va, 3))));
      });
      plots.append(h("h2", {}, `Matched-step: ${names[0]} vs ${names[1]}`),
        h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "step"),
          h("th", {}, names[0]), h("th", {}, names[1]), h("th", {}, "Δ (B−A)"))),
          h("tbody", {}, rows)));
    }
  }
  syncState();
  await draw();
}

/* =========================================================== COMPARE ===== */
async function pageCompare(root) {
  if (!state.runsIndex) state.runsIndex = await api("/api/runs");
  const runs = state.runsIndex.runs;
  root.innerHTML = "";
  root.append(h("h1", {}, "Compare runs"),
    h("p", {}, "Overlay two runs' bpc_val curves and diff them at the steps they share. ",
      "Every number is computed at request time from the committed CSVs in results/."));
  if (runs.length < 2) {
    warnbox(root, "Need at least two committed training CSVs in results/ to compare. ",
      "Runs land there via the training workflow: ", h("code", {}, "gh workflow run train.yml -f config=<rung>.yaml"));
    return;
  }

  // seed A/B from the hash query (#/compare?a=..&b=..) or the eqc pair
  const qp = new URLSearchParams(location.hash.split("?")[1] || "");
  const nameOf = v => runs.some(r => r.name === v) ? v : null;
  const defaultA = nameOf(qp.get("a")) ||
    (runs.find(r => r.name === "train_e0_plain_ensemble_eqc_seed0")?.name) || runs[0].name;
  const defaultB = nameOf(qp.get("b")) ||
    (runs.find(r => r.name === "train_e1_weights_passes_eqc_seed0")?.name) ||
    (runs.find(r => r.name !== defaultA) || runs[1]).name;

  const selA = h("select", {}, runs.map(r =>
    h("option", { value: r.name, selected: r.name === defaultA ? "" : null }, r.name)));
  const selB = h("select", {}, runs.map(r =>
    h("option", { value: r.name, selected: r.name === defaultB ? "" : null }, r.name)));
  const swap = h("button", { title: "swap A and B" }, "⇄ swap");
  const view = h("div", {});
  root.append(h("div", { class: "toolbar" },
    h("label", { class: "muted" }, "A ", selA), swap,
    h("label", { class: "muted" }, "B ", selB)), view);

  const show = async () => {
    view.innerHTML = "";
    view.append(h("div", { class: "loading" }, "· · ·"));
    let d;
    try { d = await api(`/api/compare?a=${encodeURIComponent(selA.value)}&b=${encodeURIComponent(selB.value)}`); }
    catch (e) { view.innerHTML = ""; errbox(view, e); return; }
    view.innerHTML = "";
    if (d.error) { errbox(view, d.error); return; }
    syncHash();
    // vocab honesty banner
    if (d.vocab_mismatch === true) {
      warnbox(view, h("b", {}, "Different vocabularies — compare with care. "),
        "A runs ", h("code", {}, d.a.vocab_file || d.a.vocab_sha256?.slice(0, 12) + "…"),
        " while B runs ", h("code", {}, d.b.vocab_file || d.b.vocab_sha256?.slice(0, 12) + "…"),
        " (declared in configs / eval vocab_sha256). bpc across different tokenizers is NOT directly comparable.");
    } else if (d.vocab_mismatch === null) {
      warnbox(view, "vocab identity unknown for at least one of these runs (no eval vocab_sha256 and the committed configs declare no vocab_file) — the curves are plotted without a vocab claim.");
    } else {
      view.append(h("p", { class: "legend" }, "same declared vocabulary (",
        h("code", {}, d.a.vocab_file || d.a.vocab_sha256?.slice(0, 12) + "…"), ") — curves are comparable."));
    }
    if (!d.n_matched) {
      warnbox(view, "These two runs share no eval steps — no overlay or delta view is possible. ");
      return;
    }
    // overlay chart: bpc_val vs step (B dashed)
    view.append(chartBox("bpc_val overlay — A solid, B dashed", {
      datasets: [
        { label: `A ${d.a.name}`, data: d.a.points.map(([s, v]) => ({ x: s, y: v })),
          borderColor: "#4fc3f7", pointRadius: 2.5, tension: 0.15 },
        { label: `B ${d.b.name}`, data: d.b.points.map(([s, v]) => ({ x: s, y: v })),
          borderColor: "#ffb74d", borderDash: [6, 4], pointRadius: 2.5, tension: 0.15 },
      ],
    }, "line", {
      sub: `at the ${d.n_matched} shared eval step${d.n_matched === 1 ? "" : "s"}`,
      scales: { x: { type: "linear", ...AXIS_GRID, border: { color: "#26304a" },
                     title: { display: true, text: "step", color: "#8a95ad" } },
                y: { ...AXIS_GRID, border: { color: "#26304a" },
                     title: { display: true, text: "bpc", color: "#8a95ad" } } } }));
    // summary
    view.append(h("div", { class: "cards" },
      card(fmt(d.mean_a, 5), "mean bpc_val over matched steps — A"),
      card(fmt(d.mean_b, 5), "mean bpc_val over matched steps — B"),
      card(fmt(d.mean_delta, 5),
        `mean Δ (B−A) — ${d.better === "b" ? "B lower (better)" : d.better === "a" ? "A lower (better)" : "tie"}`,
        d.better === "b" ? "" : d.better === "a" ? "" : "")));
    // delta lollipop/bar per matched step
    view.append(chartBox("Δ bpc at matched steps (B − A; below 0 = B better)", {
      datasets: [{ label: "Δ (B−A)",
        data: d.deltas.map(t => ({ x: t.step, y: t.delta })),
        backgroundColor: d.deltas.map(t => t.delta <= 0 ? "#81c784cc" : "#e57373cc"),
        borderColor: d.deltas.map(t => t.delta <= 0 ? "#81c784" : "#e57373"),
        borderWidth: 1, borderRadius: 4, maxBarThickness: 26 }],
    }, "bar", {
      sub: `matched steps: ${d.matched_steps.join(", ")}`,
      scales: { x: { type: "linear", ...AXIS_GRID, border: { color: "#26304a" },
                     title: { display: true, text: "step", color: "#8a95ad" } },
                y: { ...AXIS_GRID, border: { color: "#26304a" },
                     title: { display: true, text: "Δ bpc (B−A)", color: "#8a95ad" } } },
      legend: false }));
    // matched table
    view.append(h("h2", {}, "Matched steps"),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "step"),
        h("th", {}, `A — ${d.a.name}`), h("th", {}, `B — ${d.b.name}`), h("th", {}, "Δ (B−A)"))),
        h("tbody", {}, d.deltas.map(t => h("tr", { class: t.delta <= 0 ? "best" : "" },
          h("td", { class: "num" }, String(t.step)),
          h("td", { class: "num" }, fmt(t.a)),
          h("td", { class: "num" }, fmt(t.b)),
          h("td", { class: "num" }, fmt(t.delta, 3)))))));
  };
  const syncHash = () => history.replaceState(null, "",
    `#/compare?a=${encodeURIComponent(selA.value)}&b=${encodeURIComponent(selB.value)}`);
  selA.addEventListener("change", show);
  selB.addEventListener("change", show);
  swap.addEventListener("click", () => {
    const t = selA.value; selA.value = selB.value; selB.value = t; show();
  });
  await show();
}

/* ===================================================== ARCHITECTURE ======= */
const NS = "http://www.w3.org/2000/svg";
function svgEl(tag, attrs = {}, text) {
  const el = document.createElementNS(NS, tag);
  for (const [k, v] of Object.entries(attrs)) el.setAttribute(k, v);
  if (text !== undefined) el.textContent = text;
  return el;
}

function gnode(x, y, w, hh, title, shape, sub, key, container) {
  const g = svgEl("g", { class: "gnode", "data-key": key });
  g.append(svgEl("rect", { class: container ? "container" : "node",
    x: x - w / 2, y: y - hh / 2, width: w, height: hh, rx: 9 }));
  const lines = [[title, "title", 0]];
  if (shape) lines.push([shape, "shape", 16]);
  if (sub) lines.push([sub, "sub", shape ? 29 : 16]);
  const n = lines.length;
  lines.forEach(([t, cls, dy], i) => {
    const base = container ? y - hh / 2 + 16 : y + 2 - (n - 1) * 7;
    g.append(svgEl("text", { class: cls, x, y: base + dy,
      "text-anchor": "middle" }, t));
  });
  return g;
}
function edge(x1, y1, x2, y2, label, cls) {
  const g = svgEl("g");
  const midY = (y1 + y2) / 2;
  const d = (y2 > y1)
    ? `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2 - 5}`
    : `M ${x1} ${y1} C ${x1} ${midY}, ${x2} ${midY}, ${x2} ${y2 + 5}`;
  g.append(svgEl("path", { class: "gedge" + (cls ? " " + cls : ""), d }));
  if (label) g.append(svgEl("text", { class: "gedge label", x: (x1 + x2) / 2,
    y: midY - 4, "text-anchor": "middle" }, label));
  return g;
}

const ARCH_DETAIL = [
  { key: "tok", t: "Text → tokens (UBM)",
    d: "NFC-normalize, strip boilerplate, encode with the fixed 512-base UBM vocab (256 bytes + 8 specials + 248 learned merges).",
    shape: "chars → ids [T] (+ expansion lengths for bpc accounting)",
    what: "One fixed vocabulary for every rung: comparisons never mix tokenizers. Each token credits the characters its decode expands to, so Σ expansion lengths = text length exactly.",
    formula: "token tₖ ← longest merge path in UBM;  bpc = NLL / Σ expansion_len(t̂)",
    knobs: "UBM/vocab_v1.json (sha256 recorded in every checkpoint; mismatch fails loudly)" },
  { key: "embed", t: "Embedding (tied)",
    d: "Every token id → a d-dim vector; the SAME table is reused as the output projection.",
    shape: "E [V=512, d] → rows = E(ids) [B, T, d]",
    what: "Tied embeddings keep the parameter count small and let every dot's readout point at real token vectors.",
    formula: "rows = E(ids)",
    knobs: "model.d (d_q, d_v for query/value projections)" },
  { key: "region", t: "Region: window memory + N dots",
    d: "Each position keeps a W-token working window (span); N independent dots each run a gated state update from that window.",
    shape: "rows [B, T, d] → states s [B, T, N, d_s]",
    what: "A dot is a tiny gated reader with its OWN parameters for this round (pass) — dots are the independent 'experts' whose disagreement the consensus machinery harvests. Region 0 uses the first entry of model.spans as its window.",
    formula: "z = σ(W_z x + U_z s);  s' = (1−z)·tanh(W n + U r s) + z·s",
    knobs: "model.n_regions, n_dots, spans[0] (window), d_s; dot_dropout" },
  { key: "pass", t: "Dot heads: prediction, confidence, trust",
    d: "Each dot emits its own full next-token distribution plus two scalar quality signals.",
    shape: "log πₖ [B, T, N, V];  γₖ [B, T, N];  trustₖ [B, T, N]",
    what: "πₖ = the dot's private prediction (through the tied E). γₖ = self-rated confidence. The trust term is the dot's raw vote entering the score below.",
    formula: "logitsₖ = (P·sₖ)·Eᵀ;  log πₖ = log softmax(logitsₖ);  γₖ = c_w·sₖ + c_b;  trustₖ = g_w·sₖ + g_b",
    knobs: "shared per round; params for round p are indexed by pass (share_pass_params)" },
  { key: "consensus", t: "Consensus: scores → trust → weights → pooled opinion",
    d: "Turn the N private predictions into N mixing weights, softly, every pass, then pool log-linearly.",
    shape: "weights w [B, T, N] → pooled log-b [B, T, V]",
    what: "Score each dot by how much it DISAGREES with the leave-one-out crowd (novelty) plus its confidence γ and its trust carry-over. softmax with temperature τ_p mixes. The pooled mixture is the region's opinion of the world this pass.",
    formula: "uₖ = α·(−JS(πₖ ‖ π_loo)) + β·γₖ + Tₖ;  T^p = ρ·T^(p−1) + u^p;  wₖ = softmax(Tₖ / τ_p), floor ε = 0.1/N;  log_b^p = Σₖ wₖ·log πₖ^p",
    knobs: "α, β (learned); ρ (trust carry); τ_p = τ0·κ^(p−1), τ0 annealed τ_max→τ_min; weight_floor_fraction" },
  { key: "err", t: "Pass-to-pass feedback: pooled opinion + error features",
    d: "Pass 1: dots predict blind. Pass p>1: every dot READS the previous pass's pooled opinion and its own disagreement numbers.",
    shape: "h_pool [B, T, d_s] broadcast;  err [B, T, N, 3]",
    what: "This is the iterative-refinement core: the crowd's pooled answer becomes an input, and each dot sees (crowd mean, own-vs-crowd JS) so it can correct. The diagram's dashed orange arrows are exactly this loop.",
    formula: "pooled = Σₖ wₖ·sₖ;  errₖ = features(πₖ, crowd, JSₖ)",
    knobs: "model.n_passes, pooling=log_linear|mixture" },
  { key: "messages", t: "Messages between passes",
    d: "After each pass the region writes a summary of its states; dots read it as extra input on the NEXT pass.",
    shape: "msg [B, T, ·] written at pass p, read at pass p+1",
    what: "Lets dots coordinate beyond the pooled opinion — ablatable. At E1 this is OFF (use_messages: false); the diagram shows it as a ghost badge.",
    formula: "msg_write = M(s^p);  input at p+1 += msg_read",
    knobs: "model.use_messages; message_dim, message_topology=ring, message_dropout 0.2" },
  { key: "chain", t: "Chain region r → r+1",
    d: "The next region's input rows are built from THIS region's final weighted summary.",
    shape: "rows_{r+1} = LN(W·summary_r) [B, T, d]",
    what: "Region 1 sees raw embeddings; region 2 sees what region 1 concluded; region 3 sees region 2. Early regions ≈ spelling, later ≈ meaning (the win-map diagnostic tests this).",
    formula: "rows_{r+1} = chain_ln_r(chain_proj_r(summary_r))",
    knobs: "chain_proj/chain_ln per region (present when n_regions > 1)" },
  { key: "gate", t: "Cross-region competition (D5)",
    d: "Regions vote on which region should dominate the final answer.",
    shape: "ω [B, T, R]",
    what: "Each region's confidence (negative entropy of its consensus) plus a learned gate scalar → softmax over regions.",
    formula: "score_r = α_region·(−H(log_b_r)) + gate_r;  ω = softmax(score);  log_y = Σ_r ω_r·log_b_r",
    knobs: "alpha_region, region_gate (on when model.use_cross_region)" },
  { key: "out", t: "Final prediction",
    d: "The model's answer: the (gated) mixture of the last regions' pooled distributions, in the tied embedding's basis.",
    shape: "log_y [B, T, V=512]",
    what: "Single-region rungs output the last region's pooled log-mixture directly; multi-region rungs mix by ω.",
    formula: "log_y = log_b^(R) (single region)  |  Σ_r ω_r·log_b_r (multi-region)",
    knobs: "—" },
  { key: "loss", t: "Loss + the λ family",
    d: "NLL on the pooled answer, plus auxiliary terms that keep the machinery honest.",
    shape: "scalar loss",
    what: "L_region keeps early-region predictions useful; L_dot (boosted) trains dots against the pooled answer; L_corr decorrelates dots; L_conf calibrates γ against actual errors; L_w REWARDS weight entropy so the ensemble stays diverse. Active terms are highlighted from the live config.",
    formula: "L = NLL(log_y) + λ_r·L_region + λ_d·L_dot + λ_c·L_corr + λ_k·L_conf − λ_w·H(w)",
    knobs: "λ_r, λ_d, λ_c, λ_k, λ_w with ramp; the E1-no-λ pilot turns them all off" },
  { key: "train", t: "Training loop & honest stopping",
    d: "AdamW + grad clip; eval every eval_every steps on a 100k-char val sample; best point re-scored on FULL validation at the tau of that step.",
    shape: "CSV: step, lr, loss, bpc_val, bpc_p1..p3, bpc_train, tau0, weight_entropy, L_w, elapsed",
    what: "Early stop = 6 evals without ≥0.005 bpc improvement (noise floor ≈0.002). stop_kind records patience vs wall_cap vs reached-target — a wall-capped 'best' is NOT a plateau.",
    formula: "stop when since_best ≥ patience_evals AND best improvement < min_improvement",
    knobs: "training.lr_schedule, eval_chars, early_stop{patience_evals, min_improvement, wall_hours}" },
];

function buildDiagram(cfg, onPick) {
  const m = cfg.model || {}, cons = cfg.consensus || {}, tr = cfg.training || {};
  const R = Math.max(1, m.n_regions ?? 1), N = Math.max(1, m.n_dots ?? 8),
        P = Math.max(1, m.n_passes ?? 3), span = (m.spans || [])[0] ?? 32;
  const W = 920;
  const yTok = 30, yEmb = 96;
  const regY0 = 156;
  const passH = 116;
  const topPad = 56;
  const regH = topPad + P * passH + 20;
  const regY1 = regY0 + regH;
  const lastConsY = regY0 + topPad + (P - 1) * passH + 32;
  let y = regY1 + 46;
  const hasGate = !!m.use_cross_region;
  const yGate = hasGate ? y : 0; if (hasGate) y += 66;
  const yOut = y; y += 66;
  const yLoss = y; y += 66;
  const yTrain = y; y += 78;
  const H = yTrain + 40;
  const svg = svgEl("svg", { viewBox: `0 0 ${W} ${H}`, width: "100%",
    height: H, role: "img", "aria-label": "IECNN architecture diagram" });
  const defs = svgEl("defs", {});
  const mk = svgEl("marker", { id: "arrow", viewBox: "0 0 10 10", refX: 9,
    refY: 5, markerWidth: 7, markerHeight: 7, orient: "auto-start-reverse" });
  mk.append(svgEl("path", { d: "M 0 0 L 10 5 L 0 10 z", fill: "#4a5a78" }));
  defs.append(mk); svg.append(defs);

  const nodes = [];
  nodes.push(gnode(W / 2, yTok, 320, 46, "1 · Text → UBM ids",
    `V = 512 (256 bytes + 8 specials + 248 merges)`, null, "tok"));
  nodes.push(gnode(W / 2, yEmb, 320, 52, "2 · Tied embedding E",
    `E [V=512, d=${m.d ?? 128}] → rows [B,T,d]`, "shared as the output projection", "embed"));
  svg.append(edge(W / 2, yTok + 23, W / 2, yEmb - 26));
  // region container
  const cont = svgEl("g", { class: "gnode", "data-key": "region" });
  cont.append(svgEl("rect", { class: "container",
    x: 40, y: regY0, width: W - 80, height: regH, rx: 12 }));
  cont.append(svgEl("text", { class: "title", x: 58, y: regY0 + 20 },
    `3 · Region 1..${R} — ${N} dots, window span=${span}, d_s=${m.d_s ?? 128}`));
  svg.append(cont);
  svg.append(edge(W / 2, yEmb + 26, W / 2, regY0 + 2));
  // passes inside the region
  const passElems = [];
  for (let p = 0; p < P; p++) {
    const py = regY0 + topPad + p * passH;
    const gp = gnode(230, py, 300, 64,
      `Pass ${p + 1} · ${N} dots`,
      `window keys/vals + rel-dist → s [B,T,${N},${m.d_s ?? 128}]`,
      `logits via Eᵀ · γₖ · trustₖ`, "pass");
    const gc = gnode(660, py, 340, 64,
      `Consensus ${p + 1} → weights → pooled`,
      `w = softmax(Tₖ/τ${p + 1}), floor ${fmt(0.1 / N, 3)}`,
      `log_b^(p) = Σₖ wₖ·log πₖ^p (log-linear)`, "consensus");
    svg.append(gp, gc);
    svg.append(edge(382, py, 488, py, "uₖ scores"));
    passElems.push(gp);
    if (p < P - 1) {
      svg.append(edge(660, py + 32, 230, py + passH - 34,
        "pooled + err", "flow"));
    }
  }
  // messages badge
  if (m.use_messages) {
    svg.append(edge(660, regY0 + topPad, 660, regY0 + topPad + passH - 34, "msg ring"));
    svg.append(svgEl("text", { class: "klabel", x: 872, y: regY0 + 22,
      "text-anchor": "end" }, "messages ON"));
  } else {
    svg.append(svgEl("text", { class: "sub", x: 872, y: regY0 + 22,
      "text-anchor": "end" }, "messages OFF"));
  }
  // region output
  svg.append(edge(660, lastConsY, W / 2, yOut - 24));
  nodes.push(gnode(W / 2, yOut, 340, 46, "4 · Region output",
    hasGate ? "gated mix over R regions" : `log_y = log_b^(last pass) [B,T,V=512]`,
    null, "out"));
  if (hasGate) {
    svg.append(gnode(W / 2, yGate, 340, 50, "Region gate (D5)",
      "ω = softmax(α·(−H(log_b)) + gate) [B,T,R]", null, "gate"));
    svg.append(edge(W / 2, yGate + 25, W / 2, yOut - 23));
  }
  // loss
  const active = ["lambda_r", "lambda_d", "lambda_c", "lambda_k", "lambda_w"]
    .filter(k => (tr[k] ?? 0) > 0).map(k => `${k.slice(7)}=${tr[k]}`);
  nodes.push(gnode(W / 2, yLoss, 420, 50, "5 · Loss",
    active.length ? `NLL + active λ: ${active.join(" · ")}` : "NLL only (all λ off — no-λ ablation)",
    null, "loss"));
  svg.append(edge(W / 2, yOut + 23, W / 2, yLoss - 25));
  nodes.push(gnode(W / 2, yTrain, 520, 56, "6 · Training loop",
    `AdamW · ${tr.steps ?? "?"} steps · warmup ${tr.warmup_steps ?? "?"} · eval every ${tr.eval_every ?? "?"}`,
    `early stop: patience ${tr.early_stop?.patience_evals ?? "?"} evals / wall ${tr.early_stop?.wall_hours ?? "?"} h`, "train"));

  if (R > 1) {
    svg.append(svgEl("text", { class: "sub", x: 62, y: regY1 + 6 },
      `chain_proj + chain_ln feed region r's summary into region r+1 (${R} regions total) — diagram shows region 1.`));
  }
  for (const n of nodes.concat(passElems)) svg.append(n);
  return svg;
}

async function pageArchitecture(root) {
  if (!state.arch) state.arch = await api("/api/arch");
  const arch = state.arch;
  root.innerHTML = "";
  root.append(h("h1", {}, "Architecture — the full IECNN pipeline"),
    h("p", {}, "Built live from the committed configs in ",
      h("code", {}, "configs/"), " (including the ",
      h("code", {}, "base:"), " inheritance chain), so the diagram cannot drift from what actually runs. ",
      "Click any stage for shapes and formulas."));

  const sel = h("select", {}, (arch.configs || []).map(c =>
    h("option", { value: c, selected: c === arch.config ? "" : null }, c)));
  root.append(h("div", { class: "toolbar" },
    h("label", { class: "muted", for: "rungsel" }, "rung config: "), sel));
  if (!(arch.configs || []).length) {
    warnbox(root, "No committed configs parsed yet — expected configs/*.yaml in the repo. ",
      "Architecture views are built from those files; commit a config (e.g. configs/base.yaml) and reload.");
    return;
  }

  const refresh = name => {
    const r = (arch.resolved || {})[name] || arch.active;
    if (!r) { errbox(root, new Error(`config ${name} missing from /api/arch`)); return; }
    const cfg = r.config;
    const m = cfg.model || {}, cons = cfg.consensus || {}, tr = cfg.training || {};
    const N = m.n_dots ?? 8, P = m.n_passes ?? 3;
    // hyperparam chips
    const chips = h("div", { class: "hpchips" });
    const chip = (k, v) => chips.append(h("span", { class: "hpchip" }, k + " ", h("b", {}, String(v))));
    chip("kind", m.kind ?? "iecnn");
    chip("R", m.n_regions ?? 1); chip("dots", N); chip("passes", P);
    chip("d", m.d ?? 128); chip("d_s", m.d_s ?? 128);
    chip("window span", (m.spans || [])[0] ?? "?");
    chip("pooling", m.pooling ?? "log_linear");
    chip("weights", m.use_weights ? "learned" : "uniform");
    chip("τ range", `${fmt(cons.tau_max, 2)}→${fmt(cons.tau_min, 2)} (κ=${fmt(cons.kappa, 2)})`);
    chip("ρ trust", fmt(cons.rho, 2));
    chip("messages", m.use_messages ? "ON" : "OFF");
    chip("cross-region gate", m.use_cross_region ? "ON" : "OFF");
    ["lambda_r", "lambda_d", "lambda_c", "lambda_k", "lambda_w"].forEach(k => {
      const v = tr[k] ?? 0;
      chips.append(h("span", { class: "hpchip" }, `λ_${k.slice(7)} `,
        h("b", { style: v > 0 ? "color:var(--good)" : "color:var(--muted)" }, String(v))));
    });
    chip("steps", tr.steps ?? "?"); chip("eval every", tr.eval_every ?? "?");
    const inherited = (r.chain || []).length ? ` (inherits: ${r.chain.join(" → ")})` : "";
    root.querySelectorAll(".hpchips, .archwrap, .muted.inh").forEach(e => e.remove());
    root.append(h("div", { class: "muted inh" },
      `config ${name}${inherited}`), chips);

    const wrap = h("div", { class: "archwrap" });
    const dia = h("div", { class: "diagram" });
    const detail = h("div", { class: "detail" });
    wrap.append(dia, detail);
    root.append(wrap);
    let nonIecnn = false;
    try {
      if ((m.kind ?? "iecnn") !== "iecnn") {
        nonIecnn = true;
        dia.append(h("p", { class: "warn", style: "padding:12px" },
          `This rung is a ${m.kind} baseline, not an IECNN — the diagram below still shows the IECNN machinery for reference; shapes may not apply.`));
      }
      dia.append(buildDiagram(cfg, key => showDetail(key)));
    } catch (e) { errbox(dia, e); }
    const showDetail = key => {
      const s = ARCH_DETAIL.find(x => x.key === key) || ARCH_DETAIL[0];
      $$(".gnode", dia).forEach(g => g.classList.toggle("sel", g.dataset.key === key));
      detail.innerHTML = "";
      detail.append(h("h3", {}, s.t),
        h("div", { class: "f" }, s.shape),
        h("p", {}, s.what),
        h("div", { class: "f" }, s.formula),
        h("p", { class: "knobs" }, h("b", {}, "knobs: "), s.knobs));
    };
    // τ anneal across passes (consensus detail extra)
    $$(".gnode", dia).forEach(g => g.addEventListener("click", () => showDetail(g.dataset.key)));
    showDetail("tok");
  };
  sel.addEventListener("change", () => refresh(sel.value));
  refresh(arch.config);
  await replayPanel(root);
}

/* ---- replay: pass-by-pass animation over real viz-export data -----------
   Uses the committed results/viz/*.json sample positions (real per-pass
   dot weights/gamma/predictions from the checkpoint) — no synthetic data.
   With prefers-reduced-motion there is no autoplay; the slider works. */
async function replayPanel(root) {
  let list = [];
  try { list = (await api("/api/viz")).exports || []; } catch { list = []; }
  const sec = h("div", {});
  root.append(h("h2", {}, "Replay — pass-by-pass forward pass"), sec);
  if (!list.length) {
    warnbox(sec, "No checkpoint viz exports yet (results/viz/*.json) — the replay needs a real per-pass export. ",
      h("code", {}, "gh workflow run viz.yml -f artifact=<checkpoint artifact>"),
      " commits one and this panel fills in.");
    return;
  }
  const sel = h("select", {}, list.map(n => h("option", { value: n }, n)));
  sec.append(h("p", { class: "legend" },
    "Animate the sample text through the real checkpoint: each position runs pass 1 → pass 3, showing what the dots read, weigh and predict at each pass; the pooled consensus is revealed at the last pass."),
    h("div", { class: "toolbar" }, h("label", { class: "muted" }, "export: "), sel));
  const stage = h("div", {});
  sec.append(stage);
  const reduceMotion = window.matchMedia &&
    window.matchMedia("(prefers-reduced-motion: reduce)").matches;
  const showExport = async () => {
    stage.innerHTML = "";
    stage.append(h("div", { class: "loading" }, "· · ·"));
    let v;
    try { v = state.vizData[sel.value] || await api(`/api/viz/${sel.value}`); }
    catch (e) { stage.innerHTML = ""; errbox(stage, e); return; }
    state.vizData[sel.value] = v;
    stage.innerHTML = "";
    const S = v.sample;
    if (!S || !Array.isArray(S.positions) || !S.positions.length) {
      warnbox(stage, "this export has no sample.positions payload — nothing to replay"); return;
    }
    const positions = S.positions, P = (S.passes || ["pass 1", "pass 2", "pass 3"]).length;
    const total = positions.length * P;
    const slider = h("input", { type: "range", class: "posbar", min: "0",
      max: String(total - 1), value: "0", "aria-label": "replay step (position × pass)" });
    const play = h("button", { title: reduceMotion ?
      "autoplay disabled by prefers-reduced-motion" : "play the replay" },
      reduceMotion ? "▶ (motion off)" : "▶ play");
    const stopLbl = h("span", { class: "muted" });
    const board = h("div", {});
    stage.append(h("div", { class: "toolbar" }, slider, play, stopLbl), board);
    let timer = null;
    const draw = () => {
      const t = +slider.value;
      const pi = Math.floor(t / P), pass = t % P;
      const Pd = positions[Math.min(pi, positions.length - 1)];
      stopLbl.textContent = `position ${pi + 1}/${positions.length} · ${S.passes?.[pass] || "pass " + (pass + 1)}`;
      board.innerHTML = "";
      // pass-by-pass: the dots' weights/gamma so far, predictions revealed per pass
      const dots = Array.isArray(Pd.dots) ? Pd.dots : [];
      const grid = h("div", { class: "dotgrid" });
      const wMax = Math.max(1e-9, ...dots.map(d => d.weights?.[pass] ?? 0));
      dots.forEach((d, k) => {
        const w = d.weights?.[pass] ?? 0, g = d.gamma?.[pass] ?? 0;
        const top = (d.top || [])[pass] || [];
        grid.append(h("div", { class: "dotcard" + (w / wMax > 0.75 ? " hit" : "") },
          h("div", { class: "hd" },
            h("strong", { style: `color:${PALETTE[k % PALETTE.length]}` }, `dot ${k}`),
            h("span", { class: "muted" }, `γ ${fmt(g, 3)}`)),
          h("div", { class: "replaybar" },
            h("div", { style: `width:${Math.round(100 * w / wMax)}%;background:${HEAT(w / wMax)}`,
                       title: `w = ${fmt(w, 4)}` })),
          h("div", { class: "muted" }, top.length ?
            `p${pass + 1} top: ` + top.slice(0, 3).map(([tok, pr]) =>
              `${tokDisp(tok)} ${fmt(pr * 100, 0)}%`).join(" · ") :
            "(no predictions logged for this pass)")));
      });
      board.append(grid);
      // pooled answer revealed only at the final pass
      if (pass === P - 1) {
        const pool = Pd.pooled_top || [];
        board.append(h("div", { class: "explain" },
          h("b", {}, "pooled consensus — "),
          pool.map(([tok, prob]) =>
            h("span", { class: "pill" + (tok === Pd.next ? " ok" : "") },
              `${tokDisp(tok)} ${fmt(prob * 100, 1)}%`)).slice(0, 4),
          h("span", { class: "muted" },
            pool[0] && pool[0][0] === Pd.next
              ? "  ✓ matches the target" : `  ✗ target was “${tokDisp(Pd.next)}”`)));
      } else {
        board.append(h("p", { class: "muted" },
          `pass ${pass + 2} and the pooled consensus come next — keep playing or scrub the slider`));
      }
    };
    play.addEventListener("click", () => {
      if (timer) { clearInterval(timer); timer = null; play.textContent = "▶ play"; return; }
      if (reduceMotion) {                    // honour prefers-reduced-motion
        stopLbl.textContent = "autoplay disabled by prefers-reduced-motion — use the slider";
        return;
      }
      let t = +slider.value;
      if (t >= total - 1) { t = 0; slider.value = "0"; draw(); }
      timer = setInterval(() => {
        t = +slider.value + 1;
        if (t >= total) { clearInterval(timer); timer = null; play.textContent = "▶ replay"; return; }
        slider.value = String(t);
        draw();
      }, 700);
    });
    slider.addEventListener("input", draw);
    draw();
  };
  sel.addEventListener("change", showExport);
  await showExport();
}

/* ======================================================= CHECKPOINTS ====== */
async function pageCheckpoints(root) {
  root.innerHTML = "";
  root.append(h("h1", {}, "Weights & dots — checkpoint inspector"));
  let list;
  try { list = (await api("/api/viz")).exports; }
  catch { list = []; }
  if (!list.length) {
    warnbox(root, "No checkpoint exports yet. Real weights/dots live inside torch checkpoints that only exist as CI artifacts — per the no-local-torch rule they are exported ON THE RUNNER by the viz workflow. ",
      h("code", {}, "gh workflow run viz.yml -f artifact=<checkpoint artifact>"),
      " writes results/viz/<run>.json and this page fills in.");
    return;
  }
  const sel = h("select", {}, list.map(n => h("option", { value: n }, n)));
  const view = h("div", {});
  root.append(h("div", { class: "toolbar" },
    h("label", { class: "muted" }, "export: "), sel), view);
  const show = async () => {
    view.innerHTML = "";
    view.append(h("div", { class: "loading" }, "· · ·"));
    try {
      const v = state.vizData[sel.value] || await api(`/api/viz/${sel.value}`);
      state.vizData[sel.value] = v;
      view.innerHTML = "";
      renderViz(view, v);
    } catch (e) { view.innerHTML = ""; errbox(view, e); }
  };
  sel.addEventListener("change", show);
  await show();
}

async function renderViz(root, v) {
  root.innerHTML = "";
  if (v.error) { errbox(root, v.error); return; }
  const M = v.meta || {};
  root.append(h("div", { class: "cards" },
    card(M.run || v.run || "?", "run"),
    card(`step ${M.step ?? "?"}`, "checkpoint step"),
    card(M.stop_kind ?? "?", "stop kind", "card"),
    card((M.params ?? 0).toLocaleString(), "parameters"),
    card(`τ ${fmt(M.tau0 ?? 1, 3)}`, "eval temperature tau0"),
    card(`${M.R ?? "?"}R · ${M.n_dots ?? "?"} dots · ${M.n_passes ?? "?"} passes`, "geometry")));

  // ---- embedding map
  const pts = v.embeddings?.points;
  if (Array.isArray(pts) && pts.length) {
    const labels = v.embeddings.labels;
    root.append(h("h2", {}, "Token embedding map (2-D projection of the tied table)"),
      h("p", { class: "legend" }, `Each point is a vocab entry (${pts.length}). Nearby points are embedded close together. Hover to read tokens. `,
        "Byte-fallback tokens (ids 0–255) display as their byte: printable ASCII as-is, otherwise hex ", h("code", {}, "<xx>"), " — 129 of 512 entries display that way, by design."));
    const cv = h("canvas", { style: "max-height:420px" });
    root.append(h("div", { class: "chartbox" }, cv));
    if (HAS_CHART) requestAnimationFrame(() => {
      try {
        new Chart(cv, { type: "scatter",
          data: { datasets: [{ data: pts.map((p, i) => ({ x: p[0], y: p[1],
            label: idDisp(i, labels) })),
            pointRadius: 2.2, pointHoverRadius: 5,
            backgroundColor: "#4fc3f7aa" }] },
          options: { responsive: true, maintainAspectRatio: false,
            plugins: { legend: { display: false },
              tooltip: { callbacks: { label: c => c.raw.label } } },
            scales: { x: { ...AXIS_GRID, type: "linear", ticks: AXIS_TICK },
                      y: { ...AXIS_GRID, type: "linear", ticks: AXIS_TICK } } } });
      } catch (e) { /* decorative */ }
    });
  }

  // ---- sample forward inspector
  const S = v.sample;
  if (!S || !Array.isArray(S.positions) || !S.positions.length) return;
  const passes = S.passes || ["pass 1", "pass 2", "pass 3"];
  const nPos = S.positions.length;
  // corpus-verified correctness of the pooled answer over the whole sample
  let hits = 0, scored = 0;
  for (const P of S.positions) {
    const top = P.pooled_top && P.pooled_top[0];
    if (top && P.next !== undefined) { scored++; if (top[0] === P.next) hits++; }
  }
  root.append(h("h2", {}, "Sample forward — dots in action"),
    h("p", {}, "Text fed through the real checkpoint: ",
      h("details", { class: "rawjson" },
        h("summary", {}, `${esc((S.text || "").length)} chars of sample text`),
        h("pre", {}, esc((S.text || "").slice(0, 800)) + " …")),
      "Drag the position slider to inspect any token: how much each dot trusted itself (weights), its confidence (γ), and what each dot actually predicted. ",
      "Token display: raw byte tokens show as ", h("code", {}, "<xx>"), " hex; ",
      h("code", {}, "<fffd>"), " marks a byte the runner-side export decode already lost; ",
      h("code", {}, "\\n"), " is a newline. The · dots mark token boundaries — the model reads in ~2-char chunks by design. ",
      h("span", { class: "pill " + (hits / Math.max(1, scored) >= .5 ? "ok" : "") },
        `pooled top-1 correct on ${hits}/${scored} positions (${fmt(100 * hits / Math.max(1, scored), 1)}%)`)));

  // ---- consensus audit: when a dot held the right answer but the pooled
  // consensus overruled it. Stats computed server-side from this exact
  // payload (v.audit); the panel is skipped entirely if absent.
  const A = v.audit;
  const heldMap = new Map();
  if (A && Array.isArray(A.per_position))
    for (const e of A.per_position) if (e && Number.isInteger(e.i)) heldMap.set(e.i, e);
  if (A && A.sampled > 0) {
    const panel = h("div", { class: "explain audit" },
      h("b", {}, `Consensus audit — ${A.sampled}-position sample. `),
      "Sometimes a dot holds the right answer but doesn't get chosen: ",
      "a correct dot was outvoted here — the learned weights muted it (see its weight row) or a sharper wrong dot won. ",
      "Definitions: oracle best-dot = at least one dot predicted the target (any pass); ",
      "outvoted = a dot held it (", h("b", {}, "top-1", ), " stronger, or in its ",
      h("b", {}, "top-5"), ") while the pooled answer missed.");
    panel.append(h("div", { class: "cards" },
      card(`${A.pooled_top1}/${A.sampled} · ${fmt(100 * A.pooled_top1_acc, 1)}%`,
        "pooled top-1 accuracy"),
      card(`${A.oracle_top1}/${A.sampled} · ${fmt(100 * A.oracle_top1_acc, 1)}%`,
        "oracle best-dot top-1"),
      card(`${A.oracle_top5}/${A.sampled} · ${fmt(100 * A.oracle_top5_acc, 1)}%`,
        "oracle best-dot top-5"),
      card(`${A.outvoted ?? (A.outvoted_top1 + A.outvoted_top5)}`,
        "outvoted positions",
        (A.outvoted || (A.outvoted_top1 + A.outvoted_top5)) ? "warn" : "")));
    if (A.outvoted) {
      const chips = h("div", { class: "chips" });
      for (const e of A.per_position) {
        const w = e.held?.[0]?.weight;
        const btn = h("button", { class: "chip", title:
            `dot ${e.held?.map(x => x.dot).join(", ")} held the target ` +
            `${e.level === "top1" ? "as top-1" : "in top-5"}${w != null ? ` at w=${fmt(w, 4)}` : ""} — pooled answer missed` },
          `pos ${e.i} · ${e.level === "top1" ? "held top-1" : "held top-5"}`);
        // bar is declared below; safe because this only runs on click
        btn.addEventListener("click", () => {
          bar.value = String(e.i);
          bar.dispatchEvent(new Event("input"));
          window.scrollTo({ top: bar.getBoundingClientRect().top + window.scrollY - 80, behavior: "smooth" });
        });
        chips.append(btn);
      }
      panel.append(h("div", { class: "muted", style: "margin:4px 0 2px" },
        `outvoted positions (click to inspect): ${A.outvoted_top1} held top-1, ${A.outvoted_top5} held top-5 only`), chips);
    }
    // inserted below the intro so it reads as the section summary
    root.append(panel);
  }

  const bar = h("input", { type: "range", class: "posbar", min: "0",
    max: String(nPos - 1), value: "0", "aria-label": "token position" });
  const posLabel = h("div", { class: "muted" });
  const zone = h("div", {});
  root.append(bar, posLabel, zone);

  // ---- linked dot focus: clicking a dot chip highlights that dot's traces
  // across ALL charts on this page (weights heat cells, γ line, per-pass
  // weight trace, per-dot prediction cards)
  let focusDot = null;
  const dotCount = (S.positions[0]?.dots || []).length;
  const dotChips = h("div", { class: "chips", role: "group",
    "aria-label": "focus a dot across all charts" });
  const chipEls = [];
  for (let k = 0; k < dotCount; k++) {
    const b = h("button", { class: "chip", "aria-pressed": "false",
      title: `highlight dot ${k} across the weights, γ and prediction charts` },
      h("span", { class: "dot", style: `background:${PALETTE[k % PALETTE.length]}` }),
      `dot ${k}`);
    b.addEventListener("click", () => {
      focusDot = focusDot === k ? null : k;
      chipEls.forEach((c, j) => {
        c.classList.toggle("on", focusDot === j);
        c.setAttribute("aria-pressed", String(focusDot === j));
      });
      redraw();
      drawGamma();
    });
    chipEls.push(b);
    dotChips.append(b);
  }
  root.append(h("div", { class: "muted", style: "margin:2px 0 -6px" },
    "focus a dot — its traces light up in every chart below:"), dotChips);

  const redraw = () => {
    const i = Math.min(+bar.value, nPos - 1), P = S.positions[i];
    if (!P) return;
    posLabel.textContent = `position ${i} / ${nPos - 1} · token “${tokDisp(P.token)}” → target next “${tokDisp(P.next)}” (id ${P.next_id ?? "?"}${P.next_id !== undefined && P.next_id < 256 ? ` = byte ${byteDisp(P.next_id)}` : ""})`;
    zone.innerHTML = "";
    // outvoted badge: a dot here held the true next token but the pooled
    // consensus picked something else
    const he = heldMap.get(i);
    if (he) {
      const first = (he.held || [])[0] || {};
      zone.append(h("div", { class: "outvoted" },
        h("span", { class: "pill warn" },
          `⚠ outvoted · dot ${he.held?.map(x => x.dot).join(", ")} held ` +
          (he.level === "top1" ? "top-1" : "top-5") +
          (first.weight != null ? ` at w=${fmt(first.weight, 4)}` : "")),
        h("span", { class: "muted" },
          he.level === "top1"
            ? " — the correct dot was muted by the learned weights (see its near-zero weight below) or beaten by a sharper wrong dot"
            : " — a dot had the answer in its top-5 but the pooled consensus missed")));
    }
    // token context strip — one token per chunk, · separators make the
    // chunking visible; byte-fallback tokens render as hex escapes
    const toks = Array.isArray(S.tokens) ? S.tokens : null;
    if (toks && toks.length) {
      const a = Math.max(0, i - 14), b = Math.min(toks.length, i + 8);
      const strip = h("div", { class: "tokline" });
      if (a > 0) strip.append("… ");
      toks.slice(a, b).forEach((t, j) => {
        const idx = a + j, disp = tokDisp(t), escaped = disp !== t;
        const cls = (idx === i ? "cur" : idx === i + 1 ? "tgt" : "") +
          (escaped ? " btk" : "");
        if (cls) strip.append(h("span", { class: cls.trim() }, disp));
        else strip.append(disp);
        strip.append(h("span", { class: "tsep" }, "·"));
      });
      if (b < toks.length) strip.append("…");
      zone.append(strip);
    }
    // weights heatmap: passes × dots (normalized per pass row)
    const dots = Array.isArray(P.dots) ? P.dots : [];
    const tbl = h("table", { class: "heat" });
    const head = h("tr", {}, h("th", { class: "rowh" }, "pass"));
    dots.forEach((_, k) => head.append(h("th", { class: "rowh" }, `dot ${k}`)));
    tbl.append(head);
    const rowMax = passes.map((_, pi) =>
      Math.max(1e-9, ...dots.map(d => d.weights?.[pi] ?? 0)));
    // cells belonging to the outvoted holder dot(s) get an amber outline
    const heldCells = new Set((he?.held || []).map(x => `${x.dot}:${x.pass}`));
    passes.forEach((pname, pi) => {
      const tr = h("tr", {}, h("td", { class: "rowh" }, esc(pname)));
      dots.forEach((d, k) => {
        const w = d.weights?.[pi] ?? 0, g = d.gamma?.[pi] ?? 0;
        const held = heldCells.has(`${k}:${pi}`);
        const dimmed = focusDot !== null && focusDot !== k;
        tr.append(h("td", {}, h("div", { class: "cell" + (held ? " heldc" : "") +
            (dimmed ? " dim" : "") + (focusDot === k ? " focused" : ""),
          style: `background:${HEAT(w / rowMax[pi])}`,
          title: `w=${fmt(w, 4)} γ=${fmt(g, 3)} (row-normalised ${fmt(w / rowMax[pi], 2)})` +
            (held ? " ← this dot held the correct next token" : "")
        }, fmt(w, 2))));
      });
      tbl.append(tr);
    });
    zone.append(h("h3", {}, "dot weights per pass (brighter = more weight; hover for γ; amber outline = held the correct answer)"), tbl);

    // pooled answer
    const pool = P.pooled_top || [];
    zone.append(h("h3", {}, "pooled region answer (log-linear over dots)"),
      h("div", { class: "dotcard" },
        ...pool.map(([tok, prob]) =>
          h("span", { class: "pill" + (tok === P.next ? " ok" : "") +
            (tokEscaped(tok) ? " esc" : "") },
            `${tokDisp(tok)} ${fmt(prob * 100, 1)}%`)),
        h("div", { class: "muted" },
          pool[0] && pool[0][0] === P.next ? "✓ top-1 == target" : `✗ top-1 ≠ target “${tokDisp(P.next)}”`)));

    // per-dot predictions at selected pass
    const pSel = h("select", {}, passes.map((pn, pi) =>
      h("option", { value: String(pi), selected: pi === passes.length - 1 }, esc(pn))));
    const cardsBox = h("div", { class: "dotgrid" });
    const fillDots = () => {
      const pl = +pSel.value;
      cardsBox.innerHTML = "";
      dots.forEach((d, k) => {
        const top = (d.top || [])[pl] || [];
        const hit = top.some(t => t[0] === P.next);
        const dimmed = focusDot !== null && focusDot !== k;
        cardsBox.append(h("div", { class: "dotcard" + (hit ? " hit" : "") +
            (dimmed ? " dim" : "") + (focusDot === k ? " focused" : "") },
          h("div", { class: "hd" },
            h("strong", {}, `dot ${k}`, hit ? "  ✓" : ""),
            h("span", { class: "muted" }, `w=${fmt(d.weights?.[pl], 3)} γ=${fmt(d.gamma?.[pl], 3)}`)),
          h("div", { class: "muted" }, `${esc(passes[pl])} top predictions: `),
          ...top.map(([tok, prob]) =>
            h("span", { class: "pill" + (tok === P.next ? " ok" : "") +
              (tokEscaped(tok) ? " esc" : "") },
              `${tokDisp(tok)} ${fmt(prob * 100, 1)}%`))));
      });
    };
    pSel.addEventListener("change", fillDots);
    fillDots();
    zone.append(h("h3", {}, "what each dot predicted",
      h("span", { class: "muted" }, " — pass: "), pSel), cardsBox);
  };
  bar.addEventListener("input", redraw);
  redraw();

  // per-position scalar charts
  const nChart = S.positions.map((_, i) => i);
  if (Array.isArray(v.weight_entropy_over_positions) &&
      v.weight_entropy_over_positions.length) {
    root.append(h("h2", {}, "Weight entropy along the sample"),
      h("p", { class: "legend" }, `High entropy = dots disagree and share the load; near 0 = one dot dominates. Uniform baseline for ${M.n_dots ?? 8} dots = ${fmt(Math.log(M.n_dots ?? 8), 3)} nats.`));
    root.append(chartBox("H(w) per token position", {
      labels: nChart,
      datasets: [{ label: "H(w)", data: v.weight_entropy_over_positions,
                   borderColor: "#ffb74d", pointRadius: 0, tension: .2 }],
    }, "line", { xlabel: "token position", ylabel: "entropy (nats)" }));
  }
  const omega = Array.isArray(v.omega_over_positions) ? v.omega_over_positions : null;
  if (omega && omega.length) {
    const flat = uniq(omega.map(x => (typeof x === "number" ? fmt(x, 6) : JSON.stringify(x)))).length <= 1;
    if (flat) {
      root.append(h("h2", {}, "Region gate ω"),
        h("p", { class: "legend" }, `Constant over the sample (single region, R=${M.R ?? 1}) — cross-region gating is inactive here.`));
    } else {
      root.append(chartBox("region gate ω per position", {
        labels: nChart, datasets: [{ label: "ω", data: omega,
          borderColor: "#ba68c8", pointRadius: 0, tension: .2 }],
      }, "line", { xlabel: "token position", ylabel: "ω" }));
    }
  }
  // gamma + weight trajectory chart per dot — rebuilt on dot focus so the
  // focused dot's traces stay highlighted across the page's charts
  const gammaHost = h("div", {});
  const drawGamma = () => {
    gammaHost.innerHTML = "";
    if (!S.positions[0]?.dots?.length) return;
    const nd = S.positions[0].dots.length;
    const gsets = [];
    for (let k = 0; k < nd; k++) {
      const focused = focusDot === k, dimmed = focusDot !== null && !focused;
      gsets.push({ label: `γ dot ${k}`,
        data: S.positions.map(P => P.dots[k]?.gamma?.[0] ?? null),
        borderColor: PALETTE[k % PALETTE.length],
        borderDash: focused ? [] : DASHES[k % DASHES.length],
        borderWidth: focused ? 3 : dimmed ? 1 : 1.6,
        pointRadius: focused ? 2.5 : 0,
        backgroundColor: focused ? PALETTE[k % PALETTE.length] : "transparent",
        tension: .2 });
    }
    gammaHost.append(chartBox("dot confidence γ (pass 1) along the sample" +
      (focusDot !== null ? ` — dot ${focusDot} focused` : ""), {
      labels: nChart, datasets: gsets,
    }, "line", { xlabel: "token position", ylabel: "γ" }));
    if (focusDot !== null) {
      // the focused dot's weight per pass along the whole sample
      gammaHost.append(chartBox(
        `dot ${focusDot} — weight per pass along the sample`, {
        labels: nChart,
        datasets: passes.map((pn, pi) => ({
          label: esc(pn),
          data: S.positions.map(P => P.dots[focusDot]?.weights?.[pi] ?? null),
          borderColor: PASS_COLORS[pi % PASS_COLORS.length],
          pointRadius: 0, tension: .2 })),
      }, "line", { xlabel: "token position", ylabel: "weight" }));
    }
  };
  root.append(gammaHost);
  drawGamma();
}

/* ========================================================= SCORES ========= */
async function pageScores(root) {
  const [evals, runs, summaries, reports] = await Promise.all(
    [api("/api/evals"), api("/api/runs"), api("/api/summaries"),
     api("/api/reports")]);
  root.innerHTML = "";
  root.append(h("h1", {}, "Re-score reports"),
    h("p", {}, "Each checkpoint's best step was re-scored on the FULL validation set at that step's τ0. ",
      "Provenance (run · step · τ · vocab sha · wall/compute) is shown with every number; ",
      h("code", {}, "vocab_sha256"), " is checked across every report below and against the viz export + diagnosis table."));

  const entries = Object.entries(evals);
  if (!entries.length) {
    warnbox(root, "No eval_*.json reports in results/ yet — they are produced runner-side by the re-score/diagnose workflows; committed files land in results/ and this page fills in. ",
      h("code", {}, "gh workflow run eval.yml -f artifact=<checkpoint artifact> -f corpus_config=<rung>_pilot.yaml"));
    return;
  }

  // corpus val sha for the val-provenance check
  const valSha = ((reports.split_info || {}).parts || {}).val?.sha256 || "";

  // vocab sha consistency across NON-retracted reports only
  const live = entries.filter(([, v]) => !v.retracted);
  const shas = new Set(live.map(([, v]) => v.vocab_sha256).filter(Boolean));
  const shaOK = shas.size === 1;
  const sha = [...shas][0] || "";
  root.append(h("div", { class: "cards" },
    h("div", { class: "card" },
      h("div", { class: "big", style: `color:${shaOK ? "var(--good)" : "var(--bad)"}` },
        shaOK ? "✓ consistent" : "✗ MISMATCH"),
      h("div", { class: "lbl" }, `vocab_sha256 across ${live.length} active reports`),
      h("div", { class: "lbl muted" }, sha ? sha.slice(0, 16) + "…" : "—"))));

  // wall/compute for each eval: summary JSON first, then the run CSV's
  // last elapsed_s
  const computeFor = async (name, v) => {
    const ck = (v.ckpt_dir || "").split("/").filter(Boolean).pop() || "";
    const summ = summaries[ck];
    if (summ && summ.steps_final != null) {
      return { text: `steps_final ${summ.steps_final}` +
        (summ.train_seconds_total != null
          ? ` · train ${fmt(summ.train_seconds_total, 1)} s` : ""),
        src: `checkpoints/${ck}/summary.json` };
    }
    const run = runs.runs.find(r => r.name === ck);
    if (run) {
      try {
        const d = await api(`/api/run/${run.name}`);
        const el = (d.series?.elapsed_s || []).filter(x => x !== null);
        if (el.length) return { text: `last eval at ${fmt(el[el.length - 1], 1)} s wall (${d.n_points} eval pts)`,
                                src: `results/${run.name}.csv` };
      } catch { /* fall through */ }
    }
    return null;
  };

  for (const [name, v] of entries) {
    const retracted = !!v.retracted;
    const vs = v.val_sha256 || v.provenance?.val_sha256;
    const verified = !!(vs && valSha &&
      (vs === valSha || vs.slice(0, 16) === valSha.slice(0, 16)));
    const mismatch = !!(vs && valSha && !verified);
    const passes = v.per_pass_bpc || [];
    const ck = (v.ckpt_dir || "").split("/").filter(Boolean).pop() || "—";
    const sec = h("div", {});
    root.append(sec);
    const h2 = h("h2", {}, name, h("span", { class: "pill" }, "full-val"),
      verified ? h("span", { class: "pill ok" }, "corpus-verified ✓")
      : mismatch ? h("span", { class: "pill warn" }, "✗ val corpus mismatch — hidden")
      : h("span", { class: "pill warn" }, "unverified corpus"));
    if (retracted) h2.append(h("span", { class: "pill warn" }, "RETRACTED"));
    sec.append(h2);

    if (mismatch) {
      // a declared val sha that does NOT match the committed corpus: the
      // number is hidden behind the flag
      sec.append(h("div", { class: "errbox" },
        h("b", {}, "val_sha256 mismatch — number hidden. "),
        `this report declares val sha ${shaShort(vs)} but the committed val split is ${shaShort(valSha)}; the score was computed against a different corpus and is not shown. `));
      sec.append(h("details", { class: "rawjson" },
        h("summary", {}, `raw JSON — ${name}`),
        h("pre", {}, JSON.stringify(v, null, 1))));
      continue;
    }

    const comp = await computeFor(name, v);
    const prov = h("div", { class: "hpchips" },
      provChip("run", ck),
      provChip("best step", v.best_step ?? "—"),
      provChip("τ0", fmt(v.tau0_used, 4)),
      provChip("vocab sha", shaShort(v.vocab_sha256),
        "full sha256: " + (v.vocab_sha256 || "—")));
    if (comp) {
      prov.append(provChip("compute", comp.text));
      prov.append(h("span", { class: "muted" }, "(from "), rawLink(comp.src),
        h("span", { class: "muted" }, ")"));
    }
    sec.append(prov);

    const cards = h("div", { class: "cards" });
    const bigNum = h("div", { class: "big" + (retracted ? " struck" : "") },
      fmt(v.best_full_val_bpc_at_best_tau, 5));
    cards.append(h("div", { class: "card" + (retracted ? " mutedcard" : "") },
      bigNum, h("div", { class: "lbl" }, "full-val bpc @ best τ")),
      card(String(v.best_step ?? "—"), "best step"),
      card(fmt(v.tau0_used, 4), "τ0 used"),
      card(fmt(v.weight_entropy, 4), "weight entropy at best step"),
      card(fmt(v.hard_bpc_at_best_tau, 5), "hard (top-m) bpc"),
      card(fmt(v.mode_gap, 5), "mode gap (hard − soft)"),
      card(fmt(v.bpc_train_slice, 5), "train-slice bpc"),
      card(shaShort(v.vocab_sha256), "vocab_sha256"));
    if (retracted) {
      const r = v.retracted;
      cards.append(h("div", { class: "card mutedcard warncard" },
        h("div", { class: "lbl", style: "color:var(--accent2)" },
          `RETRACTED ${r.date ? "· " + r.date : ""}`),
        h("div", { class: "lbl" }, r.reason || "retracted"),
        r.superseded_by ? h("div", { class: "lbl muted" },
          `superseded by: ${r.superseded_by}`) : null));
    }
    sec.append(cards);
    if (retracted) sec.append(h("p", { class: "muted" },
      "This report's numbers are excluded from every aggregate, best pick and gate on this dashboard."));
    if (passes.length) {
      sec.append(chartBox(`Per-pass bpc — ${name}`, {
        labels: passes.map((_, i) => `pass ${i + 1}`),
        datasets: [{ label: "bpc", data: passes,
          backgroundColor: ["#7986cbbb", "#4fc3f7bb", "#ffd54fbb"],
          borderColor: ["#7986cb", "#4fc3f7", "#ffd54f"],
          borderWidth: 1, borderRadius: 6 }],
      }, "bar", { sub: "each pass should improve on the last (iterative refinement)", ylabel: "bpc", legend: false }));
    }
    // compare against the run's own sampled val curve
    const run = runs.runs.find(r => r.name === name.replace(/^eval_/, ""));
    if (run) {
      sec.append(h("p", {}, `Sampled val curve for ${run.name}: best ${fmt(run.best_val)} at step ${run.best_step ?? "?"} (100k-char sample — expect the full-val re-score to sit nearby). `,
        h("a", { href: `#/runs/${run.name}` }, "open curve →")));
    }
    sec.append(sourceLine([`results/${name}.json`]));
    sec.append(h("details", { class: "rawjson" },
      h("summary", {}, `raw JSON — ${name}`),
      h("pre", {}, JSON.stringify(v, null, 1))));
  }

  // side-by-side comparison of the first two ACTIVE (non-retracted) evals
  if (live.length >= 2) {
    const [nA, vA] = live[0], [nB, vB] = live[1];
    const row = (label, key, d = 5) => {
      const a = vA[key], b = vB[key];
      return h("tr", {},
        h("td", {}, label),
        h("td", { class: "num" }, fmt(a, d)),
        h("td", { class: "num" }, fmt(b, d)),
        h("td", { class: "num" }, a !== undefined && b !== undefined ? fmt(b - a, d) : "—"));
    };
    root.append(h("h2", {}, `${nA} vs ${nB}`),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "metric"),
        h("th", {}, nA), h("th", {}, nB), h("th", {}, "Δ (B−A)"))),
        h("tbody", {},
          row("full-val bpc", "best_full_val_bpc_at_best_tau"),
          row("best step", "best_step", 0),
          row("τ0 used", "tau0_used"),
          row("weight entropy", "weight_entropy"),
          row("train-slice bpc", "bpc_train_slice"),
          (vA.per_pass_bpc || []).map((_, i) => h("tr", {},
              h("td", {}, `pass ${i + 1} bpc`),
              h("td", { class: "num" }, fmt(vA.per_pass_bpc[i])),
              h("td", { class: "num" }, fmt(vB.per_pass_bpc?.[i])),
              h("td", { class: "num" },
                vB.per_pass_bpc?.[i] !== undefined ? fmt(vB.per_pass_bpc[i] - vA.per_pass_bpc[i]) : "—"))))));
  }
  for (const [name, v] of entries) {
    root.append(h("details", { class: "rawjson" }, h("summary", {}, `raw JSON — ${name}`),
      h("pre", {}, JSON.stringify(v, null, 1))));
  }
}

/* ======================================================== DIAGNOSIS ======= */
const PROBE_LABELS = {
  A1_live_trainmode_seq128_head100k: ["A1", "live model, train-mode, seq128, head-100k val"],
  A2_live_evalmode_seq128_head100k: ["A2", "same but eval-mode (dropout off) — should equal A1"],
  A3_live_evalmode_seq256_head100k: ["A3", "eval-mode at the trained seq_len 256 — should equal A1/A2"],
  A4_live_evalmode_seq128_after100k: ["A4", "tail of the 100k val window (seq128)"],
  A5_live_evalmode_seq256_after100k: ["A5", "tail of the 100k val window (seq256)"],
  A1_in_loop_reference: ["A1*", "the in-loop eval recorded at step 540 (train-mode, seq128, head-100k)"],
  rescore_reference: ["R", "runner re-score note (older corpus — see re-score reports page for the verified numbers)"],
  T1_live_trainmode_seq128_train100k: ["T1", "train-slice probe, train-mode, seq128"],
  T2_live_evalmode_seq256_train100k: ["T2", "train-slice probe, eval-mode, seq256 — should equal T1"],
  A6_live_full_val_canonical: ["A6", "CANONICAL: full val, eval-mode, seq256 (this is the number of record)"],
  B1_max_param_diff: ["B1", "max |Δparam| after the probe run — must be 0 (probes don't train)"],
  B2_fresh_full_val: ["B2", "fresh reload, full val — must equal A6"],
  C1_max_param_diff_roundtrip: ["C1", "max |Δparam| after save→reload roundtrip — must be 0"],
  C2_roundtrip_full_val: ["C2", "roundtrip reload, full val — must equal A6"],
  D1_live_fastOFF_head100k: ["D1", "fast-math OFF, head-100k"],
  D2_live_fastON_head100k: ["D2", "fast-math ON, head-100k — allowed tiny numeric drift vs D1"],
  D3_rt_fastOFF_head100k: ["D3", "roundtrip reload, fast OFF"],
  D4_rt_fastON_head100k: ["D4", "roundtrip reload, fast ON"],
  E1_forward_max_abs_diff_live_vs_roundtrip: ["E1", "max |Δlogits| live vs roundtrip forward — must be 0"],
};

async function pageDiagnosis(root) {
  const [diag, profiles] = await Promise.all(
    [api("/api/diagnosis"), api("/api/profiles")]);
  root.innerHTML = "";
  root.append(h("h1", {}, "Diagnosis — reload & causality probes"),
    h("p", {}, "Runner-side probe tables: the live model and a save→reload roundtrip are evaluated on identical windows to prove the checkpoint reload is bit-exact and that eval-mode/seq-len/fast-math do not move the numbers."));
  if (!Object.keys(diag.data || {}).length &&
      !Object.keys(profiles.data || {}).length) {
    warnbox(root, "No committed diagnosis files yet (results/diagnosis/*.json + results/profiles/*). ",
      "Dispatch a reload probe: ",
      h("code", {}, "gh workflow run diagnose.yml -f artifact=e1_weights_passes_pilot.yaml-seeds-0-pilot"),
      " — the tables land in results/ and this page fills in.");
    return;
  }

  for (const [name, d] of Object.entries(diag.data || {})) {
    const meta = d.meta || {};
    root.append(h("h2", {}, name));
    root.append(h("div", { class: "cards" },
      card(String(meta.best_step ?? "?"), "best step"),
      card(fmt(meta.tau0, 4), "τ0"),
      card(String(meta.eval_chars ?? "?"), "eval chars"),
      card(String(meta.seq_len_trained ?? "?"), "seq_len trained"),
      card((meta.vocab_sha256 || "").slice(0, 12) + "…", "vocab_sha256")));

    // causality panel
    const caus = d.causality || {};
    root.append(h("h3", {}, "Causality: reload is bit-identical"),
      h("div", { class: "cards" }, ["live", "reloaded"].map(k => {
        const c = caus[k] || {};
        const ok = c.bit_identical === true || c.max_abs_diff_leq_t0 === 0;
        return h("div", { class: "card" },
          h("div", { class: "big", style: `color:${ok ? "var(--good)" : "var(--bad)"}` },
            ok ? "✓ bit-identical" : "✗ differs"),
          h("div", { class: "lbl" }, `${k}: max |Δ| (≤ t0) = ${fmt(c.max_abs_diff_leq_t0, 6)}`));
      })));

    // probe table
    const probes = d.probes || {};
    const rows = [];
    for (const [k, v] of Object.entries(probes)) {
      const [tag, desc] = PROBE_LABELS[k] || [k, ""];
      const isNum = typeof v === "number";
      const bitIdent = isNum && v === 0;
      const val = isNum ? fmt(v, 6) : fmt(v.bpc, 5);
      const passCols = (!isNum && Array.isArray(v.passes)) ? v.passes : null;
      const badge = bitIdent ? h("span", { class: "pill ok" }, "bit-identical ✓")
        : (k.startsWith("D") && isNum === false ? null : null);
      rows.push(h("tr", { class: bitIdent ? "best" : "" },
        h("td", {}, h("strong", {}, tag)),
        h("td", { class: "num" }, val),
        h("td", {}, passCols ? passCols.map((p, i) =>
          h("span", { class: "pill" }, `p${i + 1} ${fmt(p, 3)}`)) : "—"),
        h("td", { class: "muted" }, desc + (v && v.note ? ` — ${v.note}` : "")),
        h("td", {}, badge ?? "")));
    }
    // fast-math pair delta
    const d1 = probes.D1_live_fastOFF_head100k, d2 = probes.D2_live_fastON_head100k;
    if (typeof d1 === "number" && typeof d2 === "number") {
      rows.push(h("tr", {},
        h("td", {}, h("strong", {}, "ΔD")),
        h("td", { class: "num" }, fmt(Math.abs(d1 - d2), 12)),
        h("td", {}, "—"),
        h("td", { class: "muted" }, "|D1 − D2| fast-math drift (expected ~1e-9, harmless)"),
        h("td", {}, h("span", { class: "pill" }, "informational"))));
    }
    root.append(h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "probe"), h("th", {}, "bpc / value"),
        h("th", {}, "per-pass"), h("th", {}, "what it checks"), h("th", {}, "status"))),
      h("tbody", {}, rows)));

    // A6/B2/C2 agreement highlight
    const trio = ["A6_live_full_val_canonical", "B2_fresh_full_val", "C2_roundtrip_full_val"]
      .map(k => probes[k]?.bpc).filter(x => typeof x === "number");
    if (trio.length === 3 && new Set(trio).size === 1) {
      root.append(h("p", {}, h("span", { class: "pill ok" },
        `A6 = B2 = C2 = ${fmt(trio[0], 5)} — fresh reload, roundtrip and canonical full-val agree exactly.`)));
    }
  }

  // profile tables
  for (const [name, text] of Object.entries(profiles.data || {})) {
    const lines = String(text).split("\n").filter(Boolean);
    const tbl = h("table", {},
      h("thead", {}, h("tr", {}, h("th", {}, "component"), h("th", {}, "median ms"),
        h("th", {}, "samples"))),
      h("tbody", {}, lines.map(line => {
        const m = line.match(/^(\S+)\s+median\s+([\d.]+)\s+ms\s+\(n=(\d+)\)\s*$/);
        if (!m) return null;
        return h("tr", {},
          h("td", {}, m[1]),
          h("td", { class: "num" }, fmt(+m[2], 1)),
          h("td", { class: "num" }, m[3]));
      }).filter(Boolean)));
    root.append(h("h2", {}, `Runner profile — ${name}`),
      h("p", { class: "legend" }, "median wall-time per component on the CI runner (torch compute happens there, never locally)."));
    if (tbl.querySelectorAll("tbody tr").length) root.append(tbl);
    else root.append(h("pre", {}, String(text)));
  }
}

/* ============================================================= VOCAB ====== */
async function pageVocab(root) {
  const [rep, voc] = await Promise.all([api("/api/reports"), api("/api/vocab")]);
  root.innerHTML = "";
  root.append(h("h1", {}, "Vocabulary (UBM)"));
  const c = rep.vocab_v1_compression || {};
  const st = Object.fromEntries((c.stats || []).map(x => [x.split, x]));
  const comp = voc.composition || {};
  if (!comp.bases && !(voc.bases || []).length) {
    warnbox(root, "No committed vocabulary yet (expected UBM/vocab_v1.json + the compression history in results/). ",
      "Build it runner-side: ", h("code", {}, "gh workflow run vocab.yml"),
      " — the vocab file and stats are committed back and this page fills in.");
    return;
  }
  const nByte = comp.byte ?? (voc.bases || []).filter(b => b.kind === "byte").length;
  const nSpec = comp.special ?? 8;
  const nMerge = comp.merge ?? (voc.bases || []).filter(b => b.kind === "merge").length;

  // composition bar: byte-fallback vs specials vs learned merges
  const seg = (n, color, label) => n > 0 ? h("div", {
    style: `flex:${n};background:${color};min-width:3px`,
    title: `${label}: ${n}` }) : null;
  const compBar = h("div", { class: "compbar" },
    seg(nByte, "#e5737388", "byte-fallback"),
    seg(nSpec, "#8a95ad", "specials"),
    seg(nMerge, "#81c784", "learned merges"));
  const compLbl = h("div", { class: "muted", style: "font-size:11px;margin:-2px 0 6px" },
    `${nByte} byte-fallback · ${nSpec} specials · ${nMerge} learned merges (of ${voc.size ?? 512})`);

  root.append(h("div", { class: "cards" },
    card(fmt(st.train?.chars_per_token, 3), "chars/token (train)"),
    card(fmt(st.val?.chars_per_token, 3), "chars/token (val)"),
    card(voc.size ?? 512, "vocab size"),
    card(`${st.train ? (st.train.tokens / 1e6).toFixed(1) + "M" : "—"}`,
      "train tokens"),
    card(`${c.merges ?? nMerge} merges`, "learned beyond 256 bytes + 8 specials")));
  root.append(compBar, compLbl);

  // plain-English explainer for the chunking Tolu noticed
  root.append(h("div", { class: "explain" },
    h("b", {}, "Why the model reads in ~2-character pieces — by design. "),
    "At vocab size 512 the tokenizer compresses text to ",
    h("b", {}, `${fmt(st.train?.chars_per_token, 2)} chars/token (train), ${fmt(st.val?.chars_per_token, 2)} (val)`),
    " — i.e. the model reads text in roughly two-character chunks, and ",
    h("b", {}, `${nByte} of the ${voc.size ?? 512} slots are single-byte fallback`),
    ", so ordinary short words are multi-token: “the” = ", h("code", {}, "\" t\""), " + ",
    h("code", {}, "\"he\""), ". This is intentional at pilot scale: the UBM vocabulary is FIXED across every rung so bpc comparisons never mix tokenizers, and half the vocab being raw bytes guarantees nothing can ever be un-representable. ",
    "Rung E5 (ubm promotion) later promotes frequent whole tokens into the embedding during training, which is where chunk granularity improves."));

  // display legend for the hex escapes
  root.append(h("div", { class: "explain" },
    h("b", {}, "Token display. "),
    "ids 0–255 are raw single bytes: printable ASCII shows as-is; control bytes show as ",
    h("code", {}, "\\n"), ", ", h("code", {}, "\\t"), "; high bytes (0x80–0xFF) and the one merge whose bytes are not valid UTF-8 show as ",
    h("code", {}, "<xx>"), " hex — hover any row for the exact bytes. ",
    `${(voc.bases || []).filter(b => b.escaped).length} of ${voc.size ?? 512} entries display as escapes. In checkpoint exports the runner-side decode has already replaced invalid bytes with U+FFFD; the inspector shows those as `,
    h("code", {}, "<fffd>"), "."));

  if (Array.isArray(c.first_20_merges)) {
    root.append(h("h2", {}, "First 20 merges learned"),
      h("p", { class: "legend" }, "shown as JSON-escaped strings — note how they are ~2-char pieces, matching the fertility stats above"),
      h("div", { class: "chips" }, c.first_20_merges.map(m =>
        h("span", { class: "chip on" }, JSON.stringify(m)))));
  }
  if (rep.first20_merges) {
    root.append(h("details", { class: "rawjson" }, h("summary", {}, "merge-freq dump (results/ubm_first20_merges.txt)"),
      h("pre", {}, rep.first20_merges)));
  }
  const q = h("input", { placeholder: "filter tokens… (type e.g. the, or a hex escape like e2)" });
  const info = h("span", { class: "muted" });
  const tbl = h("table", {});
  const more = h("button", {}, "show more");
  root.append(h("h2", {}, `All ${voc.size} bases`),
    h("p", {}, "Merges are shown as their decoded text; byte/special tokens as readable escapes."), q, info,
    tbl, h("div", { class: "toolbar" }, more));
  let limit = 120;
  const fill = () => {
    const query = q.value;
    tbl.innerHTML = "";
    tbl.append(h("thead", {}, h("tr", {}, h("th", {}, "id"), h("th", {}, "text"),
      h("th", {}, "kind"), h("th", {}, "bytes"))));
    const body = h("tbody", {});
    let shown = 0, matched = 0;
    for (const b of (voc.bases || [])) {
      if (query && !(b.text.includes(query) || (b.hex || "").includes(query))) continue;
      matched++;
      if (shown < limit) {
        body.append(h("tr", {}, h("td", { class: "num" }, String(b.id)),
          h("td", { class: b.escaped ? "btk" : "",
            title: b.hex ? `raw bytes (hex): ${b.hex}` : undefined },
            JSON.stringify(b.text)),
          h("td", {}, b.kind || (b.merge ? "merge" : "byte")),
          h("td", { class: "num" }, String(b.n_bytes ?? 0))));
        shown++;
      }
    }
    tbl.append(body);
    info.textContent = `${matched} match${matched === 1 ? "" : "es"} — showing ${shown}`;
    more.style.display = matched > shown ? "" : "none";
  };
  q.addEventListener("input", () => { limit = 120; fill(); });
  more.addEventListener("click", () => { limit += 240; fill(); });
  fill("");
  root.append(h("details", { class: "rawjson" }, h("summary", {}, "raw compression report"),
    h("pre", {}, JSON.stringify(c, null, 1))));
}

/* ========================================================== CORPUS ======== */
async function pageCorpus(root) {
  const rep = await api("/api/reports");
  root.innerHTML = "";
  root.append(h("h1", {}, "Corpus & evaluation split"),
    h("p", {}, "Corpus v1 split = whole held-out books (no book in two parts; dedup by id and body sha16; boilerplate stripped)."));
  const sp = rep.split_info || {};
  if (!(sp.parts || {}).train) {
    warnbox(root, "No data/split_info.json committed yet — the corpus split is produced by the vocab workflow and committed back to data/. ",
      h("code", {}, "gh workflow run vocab.yml"));
    return;
  }
  const partsBox = h("div", { class: "cards" });
  const colorFor = { train: "var(--good)", val: "var(--accent)", test: "var(--accent2)" };
  for (const [name, d] of Object.entries(sp.parts || {})) {
    partsBox.append(h("div", { class: "card" },
      h("div", { class: "big", style: `color:${colorFor[name] || "var(--text)"}` },
        (d.chars ?? 0).toLocaleString()),
      h("div", { class: "lbl" }, `${name} — ${(d.books?.length ?? d.n_books ?? "?")} books`),
      h("div", { class: "lbl muted" }, `sha ${(d.sha256 || "").slice(0, 16)}…`)));
  }
  root.append(partsBox,
    h("p", {}, esc(sp.source || ""), " — ", esc(sp.split_design || "")),
    h("details", { class: "rawjson" }, h("summary", {}, "raw split_info.json"),
      h("pre", {}, JSON.stringify(sp, null, 1))));

  const ubm = rep.ubm_compression;
  if (Array.isArray(ubm) && ubm.length) {
    root.append(h("h2", {}, "UBM compression history"),
      h("p", { class: "legend" }, "each row is a UBM training stage — chars/token should rise as merges accumulate (compression ↑ = fewer tokens per char)."),
      h("table", {}, h("thead", {}, h("tr", {}, h("th", {}, "vocab size"),
        h("th", {}, "merges"), h("th", {}, "val chars"), h("th", {}, "val tokens"),
        h("th", {}, "chars/token"), h("th", {}, "roundtrip"))),
        h("tbody", {}, ubm.map(u => h("tr", {},
          h("td", { class: "num" }, String(u.vocab_size)),
          h("td", { class: "num" }, String(u.merges)),
          h("td", { class: "num" }, (u.val_chars ?? 0).toLocaleString()),
          h("td", { class: "num" }, (u.val_tokens ?? 0).toLocaleString()),
          h("td", { class: "num" }, fmt(u.chars_per_token, 3)),
          h("td", {}, u.roundtrip ? "✓" : "✗"))))));
    root.append(chartBox("chars/token vs vocab size (val sample)", {
      labels: ubm.map(u => String(u.vocab_size)),
      datasets: [{ label: "chars/token", data: ubm.map(u => u.chars_per_token),
        borderColor: "#4fc3f7", backgroundColor: "#4fc3f733", fill: true,
        tension: .2, pointRadius: 4 }],
    }, "line", { xlabel: "vocab size", ylabel: "chars/token" }));
  } else if (ubm) {
    root.append(h("h2", {}, "UBM compression history"), h("pre", {}, JSON.stringify(ubm, null, 1)));
  }
}

/* ====================================================== PLAYGROUND ====== */
/* Prompt the real checkpoint via the `generate` GitHub Actions workflow:
   dispatch → the runner (torch lives there, never here) generates and
   commits results/generation/gen-*.json → we poll until it lands. */
function pagePlayground(root) {
  root.innerHTML = "";
  root.append(h("h1", {}, "Playground — prompt the real model"),
    h("p", { class: "legend" },
      "Generation runs on GitHub Actions (torch stays on the runner — never on this box). ",
      "Dispatched → queued → torch install (~3–6 min) → generation (seconds) → the result is committed to ",
      h("code", {}, "results/generation/"), " on main and shows up below. ",
      h("b", {}, "Quality note: "),
      "the current checkpoint is E1@step-540 — early training, so gibberish output is the model's real state until the plateau runs land."));

  const ta = h("textarea", { rows: "4", id: "prompt",
    placeholder: "Type a prompt… e.g. [ Illustration] or anything, really" });
  const mk = (id, val, extra) => h("input",
    Object.assign({ id, value: val, spellcheck: "false" }, extra || {}));
  const inMax = mk("maxnew", "200", { type: "number", min: "1", max: "800" });
  const inTemp = mk("temp", "0.8", { type: "number", min: "0.1", max: "2", step: "0.1" });
  const inTopk = mk("topk", "40", { type: "number", min: "0", max: "500" });
  const inSeed = mk("seed", "0", { type: "number" });
  const inArt = mk("artifact", "e1_weights_passes_pilot.yaml-seeds-0-pilot");
  const ctl = (lbl, el) => h("label", { class: "pctl" }, h("span", {}, lbl), el);
  const btn = h("button", { class: "genbtn" }, "Generate");
  const status = h("div", { class: "genstatus" });
  const out = h("div", { class: "genout" });
  root.append(ta,
    h("div", { class: "toolbar" },
      ctl("max new tokens", inMax), ctl("temperature", inTemp),
      ctl("top-k", inTopk), ctl("seed", inSeed), ctl("checkpoint artifact", inArt),
      btn),
    status, out);

  const showResult = (res) => {
    out.innerHTML = "";
    const meta = res.meta || {};
    const nTok = Array.isArray(res.generated_ids) ? res.generated_ids.length : res.n_gen_tokens;
    out.append(h("div", { class: "cards" },
      card(`${meta.run || "—"}`, "run"),
      card(`${meta.step ?? "—"} · ${meta.stop_kind ?? "—"}`, "checkpoint step"),
      card(`${fmt(meta.tau0, 2)} τ0 · ${meta.temperature ?? "—"} T · top-k ${meta.top_k ?? "—"} · seed ${meta.seed ?? "—"}`,
        "sampling"),
      card(nTok != null ? String(nTok) : "—", "generated tokens"),
      card(res.elapsed_s != null ? `${fmt(res.elapsed_s, 2)}s` : "—", "generation time")));
    out.append(h("h3", {}, "prompt"),
      h("pre", { class: "genprompt" }, esc(res.prompt ?? "")));
    out.append(h("h3", {}, "generated continuation"),
      h("pre", { class: "gentext" }, esc(res.generated ?? "(empty)")));
    // ---- token timeline ------------------------------------------------
    // color = which dot produced the token (expert_dots per-token index),
    // when the export carries it. Per-token CONFIDENCE coloring needs
    // per-token logprobs in the generation JSON, which the committed
    // generate.py does not write — stub noted below, generate.py untouched.
    const ids = Array.isArray(res.generated_ids) ? res.generated_ids : null;
    if (ids && ids.length) {
      const dots = Array.isArray(res.expert_dots) ? res.expert_dots : null;
      const strip = h("div", { class: "tl" });
      ids.forEach((id, i) => {
        const k = dots ? dots[i] : null;
        strip.append(h("span", {
          class: "tl-chip" + (k == null ? "" : " hasdot"),
          style: k != null ? `border-color:${PALETTE[k % PALETTE.length]};color:${PALETTE[k % PALETTE.length]}`
            : `background:rgba(79,195,247,${(0.12 + 0.55 * (i % 6) / 5).toFixed(2)})`,
          title: `token ${i + 1}/${ids.length} · id ${id}` +
            (k != null ? ` · dot ${k}` : "") },
          String(id)));
      });
      out.append(h("h3", {}, "token timeline"),
        h("p", { class: "legend" },
          "each chip is one generated token id",
          dots ? " — colored by the dot that produced it (expert_dots)." : "."),
        strip);
      if (dots) {
        const legend = h("div", { class: "chips" });
        [...new Set(dots)].sort((a, b) => a - b).forEach(k =>
          legend.append(h("span", { class: "chip on" },
            h("span", { class: "dot", style: `background:${PALETTE[k % PALETTE.length]}` }),
            `dot ${k}`)));
        out.append(legend);
      }
      out.append(h("p", { class: "muted stub" },
        "Note: coloring chips by model confidence (per-token logprob) is not possible yet — the committed generate.py writes no per-token logprobs into results/generation/*.json. Adding `logprobs` per token there is a generate.py change (not made in this dashboard-only task); the position coloring above uses the expert_dots index the export already carries.",
        dots ? "" : " This export also has no expert_dots — chips are plain position-tinted."));
    }
  };

  /* ---- history: locally committed generations ---- */
  const hist = h("div", {});
  root.append(h("h2", {}, "History — committed generations"),
    h("p", { class: "legend" }, "results/generation/*.json as committed to main by past generate runs. Click to load."),
    hist);
  const loadHist = async () => {
    hist.innerHTML = "";
    try {
      const g = await api("/api/generations");
      const items = g.generations || [];
      if (!items.length) {
        hist.append(h("p", { class: "muted" },
          "no committed generations yet (expected results/generation/gen-*.json) — generate one above, or run ",
          h("code", {}, "gh workflow run generate.yml -f prompt='…' -f artifact=<checkpoint artifact>"),
          " and pull once it commits."));
        return;
      }
      for (const it of items) {
        const m = it.meta || {};
        const b = h("button", { class: "chip", title: `${it.prompt || ""}` },
          `${it.file} · ${(m.run || "?").split("_seed")[0]} · step ${m.step ?? "?"} · ${fmt(it.elapsed_s, 1)}s`);
        b.addEventListener("click", () => {
          showResult(it);
          window.scrollTo({ top: 0, behavior: "smooth" });
        });
        hist.append(b, " ");
      }
    } catch (e) {
      hist.append(h("div", { class: "errbox" }, `history unavailable: ${e.message}`));
    }
  };
  loadHist();

  /* ---- dispatch + poll ---- */
  const say = (t, cls) => { status.textContent = t; status.className = "genstatus" + (cls ? " " + cls : ""); };
  btn.addEventListener("click", async () => {
    btn.disabled = true;
    out.innerHTML = "";
    const body = { prompt: ta.value, max_new_tokens: inMax.value,
      temperature: inTemp.value, top_k: inTopk.value, seed: inSeed.value,
      artifact: inArt.value.trim() };
    let resp;
    try {
      say("dispatching…");
      resp = await apiSend("/api/generate", body);
    } catch (e) {
      say(`✗ ${e.message}`, "bad"); btn.disabled = false; return;
    }
    const t0 = Date.parse(resp.dispatched_at);
    // find the run this dispatch created: created_at after our dispatch,
    // within a 2-min window so an older in-flight run is never grabbed
    let runId = null;
    for (let tries = 0; tries < 25 && !runId; tries++) {
      say("dispatched… looking for the workflow run…");
      await new Promise(r => setTimeout(r, 8000));
      if (!document.contains(status)) return;            // user navigated away
      try {
        const d = await api("/api/generate/runs");
        for (const r of d.runs || []) {
          const dt = Date.parse(r.created_at) - t0;
          if (dt >= 0 && dt <= 120000) { runId = r.id; break; }
        }
      } catch (e) { say(`✗ ${e.message}`, "bad"); btn.disabled = false; return; }
    }
    if (!runId) { say("✗ no workflow run appeared within 3 min — check the Actions tab", "bad"); btn.disabled = false; return; }
    say(`run ${runId} found — waiting for the runner (torch install dominates, ~3–6 min)…`);
    // poll the result until done/failed
    for (let tries = 0; tries < 120; tries++) {
      await new Promise(r => setTimeout(r, 8000));
      if (!document.contains(status)) return;
      let d;
      try { d = await api(`/api/generate/result?run_id=${runId}`); }
      catch (e) { say(`✗ ${e.message}`, "bad"); btn.disabled = false; return; }
      if (d.status === "done") {
        say(`✓ done — ${d.file} committed${d.result?.elapsed_s != null ? ` · generated in ${fmt(d.result.elapsed_s, 2)}s` : ""}`, "ok");
        showResult(d.result || {});
        loadHist();
        btn.disabled = false;
        return;
      }
      if (d.status === "failed") {
        say(`✗ run concluded ${d.conclusion || "unsuccessfully"}${d.note ? ` — ${d.note}` : ""} — check the Actions log`, "bad");
        btn.disabled = false;
        return;
      }
      say(`run ${runId}: ${d.run_status || "generating…"} — torch install dominates (~3–6 min), generation itself is seconds…`);
    }
    say("✗ still pending after ~16 min — check the Actions tab", "bad");
    btn.disabled = false;
  });
}

/* ---------------- boot ---------------- */
$("#navtoggle").addEventListener("click", () => {
  const hidden = $("#sidebar").classList.toggle("hidden");
  $("#navtoggle").setAttribute("aria-expanded", String(!hidden));
});
/* mobile: the sidebar is an overlay; a nav click or Escape dismisses it */
const closeMobileNav = () => {
  if (window.matchMedia && window.matchMedia("(max-width: 800px)").matches)
    $("#sidebar").classList.add("hidden");
  $("#navtoggle").setAttribute("aria-expanded", "false");
};
$$(".nav", $("#sidebar")).forEach(a =>
  a.addEventListener("click", closeMobileNav));
window.addEventListener("keydown", e => {
  if (e.key === "Escape") closeMobileNav();
});
(function boot() {
  try {
    const head = $("#git-head");
    fetch("/api/runs").then(r => r.json()).then(d => {
      if (d && d.git_head) head.textContent = d.git_head;
      else head.textContent = "(git unavailable)";
    }).catch(() => { head.textContent = "(git unavailable)"; });
  } catch (e) { /* footer only */ }
  route();
})();
