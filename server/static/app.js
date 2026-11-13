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
const fmt = (x, d = 4) => (x === null || x === undefined ||
  (typeof x === "number" && !isFinite(x))) ? "—" : Number(x).toFixed(d);
const esc = s => String(s ?? "");
const uniq = a => [...new Set(a)];

const PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#4db6ac",
                 "#f06292", "#fff176", "#64b5f6", "#aed581"];
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
if (HAS_CHART) {
  Chart.defaults.color = "#8a95ad";
  Chart.defaults.borderColor = "#26304a";
  Chart.defaults.font.family =
    "-apple-system, 'Segoe UI', Roboto, Helvetica, Arial, sans-serif";
  Chart.defaults.plugins.legend.labels.boxWidth = 10;
  Chart.defaults.plugins.legend.labels.usePointStyle = true;
  Chart.defaults.plugins.tooltip.backgroundColor = "#1d2431";
  Chart.defaults.plugins.tooltip.borderColor = "#2a3345";
  Chart.defaults.plugins.tooltip.borderWidth = 1;
  Chart.defaults.plugins.tooltip.titleColor = "#dce3f2";
}

const AXIS_GRID = { color: "#1c2434", tickColor: "#5b657d" };
const AXIS_TICK = { color: "#8a95ad" };

function chartBox(title, data, type = "line", opts = {}) {
  const box = h("div", { class: "chartbox" },
    h("div", { class: "ct" }, h("h3", {}, title),
      opts.sub ? h("span", { class: "sub" }, opts.sub) : null));
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
      new Chart(cv, { type, data, options: {
        responsive: true, maintainAspectRatio: false,
        interaction: { mode: "nearest", intersect: false },
        plugins: {
          legend: { display: opts.legend !== false,
                    position: opts.legendPos || "top" },
          tooltip: opts.tooltip || undefined,
        },
        scales: opts.scales || {
          x: axis({ type: opts.xtype || "category",
                    title: { display: !!opts.xlabel, text: opts.xlabel || "",
                             color: "#8a95ad" } }),
          y: axis({ type: "linear",
                    title: { display: !!opts.ylabel, text: opts.ylabel || "",
                             color: "#8a95ad" } }),
        },
      }});
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

/* Runs live on different step grids — align every series onto the sorted
   union of steps so multi-run charts stay honest (missing points = null). */
function alignSteps(datasets, key) {
  const steps = uniq(datasets.flatMap(d => d.series.step)).sort((a, b) => a - b);
  const series = datasets.map(d => {
    const m = new Map(d.series.step.map((s, i) => [s, d.series[key]?.[i] ?? null]));
    return steps.map(s => (m.has(s) ? m.get(s) : null));
  });
  return { steps, series };
}

/* ---------------- shared state ---------------- */
const state = { runsIndex: null, arch: null, sel: null, runData: {}, vizData: {} };

/* ---------------- router ---------------- */
const PAGES = {
  dashboard:    { fn: pageDashboard,    title: "Dashboard" },
  runs:         { fn: pageRuns,         title: "Training runs" },
  scores:       { fn: pageScores,       title: "Re-score reports" },
  architecture: { fn: pageArchitecture, title: "Architecture" },
  checkpoints:  { fn: pageCheckpoints,  title: "Weights & dots" },
  diagnosis:    { fn: pageDiagnosis,    title: "Diagnosis" },
  vocab:        { fn: pageVocab,        title: "Vocab (UBM)" },
  corpus:       { fn: pageCorpus,       title: "Corpus & split" },
};
function route() {
  const parts = location.hash.replace(/^#\/?/, "").split("/").filter(Boolean);
  const name = PAGES[parts[0]] ? parts[0] : "dashboard";
  $$(".nav").forEach(a => a.classList.toggle("active", a.dataset.page === name));
  $("#pgtitle").textContent = PAGES[name].title;
  const page = $("#page");
  page.innerHTML = "";
  page.append(h("div", { class: "loading" }, "· · ·"));
  Promise.resolve(PAGES[name].fn(page, parts[1])).catch(e => {
    page.innerHTML = "";
    page.append(h("h1", {}, PAGES[name].title),
      h("div", { class: "errbox" }, "Could not build this page: ",
        h("code", {}, String(e && e.message || e))));
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

/* ======================================================== DASHBOARD ======= */
const PILOT_RUNGS = [
  { label: "B0 GRU",              rung: "b0_gru_pilot",                  color: "#81c784" },
  { label: "E0 plain ensemble",   rung: "e0_plain_ensemble_pilot",       color: "#4fc3f7" },
  { label: "E1 weights+passes",   rung: "e1_weights_passes_pilot",       color: "#ffb74d" },
  { label: "E1 no-λ (ablation)",  rung: "e1_weights_passes_nolambda_pilot", color: "#e57373" },
];

async function pageDashboard(root) {
  const [runs, evals, arch] = await Promise.all(
    [api("/api/runs"), api("/api/evals"), api("/api/arch")]);
  state.runsIndex = runs; state.arch = arch;
  root.innerHTML = "";
  root.append(h("h1", {}, "Dashboard"),
    h("p", {}, "Read straight from the repository (results CSVs, eval re-scores, configs). ",
      "No training happens on this machine — all torch compute runs on GitHub Actions."));

  const byName = name => runs.runs.find(r => r.name === name);
  const evalFor = rung => Object.entries(evals).find(([k]) => k.includes(rung));

  // headline cards — numbers pulled from live eval re-scores / CSVs, never hardcoded
  const cards = h("div", { class: "cards" });
  for (const pr of PILOT_RUNGS) {
    const ev = evalFor(pr.rung);
    let big, lbl, sub;
    if (ev) {
      const [k, v] = ev;
      big = fmt(v.best_full_val_bpc_at_best_tau, 5);
      lbl = `${pr.label} — full-val bpc (re-scored)`;
      sub = `eval ${k} · best step ${v.best_step} · τ0 ${fmt(v.tau0_used, 4)}`;
    } else {
      const r = byName("train_" + pr.rung + "_seed0");
      big = r && r.best_val !== null ? fmt(r.best_val, 5) : "—";
      lbl = `${pr.label} — best val bpc (100k-char sample)`;
      sub = r ? `${r.points} eval pts · last step ${r.last_step}` : "no data yet";
    }
    cards.append(h("div", { class: "card click", onclick: () => { location.hash = "#/scores"; } },
      h("div", { class: "big", style: `color:${pr.color}` }, big),
      h("div", { class: "lbl" }, lbl),
      h("div", { class: "lbl muted" }, sub)));
  }
  root.append(cards);

  // ladder chart
  const ladder = {
    labels: PILOT_RUNGS.map(x => x.label),
    datasets: [{ label: "bpc (lower is better)",
                 data: PILOT_RUNGS.map(pr => {
                   const ev = evalFor(pr.rung);
                   if (ev) return ev[1].best_full_val_bpc_at_best_tau;
                   const r = byName("train_" + pr.rung + "_seed0");
                   return r && r.best_val !== null ? r.best_val : null;
                 }),
                 backgroundColor: PILOT_RUNGS.map(x => x.color + "cc"),
                 borderColor: PILOT_RUNGS.map(x => x.color),
                 borderRadius: 6, maxBarThickness: 72 }],
  };
  root.append(chartBox("The ladder so far — E0 → E1 (equal-compute pilots)",
    ladder, "bar", { sub: "full-val re-score where available, else best sampled val", ylabel: "bpc" }));

  // sparkline strip of the pilot curves
  const spark = h("div", { class: "sparkrow" });
  for (const pr of PILOT_RUNGS) {
    const r = byName("train_" + pr.rung + "_seed0");
    if (!r) continue;
    spark.append(h("div", { class: "spark", onclick: () => {
        location.hash = `#/runs/${r.name}`; } },
      h("div", { class: "t", style: `color:${pr.color}` }, pr.label),
      h("div", { class: "v" }, fmt(r.best_val, 4), h("span", { class: "s" }, " best val")),
      h("div", { class: "s" }, `seed 0 · ${r.points} eval pts · step ${r.last_step}`),
      h("div", { class: "spw" }, h("canvas", {}))));
  }
  // fill sparklines after the strip is in the DOM (needs /api/run data)
  if (spark.children.length) {
    root.append(spark);
    Promise.all(PILOT_RUNGS.map(pr => byName("train_" + pr.rung + "_seed0")
      ? api(`/api/run/${byName("train_" + pr.rung + "_seed0").name}`).catch(() => null) : null))
      .then(ds => {
        $$(".spark .spw canvas", spark).forEach((cv, i) => {
          const d = ds[i];
          if (d && HAS_CHART && d.series?.bpc_val) {
            const s = d.series.bpc_val.filter(v => v !== null);
            const c = PILOT_RUNGS[i].color;
            new Chart(cv, { type: "line", data: {
                labels: s.map((_, j) => j),
                datasets: [{ data: s, borderColor: c, borderWidth: 1.6,
                             pointRadius: 0, tension: .25 }] },
              options: { animation: false, maintainAspectRatio: false,
                plugins: { legend: { display: false } },
                scales: { x: { display: false }, y: { display: false } } } });
          }
        });
      });
  }

  // eval report highlights
  const evs = Object.entries(evals);
  if (evs.length) {
    const box = h("div", { class: "cards" });
    for (const [k, v] of evs) {
      box.append(h("div", { class: "card click",
          onclick: () => { location.hash = "#/scores"; } },
        h("div", { class: "big", style: "color:var(--accent)" },
          fmt(v.best_full_val_bpc_at_best_tau, 5)),
        h("div", { class: "lbl" }, `${k} — full-val bpc @ best τ`),
        h("div", { class: "lbl muted" },
          `step ${v.best_step} · passes ${fmt(v.per_pass_bpc?.[0], 3)} → ${fmt(v.per_pass_bpc?.[2], 3)}`)));
    }
    root.append(h("h2", {}, "Checkpoint re-scores"), box);
  }

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
  root.append(h("h2", {}, "All runs in results/"), tbl,
    h("div", { class: "muted" }, `git ${esc(runs.git_head)}`));
}

/* ============================================================= RUNS ======= */
async function pageRuns(root, preselect) {
  if (!state.runsIndex) state.runsIndex = await api("/api/runs");
  const runs = state.runsIndex.runs;
  if (!state.sel) {
    state.sel = new Set(runs.filter(r => /pilot/.test(r.name)).map(r => r.name));
    if (preselect && runs.some(r => r.name === preselect)) state.sel.add(preselect);
  }
  root.innerHTML = "";
  root.append(h("h1", {}, "Training runs"),
    h("p", {}, "Curves come from the committed results/*.csv. Click chips to toggle runs. ",
      "Grad-norm columns (gn_*) only exist on runs trained after 2026-11-13 — the chart appears automatically when the selected runs have them."));

  const chips = h("div", { class: "chips" });
  for (const r of runs) {
    const on = state.sel.has(r.name);
    const chip = h("span", { class: "chip" + (on ? " on" : ""), onclick: () => {
      state.sel.has(r.name) ? state.sel.delete(r.name) : state.sel.add(r.name);
      chip.classList.toggle("on");
      draw();
    }}, h("span", { class: "dot", style: `background:${rungColor(r.rung)}` }),
      `${r.rung} · s${r.seed}`, h("span", { class: "n" }, `best ${fmt(r.best_val)}`));
    chips.append(chip);
  }
  root.append(chips);
  const plots = h("div", {});
  root.append(plots);
  let lastReq = 0;
  async function draw() {
    const req = ++lastReq;
    const names = runs.filter(r => state.sel.has(r.name)).map(r => r.name);
    plots.innerHTML = "";
    if (!names.length) { plots.append(h("p", { class: "muted" }, "no runs selected")); return; }
    let datasets;
    try {
      datasets = await Promise.all(names.map(n => api(`/api/run/${n}`)));
    } catch (e) { errbox(plots, e); return; }
    if (req !== lastReq) return;             // a newer selection superseded us
    datasets.forEach((d, i) => state.runData[names[i]] = d);
    const N = names.length;
    // main: val vs train bpc (aligned on the union step grid)
    const aV = alignSteps(datasets, "bpc_val"), aT = alignSteps(datasets, "bpc_train");
    plots.append(chartBox("Validation bpc (solid) vs train bpc (dashed)", {
      labels: aV.steps,
      datasets: datasets.flatMap((d, i) => {
        const c = rungColor(d.rung), lbl = `${d.rung}·s${d.seed}`;
        return [
          { label: `val ${lbl}`, data: aV.series[i], borderColor: c,
            pointRadius: 0, tension: 0.15, spanGaps: true },
          { label: `train ${lbl}`, data: aT.series[i], borderColor: c,
            borderDash: [4, 4], pointRadius: 0, tension: 0.15, spanGaps: true,
            hidden: N > 2 },
        ];
      }),
    }, "line", { ylabel: "bpc", xlabel: "step" }));
    // per-pass
    const has = (d, k) => Array.isArray(d.series?.[k]) && d.series[k].some(v => v !== null);
    const pp = datasets.filter(d => has(d, "bpc_p1"));
    if (pp.length) {
      const aP1 = alignSteps(pp, "bpc_p1"), aP2 = alignSteps(pp, "bpc_p2"),
            aP3 = alignSteps(pp, "bpc_p3");
      plots.append(chartBox("Per-pass bpc (later passes should beat earlier ones)", {
        labels: aP1.steps,
        datasets: pp.flatMap((d, i) => [
          { label: `p1 ${d.rung}`, data: aP1.series[i], borderColor: "#7986cb",
            pointRadius: 0, tension: .15, spanGaps: true },
          { label: `p2 ${d.rung}`, data: aP2.series[i], borderColor: "#4fc3f7",
            pointRadius: 0, tension: .15, spanGaps: true },
          { label: `p3 ${d.rung}`, data: aP3.series[i], borderColor: "#ffd54f",
            pointRadius: 0, tension: .15, spanGaps: true },
        ]),
      }, "line", { ylabel: "bpc", xlabel: "step" }));
    }
    // tau0 + weight entropy
    const ie = datasets.filter(d => has(d, "tau0"));
    if (ie.length) {
      const aTau = alignSteps(ie, "tau0"), aEnt = alignSteps(ie, "weight_entropy");
      plots.append(chartBox("τ0 (left) and weight entropy (right axis)", {
        labels: aTau.steps,
        datasets: ie.flatMap((d, i) => [
          { label: `tau0 ${d.rung}`, data: aTau.series[i], borderColor: rungColor(d.rung),
            yAxisID: "y", pointRadius: 0, spanGaps: true },
          { label: `w-entropy ${d.rung}`, data: aEnt.series[i],
            borderColor: rungColor(d.rung), borderDash: [4, 4], yAxisID: "y1",
            pointRadius: 0, spanGaps: true },
        ]),
    }, "line", { scales: {
      x: AXIS_GRID,
      y: { ...AXIS_GRID, title: { display: true, text: "tau0" } },
      y1: { position: "right", ...AXIS_GRID,
            title: { display: true, text: "entropy (nats)" }, grid: { drawOnChartArea: false } } } }));
    }
    // grad norms (only when the columns exist)
    const gn = datasets.filter(d => has(d, "gn_total"));
    if (gn.length) {
      const aGn = alignSteps(gn, "gn_total");
      plots.append(chartBox("Pre-clip gradient norms per component", {
        labels: aGn.steps,
        datasets: gn.flatMap((d, i) => ["gn_emb", "gn_regions", "gn_chain", "gn_gate",
          "gn_messages", "gn_other"].filter(k => has(d, k)).map(k => {
            const a = alignSteps(gn, k);
            return {
              label: `${d.rung} ${k.slice(3)}`, data: a.series[i],
              borderColor: rungColor(d.rung), pointRadius: 0, spanGaps: true };
          })),
      }, "line", { ylabel: "L2 norm", xlabel: "step" }));
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
  await draw();
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
    root.append(h("h2", {}, "Token embedding map (2-D projection of the tied table)"),
      h("p", { class: "legend" }, `Each point is a vocab entry (${pts.length}). Nearby points are embedded close together. Hover to read tokens.`));
    const cv = h("canvas", { style: "max-height:420px" });
    root.append(h("div", { class: "chartbox" }, cv));
    if (HAS_CHART) requestAnimationFrame(() => {
      try {
        new Chart(cv, { type: "scatter",
          data: { datasets: [{ data: pts.map((p, i) => ({ x: p[0], y: p[1],
            label: v.embeddings.labels?.[i] ?? String(i) })),
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
      h("span", { class: "pill " + (hits / Math.max(1, scored) >= .5 ? "ok" : "") },
        `pooled top-1 correct on ${hits}/${scored} positions (${fmt(100 * hits / Math.max(1, scored), 1)}%)`)));

  const bar = h("input", { type: "range", class: "posbar", min: "0",
    max: String(nPos - 1), value: "0", "aria-label": "token position" });
  const posLabel = h("div", { class: "muted" });
  const zone = h("div", {});
  root.append(bar, posLabel, zone);

  const redraw = () => {
    const i = Math.min(+bar.value, nPos - 1), P = S.positions[i];
    if (!P) return;
    posLabel.textContent = `position ${i} / ${nPos - 1} · token “${esc(P.token)}” → target next “${esc(P.next)}” (id ${P.next_id ?? "?"})`;
    zone.innerHTML = "";
    // token context strip
    const toks = Array.isArray(S.tokens) ? S.tokens : null;
    if (toks && toks.length) {
      const a = Math.max(0, i - 14), b = Math.min(toks.length, i + 8);
      const strip = h("div", { class: "tokline" });
      if (a > 0) strip.append("… ");
      toks.slice(a, b).forEach((t, j) => {
        const idx = a + j;
        if (idx === i) strip.append(h("span", { class: "cur" }, esc(t)));
        else if (idx === i + 1) strip.append(h("span", { class: "tgt" }, esc(t)));
        else strip.append(esc(t));
        strip.append(" ");
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
    passes.forEach((pname, pi) => {
      const tr = h("tr", {}, h("td", { class: "rowh" }, esc(pname)));
      dots.forEach((d, k) => {
        const w = d.weights?.[pi] ?? 0, g = d.gamma?.[pi] ?? 0;
        tr.append(h("td", {}, h("div", { class: "cell",
          style: `background:${HEAT(w / rowMax[pi])}`,
          title: `w=${fmt(w, 4)} γ=${fmt(g, 3)} (row-normalised ${fmt(w / rowMax[pi], 2)})`
        }, fmt(w, 2))));
      });
      tbl.append(tr);
    });
    zone.append(h("h3", {}, "dot weights per pass (brighter = more weight; hover for γ)"), tbl);

    // pooled answer
    const pool = P.pooled_top || [];
    zone.append(h("h3", {}, "pooled region answer (log-linear over dots)"),
      h("div", { class: "dotcard" },
        ...pool.map(([tok, prob]) =>
          h("span", { class: "pill" + (tok === P.next ? " ok" : "") },
            `${esc(tok)} ${fmt(prob * 100, 1)}%`)),
        h("div", { class: "muted" },
          pool[0] && pool[0][0] === P.next ? "✓ top-1 == target" : `✗ top-1 ≠ target “${esc(P.next)}”`)));

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
        cardsBox.append(h("div", { class: "dotcard" + (hit ? " hit" : "") },
          h("div", { class: "hd" },
            h("strong", {}, `dot ${k}`, hit ? "  ✓" : ""),
            h("span", { class: "muted" }, `w=${fmt(d.weights?.[pl], 3)} γ=${fmt(d.gamma?.[pl], 3)}`)),
          h("div", { class: "muted" }, `${esc(passes[pl])} top predictions: `),
          ...top.map(([tok, prob]) =>
            h("span", { class: "pill" + (tok === P.next ? " ok" : "") },
              `${esc(tok)} ${fmt(prob * 100, 1)}%`))));
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
  // gamma + weight trajectory chart per dot
  if (S.positions[0]?.dots?.length) {
    const nd = S.positions[0].dots.length;
    const gsets = [];
    for (let k = 0; k < nd; k++)
      gsets.push({ label: `γ dot ${k}`, data: S.positions.map(P => P.dots[k]?.gamma?.[0] ?? null),
                   borderColor: PALETTE[k % PALETTE.length], pointRadius: 0, tension: .2 });
    root.append(chartBox("dot confidence γ (pass 1) along the sample", {
      labels: nChart, datasets: gsets,
    }, "line", { xlabel: "token position", ylabel: "γ" }));
  }
}

/* ========================================================= SCORES ========= */
async function pageScores(root) {
  const [evals, runs] = await Promise.all([api("/api/evals"), api("/api/runs")]);
  root.innerHTML = "";
  root.append(h("h1", {}, "Re-score reports (corpus-verified)"),
    h("p", {}, "Each checkpoint's best step was re-scored on the FULL validation set at that step's τ0 — the numbers here supersede the sampled val curve. ",
      "vocab_sha256 is checked across every report below and against the viz export + diagnosis table: a mismatch would fail loudly runner-side."));

  const entries = Object.entries(evals);
  if (!entries.length) { warnbox(root, "No eval_*.json reports in results/ yet."); return; }

  // vocab sha consistency
  const shas = new Set(entries.map(([, v]) => v.vocab_sha256).filter(Boolean));
  const shaOK = shas.size === 1;
  const sha = [...shas][0] || "";
  root.append(h("div", { class: "cards" },
    h("div", { class: "card" },
      h("div", { class: "big", style: `color:${shaOK ? "var(--good)" : "var(--bad)"}` },
        shaOK ? "✓ consistent" : "✗ MISMATCH"),
      h("div", { class: "lbl" }, `vocab_sha256 across ${entries.length} reports`),
      h("div", { class: "lbl muted" }, sha ? sha.slice(0, 16) + "…" : "—"))));

  for (const [name, v] of entries) {
    const passes = v.per_pass_bpc || [];
    root.append(h("h2", {}, name,
      h("span", { class: "pill ok" }, "full-val")));
    root.append(h("div", { class: "cards" },
      card(fmt(v.best_full_val_bpc_at_best_tau, 5), "full-val bpc @ best τ"),
      card(String(v.best_step ?? "—"), "best step"),
      card(fmt(v.tau0_used, 4), "τ0 used"),
      card(fmt(v.weight_entropy, 4), "weight entropy at best step"),
      card(fmt(v.hard_bpc_at_best_tau, 5), "hard (top-m) bpc"),
      card(fmt(v.mode_gap, 5), "mode gap (hard − soft)"),
      card(fmt(v.bpc_train_slice, 5), "train-slice bpc"),
      card((v.vocab_sha256 || "").slice(0, 12) + "…", "vocab_sha256")));
    if (passes.length) {
      root.append(chartBox(`Per-pass bpc — ${name}`, {
        labels: passes.map((_, i) => `pass ${i + 1}`),
        datasets: [{ label: "bpc", data: passes,
          backgroundColor: ["#7986cbbb", "#4fc3f7bb", "#ffd54fbb"],
          borderColor: ["#7986cb", "#4fc3f7", "#ffd54f"],
          borderWidth: 1, borderRadius: 6 }],
      }, "bar", { sub: "each pass should improve on the last (iterative refinement)", ylabel: "bpc", legend: false }));
    }
    // compare against the run's own sampled val curve
    const runName = name.replace(/^eval_/, "").replace(/_seed(\d+)$/, "_seed$1");
    const run = runs.runs.find(r => r.name === runName);
    if (run) {
      root.append(h("p", {}, `Sampled val curve for ${run.name}: best ${fmt(run.best_val)} at step ${run.best_step ?? "?"} (100k-char sample — expect the full-val re-score to sit nearby). `,
        h("a", { href: `#/runs/${run.name}` }, "open curve →")));
    }
  }

  // side-by-side comparison when there are exactly 2+ evals
  if (entries.length >= 2) {
    const [nA, vA] = entries[0], [nB, vB] = entries[1];
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
  root.append(h("div", { class: "cards" },
    card(voc.size ?? 512, "vocab size"),
    card(`${c.merges ?? "?"} merges`, "256 bytes + 8 specials + merges"),
    card(fmt(st.train?.chars_per_token, 3), "chars/token (train)"),
    card(fmt(st.val?.chars_per_token, 3), "chars/token (val)")));
  if (Array.isArray(c.first_20_merges)) {
    root.append(h("h2", {}, "First 20 merges learned"),
      h("p", { class: "legend" }, "shown as JSON-escaped strings"),
      h("div", { class: "chips" }, c.first_20_merges.map(m =>
        h("span", { class: "chip on" }, JSON.stringify(m)))));
  }
  if (rep.first20_merges) {
    root.append(h("details", { class: "rawjson" }, h("summary", {}, "merge-freq dump (results/ubm_first20_merges.txt)"),
      h("pre", {}, rep.first20_merges)));
  }
  const q = h("input", { placeholder: "filter tokens… (type e.g. the)" });
  const info = h("span", { class: "muted" });
  const tbl = h("table", {});
  const more = h("button", {}, "show more");
  root.append(h("h2", {}, `All ${voc.size} bases`),
    h("p", {}, "Merges are shown as their decoded text; byte/special tokens as escaped text."), q, info,
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
      if (query && !(b.text.includes(query) || (b.raw_text || "").includes(query))) continue;
      matched++;
      if (shown < limit) {
        body.append(h("tr", {}, h("td", { class: "num" }, String(b.id)),
          h("td", {}, JSON.stringify(b.text)),
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

/* ---------------- boot ---------------- */
$("#navtoggle").addEventListener("click", () => $("#sidebar").classList.toggle("hidden"));
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
