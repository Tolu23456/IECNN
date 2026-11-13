/* IECNN Explorer — reads only /api/* served by server/server.py. */
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
  for (const kid of kids.flat()) {
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
const fmt = (x, d = 4) => (x === null || x === undefined || isNaN(x)) ? "—" :
  Number(x).toFixed(d);
const uniq = a => [...new Set(a)];
const PALETTE = ["#4fc3f7", "#ffb74d", "#81c784", "#e57373", "#ba68c8", "#4db6ac",
                 "#f06292", "#fff176", "#64b5f6", "#aed581"];
function rungColor(rung) {
  if (/^b0/.test(rung)) return "#81c784";
  if (/^e0/.test(rung)) return "#4fc3f7";
  if (/nolambda/.test(rung)) return "#e57373";
  if (/^e1/.test(rung)) return "#ffb74d";
  if (/^e[2-9]/.test(rung)) return "#ba68c8";
  return "#8a95ad";
}
const HEAT = w => {                       // weight in [0,1] -> blue-hot color
  const t = Math.max(0, Math.min(1, w));
  const r = Math.round(30 + 225 * t), g = Math.round(60 + 60 * t),
        b = Math.round(150 - 90 * t);
  return `rgb(${r},${g},${b})`;
};

Chart.defaults.color = "#8a95ad";
Chart.defaults.borderColor = "#2a3448";
Chart.defaults.plugins.legend.labels.boxWidth = 10;

/* ---------------- shared state ---------------- */
const state = { runsIndex: null, arch: null, runData: {} };

/* ---------------- router ---------------- */
const PAGES = {
  dashboard: pageDashboard, architecture: pageArchitecture, runs: pageRuns,
  checkpoints: pageCheckpoints, vocab: pageVocab, corpus: pageCorpus,
};
function route() {
  const name = (location.hash.replace(/^#\//, "") || "dashboard").split("/")[0];
  $$(".nav").forEach(a => a.classList.toggle("active", a.hash === `#/${name}`));
  const page = $("#page");
  page.innerHTML = "";
  page.append(h("div", { class: "loading" }, "loading…"));
  (PAGES[name] || pageDashboard)(page).catch(e => {
    page.innerHTML = "";
    page.append(h("p", { class: "err" }, String(e)));
  });
}
window.addEventListener("hashchange", route);

/* ======================================================== DASHBOARD ======= */
async function pageDashboard(root) {
  const [runs, evals, arch] = await Promise.all(
    [api("/api/runs"), api("/api/evals"), api("/api/arch")]);
  state.runsIndex = runs; state.arch = arch;
  root.innerHTML = "";
  root.append(h("h1", {}, "Dashboard"));
  root.append(h("p", {}, "Everything below is read straight from the repository ",
    "(results CSVs, eval reports, vocab). No training happens on this machine."));

  // headline numbers from the four-way pilot
  const headline = [
    ["B0 GRU", "1.7330", "patience @9360 steps (best 8400)"],
    ["E0 plain ensemble", "2.4682", "wall cap @2640 — still falling"],
    ["E1 weights+passes", "3.7772", "wall cap @541 actual steps"],
    ["E1 no-lambdas", "3.9669", "wall cap @361 actual steps"],
  ];
  const cards = h("div", { class: "cards" });
  for (const [name, bpc, note] of headline) {
    cards.append(h("div", { class: "card" },
      h("div", { class: "big", style: `color:${rungColor(name.toLowerCase().replace(/ /g, "_"))}` }, bpc),
      h("div", { class: "lbl" }, `${name} — full-val bpc`), h("div", { class: "lbl muted" }, note)));
  }
  const s = runs.runs[0] ? `git ${runs.git_head}` : "";
  root.append(cards, h("div", { class: "muted" }, s));

  // ladder chart
  const ladder = {
    labels: headline.map(x => x[0]),
    datasets: [{ label: "full-val bpc (best point, best-step tau)",
                 data: headline.map(x => +x[1]),
                 backgroundColor: headline.map(x => rungColor(x[0].toLowerCase().replace(/ /g, "_"))) }],
  };
  root.append(chartBox("The ladder so far — lower is better", ladder, "bar"));

  // latest runs table
  const tbl = h("table", {},
    h("tr", {}, h("th", {}, "run"), h("th", {}, "seed"), h("th", {}, "eval points"),
      h("th", {}, "last step"), h("th", {}, "best val (sample)")),
    runs.runs.map(r => h("tr", {},
      h("td", {}, h("a", { href: `#/runs` }, r.rung + (r.seed !== null ? ` · s${r.seed}` : ""))),
      h("td", { class: "num" }, String(r.seed ?? "—")),
      h("td", { class: "num" }, String(r.points)),
      h("td", { class: "num" }, String(r.last_step ?? "—")),
      h("td", { class: "num" }, fmt(r.best_val, 4)))));
  root.append(h("h2", {}, "Runs in results/"), tbl);

  // eval reports
  for (const [k, v] of Object.entries(evals)) {
    root.append(h("h2", {}, `Checkpoint re-score — ${k}`),
      h("pre", {}, JSON.stringify(v, null, 1)));
  }
}

function chartBox(title, data, type = "line", opts = {}) {
  const box = h("div", { class: "chartbox" }, h("h3", {}, title));
  const cv = h("canvas", {});
  box.append(cv);
  requestAnimationFrame(() => new Chart(cv, { type, data, options: {
    responsive: true, maintainAspectRatio: false,
    scales: opts.scales || { x: { title: { display: true, text: opts.xlabel || "step" } },
                              y: { title: { display: true, text: opts.ylabel || "" } } },
    ...opts.plugins,
  }}));
  return box;
}

/* ===================================================== ARCHITECTURE ======= */
const STAGES = [
  { id: "text", t: "1 · Text → tokens (UBM)", d: "NFC-normalize, strip boilerplate, encode with the fixed 512-base UBM vocab (256 bytes + 8 specials + 248 learned merges).",
    shape: "chars → ids [T] (+ expansion lengths for bpc accounting)",
    what: "One fixed vocabulary for every rung: comparisons never mix tokenizers. Each token credits the characters its decode expands to, so Σ expansion lengths = text length exactly.",
    formula: "token tₖ ← longest merge path in UBM;  bpc = NLL / Σ expansion_len(t̂)",
    knobs: "UBM/vocab_v1.json (sha256 recorded in every checkpoint; mismatch fails loudly)" },
  { id: "embed", t: "2 · Embedding (tied)", d: "Every token id → a d-dim vector; the SAME table is reused as the output projection.",
    shape: "E [V=512, d=64]; rows = E(ids) [B, T, d]",
    what: "Tied embeddings keep the parameter count small and let every dot's readout point at real token vectors.",
    formula: "rows = E(ids)",
    knobs: "model.d_model" },
  { id: "region", t: "3 · Region r: window memory + N dots", d: "Each position keeps a W-token working window; N independent dots each run a gated state update from that window.",
    shape: "rows [B, T, d] → states s [B, T, N, d_s]  (R=3 regions, N=4 dots, W=16 window)",
    what: "A dot is a tiny gated reader with its OWN parameters for this round (pass) — dots are the independent 'experts' whose disagreement the consensus machinery will harvest.",
    formula: "z = σ(W_z x + U_z s);  s' = (1−z)·tanh(W n + U r s) + z·s",
    knobs: "model.n_regions, n_dots, window, d_s; dot_dropout (E4: one random dot masked on 5% of steps)" },
  { id: "dotout", t: "4 · Dot heads: prediction, confidence, trust", d: "Each dot emits its own full next-token distribution plus two scalar quality signals.",
    shape: "log πₖ [B, T, N, V];  γₖ [B, T, N];  trust-termₖ [B, T, N]",
    what: "πₖ = the dot's private prediction (through the tied E). γₖ = self-rated confidence. The trust term is the dot's raw vote entering the score below.",
    formula: "logitsₖ = (P·sₖ)·Eᵀ;  log πₖ = log softmax(logitsₖ);  γₖ = c_w·sₖ + c_b;  trustₖ = g_w·sₖ + g_b",
    knobs: "shared per round; params for round p are indexed by pass" },
  { id: "consensus", t: "5 · Consensus: scores → trust → weights", d: "Turn the N private predictions into N mixing weights, softly, every pass.",
    shape: "weights w [B, T, N] (per pass)",
    what: "Score each dot by how much it DISAGREES with the leave-one-out crowd (novelty) plus its confidence γ and its trust carry-over. softmax with temperature τ_p mixes.",
    formula: "uₖ = α·(−JS(πₖ ‖ π_loo)) + β·γₖ + Tₖ;   T^p = ρ·T^(p−1) + u^p;   wₖ = softmax(Tₖ / τ_p) with floor ε = 0.1/N",
    knobs: "α, β (learned); ρ=0.6; τ_p = τ0·κ^(p−1), τ0 annealed 1.0→0.2; floor active until final 10% of steps" },
  { id: "passes", t: "6 · Passes 1..P: pooled opinion + error features", d: "Pass 1: dots predict blind. Pass p>1: every dot READS the previous pass's pooled opinion and its own disagreement numbers.",
    shape: "h_pool [B, T, d_s] broadcast to all dots; err [B, T, N, 3]",
    what: "This is the 'iterative refinement' core: the crowd's pooled answer becomes an input, and each dot sees (crowd mean, own-vs-crowd JS) so it can correct.",
    formula: "pooled = Σₖ wₖ·sₖ ;  errₖ = features(πₖ, crowd, JSₖ);  log_b^p = Σₖ wₖ·log πₖ^p (log-linear pooling)",
    knobs: "model.n_passes=3, pooling=log_linear|mixture; share_pass_params" },
  { id: "messages", t: "7 · Messages between passes", d: "After each pass the region writes a summary of its states; dots read it as extra input on the NEXT pass.",
    shape: "msg [B, T, ·] written at pass p, read at pass p+1",
    what: "Lets dots coordinate beyond the pooled opinion — ablatable (use_messages).",
    formula: "msg_write = M(s^p);  input at p+1 += msg_read",
    knobs: "model.use_messages; message_dropout 0.2 (regularization parity across rungs)" },
  { id: "chain", t: "8 · Chain region r → r+1", d: "The next region's input rows are built from THIS region's final weighted summary.",
    shape: "rows_{r+1} = LN(W·summary_r) [B, T, d]",
    what: "Region 1 sees raw embeddings; region 2 sees what region 1 concluded; region 3 sees region 2. Early regions ≈ spelling, later ≈ meaning (the win-map diagnostic tests this).",
    formula: "rows_{r+1} = chain_ln_r(chain_proj_r(summary_r))",
    knobs: "chain_proj/chain_ln per region" },
  { id: "gates", t: "9 · Cross-region competition (D5)", d: "Regions vote on which region should dominate the final answer.",
    shape: "ω [B, T, R]",
    what: "Each region's confidence (negative entropy of its consensus) plus a learned gate scalar → softmax over regions.",
    formula: "score_r = α_region·(−H(log_b_r)) + gate_r;  ω = softmax(score);  log_y = Σ_r ω_r·log_b_r",
    knobs: "alpha_region, region_gate (rung configs with use_cross_region)" },
  { id: "loss", t: "10 · Loss + the λ family", d: "NLL on the pooled answer, plus auxiliary terms that keep the machinery honest.",
    shape: "scalar loss",
    what: "L_region keeps early-region predictions useful; L_dot (boosted) trains dots against the pooled answer; L_corr decorrelates dots; L_conf calibrates γ against actual errors; L_w REWARDS weight entropy so the ensemble stays diverse.",
    formula: "L = NLL(log_y) + λ_r·L_region + λ_d·L_dot + λ_c·L_corr + λ_k·L_conf − λ_w·H(w)",
    knobs: "λ_r, λ_d, λ_c, λ_k, λ_w with ramp; E1-no-λ pilot turns them all off" },
  { id: "train", t: "11 · Training loop & honest stopping", d: "AdamW + grad clip; eval every eval_every steps on a 100k-char val sample; best point re-scored on FULL validation at the tau of that step.",
    shape: "CSV: step, lr, loss, bpc_val, bpc_p1..p3, bpc_train, tau0, weight_entropy, L_w, elapsed, gn_* (per-component grad norms)",
    what: "Early stop = 6 evals without ≥0.005 bpc improvement (noise floor ≈0.002). stop_kind records patience vs wall_cap vs reached-target — a wall-capped 'best' is NOT a plateau. Warmup 500 then constant LR for the next wave (one identical schedule per rung).",
    formula: "patience: stop when since_best ≥ 6 evals AND best_full improvement < 0.005",
    knobs: "training.lr_schedule=constant|cosine, eval_chars=100000, early_stop{patience_evals, min_improvement, wall_hours}" },
];

async function pageArchitecture(root) {
  const arch = state.arch || await api("/api/arch");
  root.innerHTML = "";
  root.append(h("h1", {}, "How IECNN works"));
  root.append(h("p", {}, "Click any stage. Numbers are read from the committed ",
    h("code", {}, "configs/e1_weights_passes.yaml"),
    ` — so the page cannot drift from what actually runs. Current rung: R=${arch.R} regions × ${arch.n_dots} dots × ${arch.n_passes} passes, d=${arch.d_model}, d_s=${arch.d_s}, window=${arch.window}, pooling=${arch.pooling}.`));
  const flow = h("div", { class: "flow" });
  const detail = h("div", { class: "detail" }, h("em", { class: "muted" }, "pick a stage above"));
  STAGES.forEach((s, i) => {
    const el = h("div", { class: "stage", onclick: () => {
      $$(".stage").forEach(x => x.classList.remove("sel"));
      el.classList.add("sel");
      detail.innerHTML = "";
      detail.append(h("h3", {}, s.t),
        h("div", { class: "f" }, s.shape),
        h("p", {}, s.what),
        h("div", { class: "f" }, s.formula),
        h("p", {}, h("strong", {}, "knobs: "), s.knobs));
      detail.scrollIntoView({ behavior: "smooth", block: "nearest" });
    }}, h("div", { class: "t" }, s.t), h("div", { class: "d" }, s.d),
        h("div", { class: "shape" }, s.shape));
    flow.append(el);
    if (i < STAGES.length - 1) flow.append(h("div", { class: "arrow" }, "▼"));
  });
  root.append(flow, detail);
}

/* ============================================================= RUNS ======= */
async function pageRuns(root) {
  if (!state.runsIndex) state.runsIndex = await api("/api/runs");
  const runs = state.runsIndex.runs;
  root.innerHTML = "";
  root.append(h("h1", {}, "Training runs"),
    h("p", {}, "Curves come from the committed results/*.csv. Grad-norm columns (gn_*) exist only on runs trained after 2026-11-13."));

  const sel = h("select", { multiple: true, size: Math.min(10, runs.length), style: "width:340px" },
    runs.map(r => h("option", { value: r.name, selected: /pilot/.test(r.name) },
      `${r.rung} · seed ${r.seed} · ${r.points} pts · best ${fmt(r.best_val)}`)));
  const plots = h("div", {});
  root.append(h("h2", {}, "Pick runs (ctrl-click for several)"), sel,
    h("button", { onclick: () => draw(sel) }, "plot"), plots);

  function draw(sel) {
    plots.innerHTML = "";
    const names = [...sel.selectedOptions].map(o => o.value);
    if (!names.length) return;
    Promise.all(names.map(n => api(`/api/run/${n}`))).then(datasets => {
      datasets.forEach((d, i) => state.runData[names[i]] = d);
      const N = names.length;
      // main: val vs train bpc
      root_append(plots, chartBox("Validation bpc (solid) vs train bpc (dashed)", {
        labels: datasets[0].series.step,
        datasets: datasets.flatMap((d, i) => {
          const c = rungColor(d.rung), lbl = `${d.rung}·s${d.seed}`;
          return [
            { label: `val ${lbl}`, data: d.series.bpc_val, borderColor: c,
              pointRadius: 0, tension: 0.15, spanGaps: true },
            { label: `train ${lbl}`, data: d.series.bpc_train, borderColor: c,
              borderDash: [4, 4], pointRadius: 0, tension: 0.15, spanGaps: true,
              hidden: N > 2 },
          ];
        }),
      }, "line", { ylabel: "bpc" }));
      // per-pass
      const pp = datasets.filter(d => d.series.bpc_p1?.some(v => v !== null));
      if (pp.length) root_append(plots, chartBox("Per-pass bpc (later passes should beat earlier ones)", {
        labels: pp[0].series.step,
        datasets: pp.flatMap(d => [
          { label: `p1 ${d.rung}`, data: d.series.bpc_p1, borderColor: "#7986cb",
            pointRadius: 0, tension: .15, spanGaps: true },
          { label: `p2 ${d.rung}`, data: d.series.bpc_p2, borderColor: "#4fc3f7",
            pointRadius: 0, tension: .15, spanGaps: true },
          { label: `p3 ${d.rung}`, data: d.series.bpc_p3, borderColor: "#ffd54f",
            pointRadius: 0, tension: .15, spanGaps: true },
        ]),
      }, "line", { ylabel: "bpc" }));
      // tau0 + weight entropy
      const ie = datasets.filter(d => d.series.tau0?.some(v => v !== null));
      if (ie.length) root_append(plots, chartBox("τ0 (left) and weight entropy (right axis)", {
        labels: ie[0].series.step,
        datasets: ie.flatMap(d => [
          { label: `tau0 ${d.rung}`, data: d.series.tau0, borderColor: rungColor(d.rung),
            yAxisID: "y", pointRadius: 0, spanGaps: true },
          { label: `w-entropy ${d.rung}`, data: d.series.weight_entropy,
            borderColor: rungColor(d.rung), borderDash: [4, 4], yAxisID: "y1",
            pointRadius: 0, spanGaps: true },
        ]),
      }, "line", { scales: {
        y: { title: { display: true, text: "tau0" } },
        y1: { position: "right", title: { display: true, text: "entropy" }, grid: { drawOnChartArea: false } } } }));
      // grad norms
      const gn = datasets.filter(d => d.series.gn_total?.some(v => v !== null));
      if (gn.length) root_append(plots, chartBox("Pre-clip gradient norms per component", {
        labels: gn[0].series.step,
        datasets: gn.flatMap(d => ["gn_emb", "gn_regions", "gn_chain", "gn_gate",
          "gn_messages", "gn_other"].map(k => ({
          label: `${d.rung} ${k.slice(3)}`, data: d.series[k],
          borderColor: rungColor(d.rung), pointRadius: 0, spanGaps: true }))),
      }, "line", { ylabel: "L2 norm" }));

      // matched-step table between the first two selected runs
      if (names.length >= 2) {
        const A = state.runData[names[0]], B = state.runData[names[1]];
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
          h("table", {}, h("tr", {}, h("th", {}, "step"),
            h("th", {}, names[0]), h("th", {}, names[1]), h("th", {}, "Δ (B−A)")), rows));
      }
    });
  }
  // auto-plot the pilots on first load
  const auto = [...sel.options].filter(o => o.selected).map(o => o.value);
  if (auto.length) { [...sel.selectedOptions].forEach(o => o.selected = false);
    auto.slice(0, 2).forEach(n => [...sel.options].find(o => o.value === n).selected = true);
    draw(sel); }
}
function root_append(parent, el) { parent.append(el); }

/* ======================================================= CHECKPOINTS ====== */
async function pageCheckpoints(root) {
  root.innerHTML = "";
  root.append(h("h1", {}, "Weights & dots — checkpoint inspector"));
  let list;
  try { list = (await api("/api/viz")).exports; }
  catch { list = []; }
  if (!list.length) {
    root.append(h("p", { class: "warn" },
      "No checkpoint exports yet. Real weights/dots live inside torch checkpoints that only exist as CI artifacts — per the no-local-torch rule they are exported ON THE RUNNER by the viz workflow. ",
      h("code", {}, "gh workflow run viz.yml -f artifact=<checkpoint artifact>"),
      " writes results/viz/<run>.json and this page fills in."));
    return;
  }
  const sel = h("select", {}, list.map(n => h("option", { value: n }, n)));
  const view = h("div", {});
  root.append(h("p", {}, "Export: "), sel, view);
  const show = async () => renderViz(view, await api(`/api/viz/${sel.value}`));
  sel.addEventListener("change", show);
  await show();
}

async function renderViz(root, v) {
  root.innerHTML = "";
  if (v.error) { root.append(h("p", { class: "err" }, v.error)); return; }
  root.append(h("div", { class: "cards" },
    card(v.meta?.run || v.run, "run"),
    card(`step ${v.meta?.step ?? "?"}`, "checkpoint step"),
    card(v.meta?.stop_kind ?? "?", "stop kind"),
    card(`${(v.meta?.params ?? 0).toLocaleString()}`, "parameters"),
    card(`τ ${fmt(v.meta?.tau0 ?? 1, 3)}`, "eval temperature")));

  // ---- embedding map
  if (v.embeddings?.points) {
    root.append(h("h2", {}, "Token embedding map (2-D projection of the tied table)"),
      h("p", { class: "legend" }, "Each point is a vocab entry (512). Nearby points are embedded close together. Hover to read tokens."));
    const cv = h("canvas", { style: "max-height:420px" });
    root.append(h("div", { class: "chartbox" }, cv));
    const pts = v.embeddings.points;
    requestAnimationFrame(() => new Chart(cv, { type: "scatter",
      data: { datasets: [{ data: pts.map((p, i) => ({ x: p[0], y: p[1],
        label: v.embeddings.labels?.[i] ?? String(i) })),
        pointRadius: 2.5, backgroundColor: "#4fc3f7aa" }] },
      options: { plugins: { tooltip: { callbacks: {
        label: c => c.raw.label } } },
        scales: { x: { type: "linear" }, y: { type: "linear" } } } }));
  }

  // ---- sample forward inspector
  const S = v.sample;
  if (!S) return;
  root.append(h("h2", {}, "Sample forward — dots in action"),
    h("p", {}, `Text fed through the real checkpoint: “${S.text}”`,
      h("br"), "Drag the position slider to inspect any token: how much each dot trusted itself (weights), its confidence (γ), and what each dot actually predicted."));
  const bar = h("input", { type: "range", class: "posbar", min: 0,
    max: String(S.positions.length - 1), value: "0" });
  const posLabel = h("div", { class: "muted" });
  const zone = h("div", {});
  root.append(bar, posLabel, zone);
  const redraw = () => {
    const i = +bar.value, P = S.positions[i];
    posLabel.textContent = `position ${i} · token “${P.token}” · target next “${P.next}” (id ${P.next_id})`;
    zone.innerHTML = "";
    // weights heatmap passes × dots
    const tbl = h("table", { class: "heat" });
    tbl.append(h("tr", {}, h("th", {}), P.dots.map((_, k) => h("th", {}, `dot ${k}`))));
    S.passes.forEach((pname, pi) => {
      const tr = h("tr", {}, h("td", {}, pname));
      P.dots.forEach((d, k) => {
        const w = d.weights[pi], g = d.gamma[pi];
        tr.append(h("td", {}, h("div", { class: "cell",
          style: `background:${HEAT(w / (Math.max(...P.dots.map(x => Math.max(...x.weights)))) || 1)}`,
          title: `w=${fmt(w,3)} γ=${fmt(g,2)}` }, `${fmt(w, 2)}`)));
      });
      tbl.append(tr);
    });
    zone.append(h("h3", {}, "dot weights per pass (brighter = more weight)"), tbl);
    // per-dot predictions at last pass
    const pl = S.passes.length - 1;
    const cardsBox = h("div", {});
    P.dots.forEach((d, k) => {
      cardsBox.append(h("div", { class: "dotcard" },
        h("strong", {}, `dot ${k}`), `  γ=${fmt(d.gamma[pl], 2)}  w=${fmt(d.weights[pl], 3)}`,
        h("div", { class: "muted" }, "top predictions (last pass): "),
        ...d.top[pl].map(([tok, prob]) =>
          h("span", { class: "pill" }, `${tok} ${fmt(prob * 100, 1)}%`))));
    });
    zone.append(h("h3", {}, "what each dot predicted"), cardsBox);
    // pooled answer
    zone.append(h("div", { class: "dotcard" }, h("strong", {}, "pooled region answer: "),
      ...P.pooled_top.map(([tok, prob]) => h("span", { class: "pill" }, `${tok} ${fmt(prob * 100, 1)}%`)),
      h("div", { class: "muted" }, P.pooled_top[0] && P.pooled_top[0][0] === P.next ?
        " ✓ correct" : ` ✗ (target “${P.next}”)`)));
  };
  bar.addEventListener("input", redraw);
  redraw();

  if (v.weight_entropy_over_positions) {
    root.append(h("h2", {}, "Weight entropy along the sample"),
      h("p", { class: "legend" }, "High entropy = dots disagree and share the load; near 0 = one dot dominates. Compare with the uniform baseline log(N)."));
    root.append(chartBox("entropy of dot weights per position", {
      labels: v.sample.positions.map((_, i) => i),
      datasets: [{ label: "H(w)", data: v.weight_entropy_over_positions,
                   borderColor: "#ffb74d", pointRadius: 0, tension: .2 }],
    }, "line", { xlabel: "token position", ylabel: "entropy (nats)" }));
  }
}
function card(txt, lbl) {
  return h("div", { class: "card" }, h("div", { class: "big" }, String(txt)),
    h("div", { class: "lbl" }, lbl));
}

/* ============================================================= VOCAB ====== */
async function pageVocab(root) {
  const [rep, voc] = await Promise.all([api("/api/reports"), api("/api/vocab")]);
  root.innerHTML = "";
  root.append(h("h1", {}, "Vocabulary (UBM)"));
  const c = rep.vocab_v1_compression || {};
  root.append(h("div", { class: "cards" },
    card(c.vocab_size ?? voc.size, "vocab size"),
    card(fmt(c.chars_per_token_train ?? c.train_chars_per_token, 3), "chars/token train"),
    card(fmt(c.chars_per_token_val ?? c.val_chars_per_token, 3), "chars/token val"),
    card("512 = 256+8+248", "bytes + specials + merges")));
  root.append(h("h2", {}, "Compression report"), h("pre", {},
    JSON.stringify(c, null, 1) || "(report not found)"));
  if (rep.first20_merges) root.append(h("h2", {}, "First 20 merges learned"),
    h("pre", {}, rep.first20_merges));
  const q = h("input", { placeholder: "filter tokens…", oninput: () => fill(q.value) });
  const tbl = h("table", {});
  root.append(h("h2", {}, `All ${voc.size} bases`), h("p", {}, "Merges shown as a+b."),
    q, tbl);
  const fill = query => {
    tbl.innerHTML = "";
    tbl.append(h("tr", {}, h("th", {}, "id"), h("th", {}, "text"), h("th", {}, "kind"),
      h("th", {}, "bytes")));
    for (const b of voc.bases) {
      if (query && !b.text.includes(query)) continue;
      tbl.append(h("tr", {}, h("td", { class: "num" }, String(b.id)),
        h("td", {}, JSON.stringify(b.text)), h("td", {}, b.merge ? "merge" : "byte/special"),
        h("td", { class: "num" }, String(b.n_bytes))));
      if (tbl.children.length > 600) break;
    }
  };
  fill("");
}

/* ======================================================== CORPUS ========== */
async function pageCorpus(root) {
  const rep = await api("/api/reports");
  root.innerHTML = "";
  root.append(h("h1", {}, "Corpus & evaluation reports"));
  const sp = rep.split_info || {};
  root.append(h("p", {}, "Corpus v1 split = whole held-out books (no book in two parts; dedup by id and body sha16; boilerplate stripped)."));
  const partsBox = h("div", { class: "cards" });
  for (const [name, d] of Object.entries(sp.parts || {})) {
    partsBox.append(h("div", { class: "card" },
      h("div", { class: "big" }, (d.chars ?? 0).toLocaleString()),
      h("div", { class: "lbl" }, `${name} — ${d.n_books ?? "?"} books`),
      h("div", { class: "lbl muted" }, `sha ${(d.sha256 || "").slice(0, 16)}…`)));
  }
  root.append(partsBox, h("h2", {}, "split_info.json"), h("pre", {},
    JSON.stringify(sp, null, 1)));
  root.append(h("h2", {}, "UBM compression history"),
    h("pre", {}, JSON.stringify(rep.ubm_compression || {}, null, 1)));
}

/* ---------------- boot ---------------- */
route();
