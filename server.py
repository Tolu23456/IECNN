"""
IECNN Web Interface
A simple Flask app to run the IECNN pipeline interactively.
"""

import json
import numpy as np
from flask import Flask, request, jsonify, render_template_string

app = Flask(__name__)

CORPUS = [
    "neural networks learn from data using gradient descent",
    "deep learning models use many layers to extract features",
    "transformers use self-attention to model relationships",
    "natural language processing understands human language",
    "computer vision enables machines to interpret images",
    "reinforcement learning trains agents through rewards",
    "convolutional networks process spatial data efficiently",
    "recurrent networks handle sequential information over time",
    "generative models can create new data from learned distributions",
    "embeddings map words into continuous vector spaces",
]

from iecnn import IECNN

model = IECNN(
    feature_dim=128,
    num_dots=64,
    max_iterations=8,
    similarity_threshold=0.45,
    dominance_threshold=0.70,
    novelty_threshold=0.05,
    seed=42,
)
model.fit(CORPUS)

HTML = """
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8"/>
  <meta name="viewport" content="width=device-width, initial-scale=1.0"/>
  <title>IECNN — Iterative Emergent Convergent Neural Network</title>
  <style>
    *, *::before, *::after { box-sizing: border-box; margin: 0; padding: 0; }
    body {
      font-family: 'Segoe UI', system-ui, sans-serif;
      background: #0d0f1a;
      color: #e2e8f0;
      min-height: 100vh;
      padding: 2rem 1rem;
    }
    .container { max-width: 860px; margin: 0 auto; }
    header { text-align: center; margin-bottom: 2.5rem; }
    header h1 {
      font-size: 2.2rem; font-weight: 800;
      background: linear-gradient(90deg, #a78bfa, #60a5fa);
      -webkit-background-clip: text; -webkit-text-fill-color: transparent;
      background-clip: text;
    }
    header p { color: #94a3b8; margin-top: 0.4rem; }
    .card {
      background: #1a1d2e; border: 1px solid #2d3148;
      border-radius: 12px; padding: 1.5rem; margin-bottom: 1.5rem;
    }
    .card h2 { color: #a78bfa; font-size: 1rem; margin-bottom: 1rem; letter-spacing: 0.05em; text-transform: uppercase; }
    textarea {
      width: 100%; min-height: 90px; padding: 0.75rem 1rem;
      background: #0d0f1a; color: #e2e8f0;
      border: 1px solid #2d3148; border-radius: 8px;
      font-size: 0.95rem; font-family: inherit; resize: vertical;
    }
    textarea:focus { outline: none; border-color: #6d28d9; }
    button {
      margin-top: 1rem; padding: 0.65rem 1.75rem;
      background: linear-gradient(135deg, #6d28d9, #2563eb);
      color: #fff; border: none; border-radius: 8px;
      font-size: 0.95rem; font-weight: 600; cursor: pointer;
    }
    button:hover { opacity: 0.88; }
    button:disabled { opacity: 0.4; cursor: not-allowed; }
    .results { display: none; }
    .results.visible { display: block; }
    .stat-grid {
      display: grid; grid-template-columns: repeat(auto-fill, minmax(160px, 1fr));
      gap: 0.75rem; margin-bottom: 1rem;
    }
    .stat {
      background: #131625; border: 1px solid #2d3148;
      border-radius: 8px; padding: 0.75rem 1rem;
    }
    .stat-label { color: #64748b; font-size: 0.75rem; text-transform: uppercase; margin-bottom: 0.25rem; }
    .stat-value { color: #c4b5fd; font-size: 1.1rem; font-weight: 700; }
    .round-row {
      display: flex; gap: 1rem; align-items: center;
      padding: 0.5rem 0; border-bottom: 1px solid #1e2130; font-size: 0.9rem;
    }
    .round-row:last-child { border-bottom: none; }
    .round-num { color: #6d28d9; font-weight: 700; min-width: 60px; }
    .round-detail { color: #94a3b8; flex: 1; }
    .dom-bar-wrap { width: 120px; background: #0d0f1a; border-radius: 4px; height: 8px; }
    .dom-bar { height: 8px; border-radius: 4px; background: linear-gradient(90deg, #6d28d9, #2563eb); }
    .bases-list { display: flex; flex-wrap: wrap; gap: 0.4rem; margin-top: 0.5rem; }
    .base-tag {
      background: #2d1b69; color: #c4b5fd;
      border: 1px solid #4c1d95; padding: 0.2rem 0.6rem;
      border-radius: 12px; font-size: 0.8rem;
    }
    .base-tag.char { background: #1e3a5f; color: #93c5fd; border-color: #1d4ed8; }
    .base-tag.phrase { background: #1a3a1a; color: #86efac; border-color: #166534; }
    .bias-row { display: flex; align-items: center; gap: 1rem; padding: 0.35rem 0; font-size: 0.88rem; }
    .bias-label { color: #94a3b8; min-width: 150px; }
    .bias-bar-wrap { flex: 1; background: #0d0f1a; border-radius: 4px; height: 6px; }
    .bias-bar { height: 6px; border-radius: 4px; background: linear-gradient(90deg, #a78bfa, #60a5fa); }
    .bias-val { color: #c4b5fd; min-width: 45px; text-align: right; font-weight: 600; }
    .stop-badge {
      display: inline-block; padding: 0.2rem 0.75rem;
      border-radius: 12px; font-size: 0.8rem; font-weight: 600;
    }
    .stop-budget { background: #7c2d12; color: #fdba74; }
    .stop-dominance { background: #14532d; color: #86efac; }
    .stop-novelty { background: #1e3a5f; color: #93c5fd; }
    .error-box { background: #3b0a0a; border: 1px solid #7f1d1d; border-radius: 8px; padding: 1rem; color: #fca5a5; }
    #spinner { display: none; margin-left: 1rem; color: #6d28d9; }
  </style>
</head>
<body>
<div class="container">
  <header>
    <h1>IECNN</h1>
    <p>Iterative Emergent Convergent Neural Network — Live Demo</p>
  </header>

  <div class="card">
    <h2>Input Text</h2>
    <textarea id="inputText" placeholder="Enter any text to run through the IECNN pipeline…">IECNN uses neural dots that independently generate predictions and converge toward stable emergent outputs.</textarea>
    <br/>
    <button id="runBtn" onclick="runIECNN()">Run IECNN</button>
    <span id="spinner">Processing…</span>
  </div>

  <div id="errorBox" style="display:none; margin-bottom:1.5rem;"></div>

  <div id="results" class="results">
    <div class="card">
      <h2>Pipeline Summary</h2>
      <div class="stat-grid" id="summaryStats"></div>
      <div id="stopBadge" style="margin-top:0.5rem;"></div>
    </div>

    <div class="card">
      <h2>BaseMapping</h2>
      <div id="basemapInfo" style="color:#94a3b8; font-size:0.9rem; margin-bottom:0.75rem;"></div>
      <div class="bases-list" id="basesList"></div>
    </div>

    <div class="card">
      <h2>Iteration Rounds</h2>
      <div id="roundsList"></div>
    </div>

    <div class="card">
      <h2>Top Cluster</h2>
      <div id="clusterInfo"></div>
    </div>

    <div class="card">
      <h2>Bias Vector (after learning)</h2>
      <div id="biasInfo"></div>
    </div>
  </div>
</div>

<script>
async function runIECNN() {
  const text = document.getElementById('inputText').value.trim();
  if (!text) return;

  document.getElementById('runBtn').disabled = true;
  document.getElementById('spinner').style.display = 'inline';
  document.getElementById('results').classList.remove('visible');
  document.getElementById('errorBox').style.display = 'none';

  try {
    const resp = await fetch('/run', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ text })
    });
    const data = await resp.json();

    if (data.error) {
      document.getElementById('errorBox').innerHTML =
        '<div class="error-box">Error: ' + data.error + '</div>';
      document.getElementById('errorBox').style.display = 'block';
    } else {
      renderResults(data);
      document.getElementById('results').classList.add('visible');
    }
  } catch(e) {
    document.getElementById('errorBox').innerHTML =
      '<div class="error-box">Request failed: ' + e.message + '</div>';
    document.getElementById('errorBox').style.display = 'block';
  }

  document.getElementById('runBtn').disabled = false;
  document.getElementById('spinner').style.display = 'none';
}

function renderResults(d) {
  // Summary stats
  const statsHtml = [
    { label: 'Rounds', value: d.rounds },
    { label: 'Bases', value: d.num_bases },
    { label: 'Candidates', value: d.total_candidates },
    { label: 'Top Score', value: d.top_score.toFixed(4) },
    { label: 'Dominance', value: (d.dominance * 100).toFixed(1) + '%' },
    { label: 'Top Cluster Size', value: d.top_cluster_size },
  ].map(s =>
    `<div class="stat"><div class="stat-label">${s.label}</div><div class="stat-value">${s.value}</div></div>`
  ).join('');
  document.getElementById('summaryStats').innerHTML = statsHtml;

  // Stop reason badge
  const stopClass = {
    'iteration_budget': 'stop-budget',
    'convergence_dominance': 'stop-dominance',
    'low_novelty_gain': 'stop-novelty',
  }[d.stop_reason] || 'stop-budget';
  document.getElementById('stopBadge').innerHTML =
    `Stop reason: <span class="stop-badge ${stopClass}">${d.stop_reason.replace(/_/g,' ')}</span>`;

  // BaseMap
  document.getElementById('basemapInfo').textContent =
    `${d.num_bases} bases · matrix shape (${d.num_bases}, 128) · types: ` +
    Object.entries(d.base_types).map(([k,v]) => `${v} ${k}`).join(', ');
  const basesHtml = d.bases.map(b =>
    `<span class="base-tag ${b.type}">${b.token}</span>`
  ).join('');
  document.getElementById('basesList').innerHTML = basesHtml;

  // Rounds
  const roundsHtml = d.rounds_detail.map(r => {
    const domPct = Math.min(100, Math.round(r.dominance * 100));
    return `<div class="round-row">
      <span class="round-num">Round ${r.round}</span>
      <span class="round-detail">${r.candidates} candidates → ${r.clusters} clusters</span>
      <div class="dom-bar-wrap"><div class="dom-bar" style="width:${domPct}%"></div></div>
      <span style="color:#64748b;font-size:0.8rem;min-width:50px">dom ${(r.dominance*100).toFixed(0)}%</span>
    </div>`;
  }).join('');
  document.getElementById('roundsList').innerHTML = roundsHtml || '<p style="color:#64748b">No rounds recorded.</p>';

  // Top cluster
  if (d.top_cluster) {
    const src = Object.entries(d.top_cluster.sources)
      .map(([k,v]) => `${k}: ${v}`).join(', ');
    document.getElementById('clusterInfo').innerHTML =
      `<div class="stat-grid">
        <div class="stat"><div class="stat-label">Size</div><div class="stat-value">${d.top_cluster.size}</div></div>
        <div class="stat"><div class="stat-label">Score</div><div class="stat-value">${d.top_cluster.score.toFixed(4)}</div></div>
        <div class="stat"><div class="stat-label">Mean Confidence</div><div class="stat-value">${d.top_cluster.mean_conf.toFixed(4)}</div></div>
      </div>
      <p style="color:#94a3b8;font-size:0.85rem;margin-top:0.75rem;">Sources: ${src}</p>`;
  } else {
    document.getElementById('clusterInfo').innerHTML = '<p style="color:#64748b">No cluster data.</p>';
  }

  // Bias vector
  const bv = d.bias;
  const biasFields = [
    { label: 'Attention Bias', key: 'attention_bias' },
    { label: 'Granularity Bias', key: 'granularity_bias' },
    { label: 'Abstraction Bias', key: 'abstraction_bias' },
    { label: 'Inversion Bias', key: 'inversion_bias' },
    { label: 'Sampling Temperature', key: 'sampling_temperature', scale: 0.5 },
  ];
  const biasHtml = biasFields.map(f => {
    const val = bv[f.key];
    const pct = Math.min(100, Math.round(val * (f.scale ? f.scale : 1) * 100));
    return `<div class="bias-row">
      <span class="bias-label">${f.label}</span>
      <div class="bias-bar-wrap"><div class="bias-bar" style="width:${pct}%"></div></div>
      <span class="bias-val">${val.toFixed(3)}</span>
    </div>`;
  }).join('');
  document.getElementById('biasInfo').innerHTML = biasHtml;
}
</script>
</body>
</html>
"""


@app.route("/")
def index():
    return render_template_string(HTML)


@app.route("/run", methods=["POST"])
def run():
    try:
        data = request.get_json()
        text = data.get("text", "").strip()
        if not text:
            return jsonify({"error": "Empty input text."}), 400

        result = model.run(text, verbose=False)
        bmap = result.basemap
        bv = model.base_bias

        bases_info = []
        for base, btype in zip(bmap.bases, bmap.modifiers.get("type", [])):
            bases_info.append({"token": base, "type": btype})

        rounds_detail = []
        for r in result.all_rounds:
            rounds_detail.append({
                "round": r["round"],
                "candidates": r["num_candidates"],
                "clusters": r["num_clusters"],
                "dominance": r["dominance"],
            })

        top_cluster_data = None
        if result.top_cluster:
            top_cluster_data = {
                "size": result.top_cluster.size,
                "score": float(result.top_cluster.score),
                "mean_conf": float(result.top_cluster.mean_confidence),
                "sources": result.top_cluster.sources(),
            }

        total_candidates = sum(r["num_candidates"] for r in result.all_rounds)
        top_score = float(result.top_cluster.score) if result.top_cluster else 0.0
        dom = float(result.all_rounds[-1]["dominance"]) if result.all_rounds else 0.0

        response = {
            "rounds": result.iteration_summary.get("rounds_completed", 0),
            "stop_reason": result.stop_reason,
            "num_bases": len(bmap.bases),
            "base_types": bmap.metadata.get("base_types", {}),
            "bases": bases_info,
            "total_candidates": total_candidates,
            "top_score": top_score,
            "dominance": dom,
            "top_cluster_size": result.top_cluster.size if result.top_cluster else 0,
            "top_cluster": top_cluster_data,
            "rounds_detail": rounds_detail,
            "bias": {
                "attention_bias": bv.attention_bias,
                "granularity_bias": bv.granularity_bias,
                "abstraction_bias": bv.abstraction_bias,
                "inversion_bias": bv.inversion_bias,
                "sampling_temperature": bv.sampling_temperature,
            },
        }
        return jsonify(response)

    except Exception as e:
        import traceback
        return jsonify({"error": str(e), "trace": traceback.format_exc()}), 500


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
