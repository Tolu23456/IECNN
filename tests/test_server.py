"""The viz server must boot and serve every endpoint with real repo data.

Stdlib only (no torch — the server is a thin read-only layer over results/).
Runs in CI so a broken endpoint, a bad static path, a dark-theme regression
or a config-parsing drift fails the build.
"""
from __future__ import annotations

import json
import sys
import threading
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "server"))

import server as server_mod                            # noqa: E402

STATIC = ROOT / "server" / "static"


def _get(url: str):
    import urllib.error
    try:
        with urllib.request.urlopen(url, timeout=10) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:          # 4xx/5xx: urlopen raises
        return e.code, e.read()


def _start():
    srv = server_mod.ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


# ----------------------------------------------------------- basic APIs ----
def test_api_endpoints_serve_json():
    srv, base = _start()
    try:
        for ep in ("/api/runs", "/api/evals", "/api/reports", "/api/summaries",
                   "/api/arch", "/api/viz", "/api/vocab", "/api/diagnosis",
                   "/api/profiles"):
            st, body = _get(base + ep)
            assert st == 200, (ep, st)
            json.loads(body)                     # must be valid JSON
    finally:
        srv.shutdown()


def test_run_endpoint_parses_committed_csv():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/run/train_b0_gru_pilot_seed0")
        assert st == 200
        d = json.loads(body)
        assert d["rung"] == "b0_gru_pilot" and d["seed"] == 0
        assert "bpc_val" in d["series"] and "step" in d["series"]
        assert d["n_points"] > 30
    finally:
        srv.shutdown()


def test_legacy_malformed_csv_rows_are_filtered():
    """The old 5-column final-row bug must keep being dropped server-side."""
    p = ROOT / "results" / "train_e0_plain_ensemble_pilot_seed0.csv"
    assert p.exists()
    import csv as _csv
    rows = list(_csv.reader(p.read_text(encoding="utf-8").splitlines()))[1:]
    filled = [sum(1 for c in r if c != "") for r in rows]
    mode = max(set(filled), key=filled.count)
    kept = sum(1 for n in filled if n >= mode)
    info = server_mod._safe_csv(p)
    assert info["n_points"] == kept
    vals = [v for v in info["series"]["bpc_val"] if v is not None]
    # the malformed row's 2.46816 must NOT leak into the parsed series
    assert 2.46816 not in vals
    assert 2.4751 in vals


def test_unknown_run_is_a_clean_404():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/run/does_not_exist")
        assert st == 404
        assert "error" in json.loads(body)
    finally:
        srv.shutdown()


def test_unknown_viz_and_unknown_endpoint_404():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/viz/nope")
        assert st == 404 and "error" in json.loads(body)
        st, body = _get(base + "/api/definitely_not_an_endpoint")
        assert st == 404 and "error" in json.loads(body)
    finally:
        srv.shutdown()


# ----------------------------------------------------------- new content ---
def test_vocab_endpoint_decodes_real_merge_text():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/vocab")
        assert st == 200
        d = json.loads(body)
        assert d["size"] == 512
        bases = {b["id"]: b for b in d["bases"]}
        # merges must show their decoded text, not "<merge a+b>"
        assert bases[264]["text"] == " t" and bases[264]["kind"] == "merge"
        assert bases[268]["text"] == " the"
        # the 8 specials are flagged
        assert bases[256]["kind"] == "special"
        assert bases[0]["kind"] == "byte"
    finally:
        srv.shutdown()


def test_diagnosis_endpoint_parses_probes_and_causality():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/diagnosis")
        assert st == 200
        d = json.loads(body)
        assert d["files"], "expected at least one diagnosis JSON"
        for name, data in d["data"].items():
            pr = data["probes"]
            for want in ("A1_live_trainmode_seq128_head100k",
                         "A6_live_full_val_canonical",
                         "T1_live_trainmode_seq128_train100k",
                         "C1_max_param_diff_roundtrip"):
                assert want in pr, (name, want)
            # bit-identical reload is the whole point of the probe table
            assert data["causality"]["live"]["bit_identical"] is True
            assert data["causality"]["reloaded"]["bit_identical"] is True
            assert pr["C1_max_param_diff_roundtrip"] == 0.0
        first = d["data"][d["files"][0]]
        assert first["meta"]["best_step"] == 540
    finally:
        srv.shutdown()


def test_profiles_endpoint_serves_profile_text():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/profiles")
        assert st == 200
        d = json.loads(body)
        assert d["files"]
        text = d["data"][d["files"][0]]
        assert "TOTAL fwd+bwd" in text
    finally:
        srv.shutdown()


def test_viz_export_structure_for_inspector_page():
    """The W&D page relies on: meta, embeddings.points/labels, sample.positions
    with per-dot weights[3]/gamma[3]/top[3 passes x 5], pooled_top."""
    p = ROOT / "results" / "viz" / "e1_weights_passes_pilot_seed0.json"
    assert p.exists()
    v = json.loads(p.read_text(encoding="utf-8"))
    m = v["meta"]
    assert (m["R"], m["n_dots"], m["n_passes"], m["params"]) == (1, 8, 3, 1257924)
    assert len(v["embeddings"]["points"]) == 512
    assert len(v["embeddings"]["labels"]) == 512
    S = v["sample"]
    assert S["passes"] == ["pass 1", "pass 2", "pass 3"]
    pos = S["positions"]
    assert len(pos) > 10
    p0 = pos[0]
    assert len(p0["dots"]) == 8
    for d in p0["dots"]:
        assert len(d["weights"]) == 3 and len(d["gamma"]) == 3
        assert len(d["top"]) == 3 and all(len(t) == 5 for t in d["top"])
    assert len(p0["pooled_top"]) == 5
    # the export logs scalars for one extra final-token position
    assert len(v["weight_entropy_over_positions"]) in (len(pos), len(pos) + 1)


# ------------------------------------------------------------ /api/arch ----
def test_arch_resolves_config_inheritance_chain():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/arch")
        assert st == 200
        d = json.loads(body)
        assert d["config"] == "e1_weights_passes_pilot"
        act = d["active"]
        assert act["chain"] == ["e1_weights_passes_eqc", "e1_weights_passes",
                                "e0_plain_ensemble", "base"]
        c = act["config"]
        m, tr = c["model"], c["training"]
        # the real E1-pilot geometry (must match the viz export meta)
        assert (m["n_regions"], m["n_dots"], m["n_passes"]) == (1, 8, 3)
        assert m["d"] == 128 and m["d_s"] == 128
        assert m["spans"][0] == 32
        assert m["pooling"] == "log_linear" and m["use_weights"] is True
        assert m["use_messages"] is False and m["use_cross_region"] is False
        assert (tr["lambda_d"], tr["lambda_k"], tr["lambda_w"]) == (0.1, 0.05, 0.05)
        assert tr["lambda_r"] == 0.0 and tr["lambda_c"] == 0.0
        assert tr["steps"] == 2880 and tr["eval_every"] == 60
        assert c["consensus"]["kappa"] == 0.7
        # every listed config resolves
        for name in d["configs"]:
            assert name in d["resolved"], name
    finally:
        srv.shutdown()


def test_arch_query_switches_config():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/arch?config=b0_gru_pilot")
        assert st == 200
        d = json.loads(body)
        assert d["config"] == "b0_gru_pilot"
        assert d["active"]["config"]["model"]["kind"] == "gru"
        # unknown config falls back to the default, never 500s
        st, body = _get(base + "/api/arch?config=does_not_exist")
        assert st == 200
        assert json.loads(body)["config"] == "e1_weights_passes_pilot"
    finally:
        srv.shutdown()


# -------------------------------------------------- stdlib YAML subset -----
def test_yaml_subset_parser():
    load = server_mod._load_yaml_text
    d = load("name: e1\nmodel:\n  d: 128\n  use_messages: false\n"
             "  spans: [32, 64, 128, 256]\n  note: keep this # trailing comment\n"
             "training:\n  lr: 1.0e-3\n  steps: 2880\n")
    assert d["name"] == "e1"
    assert d["model"]["d"] == 128 and d["model"]["use_messages"] is False
    assert d["model"]["spans"] == [32, 64, 128, 256]
    assert d["model"]["note"] == "keep this"
    assert d["training"]["lr"] == 1.0e-3 and d["training"]["steps"] == 2880
    # unsupported constructs must raise, not silently mis-parse
    try:
        load("a:\n  - one\n  - two\n")
        raise AssertionError("block list should raise")
    except ValueError:
        pass


def test_resolve_config_matches_real_committed_configs():
    r = server_mod._resolve_config("e1_weights_passes_pilot")
    assert r is not None
    cfg = r["config"]
    assert cfg["name"] == "e1_weights_passes_pilot"
    assert cfg["model"]["n_passes"] == 3
    assert "base" not in cfg                      # chain marker removed
    assert server_mod._resolve_config("no_such_config") is None


def test_server_stays_stdlib_only():
    src = (ROOT / "server" / "server.py").read_text(encoding="utf-8")
    for banned in ("import torch", "import yaml", "import numpy",
                   "from yaml", "from torch", "import requests"):
        assert banned not in src, banned


def test_all_committed_configs_parse():
    for p in sorted((ROOT / "configs").glob("*.yaml")):
        text = p.read_text(encoding="utf-8")
        d = server_mod._load_yaml_text(text)      # must not raise
        assert isinstance(d, dict), p.name
        # rung configs carry a name; overlays (tiny.yaml) legitimately don't
        if "base" not in d:
            assert d.get("name") or d.get("model"), p.name


# ------------------------------------------------------------- static ------
def test_static_pages_served():
    srv, base = _start()
    try:
        for name, marker in (("index.html", b"IECNN"),
                             ("app.js", b"pageDashboard"),
                             ("style.css", b"#sidebar"),
                             ("chart.umd.js", b"Chart.js")):
            st, body = _get(f"{base}/{name}")
            assert st == 200 and marker in body, name
        st, _ = _get(base + "/nope.js")
        assert st == 404
    finally:
        srv.shutdown()


def test_dark_theme_is_the_default():
    css = (STATIC / "style.css").read_text(encoding="utf-8")
    assert "color-scheme: dark" in css
    # the app background is a dark colour on :root and used by body
    assert "--bg: #0d1117" in css
    assert "background: var(--bg)" in css
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    assert 'name="color-scheme" content="dark"' in html


def test_index_links_every_page():
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for page in ("dashboard", "runs", "scores", "architecture",
                 "checkpoints", "diagnosis", "vocab", "corpus"):
        assert f'data-page="{page}"' in html, page


def test_app_js_has_a_page_function_for_every_nav_item():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("pageDashboard", "pageRuns", "pageScores", "pageArchitecture",
               "pageCheckpoints", "pageDiagnosis", "pageVocab", "pageCorpus"):
        assert f"function {fn}" in js, fn


def test_static_assets_are_offline_safe():
    """No CDN references anywhere: the dashboard must work without network."""
    for f in STATIC.iterdir():
        if f.suffix not in (".html", ".js", ".css"):
            continue
        text = f.read_text(encoding="utf-8")
        for banned in ('src="http', "url(http", "@import", "fetch(\"https",
                       "fetch('https", "script.src"):
            assert banned not in text, (f.name, banned)
    # stylesheet and script tags must be root-relative (offline vendored)
    html = (STATIC / "index.html").read_text(encoding="utf-8")
    for tag in ('<link rel="stylesheet" href="/style.css">',
                '<script src="/chart.umd.js">', '<script src="/app.js">'):
        assert tag in html, tag
