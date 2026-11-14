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
                   "/api/profiles", "/api/gates", "/api/config_used"):
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
def test_byte_disp_renders_readably():
    """Byte-fallback token display: ASCII as-is, controls escaped, hex else."""
    f = server_mod._byte_disp
    assert f(0x41) == "A"                # printable ASCII as-is
    assert f(0x20) == " " and f(0x7e) == "~"
    assert f(0x0a) == "\\n"              # control bytes as visible escapes
    assert f(0x09) == "\\t" and f(0x0d) == "\\r"
    assert f(0xe2) == "<e2>"             # high bytes as lowercase hex
    assert f(0x00) == "<00>"             # non-printing controls too
    assert f(0x7f) == "<7f>"             # DEL is not printable


def test_token_display_escapes_invalid_utf8():
    """Payload display: valid UTF-8 stays text; bad bytes become <xx> escapes."""
    f = server_mod._token_display
    assert f(b" the") == " the"          # valid text untouched
    assert f(b"caf\xc3\xa9") == "café"   # multi-byte UTF-8 stays intact
    assert f(b"\xff") == "<ff>"          # lone invalid byte
    assert f(b"\xe2\x82") == "<e2><82>"  # truncated UTF-8 sequence
    assert f(b"a\x00b") == "a<00>b"
    assert f(b"\n\n") == "\\n\\n"        # control chars inside valid text
    assert f(b"hi\x7f") == "hi<7f>"
    assert f(b"") == ""


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
        # escape/hex/composition fields from the Tolu follow-up
        assert bases[65]["text"] == "A" and bases[65]["hex"] == "41"
        assert bases[65]["escaped"] is False and bases[65]["byte_fallback"] is True
        assert bases[10]["text"] == "\\n" and bases[10]["escaped"] is True
        assert bases[226]["text"] == "<e2>" and bases[226]["hex"] == "e2"
        assert bases[268]["hex"] == "20746865" and bases[268]["escaped"] is False
        assert bases[268]["byte_fallback"] is False
        assert d["composition"] == {"byte": 256, "special": 8, "merge": 248}
        assert "byte-fallback" in d["legend"]
        # exactly the non-plain-readable entries are flagged escaped:
        # 32 ctl bytes + DEL + 128 high bytes + 1 bad merge + "\n\n" merge
        bad = sum(1 for b in d["bases"] if b["escaped"])
        assert bad == 163, bad
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


# ------------------------------------------------- gates (G1–G4, step 1) ---
def _recompute_equal_compute():
    """Independent recompute of the G2 wall table from the committed CSVs.

    Deliberately does NOT call server helpers — same parse rules (markers
    skipped, first row per step wins, sorted by step) written out in full.
    """
    import csv as _csv
    def rows(p):
        raw = list(_csv.reader(p.read_text(encoding="utf-8").splitlines()))
        hdr, out, seen = raw[0], [], set()
        for r in raw[1:]:
            if not r or not r[0] or r[0].lstrip()[:1] in ("<", "=", ">"):
                continue
            try:
                step = float(r[0])
            except ValueError:
                continue
            if step in seen:
                continue
            seen.add(step)
            out.append(r)
        out.sort(key=lambda r: float(r[0]))
        return hdr, out
    def curve(p):
        hdr, rs = rows(p)
        vi, wi = hdr.index("bpc_val"), hdr.index("elapsed_s")
        pts, best = [], None
        for r in rs:
            if len(r) <= max(vi, wi) or r[vi] == "" or r[wi] == "":
                continue
            v, w = float(r[vi]), float(r[wi])
            best = v if best is None else min(best, v)
            pts.append((w, best))
        return pts
    e0 = curve(ROOT / "results" / "train_e0_plain_ensemble_eqc_seed0.csv")
    e1 = curve(ROOT / "results" / "train_e1_weights_passes_eqc_seed0.csv")
    cap = min(e0[-1][0], e1[-1][0])
    def best_at(c, w):
        b = None
        for cw, cb in c:
            if cw <= w:
                b = cb
            else:
                break
        return b
    walls = sorted({w for c in (e0, e1) for w, _ in c if w <= cap})
    table = [{"wall_s": w, "e0": best_at(e0, w), "e1": best_at(e1, w)}
             for w in walls]
    return table


def test_gates_endpoint_serves_all_four_gates():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/gates")
        assert st == 200
        d = json.loads(body)
        ids = [g["id"] for g in d["gates"]]
        assert ids == ["G1", "G2", "G3", "G4"]
        for g in d["gates"]:
            assert g["status"] in ("PASS", "PENDING", "PARTIAL", "FAIL",
                                   "NO_DATA"), g["id"]
            assert g["sources"], g["id"]
            assert g["evidence"], g["id"]
        assert d["gates"][0]["id"] == "G1"
    finally:
        srv.shutdown()


def test_g1_counts_exclude_retracted():
    srv, base = _start()
    try:
        d = json.loads(_get(base + "/api/gates")[1])
        g1 = d["gates"][0]
        evals = {}
        for p in sorted((ROOT / "results").glob("eval_*.json")):
            evals[p.stem] = json.loads(p.read_text(encoding="utf-8"))
        ret = [n for n, v in evals.items() if v.get("retracted")]
        live = [n for n, v in evals.items() if not v.get("retracted")]
        assert g1["counts"]["retracted"] == len(ret)
        # a retracted eval must never appear in the unverified-provenance list
        assert not set(g1["counts"]["missing"]) & set(ret)
        assert set(g1["counts"]["missing"]) <= set(live)
        assert g1["counts"]["unverified"] == len(g1["counts"]["missing"])
    finally:
        srv.shutdown()


def test_g2_equal_compute_matches_independent_recompute():
    srv, base = _start()
    try:
        d = json.loads(_get(base + "/api/gates")[1])
        g2 = d["gates"][1]
        table = _recompute_equal_compute()
        assert g2["matched_walls"] == len(table)
        assert len(g2["table"]) == len(table)
        last, exp = g2["table"][-1], table[-1]
        assert abs(last["wall_s"] - exp["wall_s"]) < 1e-6
        assert abs(last["e0_bpc"] - exp["e0"]) < 1e-9
        assert abs(last["e1_bpc"] - exp["e1"]) < 1e-9
        # status follows the recomputed comparison exactly
        wins = all(t["e1"] is not None and t["e0"] is not None
                   and t["e1"] <= t["e0"] + 1e-9 for t in table)
        assert g2["status"] == ("PASS" if wins else "FAIL")
        # the committed GRU reference must appear in the table rows that
        # any b0 CSV covers (b0_gru_pilot reaches 1.7358)
        if any(r.get("b0_bpc") is not None for r in g2["table"]):
            best_b0 = min(r["b0_bpc"] for r in g2["table"]
                          if r.get("b0_bpc") is not None)
            pilot = [ln.split(",") for ln in
                     (ROOT / "results" / "train_b0_gru_pilot_seed0.csv")
                     .read_text(encoding="utf-8").splitlines()]
            vi = pilot[0].index("bpc_val")
            csv_min = min(float(r[vi]) for r in pilot[1:]
                          if len(r) > vi and r[vi] != "")
            assert abs(best_b0 - min(best_b0, csv_min)) < 1e-9
            assert best_b0 <= csv_min + 1e-9
    finally:
        srv.shutdown()


def _dedupe_csv(p):
    """Same parse rule as the server: markers skipped, first row per step
    wins (the committed artifact-recovery variants appear as duplicate
    steps with a different tau0 protocol), sorted by step."""
    import csv as _csv
    raw = list(_csv.reader(p.read_text(encoding="utf-8").splitlines()))
    hdr, out, seen = raw[0], [], set()
    for r in raw[1:]:
        if not r or not r[0] or r[0].lstrip()[:1] in ("<", "=", ">"):
            continue
        try:
            step = float(r[0])
        except ValueError:
            continue
        if step in seen:
            continue
        seen.add(step)
        out.append(r)
    out.sort(key=lambda r: float(r[0]))
    return hdr, out


def test_g3_follows_committed_vocab_curves():
    srv, base = _start()
    try:
        d = json.loads(_get(base + "/api/gates")[1])
        g3 = d["gates"][2]
        assert [v["vocab"] for v in g3["vocabs"]] == ["vocab2", "vocab3",
                                                      "vocab4"]
        still = False
        for v in g3["vocabs"]:
            p = ROOT / "results" / v["csv"]
            assert p.exists()
            if v.get("best_bpc") is None:
                continue
            hdr, rows = _dedupe_csv(p)
            si, vi = hdr.index("step"), hdr.index("bpc_val")
            ev = [(float(r[si]), float(r[vi])) for r in rows
                  if len(r) > vi and r[vi] != ""]
            best = min(ev, key=lambda t: t[1])
            assert abs(v["best_bpc"] - best[1]) < 1e-9
            assert v["best_step"] == best[0]
            delta = ev[-1][1] - ev[-3][1]
            assert abs(v["last3_delta"] - round(delta, 5)) < 1e-9
            still = still or delta < -0.005
        assert g3["status"] == ("PENDING" if still else "PASS")
    finally:
        srv.shutdown()


def test_g4_matches_independent_recompute():
    srv, base = _start()
    try:
        d = json.loads(_get(base + "/api/gates")[1])
        g4 = d["gates"][3]
        imports_ok = True
        for e in g4["exports"]:
            p = ROOT / "results" / "viz" / f"{e['name']}.json"
            assert p.exists()
            pos = json.loads(p.read_text(encoding="utf-8"))["sample"]["positions"]
            n = pooled = uni = plur = 0
            for P in pos:
                tgt, pool = P.get("next"), P.get("pooled_top")
                dots = P.get("dots") or []
                if tgt is None or not pool or not dots:
                    continue
                n += 1
                if pool[0][0] == tgt:
                    pooled += 1
                sc, votes = {}, {}
                for dd in dots:
                    t1 = (dd.get("top") or [None])[0]
                    if t1:
                        votes[t1[0][0]] = votes.get(t1[0][0], 0) + 1
                        for tok, pr in t1:
                            sc[tok] = sc.get(tok, 0.0) + float(pr) / len(dots)
                if votes and max(votes, key=votes.get) == tgt:
                    plur += 1
                if sc and max(sc, key=sc.get) == tgt:
                    uni += 1
            assert e["sampled"] == n
            assert e["pooled_top1"] == pooled
            assert e["uniform_mix_top1"] == uni
            assert e["plurality_top1"] == plur
            best_proxy = max(uni, plur)
            assert e["status"] == ("PASS" if pooled > best_proxy else "FAIL")
        # gate PASS only if every export passes
        assert g4["status"] == ("PASS" if all(
            e["status"] == "PASS" for e in g4["exports"]) else "FAIL")
    finally:
        srv.shutdown()


def test_raw_endpoint_serves_results_and_blocks_traversal():
    srv, base = _start()
    try:
        # allowed: an actual committed results file, byte-identical
        p = ROOT / "results" / "eval_e1_weights_passes_pilot_seed0.json"
        st, body = _get(base + "/api/raw?path=results/"
                        + p.name)
        assert st == 200
        assert body == p.read_bytes()
        # traversal / absolute / non-allowlisted are rejected
        for q, code in (("path=../secrets", 400), ("path=/etc/passwd", 400),
                        ("path=server/server.py", 403),
                        ("path=results/does_not_exist.json", 404)):
            st, _ = _get(base + "/api/raw?" + q)
            assert st == code, (q, st)
    finally:
        srv.shutdown()


def test_read_csv_rows_skips_conflict_markers_and_dedupes():
    import tempfile
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "toy.csv"
        p.write_text(
            "step,bpc_val,elapsed_s\n"
            "240,3.9,777.6\n"
            "<<<<<<< Updated upstream\n"
            "240,3.9,999.9\n"          # duplicate step: first row must win
            "=======\n"
            "120,4.1,300.0\n"
            ">>>>>>> Stashed changes\n"
            "480,3.5,1500.0\n"
            "junk,row,here\n")
        hdr, rows = server_mod._read_csv_rows(p)
        assert [r["step"] for r in rows] == [120.0, 240.0, 480.0]
        by_step = {r["step"]: r for r in rows}
        assert by_step[240.0]["elapsed_s"] == 777.6      # first occurrence
        assert by_step[120.0]["bpc_val"] == 4.1
        assert server_mod._read_csv_rows(Path(td) / "nope.csv") is None


def test_eval_val_sha_provenance_helpers():
    """Future evals: val_sha256 (or provenance.val_sha256) is honoured."""
    assert server_mod._eval_retracted({"retracted": None}) is False
    assert server_mod._eval_retracted({"retracted": {"reason": "x"}}) is True
    assert server_mod._eval_retracted({}) is False
    assert server_mod._eval_val_sha({"val_sha256": "abc"}) == "abc"
    assert server_mod._eval_val_sha(
        {"provenance": {"val_sha256": "def"}}) == "def"
    assert server_mod._eval_val_sha({}) is None


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
                 "checkpoints", "diagnosis", "vocab", "corpus", "playground"):
        assert f'data-page="{page}"' in html, page


def test_app_js_has_a_page_function_for_every_nav_item():
    js = (STATIC / "app.js").read_text(encoding="utf-8")
    for fn in ("pageDashboard", "pageRuns", "pageScores", "pageArchitecture",
               "pageCheckpoints", "pageDiagnosis", "pageVocab", "pageCorpus",
               "pagePlayground"):
        assert f"function {fn}" in js, fn
    # playground talks to the generation endpoints
    for ep in ("/api/generate", "/api/generate/runs",
               "/api/generate/result", "/api/generations"):
        assert ep in js, ep
    # step 1: dashboard gates scoreboard + provenance-aware scores page
    assert "/api/gates" in js
    for gate_bit in ("G1", "G2", "G3", "G4", "unverified corpus", "RETRACTED"):
        assert gate_bit in js, gate_bit


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


# ------------------------------------------------------ consensus audit ----
def _pos(next_tok, pooled, dots):
    """Shorthand for a sample-forward position as the viz export stores it."""
    return {"token": "t", "next": next_tok, "next_id": 7,
            "pooled_top": pooled,
            "dots": [{"weights": d["w"], "gamma": d.get("g", [0] * 3),
                      "top": d["top"]} for d in dots]}


def test_consensus_audit_flags_outvoted_top1():
    """A dot's TOP-1 is correct, pooled consensus picks something else."""
    P = _pos(
        "z",                                # true next token
        [["x", .6], ["y", .2], ["q", .1]],  # pooled answer misses
        [{"w": [5.9e-7, 0.0, 0.0],          # dot 0: muted, but top-1 == z
          "top": [[["z", .9], ["x", .05]], [["x", .5]], [["x", .5]]]},
         {"w": [0.9, 0.8, 0.7],             # sharp wrong dot wins the pool
          "top": [[["x", .9], ["y", .05]], [["x", .9]], [["x", .9]]]}])
    a = server_mod.consensus_audit([P])
    assert a["sampled"] == 1
    assert a["pooled_top1"] == 0 and a["pooled_top1_acc"] == 0.0
    assert a["oracle_top1"] == 1 and a["oracle_top1_acc"] == 1.0
    assert a["oracle_top5"] == 1
    assert a["outvoted_top1"] == 1 and a["outvoted_top5"] == 0
    assert a["outvoted"] == 1
    assert a["per_position"] == [{"i": 0, "level": "top1",
        "held": [{"dot": 0, "pass": 0, "weight": 5.9e-7, "level": "top1"}]}]


def test_consensus_audit_levels_and_pooled_hits():
    """Level top5 vs top1; pooled hits are never outvoted; multi-holder."""
    # top-5 only: target present at slot 3 of dot 1, pass 2
    P5 = _pos("z", [["x", .6]],
              [{"w": [0.5, 0.5, 0.5], "top": [[["x", .9]]] * 3},
               {"w": [0.1, 0.1, 0.2], "top": [[["x", .9]]] * 2 +
                 [[["x", .9], ["y", .1], ["w", .1], ["z", .05]]]}])
    a = server_mod.consensus_audit([P5])
    assert a["outvoted_top5"] == 1 and a["outvoted_top1"] == 0
    assert a["oracle_top1"] == 0 and a["oracle_top5"] == 1
    assert a["per_position"][0]["level"] == "top5"
    assert a["per_position"][0]["held"] == [
        {"dot": 1, "pass": 2, "weight": 0.2, "level": "top5"}]
    # pooled hit: even when dots hold the target, nothing is outvoted
    Ph = _pos("z", [["z", .6]], [{"w": [1.0], "top": [[["z", .9]]]}])
    a = server_mod.consensus_audit([Ph])
    assert a["pooled_top1"] == 1 and a["outvoted"] == 0
    assert a["per_position"] == [] and a["oracle_top1"] == 1
    # missing pooled/dots/next are skipped, garbage input tolerated
    a = server_mod.consensus_audit(
        [{"next": "z"}, {"pooled_top": [["z", 1]]}, "junk", None, {}])
    assert a["sampled"] == 0 and a["per_position"] == []
    assert server_mod.consensus_audit([])["sampled"] == 0
    assert server_mod.consensus_audit(None)["sampled"] == 0


def test_viz_endpoint_carries_consensus_audit():
    """The live endpoint computes the audit from the committed export."""
    srv, base = _start()
    try:
        st, body = _get(base + "/api/viz/e1_weights_passes_pilot_seed0")
        assert st == 200
        d = json.loads(body)
        a = d.get("audit")
        assert isinstance(a, dict), "audit missing from /api/viz payload"
        # independently recompute from the same positions (truth, not trust)
        pos = d["sample"]["positions"]
        pooled = or1 = 0
        scored = sum(1 for P in pos if P.get("next") is not None
                     and P.get("pooled_top"))
        for P in pos:
            tgt = P.get("next")
            if tgt is None or not P.get("pooled_top"):
                continue
            pooled += P["pooled_top"][0][0] == tgt
            or1 += any(d["top"][pi][0][0] == tgt
                       for d in P["dots"] for pi in range(len(d["top"])))
        assert a["sampled"] == scored
        assert a["pooled_top1"] == pooled
        assert a["oracle_top1"] == or1
        # per_position holds exactly the outvoted positions, with valid dots
        held_idx = [e["i"] for e in a["per_position"]]
        assert held_idx == sorted(held_idx)
        assert len(held_idx) == a["outvoted"]
        for e in a["per_position"]:
            P = pos[e["i"]]
            assert P["pooled_top"][0][0] != P["next"]        # pooled missed
            assert all(h["dot"] < len(P["dots"]) for h in e["held"])
        # the headline facts for this export (committed artifact, 256-token
        # sample: the pooled consensus is much worse than the best dot)
        assert a["sampled"] == 255
        assert a["outvoted_top1"] == 30 and a["outvoted_top5"] == 50
    finally:
        srv.shutdown()


# ----------------------------------------------------------- playground ----
def _post(base: str, path: str, body) -> tuple[int, bytes]:
    req = urllib.request.Request(
        base + path, method="POST",
        data=json.dumps(body).encode("utf-8"),
        headers={"Content-Type": "application/json"})
    try:
        with urllib.request.urlopen(req, timeout=15) as r:
            return r.status, r.read()
    except urllib.error.HTTPError as e:              # noqa: F821
        return e.code, e.read()


class _FakeGH:
    """Monkeypatches the GitHub seam; no test ever touches the network."""
    def __init__(self, routes=None, token="fake-token-for-tests-only"):
        self.routes = routes or {}   # url substring -> (code, parsed_json)
        self.calls = []              # (url, method, body, token)
        self.token = token

    def __enter__(self):
        self.orig_req = server_mod._gh_request
        self.orig_cred = server_mod._gh_credential
        server_mod._gh_credential = lambda: self.token
        server_mod._gh_request = self._fake
        return self

    def __exit__(self, *exc):
        server_mod._gh_request = self.orig_req
        server_mod._gh_credential = self.orig_cred
        return False

    def _fake(self, url, method="GET", body=None, token=None, timeout=30):
        self.calls.append((url, method, body, token))
        for k, v in self.routes.items():
            if k in url:
                return v
        return 404, None


def test_playground_validation_clamps():
    """Bad requests are rejected locally — the GitHub seam is never touched."""
    with _FakeGH(routes={"dispatches": (204, None)}) as fake:
        srv, base = _start()
        try:
            bad = [
                ({"prompt": "   "}, "prompt"),
                ({"prompt": ""}, "prompt"),
                ({"prompt": "x" * 4001}, "too long"),
                ({"prompt": "ok", "max_new_tokens": 801}, "max_new_tokens"),
                ({"prompt": "ok", "max_new_tokens": 0}, "max_new_tokens"),
                ({"prompt": "ok", "temperature": 2.5}, "temperature"),
                ({"prompt": "ok", "temperature": 0.05}, "temperature"),
                ({"prompt": "ok", "top_k": 501}, "top_k"),
                ({"prompt": "ok", "artifact": "bad/artifact\nheader"}, "artifact"),
                ("not-a-dict", "JSON object"),
            ]
            for body, frag in bad:
                st, raw = _post(base, "/api/generate", body)
                d = json.loads(raw)
                assert st == 400, (body, st, d)
                assert frag in d["error"], d
            assert fake.calls == [], "invalid requests must not reach GitHub"
            # edge values are accepted exactly, defaults applied
            st, raw = _post(base, "/api/generate",
                            {"prompt": "p", "max_new_tokens": 800,
                             "temperature": 2.0, "top_k": 0, "seed": -5})
            d = json.loads(raw)
            assert st == 200 and d["ok"] is True
            assert "dispatched_at" in d
            call = fake.calls[-1]
            assert call[1] == "POST" and "dispatches" in call[0]
            assert call[2]["inputs"] == {
                "prompt": "p", "artifact":
                "e1_weights_passes_pilot.yaml-seeds-0-pilot",
                "max_new_tokens": "800", "temperature": "2.0",
                "top_k": "0", "seed": "-5"}
            assert fake.token not in raw.decode()       # token never leaks
        finally:
            srv.shutdown()


def test_playground_no_credential_and_github_errors():
    """No credential -> clear error, zero network calls; upstream 401 -> 502."""
    with _FakeGH(token=None) as fake:
        srv, base = _start()
        try:
            st, raw = _post(base, "/api/generate", {"prompt": "hi"})
            d = json.loads(raw)
            assert st == 401 and "no GitHub credential" in d["error"]
            assert fake.calls == [], "must not touch GitHub without a credential"
        finally:
            srv.shutdown()
    with _FakeGH(routes={"dispatches": (401, None)}) as fake:
        srv, base = _start()
        try:
            st, raw = _post(base, "/api/generate", {"prompt": "hi"})
            d = json.loads(raw)
            assert st == 502 and "401" in d["error"]
            # the request DID go out with the ambient token in the header…
            assert fake.calls[0][3] == fake.token
            # …but the token never reaches any response body
            assert fake.token not in raw.decode()
        finally:
            srv.shutdown()
    runs_anon = {"workflow_runs": [
        {"id": 5, "status": "queued", "conclusion": None,
         "created_at": "2026-11-14T08:00:00Z"}]}
    with _FakeGH(token=None,
                 routes={"workflows/generate.yml/runs": (200, runs_anon)}) \
            as fake:
        srv, base = _start()
        try:
            # no ambient credential: runs/result fall back to anonymous
            # (public repo) — still a clean payload, never a 500
            st, raw = _get(base + "/api/generate/runs")
            d = json.loads(raw)
            assert st == 200 and d["runs"][0]["id"] == 5
            assert fake.calls[0][3] is None          # no Authorization used
        finally:
            srv.shutdown()


def test_generate_runs_and_result_paths():
    """runs listing maps fields; result walks pending/failed/done states."""
    runs_payload = {"workflow_runs": [
        {"id": 222, "status": "completed", "conclusion": "success",
         "created_at": "2026-11-14T10:00:00Z", "head_sha": "b" * 40},
        {"id": 111, "status": "in_progress", "conclusion": None,
         "created_at": "2026-11-14T09:00:00Z", "head_sha": "a" * 40},
        {"junk": True},
    ]}
    gen_json = json.dumps({"meta": {"run": "r", "step": 540},
                           "prompt": "hi", "generated": "there",
                           "generated_ids": [1, 2, 3], "elapsed_s": 1.5})
    import base64 as _b64
    contents = {"name": "gen-20261114-100000.json",
                "content": _b64.b64encode(gen_json.encode()).decode()}
    routes = {
        "workflows/generate.yml/runs": (200, runs_payload),
        "actions/runs/222": (200, runs_payload["workflow_runs"][0]),
        "actions/runs/111": (200, runs_payload["workflow_runs"][1]),
        f"commits/{'b' * 40}": (200, {"files": [
            {"filename": "scripts/x.py"},
            {"filename": "results/generation/gen-20261114-100000.json"}]}),
        "contents/results/generation/gen-20261114-100000.json": (200, contents),
        "actions/runs/999": (200, {"status": "completed", "conclusion": "failure",
                                   "head_sha": "c" * 40}),
        "actions/runs/888": (200, {"status": "completed", "conclusion": "success",
                                   "head_sha": "d" * 40}),
        f"commits/{'d' * 40}": (200, {"files": [{"filename": "README.md"}]}),
    }
    with _FakeGH(routes=routes) as fake:
        srv, base = _start()
        try:
            st, raw = _get(base + "/api/generate/runs")
            d = json.loads(raw)
            assert st == 200
            assert d["runs"] == [
                {"id": 222, "status": "completed", "conclusion": "success",
                 "created_at": "2026-11-14T10:00:00Z"},
                {"id": 111, "status": "in_progress", "conclusion": None,
                 "created_at": "2026-11-14T09:00:00Z"}]
            assert fake.token not in raw.decode()
            # in-flight run -> pending (HTTP 200 so the client keeps polling)
            st, raw = _get(base + "/api/generate/result?run_id=111")
            d = json.loads(raw)
            assert st == 200 and d["status"] == "pending"
            # success with a committed gen file -> done, payload parsed
            st, raw = _get(base + "/api/generate/result?run_id=222")
            d = json.loads(raw)
            assert st == 200 and d["status"] == "done"
            assert d["file"] == "gen-20261114-100000.json"
            assert d["result"]["generated"] == "there"
            assert d["result"]["meta"]["step"] == 540
            assert fake.token not in raw.decode()
            # concluded failure -> failed
            st, raw = _get(base + "/api/generate/result?run_id=999")
            d = json.loads(raw)
            assert st == 200 and d["status"] == "failed"
            # succeeded but committed no generation file -> failed
            st, raw = _get(base + "/api/generate/result?run_id=888")
            d = json.loads(raw)
            assert st == 200 and d["status"] == "failed"
            assert "no generation file" in d["note"]
            # malformed run_id -> 400
            st, raw = _get(base + "/api/generate/result?run_id=../etc")
            assert st == 400
        finally:
            srv.shutdown()


def test_generations_local_listing():
    """Local history: parses real files, skips junk, empty/missing dir OK."""
    import tempfile
    sample = {"meta": {"run": "e1_weights_passes_pilot_seed0", "step": 540,
                       "stop_kind": "wall_cap", "tau0": 0.954,
                       "temperature": 0.8, "top_k": 40, "seed": 0},
              "prompt": "[ Illustration]", "generated": "hel lo wor ld",
              "generated_ids": [1] * 11, "elapsed_s": 2.25}
    with tempfile.TemporaryDirectory() as td:
        d = Path(td)
        (d / "gen-20261114-100000.json").write_text(json.dumps(sample))
        (d / "gen-20261114-090000.json").write_text("{not json")
        (d / "other.txt").write_text("ignore me")
        out = server_mod.api_generations(d)
        gens = out["generations"]
        assert [g["file"] for g in gens] == ["gen-20261114-100000.json"]
        g = gens[0]
        assert g["prompt"] == "[ Illustration]"
        assert g["generated"] == "hel lo wor ld"
        assert g["n_gen_tokens"] == 11 and g["elapsed_s"] == 2.25
        assert g["meta"]["step"] == 540
        # missing dir -> empty, no exception
        empty = server_mod.api_generations(d / "nope")
        assert empty == {"generations": []}
    # live endpoint serves the repo dir (possibly empty) without 500ing
    srv, base = _start()
    try:
        st, raw = _get(base + "/api/generations")
        d = json.loads(raw)
        assert st == 200 and isinstance(d["generations"], list)
        for g in d["generations"]:
            assert "file" in g and "generated" in g
    finally:
        srv.shutdown()
