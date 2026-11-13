"""The viz server must boot and serve every endpoint with real repo data.

Stdlib only (no torch — the server is a thin read-only layer over results/).
Runs in CI so a broken endpoint or a bad static path fails the build.
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


def _get(url: str):
    with urllib.request.urlopen(url, timeout=10) as r:
        return r.status, r.read()


def _start():
    srv = server_mod.ThreadingHTTPServer(("127.0.0.1", 0), server_mod.Handler)
    t = threading.Thread(target=srv.serve_forever, daemon=True)
    t.start()
    return srv, f"http://127.0.0.1:{srv.server_address[1]}"


def test_api_endpoints_serve_json():
    srv, base = _start()
    try:
        for ep in ("/api/runs", "/api/evals", "/api/reports", "/api/summaries",
                   "/api/arch", "/api/viz"):
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


def test_unknown_run_is_a_clean_404():
    srv, base = _start()
    try:
        st, body = _get(base + "/api/run/does_not_exist")
        assert st == 404
        assert "error" in json.loads(body)
    finally:
        srv.shutdown()


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
