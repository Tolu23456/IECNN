"""IECNN visualization server (stdlib only — no torch, no pip installs).

Serves the static dashboard in server/static/ plus JSON APIs over everything
in the repository that is safe to show: training CSVs, eval reports, vocab,
corpus split info, UBM compression reports, and checkpoint-viz exports
(produced runner-side by scripts/export_viz.py — all torch compute happens
on GitHub Actions, never on the dev box, per Tolu's standing rule).

Run from the repository root:
    python3 server/server.py            # http://localhost:8765
    python3 server/server.py --port 9000

Every endpoint is read-only. /api/* answers JSON.
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

RESULTS = ROOT / "results"
VIZ = RESULTS / "viz"
CHECKPOINTS = ROOT / "checkpoints"
UBM_DIR = ROOT / "UBM"
DATA = ROOT / "data"


# ---------------------------------------------------------------- helpers --
def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=1, default=str).encode("utf-8")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _safe_csv(path: Path):
    """CSV -> {columns, rows-as-columns} for charting; None on failure.

    Trailing malformed rows (the old 5-column final-row bug in pre-2026-11-13
    runs) are dropped: keep only rows whose populated-cell count matches the
    modal count."""
    try:
        rows = list(csv.reader(path.read_text(encoding="utf-8").splitlines()))
    except Exception:
        return None
    if not rows:
        return None
    header, body = rows[0], rows[1:]
    filled = [sum(1 for c in r if c != "") for r in body]
    if filled:
        mode = max(set(filled), key=filled.count)
        body = [r for r, n in zip(body, filled) if n >= mode]
    cols: dict = {}
    for i, name in enumerate(header):
        vals = []
        for r in body:
            if i >= len(r) or r[i] == "":
                vals.append(None)
                continue
            try:
                vals.append(float(r[i]))
            except ValueError:
                vals.append(r[i])
        cols[name] = vals
    n_points = len(body)
    return {"columns": header, "series": cols, "n_points": n_points}


def _run_name_meta(path: Path):
    """train_<rung>_seed<seed>.csv -> {rung, seed}."""
    m = re.match(r"train_(.+)_seed(\d+)\.csv$", path.name)
    return (m.group(1), int(m.group(2))) if m else (path.stem, None)


def _git_head() -> str:
    try:
        out = subprocess.run(["git", "log", "-1", "--format=%h %s"],
                             capture_output=True, text=True, cwd=ROOT)
        return out.stdout.strip() if out.returncode == 0 else "(not a git repo)"
    except Exception:
        return "(git unavailable)"


def _vocab_overview(vpath: Path):
    data = _read_json(vpath)
    if not data:
        return None
    bases = data.get("bases", [])
    out = []
    for b in bases[:2048]:
        parts = b.get("parts")
        raw = bytes(b.get("bytes", [])) if isinstance(b.get("bytes"), list) else None
        text = raw.decode("utf-8", "replace") if raw is not None else \
            ("" if parts is None else f"<merge {parts[0]}+{parts[1]}>")
        out.append({"id": b.get("id"), "text": text.replace("\n", "\\n"),
                    "merge": parts is not None,
                    "n_bytes": (len(raw) if raw is not None else 0)})
    return {"file": vpath.name, "size": len(bases), "bases": out,
            "sha_note": "checksum recorded in every checkpoint (vocab_sha256)"}


# ------------------------------------------------------------------ APIs ---
def api_runs():
    runs = []
    for p in sorted(RESULTS.glob("train_*.csv")):
        rung, seed = _run_name_meta(p)
        info = _safe_csv(p)
        if not info:
            continue
        s = info["series"]
        last = [v for v in s.get("step", []) if v is not None]
        vals = [v for v in s.get("bpc_val", []) if v is not None]
        runs.append({"name": p.stem, "rung": rung, "seed": seed,
                     "points": info["n_points"],
                     "last_step": last[-1] if last else None,
                     "best_val": min(vals) if vals else None,
                     "columns": info["columns"]})
    return {"runs": runs, "git_head": _git_head()}


def api_run(name: str):
    p = RESULTS / f"{name}.csv"
    if not p.exists():
        return {"error": f"no results CSV named {name}", "status": 404}
    info = _safe_csv(p)
    rung, seed = _run_name_meta(p)
    info["rung"], info["seed"] = rung, seed
    return info


def api_evals():
    out = {}
    for p in sorted(RESULTS.glob("eval_*.json")):
        d = _read_json(p)
        if d:
            out[p.stem] = d
    return out


def api_reports():
    return {"vocab_v1_compression": _read_json(RESULTS / "vocab_v1_compression.json"),
            "ubm_compression": _read_json(RESULTS / "ubm_compression.json"),
            "split_info": _read_json(DATA / "split_info.json"),
            "first20_merges": (RESULTS / "ubm_first20_merges.txt").read_text(
                encoding="utf-8") if (RESULTS / "ubm_first20_merges.txt").exists() else None}


def api_viz_list():
    return {"exports": sorted(p.stem for p in VIZ.glob("*.json"))}


def api_viz(name: str):
    p = VIZ / f"{name}.json"
    if not p.exists():
        return {"error": f"no viz export named {name} (dispatch the viz workflow "
                         f"to produce one: workflows/viz.yml)", "status": 404}
    return _read_json(p)


def api_summaries():
    out = {}
    if CHECKPOINTS.exists():
        for p in sorted(CHECKPOINTS.glob("*/summary.json")):
            d = _read_json(p)
            if d:
                out[p.parent.name] = d
    return out


def api_arch():
    """Machine-checkable architecture description used by the explainer page.
    Shapes/values are pulled from the committed pilot configs so the page can
    never drift from reality."""
    import yaml
    cfg = yaml.safe_load((ROOT / "configs" / "e1_weights_passes.yaml")
                         .read_text(encoding="utf-8"))
    m = cfg.get("model", {})
    cons = cfg.get("consensus", {})
    tr = cfg.get("training", {})
    return {
        "R": m.get("n_regions", 3),
        "n_dots": m.get("n_dots", 4),
        "n_passes": m.get("n_passes", 3),
        "d_model": m.get("d_model", 64),
        "d_s": m.get("d_s", 32),
        "window": m.get("window", 16),
        "pooling": m.get("pooling", "log_linear"),
        "use_weights": m.get("use_weights", True),
        "kappa": cons.get("kappa", 0.9),
        "rho": cons.get("rho", 0.6),
        "tau_warmup": cons.get("tau_warmup", 115),
        "tau_max": cons.get("tau_max", 1.0),
        "tau_min": cons.get("tau_min", 0.2),
        "lambdas": cfg.get("losses", cfg.get("lambdas", {})),
        "lr": tr.get("lr", 1e-3),
        "lr_schedule": tr.get("lr_schedule", "cosine"),
        "batch_tokens": (cfg.get("data", {}).get("seq_len", 128)) *
                        (cfg.get("data", {}).get("batch_size", 128)),
        "vocab_size": cfg.get("model", {}).get("vocab_size", 512),
    }


ROUTES = {
    "runs": lambda: (200, api_runs()),
    "evals": lambda: (200, api_evals()),
    "reports": lambda: (200, api_reports()),
    "summaries": lambda: (200, api_summaries()),
    "arch": lambda: (200, api_arch()),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "IECNN-viz/1.0"

    def log_message(self, fmt, *args):          # quieter logs
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    # ------------------------------------------------------------ helpers --
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj):
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8")

    # ---------------------------------------------------------------- GET --
    def do_GET(self):
        path = self.path.split("?")[0]
        try:
            if path == "/api/runs":
                return self._send_json(*ROUTES["runs"]())
            if path == "/api/evals":
                return self._send_json(*ROUTES["evals"]())
            if path == "/api/reports":
                return self._send_json(*ROUTES["reports"]())
            if path == "/api/summaries":
                return self._send_json(*ROUTES["summaries"]())
            if path == "/api/arch":
                return self._send_json(*ROUTES["arch"]())
            m = re.match(r"^/api/run/([\w.\-]+)$", path)
            if m:
                out = api_run(m.group(1))
                return self._send_json(out.get("status", 200), out)
            m = re.match(r"^/api/viz/([\w.\-]+)$", path)
            if m:
                out = api_viz(m.group(1))
                return self._send_json(out.get("status", 200), out)
            if path == "/api/viz":
                return self._send_json(*ROUTES.get("viz_list", (200, api_viz_list())))

            if path.startswith("/api/"):
                return self._send_json(404, {"error": f"unknown endpoint {path}"})

            # static files
            name = "index.html" if path == "/" else path.lstrip("/")
            f = (STATIC / name).resolve()
            if not str(f).startswith(str(STATIC)) or not f.is_file():
                return self._send(404, b"not found", "text/plain")
            ctype = {"html": "text/html; charset=utf-8",
                     "js": "text/javascript; charset=utf-8",
                     "css": "text/css; charset=utf-8",
                     "json": "application/json"}.get(f.suffix.lstrip("."),
                                                     "application/octet-stream")
            return self._send(200, f.read_bytes(), ctype)
        except Exception as e:                                   # noqa: BLE001
            return self._send_json(500, {"error": repr(e)})


def main() -> None:
    ap = argparse.ArgumentParser(description="IECNN visualization server")
    ap.add_argument("--port", type=int, default=8765)
    ap.add_argument("--host", default="127.0.0.1")
    args = ap.parse_args()
    httpd = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"IECNN viz server on http://{args.host}:{args.port} "
          f"(root: {ROOT})  Ctrl+C to stop")
    try:
        httpd.serve_forever()
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
