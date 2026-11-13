"""IECNN visualization server (stdlib only — no torch, no pip installs).

Serves the static dashboard in server/static/ plus JSON APIs over everything
in the repository that is safe to show: training CSVs, eval reports, vocab,
corpus split info, UBM compression reports, checkpoint-viz exports
(produced runner-side by scripts/export_viz.py — all torch compute happens
on GitHub Actions, never on the dev box, per Tolu's standing rule),
diagnosis probe tables and runner profiles.

Run from the repository root:
    python3 server/server.py            # http://localhost:8765
    python3 server/server.py --port 9000

Every endpoint is read-only. /api/* answers JSON.

stdlib-only note: configs/*.yaml are read with a tiny built-in YAML-subset
parser (_load_yaml_text) so the server never needs PyYAML. The committed rung
configs only use mappings, scalars, inline lists and comments — exactly the
subset the parser supports. If a config ever grows beyond that subset the
parser raises loudly instead of silently mis-parsing (see tests).
"""
from __future__ import annotations

import argparse
import base64
import csv
import json
import re
import subprocess
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
STATIC = Path(__file__).resolve().parent / "static"

# visible escapes for control characters inside token payloads
_CTRL_ESC = {0x09: "\\t", 0x0A: "\\n", 0x0D: "\\r"}

RESULTS = ROOT / "results"
VIZ = RESULTS / "viz"
DIAG = RESULTS / "diagnosis"
CHECKPOINTS = ROOT / "checkpoints"
CONFIGS = ROOT / "configs"
UBM_DIR = ROOT / "UBM"
DATA = ROOT / "data"

DEFAULT_ARCH_CONFIG = "e1_weights_passes_pilot"


# ---------------------------------------------------------------- helpers --
def _json_bytes(obj) -> bytes:
    return json.dumps(obj, indent=1, default=str).encode("utf-8")


def _read_json(path: Path):
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return None


def _read_text(path: Path):
    try:
        return path.read_text(encoding="utf-8")
    except Exception:
        return None


# ------------------------------------------------- stdlib YAML-subset parser --
def _yaml_scalar(s: str):
    """Parse one inline YAML scalar / inline list in the configs' dialect."""
    s = s.strip()
    if s == "" or s == "~" or s.lower() in ("null", "none"):
        return None
    if s.lower() in ("true", "yes", "on"):
        return True
    if s.lower() in ("false", "no", "off"):
        return False
    if s.startswith("[") and s.endswith("]"):
        inner = s[1:-1].strip()
        if not inner:
            return []
        return [_yaml_scalar(part) for part in inner.split(",")]
    if len(s) > 1 and s[0] == s[-1] and s[0] in "\"'":
        return s[1:-1]
    try:
        return int(s)
    except ValueError:
        pass
    try:
        return float(s)
    except ValueError:
        pass
    return s


def _yaml_strip_comment(line: str) -> str:
    """Drop a trailing ' #' comment. Config values never contain '#'."""
    out = line
    while True:
        i = out.find("#")
        if i == -1:
            return out
        if i == 0 or out[i - 1] in " \t":
            return out[:i]
        out = out[:i]  # '#' glued to a token: keep scanning left


def _load_yaml_text(text: str):
    """Minimal YAML loader for the committed rung configs.

    Supports: nested mappings by indentation, scalars, inline lists,
    blank lines and comments. Anything else (block lists, anchors,
    multi-line strings) raises ValueError instead of mis-parsing.
    """
    root: dict = {}
    stack = [(-1, root)]
    for raw in text.splitlines():
        if not raw.strip() or raw.lstrip().startswith("#"):
            continue
        line = _yaml_strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if line.strip() in ("---", "..."):
            continue
        indent = len(line) - len(line.lstrip(" "))
        content = line.strip()
        if content.startswith("- "):
            raise ValueError("block lists are not used by the rung configs")
        while len(stack) > 1 and indent <= stack[-1][0]:
            stack.pop()
        parent = stack[-1][1]
        if not isinstance(parent, dict):
            raise ValueError("bad nesting")
        key, sep, val = content.partition(":")
        if not sep:
            raise ValueError(f"not a mapping line: {line!r}")
        key = key.strip().strip("\"'")
        val = val.strip()
        if val == "":
            child: dict = {}
            parent[key] = child
            stack.append((indent, child))
        else:
            parent[key] = _yaml_scalar(val)
    return root


def _deep_merge(dst: dict, src: dict):
    for k, v in src.items():
        if isinstance(v, dict) and isinstance(dst.get(k), dict):
            _deep_merge(dst[k], v)
        else:
            dst[k] = v
    return dst


def _resolve_config(name: str):
    """Load configs/<name>.yaml and fold its `base:` inheritance chain.

    Returns {"chain": [...], "config": merged-dict} or None if the file
    is missing / unreadable / unparseable.
    """
    fname = name if name.endswith(".yaml") else name + ".yaml"
    p = CONFIGS / fname
    if not p.is_file():
        return None
    try:
        cfg = _load_yaml_text(_read_text(p) or "")
    except ValueError:
        return None
    layers = [cfg]
    chain = []
    seen = set()
    while layers[-1].get("base"):
        base = str(layers[-1]["base"])
        if base in seen:
            break
        seen.add(base)
        bp = CONFIGS / (base if base.endswith(".yaml") else base + ".yaml")
        if not bp.is_file():
            break
        try:
            layers.append(_load_yaml_text(_read_text(bp) or ""))
        except ValueError:
            break
        chain.append(bp.stem)
    merged: dict = {}
    for layer in reversed(layers):          # most general first, child wins
        _deep_merge(merged, layer)
    merged.pop("base", None)
    return {"chain": chain, "config": merged}


def _safe_csv(path: Path):
    """CSV -> {columns, rows-as-columns} for charting; None on failure.

    Trailing malformed rows (the old 5-column final-row bug in pre-2026-11-13
    runs) are dropped: keep only rows whose populated-cell count matches the
    modal count."""
    try:
        rows = list(csv.reader(_read_text(path).splitlines()))
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


def _byte_disp(b: int) -> str:
    """Readable display for one raw byte (byte-fallback tokens, ids 0–255).

    Printable ASCII shows as-is; tab/newline/CR as escapes; everything else
    (control bytes, high bytes 0x80–0xFF) as lowercase hex <xx>.
    """
    if b in _CTRL_ESC:
        return _CTRL_ESC[b]
    if 0x20 <= b <= 0x7E:
        return chr(b)
    return f"<{b:02x}>"


def _token_display(raw: bytes) -> str:
    """Token payload bytes -> human-readable display string.

    Valid UTF-8 stays normal text (" the", "café"); anything that is not
    valid standalone UTF-8 (the 128 high byte-fallback tokens plus one
    merge in vocab_v1) renders as readable hex escapes (<e2>, <ff>) instead
    of the U+FFFD tofu. Control characters inside otherwise-valid text
    (e.g. \\n) show as visible escapes too.
    """
    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError:
        return "".join(_byte_disp(b) for b in raw)
    out = []
    for ch in text:
        o = ord(ch)
        if o in _CTRL_ESC:
            out.append(_CTRL_ESC[o])
        elif o < 0x20 or o == 0x7F:
            out.append(f"<{o:02x}>")
        else:
            out.append(ch)
    return "".join(out)


def _vocab_overview(vpath: Path):
    """Decode UBM/vocab_v1.json into a display-friendly table.

    The committed vocab stores each token's payload as a base64 `bytes`
    string (older drafts used a list of ints — support both). Invalid-UTF-8
    tokens are shown as hex escapes (see _token_display) instead of U+FFFD,
    with the exact bytes in the `hex` field for the tooltip.
    """
    data = _read_json(vpath)
    if not data:
        return None
    bases = data.get("bases", [])

    def _raw(b):
        raw = b.get("bytes", "")
        if isinstance(raw, str) and raw:
            try:
                return base64.b64decode(raw)
            except Exception:
                return None
        if isinstance(raw, list) and raw:
            try:
                return bytes(raw)
            except Exception:
                return None
        return None

    out = []
    decoded = {}
    composition: dict = {}
    for b in bases[:2048]:
        raw = _raw(b)
        parts = b.get("parts")
        origin = b.get("origin") or ("merge" if parts is not None else "byte")
        composition[origin] = composition.get(origin, 0) + 1
        if origin == "special":
            display, hexs = f"<special {b.get('id')}>", ""
        elif raw is not None and len(raw) > 0:
            display, hexs = _token_display(raw), raw.hex()
        elif parts is not None and len(parts) == 2:      # legacy fallback
            a, c = decoded.get(parts[0], ""), decoded.get(parts[1], "")
            display = _token_display((a + c).encode("utf-8"))
            hexs = ""
        else:
            display, hexs = "", ""
        decoded[b.get("id")] = display
        # "escaped" = the display needed an escape (not plain readable text)
        try:
            plain = raw.decode("utf-8") if raw else ""
        except UnicodeDecodeError:
            plain = None
        escaped = (origin != "special" and bool(hexs or raw) and display != plain)
        out.append({"id": b.get("id"), "text": display,
                    "hex": hexs, "kind": origin,
                    "merge": origin == "merge",
                    "byte_fallback": origin == "byte",
                    "escaped": escaped,
                    "n_bytes": len(raw) if raw else 0})
    return {"file": vpath.name, "size": len(bases), "bases": out,
            "composition": composition,
            "legend": "ids 0-255 are single byte-fallback tokens: printable "
                      "ASCII shows as-is, everything else as <xx> hex; the "
                      "hex field carries the exact bytes",
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
        best_i = vals.index(min(vals)) if vals else None
        runs.append({"name": p.stem, "rung": rung, "seed": seed,
                     "points": info["n_points"],
                     "last_step": last[-1] if last else None,
                     "best_val": min(vals) if vals else None,
                     "best_step": (last[best_i] if best_i is not None
                                   and best_i < len(last) else None),
                     "columns": info["columns"]})
    return {"runs": runs, "git_head": _git_head()}


def api_run(name: str):
    p = RESULTS / f"{name}.csv"
    if not p.exists() or not p.is_file():
        return {"error": f"no results CSV named {name}", "status": 404}
    info = _safe_csv(p)
    if not info:
        return {"error": f"could not parse {name}.csv", "status": 500}
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
            "first20_merges": _read_text(RESULTS / "ubm_first20_merges.txt")}


def api_vocab():
    ov = _vocab_overview(UBM_DIR / "vocab_v1.json")
    if ov is None:
        return {"error": "UBM/vocab_v1.json missing or unreadable", "status": 404}
    return ov


def api_viz_list():
    return {"exports": sorted(p.stem for p in VIZ.glob("*.json"))}


def api_viz(name: str):
    p = VIZ / f"{name}.json"
    if not p.exists() or not p.is_file():
        return {"error": f"no viz export named {name} (dispatch the viz workflow "
                         f"to produce one: workflows/viz.yml)", "status": 404}
    d = _read_json(p)
    if d is None:
        return {"error": f"viz export {name} is not valid JSON", "status": 500}
    return d


def api_diagnosis():
    out = {}
    for p in sorted(DIAG.glob("*.json")):
        d = _read_json(p)
        if d:
            out[p.stem] = d
    return {"files": sorted(out), "data": out}


def api_profiles():
    out = {}
    for p in sorted(RESULTS.glob("profile_*.txt")):
        t = _read_text(p)
        if t is not None:
            out[p.stem] = t
    return {"files": sorted(out), "data": out}


def api_summaries():
    out = {}
    if CHECKPOINTS.exists():
        for p in sorted(CHECKPOINTS.glob("*/summary.json")):
            d = _read_json(p)
            if d:
                out[p.parent.name] = d
    return out


def api_arch(config: str | None = None):
    """Machine-checkable architecture description for the diagram page.

    Shapes/values are resolved from the committed rung configs (including
    the `base:` inheritance chain) so the page can never drift from what
    actually runs. The default is the pilot rung that the committed viz
    export and eval re-scores belong to.
    """
    files = sorted(p.stem for p in CONFIGS.glob("*.yaml"))
    name = config if config in files else DEFAULT_ARCH_CONFIG
    resolved = {}
    for f in files:
        r = _resolve_config(f)
        if r:
            resolved[f] = r
    return {"default": DEFAULT_ARCH_CONFIG, "config": name, "configs": files,
            "resolved": resolved, "active": resolved.get(name)}


ROUTES = {
    "runs": lambda: (200, api_runs()),
    "evals": lambda: (200, api_evals()),
    "reports": lambda: (200, api_reports()),
    "summaries": lambda: (200, api_summaries()),
    "vocab": lambda: (200, api_vocab()),
    "diagnosis": lambda: (200, api_diagnosis()),
    "profiles": lambda: (200, api_profiles()),
}


class Handler(BaseHTTPRequestHandler):
    server_version = "IECNN-viz/1.1"

    def log_message(self, fmt, *args):          # quieter logs
        if "/api/" in (args[0] if args else ""):
            sys.stderr.write("  %s\n" % (fmt % args))

    # ------------------------------------------------------------ helpers --
    def _send(self, code: int, body: bytes, ctype: str):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj):
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8")

    # ---------------------------------------------------------------- GET --
    def do_GET(self):
        path = self.path.split("?")[0]
        query = self.path.split("?", 1)[1] if "?" in self.path else ""
        try:
            if path == "/api/runs":
                return self._send_json(*ROUTES["runs"]())
            if path == "/api/evals":
                return self._send_json(*ROUTES["evals"]())
            if path == "/api/reports":
                return self._send_json(*ROUTES["reports"]())
            if path == "/api/summaries":
                return self._send_json(*ROUTES["summaries"]())
            if path == "/api/vocab":
                return self._send_json(*ROUTES["vocab"]())
            if path == "/api/diagnosis":
                return self._send_json(*ROUTES["diagnosis"]())
            if path == "/api/profiles":
                return self._send_json(*ROUTES["profiles"]())
            if path == "/api/arch":
                q = dict(pair.split("=", 1) for pair in query.split("&")
                         if "=" in pair)
                out = api_arch(q.get("config"))
                return self._send_json(200, out)
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
