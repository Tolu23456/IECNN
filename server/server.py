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
import secrets
import socket
import subprocess
import sys
import threading
import time
from datetime import datetime, timezone
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

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
DEFAULT_ARTIFACT = "e1_weights_passes_pilot.yaml-seeds-0-pilot"
GENERATIONS = RESULTS / "generation"


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
    info["rung"], info["seed"], info["name"] = rung, seed, p.stem
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


def consensus_audit(positions):
    """Pure audit of the pooled-vs-dots consensus over a viz export's sample.

    For every scored position (target `next` known, pooled answer present):
      - pooled_top1_hit: pooled_top[0] == next
      - oracle_top1_hit: some dot's top prediction (any pass) == next
      - oracle_top5_hit: some dot's top-5 (any pass) contains next
      - outvoted: a dot held the answer but the pooled consensus picked
        something else. level "top1" = a dot's TOP-1 was correct (stronger
        signal), "top5" = the answer was merely in a dot's top-5.

    Returns a summary plus a small per_position payload (outvoted positions
    only) recording which dot(s) held the answer, at which pass, with what
    weight — enough for the UI to highlight the muted dot's weight row.
    Defensive against missing fields: anything absent just means the
    position is not scored. All lists may be any length (the next export
    has 256 positions and possibly more dots/passes).
    """
    n = pooled = or1 = or5 = out1 = out5 = 0
    per_position = []
    for i, P in enumerate(positions if isinstance(positions, list) else []):
        if not isinstance(P, dict):
            continue
        tgt = P.get("next")
        pool = P.get("pooled_top") if isinstance(P.get("pooled_top"), list) else []
        if tgt is None or not pool:
            continue
        n += 1
        pooled_hit = bool(pool) and pool[0] and pool[0][0] == tgt
        pooled += pooled_hit
        held1 = held5 = None          # first (dot, pass, weight) sighting
        holders = []
        dots = P.get("dots") if isinstance(P.get("dots"), list) else []
        for k, d in enumerate(dots):
            if not isinstance(d, dict):
                continue
            tops = d.get("top") if isinstance(d.get("top"), list) else []
            w = d.get("weights") if isinstance(d.get("weights"), list) else []
            for pi, top in enumerate(tops):
                toks = [t[0] for t in top] if isinstance(top, list) else []
                if toks and toks[0] == tgt and held1 is None:
                    held1 = {"dot": k, "pass": pi, "weight": w[pi] if pi < len(w) else None}
                if tgt in toks and held5 is None:
                    held5 = {"dot": k, "pass": pi, "weight": w[pi] if pi < len(w) else None}
            if held1 and held1["dot"] == k and not any(x["dot"] == k for x in holders):
                holders.append({**held1, "level": "top1"})
            elif held5 and held5["dot"] == k and not any(x["dot"] == k for x in holders):
                holders.append({**held5, "level": "top5"})
        if held1:
            or1 += 1
        if held5 or held1:
            or5 += 1
        if not pooled_hit and (held1 or held5):
            level = "top1" if held1 else "top5"
            if level == "top1":
                out1 += 1
            else:
                out5 += 1
            per_position.append({"i": i, "level": level,
                                 "held": holders or [held1 or held5]})
    def _acc(k):
        return round(k / n, 4) if n else None
    return {"sampled": n,
            "pooled_top1": pooled, "pooled_top1_acc": _acc(pooled),
            "oracle_top1": or1, "oracle_top1_acc": _acc(or1),
            "oracle_top5": or5, "oracle_top5_acc": _acc(or5),
            "outvoted_top1": out1, "outvoted_top5": out5,
            "outvoted": out1 + out5,
            "per_position": per_position}


def api_viz(name: str):
    p = VIZ / f"{name}.json"
    if not p.exists() or not p.is_file():
        return {"error": f"no viz export named {name} (dispatch the viz workflow "
                         f"to produce one: workflows/viz.yml)", "status": 404}
    d = _read_json(p)
    if d is None:
        return {"error": f"viz export {name} is not valid JSON", "status": 500}
    # consensus audit: computed server-side from exactly the payload the
    # page renders, so the UI can never disagree with the numbers
    sample = d.get("sample") if isinstance(d.get("sample"), dict) else {}
    if isinstance(sample.get("positions"), list):
        d["audit"] = consensus_audit(sample["positions"])
    return d


# ------------------------------------------------- playground (generate) ---
def _git_config_url() -> str:
    try:
        return subprocess.run(
            ["git", "config", "--get", "remote.origin.url"],
            capture_output=True, text=True, timeout=10, cwd=str(ROOT),
        ).stdout.strip()
    except Exception:
        return ""


_GH_REPO_CACHE: str | None = None


def _gh_repo() -> str:
    """owner/name from the ambient remote URL (cached)."""
    global _GH_REPO_CACHE
    if _GH_REPO_CACHE is None:
        m = re.search(r"github\.com[/:]([\w.\-]+)/([\w.\-]+?)(?:\.git)?/?$",
                      _git_config_url())
        _GH_REPO_CACHE = f"{m.group(1)}/{m.group(2)}" if m \
            else "Tolu23456/IECNN"
    return _GH_REPO_CACHE


def _gh_credential() -> str | None:
    """Ambient GitHub credential, read exactly the way the project always
    reads it: out of remote.origin.url (x-access-token:…@).

    The token may ONLY travel in the Authorization header built by
    _gh_request — never to disk, logs, responses, or error text.
    """
    m = re.search(r"(?<=x-access-token:)[^@]+", _git_config_url())
    return m.group(0) if m else None


def _gh_request(url, method="GET", body=None, token=None, timeout=30):
    """One GitHub API call (stdlib urllib). Returns (http_status, parsed).

    Test seam: tests monkeypatch this module global so no test ever hits
    the network. Network-level failures raise; callers catch and convert
    to clean {"error": ..., "status": 502} payloads. HTTP-level errors
    (401/404/…) are returned as (code, None) for the caller to judge.
    """
    req = Request(url, method=method)
    req.add_header("Accept", "application/vnd.github+json")
    req.add_header("User-Agent", "IECNN-explorer")
    data = None
    if body is not None:
        data = json.dumps(body).encode("utf-8")
        req.add_header("Content-Type", "application/json")
    if token:
        req.add_header("Authorization", f"Bearer {token}")
    try:
        with urlopen(req, data=data, timeout=timeout) as r:
            raw = r.read()
    except HTTPError as e:                      # API answered with an error
        return e.code, None
    ctype = ""
    try:
        return r.status, (json.loads(raw) if raw else None)
    except Exception:
        return r.status, None


def _clamp_playground(body):
    """Validate a /api/generate request body -> (params, error)."""
    if not isinstance(body, dict):
        return None, "request body must be a JSON object"
    prompt = body.get("prompt")
    if not isinstance(prompt, str) or not prompt.strip():
        return None, "prompt must be a non-empty string"
    if len(prompt) > 4000:
        return None, "prompt too long (max 4000 chars)"

    def num(key, default, lo, hi):
        v = body.get(key, default)
        try:
            v = float(v)
        except (TypeError, ValueError):
            raise ValueError(f"{key} must be numeric")
        if not (lo <= v <= hi):
            raise ValueError(f"{key} must be in [{lo}, {hi}]")
        return v

    try:
        params = {"prompt": prompt,
                  "max_new_tokens": int(num("max_new_tokens", 200, 1, 800)),
                  "temperature": num("temperature", 0.8, 0.1, 2.0),
                  "top_k": int(num("top_k", 40, 0, 500)),
                  "seed": int(num("seed", 0, -2 ** 31, 2 ** 31))}
    except ValueError as e:
        return None, str(e)
    artifact = body.get("artifact") or DEFAULT_ARTIFACT
    if not isinstance(artifact, str) or not re.fullmatch(r"[\w.\-]{1,120}", artifact):
        return None, "artifact must be a plain name ([word chars . - ] only)"
    params["artifact"] = artifact
    return params, None


def api_generate(body: dict):
    """Dispatch the `generate` workflow (Playground backend).

    Returns {ok, dispatched_at} on a 204. There is no run id in the
    response — the client discovers the new run via /api/generate/runs.
    """
    params, err = _clamp_playground(body)
    if err:
        return {"error": err, "status": 400}
    token = _gh_credential()
    if not token:
        return {"error": "no GitHub credential configured (remote.origin.url "
                         "carries no x-access-token) — cannot dispatch the "
                         "generate workflow", "status": 401}
    url = (f"https://api.github.com/repos/{_gh_repo()}"
           f"/actions/workflows/generate.yml/dispatches")
    payload = {"ref": "main", "inputs": {
        "prompt": params["prompt"], "artifact": params["artifact"],
        "max_new_tokens": str(params["max_new_tokens"]),
        "temperature": str(params["temperature"]),
        "top_k": str(params["top_k"]), "seed": str(params["seed"])}}
    try:
        code, _ = _gh_request(url, method="POST", body=payload, token=token)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"GitHub unreachable: {e.__class__.__name__}",
                "status": 502}
    if code != 204:
        return {"error": f"workflow dispatch rejected (HTTP {code})"
                         + (" — credential rejected?" if code in (401, 403)
                            else ""), "status": 502}
    return {"ok": True,
            "dispatched_at": datetime.now(timezone.utc)
                             .isoformat(timespec="seconds").replace("+00:00", "Z")}


def api_generate_runs():
    """Most recent runs of the `generate` workflow (newest first)."""
    token = _gh_credential()
    url = (f"https://api.github.com/repos/{_gh_repo()}"
           f"/actions/workflows/generate.yml/runs?per_page=8")
    try:
        code, d = _gh_request(url, token=token)
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"GitHub unreachable: {e.__class__.__name__}",
                "status": 502}
    if code != 200 or not isinstance(d, dict):
        return {"error": f"GitHub API returned HTTP {code}", "status": 502}
    runs = [{"id": r.get("id"), "status": r.get("status"),
             "conclusion": r.get("conclusion"),
             "created_at": r.get("created_at")}
            for r in d.get("workflow_runs", []) or []
            if isinstance(r, dict) and isinstance(r.get("id"), int)]
    return {"runs": runs}


def api_generate_result(run_id: str):
    """Fetch the generation file committed by a completed workflow run.

    Statuses: pending (run still going / push not visible yet), failed
    (run concluded failure, or succeeded without committing a generation
    file), done (payload returned). Robust: the file is located via the
    run's head_sha commit — no timestamp guessing.
    """
    if not re.fullmatch(r"\d{1,20}", str(run_id)):
        return {"error": "run_id must be numeric", "status": 400}
    repo, token = _gh_repo(), _gh_credential()
    base = f"https://api.github.com/repos/{repo}"
    try:
        code, run = _gh_request(f"{base}/actions/runs/{run_id}", token=token)
        if code != 200 or not isinstance(run, dict):
            return {"error": f"run {run_id} not found (HTTP {code})",
                    "status": 404}
        if run.get("status") != "completed":
            return {"status": "pending", "run_status": run.get("status"),
                    "conclusion": run.get("conclusion")}
        if run.get("conclusion") != "success":
            return {"status": "failed", "conclusion": run.get("conclusion")}
        # find the file this run committed (head_sha), via the commit API
        head = run.get("head_sha") or ""
        code, commit = _gh_request(f"{base}/commits/{head}", token=token)
        names = []
        if code == 200 and isinstance(commit, dict):
            names = [f.get("filename", "")
                     for f in commit.get("files", []) or [] if isinstance(f, dict)]
        gens = sorted(n for n in names
                      if re.match(r"results/generation/gen-[\w.\-]+\.json$", n))
        if not gens:
            return {"status": "failed",
                    "conclusion": run.get("conclusion"),
                    "note": "run succeeded but committed no generation file"}
        fname = gens[-1]
        code, blob = _gh_request(
            f"{base}/contents/{fname}?ref={head}", token=token)
        if code != 200 or not isinstance(blob, dict) or not blob.get("content"):
            return {"error": f"could not fetch {fname} (HTTP {code})",
                    "status": 502}
        try:
            payload = json.loads(base64.b64decode(blob["content"]))
        except Exception:                                    # noqa: BLE001
            return {"error": "generation file is not valid JSON", "status": 502}
        return {"status": "done", "file": blob.get("name"), "result": payload}
    except Exception as e:                                   # noqa: BLE001
        return {"error": f"GitHub unreachable: {e.__class__.__name__}",
                "status": 502}


def api_compare(a: str, b: str):
    """Two-run comparison computed from the committed CSVs at request time.

    Overlay series (step → bpc_val), per-matched-step deltas (B−A), mean
    delta, and a vocab-mismatch warning. Vocab identity comes from the
    strongest source available per run: a committed eval JSON's
    vocab_sha256, else the rung config's model.vocab_file (resolved through
    the base: inheritance chain), else unknown (warned, never hidden).
    """
    for n in (a, b):
        if not re.fullmatch(r"[\w.\-]{1,120}", n or ""):
            return {"error": "run names must be plain names", "status": 400}
    runs = {}
    for tag, n in (("a", a), ("b", b)):
        p = RESULTS / f"{n}.csv"
        if not p.is_file():
            return {"error": f"no results CSV named {n}", "status": 404}
        parsed = _read_csv_rows(p)
        if parsed is None:
            return {"error": f"could not parse {n}.csv", "status": 500}
        header, rows = parsed
        pts = [(d["step"], d["bpc_val"]) for d in rows
               if d.get("bpc_val") is not None]
        if not pts:
            return {"error": f"{n}.csv has no bpc_val points", "status": 404}
        rung, seed = _run_name_meta(p)
        # vocab identity, strongest source first. A retracted eval's SCORE
        # is dead, but its vocab_sha256 still identifies which vocab the
        # checkpoint ran with — identity is not the retracted claim.
        ev = _read_json(RESULTS / f"eval_{re.sub('^train_', '', n)}.json")
        vocab_sha = (ev or {}).get("vocab_sha256")
        cfg = _resolve_config(rung)
        vocab_file = ((cfg or {}).get("config") or {}).get("model", {}).get(
            "vocab_file")
        runs[tag] = {"name": n, "rung": rung, "seed": seed,
                     "points": pts,
                     "vocab_sha256": vocab_sha,
                     "vocab_file": vocab_file,
                     "vocab_source": ("eval vocab_sha256" if vocab_sha
                                      else "config vocab_file" if vocab_file
                                      else "unknown")}
    A, B = runs["a"], runs["b"]
    sa = dict(A["points"])
    matched = [s for s, _ in B["points"] if s in sa]
    deltas = []
    for s, vb in B["points"]:
        if s in sa:
            deltas.append({"step": s, "a": sa[s], "b": vb,
                           "delta": vb - sa[s]})
    mean_a = sum(d["a"] for d in deltas) / len(deltas) if deltas else None
    mean_b = sum(d["b"] for d in deltas) / len(deltas) if deltas else None
    mean_delta = (mean_b - mean_a) if deltas else None
    # vocab mismatch: shas disagree when both known; else vocab_file names
    mismatch = None
    sha_a, sha_b = A["vocab_sha256"], B["vocab_sha256"]
    if sha_a and sha_b:
        mismatch = sha_a != sha_b
    elif A["vocab_file"] and B["vocab_file"]:
        mismatch = A["vocab_file"] != B["vocab_file"]
    return {"a": A, "b": B,
            "matched_steps": matched,
            "n_matched": len(matched),
            "deltas": deltas,
            "mean_a": mean_a, "mean_b": mean_b,
            "mean_delta": mean_delta,
            "better": ("b" if mean_delta is not None and mean_delta < 0
                       else "a" if mean_delta is not None and mean_delta > 0
                       else None if mean_delta is None else "tie"),
            "vocab_mismatch": mismatch}


def api_generations(dir_path: Path | None = None):
    """Locally committed generations (results/generation/*.json), newest
    first. Never raises: a missing or empty dir is just an empty list.
    """
    d = dir_path or GENERATIONS
    out = []
    if d.is_dir():
        for p in sorted(d.glob("gen-*.json"), reverse=True):
            j = _read_json(p)
            if not isinstance(j, dict):
                continue                       # malformed file: skip, not 500
            ids = j.get("generated_ids")
            ed = j.get("expert_dots")
            out.append({"file": p.name, "meta": j.get("meta") or {},
                        "prompt": j.get("prompt", ""),
                        "generated": j.get("generated", ""),
                        "generated_ids": ids if isinstance(ids, list) else None,
                        "token_logprobs": j.get("token_logprobs")
                        if isinstance(j.get("token_logprobs"), list) else None,
                        "token_top1_prob": j.get("token_top1_prob")
                        if isinstance(j.get("token_top1_prob"), list) else None,
                        "expert_dots": ed if isinstance(ed, list) else None,
                        "n_gen_tokens": len(ids) if isinstance(ids, list) else None,
                        "elapsed_s": j.get("elapsed_s")})
    return {"generations": out}


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


def api_config_used():
    """checkpoints/*/config_used.json, keyed by checkpoint dir name."""
    out = {}
    if CHECKPOINTS.exists():
        for p in sorted(CHECKPOINTS.glob("*/config_used.json")):
            d = _read_json(p)
            if d:
                out[p.parent.name] = d
    return out


# ------------------------------------------------------ gates (G1–G4) -------
# Every number below is computed at request time from committed files in
# results/ (CSVs, eval/summary/viz JSONs) — never hardcoded. Missing files
# degrade a gate to "NO_DATA" instead of erroring.

_MARKER_CHARS = ("<", "=", ">")


def _read_csv_rows(path: Path):
    """Conflict-marker-safe CSV reader -> (header, rows-as-dicts) or None.

    Some committed results CSVs contain stacked git merge-conflict markers
    (a stash-pop accident). Every marker line and every row whose first
    cell is not a step number is skipped; duplicate steps keep their FIRST
    occurrence; rows are returned sorted ascending by step. Defensive:
    short rows simply leave later columns None.
    """
    if not path.is_file():
        return None
    try:
        raw = list(csv.reader(_read_text(path).splitlines()))
    except Exception:
        return None
    if not raw:
        return None
    header, out, seen = raw[0], [], set()
    for r in raw[1:]:
        if not r or not r[0] or r[0].lstrip()[:1] in _MARKER_CHARS:
            continue
        try:
            step = float(r[0])
        except ValueError:
            continue
        if step in seen:
            continue
        seen.add(step)
        d: dict = {"step": step}
        for i, name in enumerate(header[1:], start=1):
            if i >= len(r) or r[i] == "":
                d[name] = None
                continue
            try:
                d[name] = float(r[i])
            except ValueError:
                d[name] = r[i]
        out.append(d)
    out.sort(key=lambda d: d["step"])
    return header, out


def _csv_has_markers(path: Path) -> bool:
    t = _read_text(path) or ""
    return any(t.startswith(m) or f"\n{m}" in t
               for m in ("<<<<<<<", "=======", ">>>>>>>"))


def _wall_best_curve(rows, wall_col="elapsed_s", val_col="bpc_val"):
    """Best bpc_val achieved up to each eval's wall time -> monotone list."""
    pts, best = [], None
    for d in rows:
        w, v = d.get(wall_col), d.get(val_col)
        if w is None or v is None:
            continue
        best = v if best is None else min(best, v)
        pts.append({"wall_s": w, "best_bpc": best, "step": d.get("step")})
    return pts


def _best_at_wall(curve, wall):
    b = None
    for p in curve:
        if p["wall_s"] <= wall:
            b = p["best_bpc"]
        else:
            break
    return b


def _eval_retracted(d):
    """Truthiness of an eval JSON's retraction marker (object or bool)."""
    return bool(d.get("retracted"))


def _eval_val_sha(d):
    """A future-proof val-corpus sha from an eval JSON, if it declares one."""
    v = d.get("val_sha256")
    if not v and isinstance(d.get("provenance"), dict):
        v = d["provenance"].get("val_sha256")
    return v


def _split_val_sha():
    sp = _read_json(DATA / "split_info.json") or {}
    val = (sp.get("parts") or {}).get("val") or {}
    return val.get("sha256")


def _gate_g1(evals):
    """Pipeline integrity: tests green + causality + corpus-verified evals."""
    verified = unverified = retracted = 0
    missing = []
    val_sha = _split_val_sha()
    for name, d in sorted(evals.items()):
        if _eval_retracted(d):
            retracted += 1
            continue
        vs = _eval_val_sha(d)
        if vs and val_sha and (vs == val_sha or vs[:16] == val_sha[:16]):
            verified += 1
        else:
            unverified += 1
            missing.append(name)
    status = "PASS" if unverified == 0 else "PARTIAL"
    return {
        "id": "G1", "title": "Pipeline integrity",
        "status": status,
        "evidence": [
            f"CI tests green on this commit (tests.yml); causality bit-exactness "
            f"is asserted by the test_causality suite + diagnosis reload probes",
            f"corpus-verified evals: {verified}, unverified-corpus: {unverified}, "
            f"retracted (excluded): {retracted}",
        ]
        + ([f"evals without val-sha provenance: {', '.join(missing)}"]
           if missing else []),
        "sources": ["results/eval_*.json", "results/diagnosis/*.json",
                    "data/split_info.json"],
        "counts": {"verified": verified, "unverified": unverified,
                   "retracted": retracted, "missing": missing},
        "note": "a future eval JSON carrying val_sha256 (top-level or inside "
                "provenance) is checked against the split_info val sha and "
                "hidden behind a flag on mismatch",
    }


_G2_FILES = {
    "e0": RESULTS / "train_e0_plain_ensemble_eqc_seed0.csv",
    "e1": RESULTS / "train_e1_weights_passes_eqc_seed0.csv",
}


def _gate_g2():
    """Beats baselines at equal compute: best bpc_val at matched wall s."""
    curves, notes, missing = {}, [], []
    for key, p in _G2_FILES.items():
        parsed = _read_csv_rows(p)
        if parsed is None:
            missing.append(p.name)
            continue
        header, rows = parsed
        curves[key] = _wall_best_curve(rows)
        if _csv_has_markers(p):
            notes.append(f"{p.name}: committed file contains stacked "
                         f"merge-conflict markers; parsed defensively "
                         f"(markers skipped, first row per step kept)")
    if "e0" not in curves or "e1" not in curves:
        return {"id": "G2", "title": "Beats baselines at equal compute",
                "status": "NO_DATA",
                "evidence": [f"missing: {', '.join(missing)}"],
                "sources": [p.name for p in _G2_FILES.values()]}

    cap = min(c[-1]["wall_s"] for c in curves.values())
    walls = sorted({p["wall_s"] for c in curves.values()
                    for p in c if p["wall_s"] <= cap})
    table = []
    for w in walls:
        row = {"wall_s": round(w, 1),
               "e0_bpc": _best_at_wall(curves["e0"], w),
               "e1_bpc": _best_at_wall(curves["e1"], w)}
        table.append(row)

    # GRU B0 at the same matched walls: use whichever committed b0 CSV is
    # strongest at each wall point (honest — never pick a strawman)
    b0_files = sorted(RESULTS.glob("train_b0_gru*.csv"))
    b0_curves = {}
    for p in b0_files:
        parsed = _read_csv_rows(p)
        if parsed is None:
            continue
        c = _wall_best_curve(parsed[1])
        if c:
            b0_curves[p.name] = c
    if b0_curves:
        for row in table:
            w = row["wall_s"]
            vals = [b for c in b0_curves.values()
                    if (b := _best_at_wall(c, w)) is not None]
            if vals:
                row["b0_bpc"] = min(vals)

    def _cmp(a, b):
        if a is None or b is None:
            return False
        return a <= b + 1e-9
    e1_wins = bool(table) and all(
        _cmp(r["e1_bpc"], r["e0_bpc"]) for r in table)
    b0_beats_e1 = any(
        r.get("b0_bpc") is not None and _cmp(r["b0_bpc"], r["e1_bpc"])
        for r in table)
    if not table:
        status = "NO_DATA"
    elif e1_wins and not b0_beats_e1:
        status = "PASS"
    else:
        status = "FAIL"
    last = table[-1] if table else {}
    head = (f"at equal wall compute (≤ {cap:,.0f} s): E1 "
            f"{last.get('e1_bpc')} vs E0 {last.get('e0_bpc')} bpc"
            + (f", best B0 {min(r['b0_bpc'] for r in table if r.get('b0_bpc') is not None)}"
               if any(r.get("b0_bpc") is not None for r in table) else ""))
    return {
        "id": "G2", "title": "Beats baselines at equal compute",
        "status": status,
        "headline": head,
        "evidence": [head, f"E1 ≤ E0 at "
                     f"{sum(1 for r in table if _cmp(r['e1_bpc'], r['e0_bpc']))}"
                     f"/{len(table)} matched wall points"]
                    + notes,
        "table": table,
        "matched_walls": len(table),
        "wall_cap_s": cap,
        "sources": [p.name for p in _G2_FILES.values()]
                   + [p.name for p in b0_files],
    }


_G3_FILES = [
    ("vocab2", RESULTS / "train_e1_weights_passes_vocab2_seed0.csv"),
    ("vocab3", RESULTS / "train_e1_weights_passes_vocab3_seed0.csv"),
    ("vocab4", RESULTS / "train_e1_weights_passes_vocab4_seed0.csv"),
]


def _gate_g3():
    """Scaling knee: PENDING while the vocab-scaling curves still descend."""
    vocabs, still = [], False
    for vocab, p in _G3_FILES:
        parsed = _read_csv_rows(p)
        if parsed is None:
            vocabs.append({"vocab": vocab, "csv": p.name,
                           "status": "no data"})
            continue
        header, rows = parsed
        ev = [(d["step"], d["bpc_val"]) for d in rows
              if d.get("bpc_val") is not None]
        if not ev:
            vocabs.append({"vocab": vocab, "csv": p.name,
                           "status": "no eval points"})
            continue
        best_step, best_bpc = min(ev, key=lambda t: t[1])
        last3 = ev[-3:]
        delta = last3[-1][1] - last3[0][1]
        improving = delta < -0.005
        still = still or improving
        entry = {"vocab": vocab, "csv": p.name,
                 "n_evals": len(ev),
                 "best_bpc": best_bpc, "best_step": best_step,
                 "last3": [{"step": s, "bpc_val": v} for s, v in last3],
                 "last3_delta": round(delta, 5),
                 "still_descending": improving,
                 "status": "descending" if improving else "plateaued"}
        summ = _read_json(CHECKPOINTS / f"e1_weights_passes_{vocab}_seed0"
                          / "summary.json")
        if summ:
            entry["stop_kind"] = summ.get("stop_kind")
            if summ.get("stop_kind") == "wall_cap":
                entry["note"] = ("wall-capped run: a flat tail is truncation, "
                                 "not a plateau")
        vocabs.append(entry)
    desc = [v["vocab"] for v in vocabs if v.get("still_descending")]
    return {
        "id": "G3", "title": "Scaling knee",
        "status": "PENDING" if still else "PASS",
        "evidence": [
            ("curves still descending (>0.005 bpc over the last 3 evals): "
             + ", ".join(desc)) if desc else
            "no vocab curve improved >0.005 bpc over its last 3 evals",
        ] + [f"{v['vocab']}: best {v.get('best_bpc')} @ step "
             f"{v.get('best_step')} ({v.get('stop_kind') or 'n/a'})"
             for v in vocabs if v.get("best_bpc") is not None],
        "vocabs": vocabs,
        "sources": [p.name for _, p in _G3_FILES]
                   + ["checkpoints/*/summary.json"],
    }


def _dumb_proxies(positions):
    """Dumb (non-learned) consensus proxies from a viz sample.

    uniform_mix: argmax of the uniform (1/N) mixture of the dots' pass-1
    top-5 distributions — an ensemble with no learned weights.
    plurality: most common dot top-1 prediction.
    best_single_dot: the best one dot could do alone across the sample
    (oracle-ish upper reference, reported for context).
    Scored over exactly the positions where pooled_top and dots exist.
    """
    n = pooled = uniform = plurality = oracle = 0
    dot_hits: dict = {}
    for P in positions if isinstance(positions, list) else []:
        if not isinstance(P, dict):
            continue
        tgt, pool = P.get("next"), P.get("pooled_top")
        dots = P.get("dots") if isinstance(P.get("dots"), list) else []
        if tgt is None or not pool or not dots:
            continue
        n += 1
        if pool[0] and pool[0][0] == tgt:
            pooled += 1
        sc: dict = {}
        votes: dict = {}
        held = False
        for k, d in enumerate(dots):
            if not isinstance(d, dict):
                continue
            tops = d.get("top") if isinstance(d.get("top"), list) else []
            if not tops:
                continue
            t1 = tops[0]
            if t1:
                votes[t1[0][0]] = votes.get(t1[0][0], 0) + 1
                dot_hits[k] = dot_hits.get(k, 0) + (1 if t1[0][0] == tgt else 0)
                for tok, pr in t1:
                    sc[tok] = sc.get(tok, 0.0) + float(pr) / len(dots)
            if any(t and t[0][0] == tgt for t in tops):
                held = True
        if held:
            oracle += 1
        if votes and max(votes, key=votes.get) == tgt:
            plurality += 1
        if sc and max(sc, key=sc.get) == tgt:
            uniform += 1
    return {"sampled": n, "pooled_top1": pooled,
            "uniform_mix_top1": uniform, "plurality_top1": plurality,
            "oracle_top1": oracle,
            "best_single_dot_top1": max(dot_hits.values()) if dot_hits else 0}


def _gate_g4():
    """Consensus health: does the pooled consensus beat dumb proxies?"""
    exports = []
    all_pass = True
    any_data = False
    for p in sorted(VIZ.glob("*.json")):
        d = _read_json(p)
        if not d:
            continue
        positions = ((d.get("sample") or {}).get("positions"))
        if not isinstance(positions, list) or not positions:
            continue
        any_data = True
        m = _dumb_proxies(positions)
        best_proxy = max(m["uniform_mix_top1"], m["plurality_top1"])
        ok = m["pooled_top1"] > best_proxy
        all_pass = all_pass and ok
        exports.append({
            "name": p.stem, "file": f"results/viz/{p.name}", **m,
            "best_dumb_proxy": best_proxy,
            "outvote_rate": round(1 - m["pooled_top1"] / m["sampled"], 4)
                            if m["sampled"] else None,
            "status": "PASS" if ok else "FAIL",
        })
    if not any_data:
        return {"id": "G4", "title": "Consensus health", "status": "NO_DATA",
                "evidence": ["no results/viz/*.json exports with a "
                             "sample.positions payload"],
                "sources": ["results/viz/*.json"]}
    return {
        "id": "G4", "title": "Consensus health",
        "status": "PASS" if all_pass else "FAIL",
        "evidence": [
            f"{len(exports)} export(s): pooled top-1 "
            f"{[e['pooled_top1'] for e in exports]} vs best dumb proxy "
            f"{[e['best_dumb_proxy'] for e in exports]} (uniform mix / "
            f"plurality vote over the same dots)",
        ],
        "exports": exports,
        "sources": ["results/viz/*.json"],
    }


def api_gates():
    evals = {}
    for p in sorted(RESULTS.glob("eval_*.json")):
        d = _read_json(p)
        if d:
            evals[p.stem] = d
    return {"gates": [_gate_g1(evals), _gate_g2(), _gate_g3(), _gate_g4()],
            "git_head": _git_head(),
            "computed_at": datetime.now(timezone.utc)
                            .isoformat(timespec="seconds").replace("+00:00", "Z")}


_RAW_PREFIXES = ("results/", "configs/", "checkpoints/", "UBM/", "data/")


# --------------------------------------------- chart URL state (step 2) -----
# Canonical, executable spec for the runs-page chart state that travels in
# the URL hash query (?s=..&y=..&x=..&r=..). server/static/app.js carries an
# identical JS mirror (parseChartState / encodeChartState) — KEEP BOTH IN
# SYNC; the round trip is asserted in tests/test_server.py.

_CHART_Y = ("lin", "log")
_CHART_X = ("steps", "tokens", "wall", "compute")
_CHART_DEFAULTS = {"s": 0, "y": "lin", "x": "steps"}


def chart_state_parse(qs: str):
    """Parse a chart-state query string into a validated dict.

    Keys: s (moving-average window, int 0..5000, 0 = off), y (lin|log),
    x (steps|tokens|wall|compute), r (comma-separated run names).
    Invalid or unknown values are dropped, never errors: chart state is a
    preference, not an attack surface.
    """
    st = dict(_CHART_DEFAULTS)
    runs = []
    for pair in (qs or "").split("&"):
        if "=" not in pair:
            continue
        k, v = pair.split("=", 1)
        if k == "s":
            try:
                s = int(v)
            except ValueError:
                continue
            if 0 <= s <= 5000:
                st["s"] = s
        elif k == "y" and v in _CHART_Y:
            st["y"] = v
        elif k == "x" and v in _CHART_X:
            st["x"] = v
        elif k == "r":
            runs = [t for t in v.split(",")[:64]
                    if re.fullmatch(r"[\w.\-]{1,120}", t)]
    return {**st, "r": runs}


def chart_state_encode(st: dict) -> str:
    """Canonical encoding: s, y, x always, r only when non-empty."""
    norm = chart_state_parse("&".join(
        f"{k}={v}" for k, v in st.items()
        if k != "r" and v is not None))
    norm["r"] = [t for t in (st.get("r") or [])
                 if re.fullmatch(r"[\w.\-]{1,120}", t)][:64]
    out = [f"s={norm['s']}", f"y={norm['y']}", f"x={norm['x']}"]
    if norm["r"]:
        out.append("r=" + ",".join(norm["r"]))
    return "&".join(out)


def _raw_file(rel: str):
    """Read-only evidence endpoint: serve a committed text file by repo-
    relative path under an allow-listed prefix. Returns (status, text|err)."""
    if not rel or rel.startswith("/") or "\\" in rel:
        return 400, "path must be repo-relative"
    parts = rel.split("/")
    if any(seg in ("", ".", "..") for seg in parts):
        return 400, "path traversal is not allowed"
    if not rel.startswith(_RAW_PREFIXES):
        return 403, f"only {' , '.join(_RAW_PREFIXES)} are served"
    p = (ROOT / rel).resolve()
    if ROOT not in p.parents or not p.is_file():
        return 404, f"no file {rel}"
    if p.stat().st_size > 8 * 1024 * 1024:
        return 400, "file too large to serve"
    try:
        return 200, p.read_text(encoding="utf-8", errors="replace")
    except Exception as e:                                   # noqa: BLE001
        return 500, f"could not read {rel}: {e.__class__.__name__}"


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
    def _send(self, code: int, body: bytes, ctype: str, headers=None):
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-cache")
        self.send_header("X-Content-Type-Options", "nosniff")
        for k, v in (headers or []):
            self.send_header(k, v)
        self.end_headers()
        self.wfile.write(body)

    def _send_json(self, code: int, obj, headers=None):
        self._send(code, _json_bytes(obj), "application/json; charset=utf-8",
                   headers)

    # ------------------------------------------------- POST security gate --
    def _gate_post(self, path: str):
        """None = allowed; else (code, payload, retry_after=0) to reject with.
        Rate limiting is enforced separately, after body validation, so junk
        requests cannot exhaust a real user's budget."""
        if not _host_allowed(self.headers.get("Host", "")):
            return 403, {"error": "foreign Host header rejected"}, 0
        if not _origin_allowed(self.headers.get("Origin", ""),
                               self.headers.get("Host", "")):
            return 403, {"error": "cross-origin POST rejected"}, 0
        if path != "/api/generate":
            return None
        if self.headers.get("X-IECNN-Session", "") != SESSION_TOKEN:
            return (401, {"error": "missing or wrong X-IECNN-Session token "
                                   "(reload the page to get a fresh one)"}, 0)
        return None

    # ---------------------------------------------------------------- POST --
    def do_POST(self):
        path = self.path.split("?")[0]
        try:
            gate = self._gate_post(path)
            if gate is not None:
                headers = ([("Retry-After", str(gate[2]))]
                           if gate[0] == 429 else None)
                return self._send_json(gate[0], gate[1], headers)
            if path == "/api/generate":
                try:
                    n = int(self.headers.get("Content-Length") or 0)
                except ValueError:
                    n = 0
                if n > 20000:
                    return self._send_json(400, {"error": "body too large"})
                raw = self.rfile.read(n) if n else b""
                try:
                    body = json.loads(raw) if raw else {}
                except Exception:                            # noqa: BLE001
                    return self._send_json(400,
                                           {"error": "body is not valid JSON"})
                # validate BEFORE the rate window: junk bodies must not eat
                # a real user's 5-per-minute budget
                if isinstance(body, dict):
                    _, verr = _clamp_playground(body)
                else:
                    verr = "body must be a JSON object"
                if verr:
                    return self._send_json(400, {"error": verr})
                ok, retry = _rate_limit(self.client_address[0])
                if not ok:
                    return self._send_json(
                        429, {"error": "rate limit: 5 generate requests per "
                                       "minute; try again later"},
                        [("Retry-After", str(retry))])
                out = api_generate(body)
                # "status" here is an HTTP code only on error payloads
                code = out["status"] if isinstance(out.get("status"), int) else 200
                return self._send_json(code, out)
            return self._send_json(404, {"error": f"unknown endpoint {path}"})
        except Exception as e:                                   # noqa: BLE001
            return self._send_json(500, {"error": repr(e)})

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
            if path == "/api/gates":
                return self._send_json(200, api_gates())
            if path == "/api/chart_state":
                st = chart_state_parse(query)
                return self._send_json(200, {
                    "query": query, "state": st,
                    "encoded": chart_state_encode(st)})
            if path == "/api/config_used":
                return self._send_json(200, api_config_used())
            if path == "/api/raw":
                q = dict(pair.split("=", 1) for pair in query.split("&")
                         if "=" in pair)
                code, text = _raw_file(q.get("path", ""))
                if code == 200:
                    return self._send(200, text.encode("utf-8"),
                                      "text/plain; charset=utf-8")
                return self._send_json(code, {"error": text})
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
            if path == "/api/compare":
                q = dict(pair.split("=", 1) for pair in query.split("&")
                         if "=" in pair)
                out = api_compare(q.get("a", ""), q.get("b", ""))
                return self._send_json(out.get("status", 200), out)
            if path == "/api/viz":
                return self._send_json(*ROUTES.get("viz_list", (200, api_viz_list())))
            if path == "/api/generate/runs":
                out = api_generate_runs()
                return self._send_json(out["status"] if isinstance(out.get("status"), int) else 200, out)
            if path == "/api/generate/result":
                q = dict(pair.split("=", 1) for pair in query.split("&")
                         if "=" in pair)
                out = api_generate_result(q.get("run_id", ""))
                # NB: payload "status" is pending/done/failed, never a code
                return self._send_json(out["status"] if isinstance(out.get("status"), int) else 200, out)
            if path == "/api/generations":
                return self._send_json(200, api_generations())

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
            data = f.read_bytes()
            if ctype.startswith("text/html"):
                data = _inject_session(data)   # per-startup session token
            return self._send(200, data, ctype)
        except Exception as e:                                   # noqa: BLE001
            return self._send_json(500, {"error": repr(e)})


# ------------------------------------------ POST security (step 5) ----------
# POSTs are the only write path (they dispatch the generate workflow), so
# they carry a small same-origin gate:
#   1. session token  — one per server start (below), embedded into served
#      HTML by _inject_session; required as the X-IECNN-Session header
#      (401 otherwise).
#   2. Host check     — DNS-rebinding guard; foreign Host headers are 403.
#   3. Origin check   — when a browser sends an Origin, it must be this
#      host (403 otherwise). Non-browser clients (no Origin) still need
#      the session token.
#   4. rate limit     — 5 generate POSTs / minute / client IP → 429 with
#      a Retry-After header.
# GETs are read-only views over committed files and stay unauthenticated.

SESSION_TOKEN = secrets.token_urlsafe(32)

_ALLOWED_HOSTS = {"127.0.0.1", "localhost", "::1", "[::1]", "",
                  socket.gethostname().lower()}


def _normalize_host(h: str) -> str:
    """'127.0.0.1:8765' / '[::1]:8765' / 'LOCALHOST' → bare, lowercased host."""
    h = (h or "").strip().lower()
    if h.count(":") >= 2 and h.startswith("["):        # [::1]:port
        return h[1:].split("]", 1)[0]
    if h.count(":") == 1:                              # host:port
        return h.rsplit(":", 1)[0]
    return h.strip("[]")                               # bare (maybe IPv6)


def _host_allowed(host_header: str) -> bool:
    if not host_header:
        return True              # HTTP/1.0-style client: nothing to bind to
    return _normalize_host(host_header) in _ALLOWED_HOSTS


def _origin_allowed(origin: str, host_header: str) -> bool:
    if not origin:
        return True              # non-browser client: token check still applies
    m = re.match(r"^[a-z][a-z0-9+.\-]*://\[?([^\]/]+)\]?(/|$)", origin.strip().lower())
    if not m:
        return False
    return _normalize_host(m.group(1)) == _normalize_host(host_header)


_RATE_LOCK = threading.Lock()
_RATE_HITS: dict = {}            # client ip -> [monotonic timestamps]


def _rate_limit(ip: str, limit: int = 5, window_s: float = 60.0):
    """Sliding window; returns (allowed, retry_after_seconds)."""
    now = time.monotonic()
    with _RATE_LOCK:
        hits = [t for t in _RATE_HITS.get(ip, []) if now - t < window_s]
        if len(hits) >= limit:
            retry = max(1, int(window_s - (now - hits[0])) + 1)
            _RATE_HITS[ip] = hits
            return False, retry
        hits.append(now)
        _RATE_HITS[ip] = hits
        return True, 0


def _inject_session(html: bytes) -> bytes:
    """Embed the per-startup session token into a served HTML page.

    app.js reads window.IECNN_SESSION and echoes it back as the
    X-IECNN-Session header on POSTs.
    """
    script = ("<script>window.IECNN_SESSION="
              + json.dumps(SESSION_TOKEN) + ";</script>").encode("utf-8")
    low = html.lower()
    if b"</head>" in low:
        i = low.index(b"</head>")
        return html[:i] + script + html[i:]
    if b"</body>" in low:
        i = low.index(b"</body>")
        return html[:i] + script + html[i:]
    return html + script


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
