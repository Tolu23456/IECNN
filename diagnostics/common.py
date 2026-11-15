"""Shared machinery for the spec-08 §6 diagnostics suite (built 2026-11-14).

Every diagnostic module exposes `run(checkpoint_path, config_path, out_dir,
**kw) -> dict`; the heavy lifting (checkpoint loading, one ordered pass over
eval text, shared math) lives here.

Design notes:
- `collect` walks the text in the SAME non-overlapping-window order as
  `train.evaluate.evaluate_bpc`, so per-position NLL and character counts are
  exactly consistent with the headline bpc numbers.
- All returned tensors are CPU fp32, flattened to a leading position axis.
"""
from __future__ import annotations

import json
import math
from pathlib import Path

import torch

from UBM import UBM
from UBM.tokenizer import fingerprint
from train.evaluate import _equal_length_batches, prepare_text, EVAL_BATCH
from train.schedules import temperature_at
from train.train import build_model


# --------------------------------------------------------------------------
# checkpoint loading
# --------------------------------------------------------------------------
def load_run(ckpt_path, config_path=None, vocab=None,
             config_fallback=None) -> dict:
    """Load (model, config, ubm, tau0) from a checkpoint dir or .pt file.

    Same guards as scripts/eval_checkpoint.py: best_state preferred, and the
    checkpoint's recorded vocab fingerprint must match the vocab file used
    here (Tolu, 2026-11-13: never silently evaluate against another vocab).
    """
    p = Path(ckpt_path)
    d = p if p.is_dir() else p.parent
    ck = torch.load(d / "checkpoint.pt" if p.is_dir() else p,
                    map_location="cpu", weights_only=False)
    if config_path:
        # The "fallback" CLI arg is a YAML rung config (diagnose_all passes
        # --config-fallback here); json.loads would die on its first char.
        # Route by extension (2026-11-15: this YAML-as-JSON bug was latent
        # behind the corpus-guard failures that died earlier in CI).
        # NO tiny.yaml overlay: load_config MERGES overlays OVER the config,
        # which silently rebuilt the model with tiny dims (7th-try failure);
        # rung configs carry their own base: chain.
        if str(config_path).endswith((".yaml", ".yml")):
            from config import load_config
            config = load_config(config_path)
        else:
            config = json.loads(Path(config_path).read_text())
    else:
        cu = d / "config_used.json"
        if cu.exists():
            config = json.loads(cu.read_text())
        elif config_fallback:
            from config import load_config
            # no tiny overlay: overlays MERGE OVER the config and would
            # rebuild the model with tiny dims (see the 7th-try CI failure)
            config = load_config(config_fallback)
        else:
            raise FileNotFoundError(
                f"no config_used.json next to {d} and no fallback given")
    vocab = vocab or config.get("ubm", {}).get("vocab_file", "UBM/vocab_v0.json")
    rec = str(ck.get("vocab_sha256", ""))
    if rec and fingerprint(vocab) != rec:
        raise RuntimeError(
            f"VOCAB MISMATCH: checkpoint trained with sha {rec[:16]}… but "
            f"'{vocab}' hashes differently — refusing diagnostics")
    ubm = UBM.load(vocab)
    model = build_model(config, len(ubm.bases))
    model.load_state_dict(ck.get("best_state") or ck["model"])
    model.eval()
    is_iecnn = config.get("model", {}).get("kind") == "iecnn"
    cons = config.get("consensus", {})
    tau0 = 1.0
    if is_iecnn:
        tau0 = temperature_at(
            int(ck.get("best_step", 0)),
            int(config.get("training", {}).get("steps", 1)),
            int(cons.get("tau_warmup", 2000)),
            float(cons.get("tau_max", 1.0)),
            float(cons.get("tau_min", 0.2)))
    return {"model": model, "config": config, "ubm": ubm, "tau0": tau0,
            "best_step": int(ck.get("best_step", 0)), "ckpt_dir": str(d)}


# --------------------------------------------------------------------------
# ordered pass over eval text
# --------------------------------------------------------------------------
@torch.no_grad()
def collect(model, ubm, text, *, region="last", passes="last", need=(),
            seq_len=None, tau0=1.0, floor=0.0, max_chars=8000,
            prepared=None) -> dict:
    """One ordered pass over `text`; returns per-POSITION tensors.

    region:  "first" | "last"   (which region's per-pass outputs to keep)
    passes:  "first" | "last" | "all"
    need:    subset of ("log_pi", "weights", "gamma", "log_b", "final",
                       "omega"); "final" = the model's final log-probs,
             "omega" = cross-region weights [P, R]. "nll" (per-position NLL
             of the FINAL output), "xs", "ys", "expand_len" are always
             included. "final"/"omega" are region-independent.

    Windows are non-overlapping (stride = context) walked in order, and
    row-major flattening matches evaluate_bpc's sequential accounting, so
    `nll.sum() / (ln2 * expand_len.sum())` IS the headline bpc restricted to
    these positions. If `max_chars` cuts mid-window, every tensor is trimmed
    to the counted prefix, keeping all outputs aligned.
    """
    device = next(model.parameters()).device
    ids, expand_len = prepared if prepared is not None else prepare_text(ubm, text)
    if not ids:
        raise ValueError("cannot collect from empty text")
    if not hasattr(model, "regions"):
        raise ValueError("collect() supports IECNN models only")
    context = seq_len or getattr(model, "context_length", 256) or 256
    if len(ids) < 2:
        raise ValueError("collect() needs at least 2 tokens")
    sel = {"first": [0], "last": None}.get(passes, "all")
    ridx = 0 if region == "first" else -1

    pp_store: dict[int, dict[str, list]] = {}
    final_a: list = []
    omega_a: list = []
    xs_a: list = []
    ys_a: list = []
    nll_a: list = []
    el_a: list = []
    n_counted = 0
    chars = 0
    stop = False
    starts = list(range(0, max(len(ids) - 1, 1), context))
    for group in _equal_length_batches(starts, ids, context, EVAL_BATCH):
        L = min(context, len(ids) - group[0] - 1)
        x = torch.tensor([ids[s:s + L] for s in group],
                         dtype=torch.long, device=device)
        y = torch.tensor([ids[s + 1:s + 1 + L] for s in group],
                         dtype=torch.long, device=device)
        out = model(x, tau0=tau0, floor=floor)
        ntok = x.shape[0] * L
        xs_a.append(x.reshape(-1).cpu())
        ys_a.append(y.reshape(-1).cpu())
        nll_a.append((-out["final_log_probs"].gather(2, y.unsqueeze(2))
                      .squeeze(2)).reshape(-1).cpu())
        if "final" in need:
            final_a.append(out["final_log_probs"].reshape(ntok, -1).cpu())
        if "omega" in need and out.get("region_weights") is not None:
            omega_a.append(out["region_weights"].reshape(ntok, -1).cpu())
        pp = out["region_outputs"][ridx]["per_pass"]
        if isinstance(sel, list):
            pidx = sel
        elif sel == "all":
            pidx = list(range(len(pp)))
        else:
            pidx = [len(pp) - 1]
        for pi_ in pidx:
            st = pp_store.setdefault(pi_, {k: [] for k in need
                                           if k not in ("final", "omega")})
            for k in need:
                if k in ("final", "omega"):
                    continue
                t = pp[pi_][k]                       # e.g. log_pi [K, T, N, V]
                st[k].append(t.reshape(ntok, *t.shape[2:]).cpu())
        for s in group:
            for i in range(L):
                chars += expand_len[s + 1 + i]
                el_a.append(expand_len[s + 1 + i])
                n_counted += 1
                if max_chars is not None and chars >= max_chars:
                    stop = True
                    break
            if stop:
                break
        if stop:
            break

    res = {
        "xs": torch.cat(xs_a)[:n_counted],
        "ys": torch.cat(ys_a)[:n_counted],
        "nll": torch.cat(nll_a)[:n_counted],
        "expand_len": torch.tensor(el_a, dtype=torch.float),
        "n_positions": n_counted,
        "n_chars": chars,
        "region": region,
        # pass_index: numeric even for passes="last" — ablation/calibration/
        # error_correlation/win_map index per_pass[pass_index]; None crashed
        # them with KeyError None (8th-try CI diagnosis, 2026-11-15)
        "pass_index": pidx[0] if isinstance(sel, list) else len(pp) - 1,
    }
    if "final" in need:
        res["final"] = torch.cat(final_a)[:n_counted]
    if "omega" in need and omega_a:
        res["omega"] = torch.cat(omega_a)[:n_counted]
    if pp_store:
        res["per_pass"] = {
            pi_: {k: torch.cat(v, 0)[:n_counted] for k, v in st.items()}
            for pi_, st in sorted(pp_store.items())}
    return res


# --------------------------------------------------------------------------
# shared math + io
# --------------------------------------------------------------------------
def gini(values) -> float:
    """Gini coefficient of a damage/importance vector (0 = all equal)."""
    v = sorted(float(x) for x in values)
    n = len(v)
    tot = sum(v)
    if n == 0 or tot <= 0:
        return 0.0
    cum = sum((i + 1) * x for i, x in enumerate(v))
    return (2.0 * cum - (n + 1) * tot) / (n * tot)


def auroc(scores, positives) -> float | None:
    """Mann-Whitney AUROC (average ranks for ties). None when one class is
    empty — callers must report that instead of a fake 0.5."""
    pos = [float(s) for s, l in zip(scores, positives) if l]
    neg = [float(s) for s, l in zip(scores, positives) if not l]
    if not pos or not neg:
        return None
    allv = pos + neg
    order = sorted(range(len(allv)), key=lambda i: allv[i])
    ranks = [0.0] * len(allv)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and allv[order[j + 1]] == allv[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    rp = sum(ranks[:len(pos)])
    u = rp - len(pos) * (len(pos) + 1) / 2.0
    return u / (len(pos) * len(neg))


def spearman(a, b) -> float | None:
    """Spearman rank correlation of two same-length sequences."""
    if len(a) != len(b) or len(a) < 2:
        return None
    ra = _ranks(a)
    rb = _ranks(b)
    n = len(a)
    ma = sum(ra) / n
    mb = sum(rb) / n
    cov = sum((x - ma) * (y - mb) for x, y in zip(ra, rb))
    va = sum((x - ma) ** 2 for x in ra) ** 0.5
    vb = sum((y - mb) ** 2 for y in rb) ** 0.5
    if va == 0 or vb == 0:
        return None
    return cov / (va * vb)


def _ranks(values) -> list:
    order = sorted(range(len(values)), key=lambda i: values[i])
    ranks = [0.0] * len(values)
    i = 0
    while i < len(order):
        j = i
        while j + 1 < len(order) and values[order[j + 1]] == values[order[i]]:
            j += 1
        avg = (i + j) / 2 + 1
        for t in range(i, j + 1):
            ranks[order[t]] = avg
        i = j + 1
    return ranks


def token_types(ubm, xs) -> list[str]:
    """Coarse per-position token classes for the by-type diagnostics
    (spec 03 §8 / 08 §6.4): whitespace, digit, punctuation, capitalized,
    long/word/short."""
    types = []
    for t in xs.tolist():
        s = ubm.decode([int(t)])
        st = s.strip()
        if not st:
            types.append("whitespace")
        elif st[0].isdigit():
            types.append("digit")
        elif not st[0].isalnum():
            types.append("punct")
        elif any(c.isupper() for c in st):
            types.append("capitalized")
        elif len(st) >= 8:
            types.append("long_word")
        elif len(st) >= 4:
            types.append("word")
        else:
            types.append("short_word")
    return types


def default_text(max_chars: int | None = None) -> str:
    t = Path("data/val.txt").read_text(encoding="utf-8")
    return t if max_chars is None else t[:max_chars]


def save_json(out_dir, name: str, payload: dict) -> Path:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{name}.json"
    path.write_text(json.dumps(payload, indent=1, default=str))
    return path


def nats_to_bpc(nats: float, chars: float) -> float:
    return nats / (math.log(2) * chars) if chars > 0 else float("nan")
