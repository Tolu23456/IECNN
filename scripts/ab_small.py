"""Arm registry: equal-wall-clock A/B of IECNN variants vs transformers.

Every arm gets the same data bytes, batch, context, optimizer and wall-clock
budget on the same 2-vCPU box. Reports params, steps, tokens seen, s/step,
val bpc at each mark, val bpc when the arm finishes, and bpc interpolated at
the token count EVERY arm reached (equal-token comparison), plus a pointer
diversity audit for IECNN arms.

Run:  PYTHONPATH=. python3 -u scripts/ab_small.py [budget_s] [arm,arm,...]
"""
from __future__ import annotations

import json
import math
import sys
import time

import torch
import torch.nn.functional as F

from models.baselines import build_transformer_baseline
from models.iecnn import build_iecnn
from models.memory import region1_rows

torch.set_num_threads(2)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
B, T = 8, 64
V = 256                      # byte-level toy vocab
MARK_S = 30.0


# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------
def iecnn(cell="gru", lean=False, n_passes=3, n_dots=8, d_s=128, d=128,
          spans=(16,), read_mode="soft", head_mode="hidden",
          per_dot_logits=True, window_mode="contiguous", window_reach=None,
          seclusion_topk=0):
    m = {
        "kind": "iecnn", "d": d, "d_q": 16, "d_v": 32, "d_s": d_s,
        "n_dots": n_dots, "n_regions": 1, "n_passes": n_passes,
        "spans": list(spans), "use_weights": True, "use_messages": False,
        "pooling": "log_linear", "head_mode": head_mode,
        "read_mode": read_mode, "share_pass_params": True,
        "dynamic_query": False, "use_cross_region": False,
        "fast_dot_math": True, "cache_attention": True,
        "cell": cell, "head_last_only": lean,
        "per_dot_logits": per_dot_logits,
        "window_mode": window_mode,
        "seclusion_topk": seclusion_topk,
    }
    if window_reach is not None:
        m["window_reach"] = window_reach
    return {"model": m,
            "consensus": {"rho": 0.0, "kappa": 0.7},
            "training": {"compile_consensus": False, "dot_dropout": 0.0},
            "data": {"seq_len": T}}


def tf(d_model=96, n_heads=4, n_layers=3):
    return {"model": {"kind": "transformer", "d_model": d_model,
                      "n_heads": n_heads, "n_layers": n_layers},
            "data": {"seq_len": T}}


ARMS = {
    # the baseline family we must beat on BOTH bpc and speed
    "tf-d96-l3":  ("transformer", lambda: tf(96, 4, 3)),
    "tf-d128-l4": ("transformer", lambda: tf(128, 4, 4)),
    # current champion lineage (GRU dot + per-pass shared head + per-dot logits)
    "iecnn-gru":  ("iecnn", lambda: iecnn("gru", False)),
    # lean cell only
    "iecnn-lug":  ("iecnn", lambda: iecnn("lug", False)),
    # last-pass-only head only
    "iecnn-lean": ("iecnn", lambda: iecnn("gru", True)),
    # no per-dot vocab logits (nothing in the toy training consumes them)
    "iecnn-nologit": ("iecnn", lambda: iecnn("gru", False, per_dot_logits=False)),
    # the full lean package: LUG + last-pass head/e_state + no per-dot logits
    "iecnn-lite": ("iecnn", lambda: iecnn("lug", True, per_dot_logits=False)),
    # smaller committee / fewer passes / smaller state
    "iecnn-lite-p2": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                             per_dot_logits=False)),
    "iecnn-lite-n4": ("iecnn", lambda: iecnn("lug", True, n_dots=4,
                                             per_dot_logits=False)),
    "iecnn-lite-ds64": ("iecnn", lambda: iecnn("lug", True, d_s=64,
                                               per_dot_logits=False)),
    "iecnn-lite-p2n4": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                               n_dots=4, per_dot_logits=False)),
    "iecnn-lite-fast": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                               n_dots=4, d_s=64, d=96,
                                               per_dot_logits=False)),
    # seclusion on top of lite
    "iecnn-lite-sec": ("iecnn", lambda: iecnn("lug", True, per_dot_logits=False,
                                              read_mode="secluded")),
    # seclusion v2: hard read + top-k straight-through gradient
    "iecnn-lite-sec-k4": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, read_mode="secluded",
        seclusion_topk=4)),
    # long-range time axis at ZERO extra compute: same 16 slots spread
    # quadratically from 0 to window_reach (toy context is 64, so 64 = full)
    "iecnn-lite-q64": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, window_mode="quad",
        window_reach=64)),
    "iecnn-lite-q32": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, window_mode="quad",
        window_reach=32)),
    "iecnn-lite-p2n4-q256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, per_dot_logits=False,
        window_mode="quad", window_reach=256)),
    # --- param-matched with tf-d96-l3 (~390k): d_s = 64 -------------------
    "iecnn-gru-ds64": ("iecnn", lambda: iecnn("gru", False, d_s=64)),
    "iecnn-lite-ds64": ("iecnn", lambda: iecnn("lug", True, d_s=64,
                                               per_dot_logits=False)),
    "iecnn-lite-ds64-q64": ("iecnn", lambda: iecnn(
        "lug", True, d_s=64, per_dot_logits=False, window_mode="quad",
        window_reach=64)),
    "iecnn-lite-ds64-q64-sec": ("iecnn", lambda: iecnn(
        "lug", True, d_s=64, per_dot_logits=False, window_mode="quad",
        window_reach=64, read_mode="secluded", seclusion_topk=4)),
    "iecnn-lite-p2n4-ds64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False)),
    "iecnn-lite-p2n4-ds64-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="quad", window_reach=64)),
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(path: str, n: int) -> torch.Tensor:
    raw = open(path, "rb").read(n)
    return torch.frombuffer(raw, dtype=torch.uint8).long()


train = load("data/toy_train.txt", 4_000_000)
val = load("data/toy_val.txt", 60_000)
print(f"tokens train={train.numel()} val={val.numel()} | B={B} T={T} V={V} "
      f"| budget {BUDGET:.0f}s/arm", flush=True)


def batches(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    while True:
        i = torch.randint(0, train.numel() - T - 1, (B,), generator=g)
        x = torch.stack([train[j:j + T] for j in i])
        y = torch.stack([train[j + 1:j + 1 + T] for j in i])
        yield x, y


@torch.no_grad()
def val_bpc(model, windows: int = 48) -> float:
    model.eval()
    tot, n = 0.0, 0
    for w in range(windows):
        i = w * T
        x = val[i:i + T].unsqueeze(0)
        y = val[i + 1:i + 1 + T].unsqueeze(0)
        if x.shape[1] < T:
            break
        lp = model(x)["final_log_probs"]
        tot += F.nll_loss(lp.reshape(-1, V), y.reshape(-1),
                          reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n / math.log(2)


@torch.no_grad()
def pointer_audit(model) -> dict:
    """How the N dots pick tokens: distinct picks + distance distribution."""
    r = model.regions[0]
    g = torch.Generator().manual_seed(7)
    i = torch.randint(0, train.numel() - T - 1, (8,), generator=g)
    x = torch.stack([train[j:j + T] for j in i])
    rows = region1_rows(model.embed(x))
    wind_k, _wv, valid = r._window_memory(rows, x.device)
    q = r.dots._sel(0, r.dots.q)
    scores = torch.einsum("bfwj,nj->bfnw", wind_k, q) / math.sqrt(r.d_q)
    scores = scores.masked_fill(~valid[None, :, None, :], float("-inf"))
    idx = scores.argmax(-1)
    distinct = torch.tensor([len(set(row.tolist()))
                             for row in idx.reshape(-1, idx.shape[-1])])
    w = wind_k.shape[2]
    dist = (w - 1 - idx).float()
    return {
        "distinct_per_pos": round(distinct.float().mean().item(), 2),
        "all_same": round((distinct == 1).float().mean().item(), 3),
        "mean_dist_back": round(dist.float().mean().item(), 2),
        "frac_window_end": round((idx == (w - 1)).float().mean().item(), 3),
    }


def run(name: str, kind: str, model) -> dict:
    torch.manual_seed(0)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    n_par = sum(p.numel() for p in model.parameters())
    gen = batches(0)
    t0 = time.time()
    step = 0
    marks = []
    next_mark = MARK_S
    while time.time() - t0 < BUDGET:
        x, y = next(gen)
        lr = 1e-3 * min(1.0, (step + 1) / 100.0)
        for grp in opt.param_groups:
            grp["lr"] = lr
        out = model(x)
        loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        el = time.time() - t0
        if el >= next_mark:
            marks.append([int(step * B * T), round(val_bpc(model), 4), step])
            next_mark += MARK_S
    el = time.time() - t0
    final = round(val_bpc(model), 4)
    row = {"arm": name, "kind": kind, "params": n_par, "steps": step,
           "tokens": step * B * T, "s_per_step": round(el / max(step, 1), 4),
           "s_per_1k_tokens": round(el / max(step * B * T / 1000, 1e-9), 3),
           "curve": marks, "bpc_final": final}
    if kind == "iecnn":
        row["pointers"] = pointer_audit(model)
    print("RESULT " + json.dumps(row), flush=True)
    return row


def bpc_at(marks, tokens: int):
    """Linear interpolation of the (tokens, bpc) curve at a token count."""
    pts = [(m[0], m[1]) for m in marks]
    if not pts or tokens < pts[0][0]:
        return None
    if tokens >= pts[-1][0]:
        return pts[-1][1]
    for (t0_, b0), (t1_, b1) in zip(pts, pts[1:]):
        if t0_ <= tokens <= t1_:
            f = (tokens - t0_) / max(t1_ - t0_, 1)
            return round(b0 + f * (b1 - b0), 4)
    return pts[-1][1]


def main():
    names = ONLY or list(ARMS)
    rows = []
    for name in names:
        kind, mk = ARMS[name]
        torch.manual_seed(0)
        cfg = mk()
        model = (build_transformer_baseline(cfg, V) if kind == "transformer"
                 else build_iecnn(cfg, V))
        print(f"--- {name} ({sum(p.numel() for p in model.parameters())} params)",
              flush=True)
        rows.append(run(name, kind, model))

    common_tok = min(r["tokens"] for r in rows)
    print(f"\n=== SUMMARY (equal wall-clock {BUDGET:.0f}s/arm, "
          f"common {common_tok/1e6:.2f}M tokens) ===", flush=True)
    hdr = f"{'arm':<18}{'params':>8}{'steps':>7}{'tokens':>9}{'s/step':>9}" \
          f"{'s/1k tok':>10}{'bpc_fin':>9}{'bpc@eqtok':>11}"
    print(hdr, flush=True)
    for r in rows:
        be = bpc_at(r["curve"], common_tok)
        print(f"{r['arm']:<18}{r['params']:>8}{r['steps']:>7}"
              f"{r['tokens']/1e6:>8.2f}M{r['s_per_step']:>9.4f}"
              f"{r['s_per_1k_tokens']:>10.2f}{r['bpc_final']:>9.4f}"
              f"{('%.4f' % be) if be is not None else '   -':>11}", flush=True)
        if "pointers" in r:
            print(f"{'':<18}pointers {r['pointers']}", flush=True)
    print("\nJSON " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
