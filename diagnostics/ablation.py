"""Diagnostic 1 — per-dot ablation damage + Gini spread (spec 08 §6.1).

Mask one dot at a time: zero its consensus weight (renormalized — exactly what
pruning to zero means under C4/D3) and recompute the FINAL region consensus of
the last pass. damage_k = mean per-position NLL increase in nats. The healthy
sign is a WIDE spread: some dots matter a lot, some little.
"""
from __future__ import annotations

import math

import torch

from models import consensus

from .common import (collect, default_text, gini, load_run, nats_to_bpc,
                     save_json)


def compute(coll: dict) -> dict:
    """Ablation damage from one collect(region='last', passes='last')."""
    pi_ = coll["pass_index"]
    log_pi = coll["per_pass"][pi_]["log_pi"].float()     # [M, N, V]
    w = coll["per_pass"][pi_]["weights"].float()         # [M, N]
    y = coll["ys"].long()
    n_dots = log_pi.shape[1]

    base = consensus.pool_log_linear(log_pi, w)          # [M, V]
    nll_base = -base.gather(1, y[:, None]).squeeze(1)
    damages = []
    for k in range(n_dots):
        w2 = w.clone()
        w2[:, k] = 0.0
        w2 = w2 / w2.sum(-1, keepdim=True).clamp_min(1e-9)
        nll_k = -consensus.pool_log_linear(log_pi, w2).gather(1, y[:, None]).squeeze(1)
        damages.append(float((nll_k - nll_base).mean()))
    mean_chars_per_token = (coll["n_chars"] / coll["n_positions"]
                            if coll["n_positions"] else 0.0)
    order = sorted(range(n_dots), key=lambda k: -damages[k])
    return {
        "n_dots": n_dots,
        "damage_nats_per_token": damages,
        "damage_bpc_equivalent": [nats_to_bpc(d, mean_chars_per_token)
                                  for d in damages],
        "gini": gini(damages),
        "most_important_dot": order[0] if order else None,
        "least_important_dot": order[-1] if order else None,
        "n_positions": coll["n_positions"],
        "n_chars": coll["n_chars"],
        "note": "mask = zero the dot's consensus weight in the final pool "
                "(renormalized); damage in nats/position",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    coll = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last",
                   need=("log_pi", "weights"), seq_len=seq_len,
                   tau0=r["tau0"], max_chars=max_chars, prepared=prepared)
    summary = compute(coll)
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "ablation", summary)
    return summary
