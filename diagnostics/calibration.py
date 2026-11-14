"""Diagnostic 2 — dot confidence calibration (spec 06 §2.5 / 08 §6.3).

Compares each dot's confidence sigmoid(gamma) with whether its argmax was
right. Healthy sign: reliability curve near the diagonal; ECE near 0.
"""
from __future__ import annotations

import torch

from .common import collect, default_text, load_run, save_json

N_BUCKETS = 10


def compute(coll: dict) -> dict:
    pi_ = coll["pass_index"]
    gamma = coll["per_pass"][pi_]["gamma"].float()       # [M, N]
    log_pi = coll["per_pass"][pi_]["log_pi"].float()
    y = coll["ys"].long()
    correct = (log_pi.argmax(-1) == y[:, None]).float()  # [M, N]
    conf = torch.sigmoid(gamma)

    flat_c = conf.reshape(-1)
    flat_a = correct.reshape(-1)
    buckets = []
    ece = 0.0
    edges = [i / N_BUCKETS for i in range(N_BUCKETS + 1)]
    for b in range(N_BUCKETS):
        lo, hi = edges[b], edges[b + 1]
        sel = (flat_c >= lo) & (flat_c < hi) if b < N_BUCKETS - 1 \
            else (flat_c >= lo) & (flat_c <= hi)
        cnt = int(sel.sum())
        if cnt:
            mc = float(flat_c[sel].mean())
            ma = float(flat_a[sel].mean())
        else:
            mc = ma = 0.0
        ece += cnt / flat_c.numel() * abs(ma - mc)
        buckets.append({"range": [lo, hi], "count": cnt,
                        "mean_confidence": round(mc, 5),
                        "accuracy": round(ma, 5)})
    n_dots = gamma.shape[1]
    return {
        "ece": round(ece, 5),
        "reliability": buckets,
        "per_dot": [{"dot": k,
                     "mean_confidence": round(float(conf[:, k].mean()), 5),
                     "accuracy": round(float(correct[:, k].mean()), 5)}
                    for k in range(n_dots)],
        "n_positions": coll["n_positions"],
        "note": "confidence = sigmoid(gamma) at the last pass of the last "
                "region; correct = dot's argmax equals the true next token",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    coll = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last",
                   need=("log_pi", "gamma"), seq_len=seq_len,
                   tau0=r["tau0"], max_chars=max_chars, prepared=prepared)
    summary = compute(coll)
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "calibration", summary)
    return summary
