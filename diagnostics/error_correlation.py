"""Diagnostic 4 — pairwise error correlation (spec 06 §2.4 / 08 §6.1).

Weighted correlation of dot errors e_k(t) = -log pi_k(y_t), weighted by how
wrong the consensus was (nu_t). Healthy sign: falls during training, stays
low; a rise when messages are on means the channel is encouraging copying.
"""
from __future__ import annotations

import torch

from .common import collect, default_text, load_run, save_json


def compute(coll: dict) -> dict:
    pi_ = coll["pass_index"]
    log_pi = coll["per_pass"][pi_]["log_pi"].float()      # [M, N, V]
    log_b = coll["per_pass"][pi_]["log_b"].float()        # [M, V]
    y = coll["ys"].long()
    m, n_dots = log_pi.shape[0], log_pi.shape[1]

    e = -log_pi.gather(2, y.view(m, 1, 1).expand(m, n_dots, 1)).squeeze(2)  # [M, N]
    nu = (1.0 - log_b.exp().gather(1, y[:, None]).squeeze(1)).clamp(0.0, 1.0)
    wsum = nu.sum().clamp_min(1e-9)

    mu = (nu[:, None] * e).sum(0) / wsum                  # [N]
    centered = (e - mu) * nu[:, None]
    cov = centered.t().matmul(e - mu) / wsum              # [N, N]
    var = cov.diagonal().clamp_min(1e-12)
    rho = cov / (var.unsqueeze(0) * var.unsqueeze(1)).sqrt().add(1e-12)
    off = rho[~torch.eye(n_dots, dtype=torch.bool)]
    pos = off[off > 0]
    return {
        "mean_pairwise_error_correlation": round(float(off.mean()), 5),
        "mean_positive_pairwise_correlation":
            round(float(pos.mean()) if pos.numel() else 0.0, 5),
        "max_pairwise_correlation": round(float(off.max()), 5),
        "matrix": [[round(float(x), 5) for x in row] for row in rho],
        "n_positions": coll["n_positions"],
        "note": "rho_ij weighted by consensus-wrongness (spec 06 2.4); "
                "high positive off-diagonals = dots fail together (collapse)",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    coll = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last",
                   need=("log_pi", "log_b"), seq_len=seq_len,
                   tau0=r["tau0"], max_chars=max_chars, prepared=prepared)
    summary = compute(coll)
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "error_correlation", summary)
    return summary
