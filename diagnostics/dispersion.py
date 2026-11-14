"""Diagnostic 3 — does disagreement predict hard tokens? (spec 08 §6.3)

The cheapest first experiment on a trained model (needs no training changes):
dispersion of region-1 dot distributions at pass 1 vs (a) whether the FINAL
output was wrong (AUROC), (b) per-token loss (Spearman), plus a halting
simulation table (what would stopping region 1 after pass 1 on easy tokens
cost/ save).
"""
from __future__ import annotations

import math

import torch

from models import consensus

from .common import (auroc, collect, default_text, load_run, save_json,
                     spearman)


def _js_mean(probs: torch.Tensor) -> torch.Tensor:
    """Mean JS(pi_k, mean_j pi_j) over dots (natural log, bounded by ln 2).

    probs [M, N, V] -> [M]
    """
    m = probs.mean(1)                                     # [M, V]
    pm = 0.5 * (probs + m.unsqueeze(1)).clamp_min(1e-12)   # [M, N, V]
    kl_p = (probs.clamp_min(1e-12) *
            (probs.clamp_min(1e-12).log() - pm.log())).sum(-1)
    kl_m = (m.unsqueeze(1).clamp_min(1e-12) *
            (m.unsqueeze(1).clamp_min(1e-12).log() - pm.log())).sum(-1)
    return (0.5 * (kl_p + kl_m)).mean(-1)


def compute(c_first: dict, c_all: dict) -> dict:
    pi_ = c_first["pass_index"]
    log_pi = c_first["per_pass"][pi_]["log_pi"].float()   # [M, N, V]
    w = c_first["per_pass"][pi_]["weights"].float()
    y = c_first["ys"].long()
    n = min(c_first["n_positions"], c_all["n_positions"])
    log_pi, w, y = log_pi[:n], w[:n], y[:n]
    probs = log_pi.exp()

    disp = _js_mean(probs)                                # [M]
    b1 = consensus.pool_log_linear(log_pi, w)
    wrong1 = (b1.argmax(-1) != y).tolist()
    final_nll = c_first["nll"][:n]

    aroc = auroc(disp.tolist(), [bool(x) for x in wrong1])
    sp = spearman(disp.tolist(), final_nll.tolist())

    # halting simulation within region 1: stop after pass 1 where dispersion
    # is low; those tokens lose (passLast - pass1) NLL, everything else keeps
    # the region's final pass.
    pp_all = c_all["per_pass"]
    last_idx = max(pp_all)
    log_b_last = pp_all[last_idx]["log_b"].float()[:n]     # [M, V]
    nll1 = (-b1.gather(1, y[:, None]).squeeze(1))
    nllL = (-log_b_last.gather(1, y[:, None]).squeeze(1))
    d_list = disp.tolist()
    n1 = nll1.tolist()
    nL = nllL.tolist()
    order = sorted(d_list)
    n_passes = last_idx + 1
    halting = []
    for pct in (10, 25, 50, 75):
        cut = order[max(0, int(len(order) * pct / 100) - 1)]
        H = [d <= cut for d in d_list]
        nH = sum(H)
        if nH == 0 or nH == n:
            continue
        loss_halt = sum(nn for nn, h in zip(n1, H) if h) / nH
        loss_would = sum(nn for nn, h in zip(nL, H) if h) / nH
        loss_other = sum(nn for nn, h in zip(nL, H) if not h) / (n - nH)
        blended = (loss_halt * nH + loss_other * (n - nH)) / n
        halting.append({
            "stop_below_percentile": pct,
            "dispersion_threshold": round(cut, 5),
            "fraction_halted": round(nH / n, 4),
            "nll_halted_pass1": round(loss_halt, 5),
            "nll_halted_passlast": round(loss_would, 5),
            "nll_blended": round(blended, 5),
            "region1_passes_saved": round(nH / n * (n_passes - 1) / n_passes, 4),
        })
    return {
        "dispersion_mean": round(float(disp.mean()), 5),
        "dispersion_std": round(float(disp.std()), 5),
        "auroc_dispersion_vs_final_wrong": (round(aroc, 5) if aroc is not None else None),
        "spearman_dispersion_vs_final_nll": (round(sp, 5) if sp is not None else None),
        "fraction_consensus1_wrong": round(sum(wrong1) / max(n, 1), 5),
        "halting_simulation": halting,
        "n_positions": n,
        "note": "dispersion = mean JS(pi_k, crowd mean) over region-1 dots at "
                "pass 1; AUROC predicts FINAL-output errors (healthy > 0.7)",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    txt = text or default_text()
    c_first = collect(r["model"], r["ubm"], txt, region="first",
                      passes="first", need=("log_pi", "weights"),
                      seq_len=seq_len, tau0=r["tau0"], max_chars=max_chars,
                      prepared=prepared)
    c_all = collect(r["model"], r["ubm"], txt, region="first", passes="all",
                    need=("log_b",), seq_len=seq_len, tau0=r["tau0"],
                    max_chars=max_chars, prepared=prepared)
    summary = compute(c_first, c_all)
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "dispersion", summary)
    return summary
