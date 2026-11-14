"""Diagnostic 5 — message influence (spec 05 §6 / 08 E3 criterion).

KL between each position's FINAL output with messages on vs with every
message zeroed (channels replaced by zero tensors — the receive path still
runs, but carries no information). Plus the bpc delta. Healthy sign: clearly
above zero; a rise in error correlation with messages on is the copying warn.
"""
from __future__ import annotations

import math
import torch

from .common import collect, default_text, load_run, save_json


class _ZeroMessages(torch.nn.Module):
    """Drop-in replacement that writes all-zero messages."""

    def __init__(self, n_dots: int, d_m: int):
        super().__init__()
        self.n_dots = n_dots
        self.d_m = d_m

    def forward(self, states, training=None):
        return states.new_zeros(states.shape[0], states.shape[1],
                                self.n_dots, self.d_m)


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    regions = [reg for reg in r["model"].regions
               if getattr(reg, "messages", None) is not None]
    if not regions:
        summary = {"skipped": True,
                   "reason": "use_messages is off in this run's config"}
        save_json(out_dir, "message_influence", summary)
        return summary

    kwargs = dict(seq_len=seq_len, tau0=r["tau0"], max_chars=max_chars,
                  prepared=prepared)
    base = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last", need=("final",), **kwargs)
    # forward again with every channel zeroed
    originals = [reg.messages for reg in regions]
    zero = _ZeroMessages(int(regions[0].dots.N), int(originals[0].d_m))
    try:
        for reg in regions:
            reg.messages = zero
        zed = collect(r["model"], r["ubm"], text or default_text(),
                      region="last", passes="last", need=("final",), **kwargs)
    finally:
        for reg, orig in zip(regions, originals):
            reg.messages = orig

    n = min(base["n_positions"], zed["n_positions"])
    lp0 = base["final"][:n].float()
    lp1 = zed["final"][:n].float()
    p0 = lp0.exp().clamp_min(1e-12)
    kl = (p0 * (lp0 - lp1)).sum(-1)                       # [positions]
    chars0 = float(base["expand_len"][:n].sum())
    chars1 = float(zed["expand_len"][:n].sum())
    bpc0 = float(base["nll"][:n].sum()) / (math.log(2) * chars0)
    bpc1 = float(zed["nll"][:n].sum()) / (math.log(2) * chars1)
    summary = {
        "mean_kl_messages_on_vs_zeroed": round(float(kl.mean()), 6),
        "max_kl": round(float(kl.max()), 6),
        "fraction_positions_kl_above_1e_3":
            round(float((kl > 1e-3).float().mean()), 5),
        "bpc_with_messages": round(bpc0, 5),
        "bpc_messages_zeroed": round(bpc1, 5),
        "bpc_delta": round(bpc0 - bpc1, 5),
        "n_positions": n,
        "note": "influence = KL(final || final with all messages zeroed); "
                "clearly above zero earns the channel its cost (rung E3)",
    }
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "message_influence", summary)
    return summary
