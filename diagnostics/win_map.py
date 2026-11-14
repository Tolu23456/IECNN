"""Diagnostic 8 — win map (spec 08 §6.1): which dot wins each position.

winner = argmax of the last pass's consensus weights in the last region,
grouped by coarse token class. Healthy sign: different dots win on different
token types (not one dot winning everywhere).
"""
from __future__ import annotations

from .common import collect, default_text, load_run, save_json, token_types


def compute(coll: dict, ubm) -> dict:
    pi_ = coll["pass_index"]
    w = coll["per_pass"][pi_]["weights"].float()           # [M, N]
    types = token_types(ubm, coll["xs"])
    n_dots = w.shape[1]
    hist: dict[str, dict] = {}
    for t, row in zip(types, w.tolist()):
        winner = row.index(max(row))
        h = hist.setdefault(t, {"counts": [0] * n_dots})
        h["counts"][winner] += 1
    table = {}
    for t, h in sorted(hist.items()):
        total = sum(h["counts"])
        table[t] = {
            "n": total,
            "winner_share": [round(c / total, 4) for c in h["counts"]],
            "dominant_dot": h["counts"].index(max(h["counts"])),
        }
    return {
        "win_map": table,
        "n_dots": n_dots,
        "n_positions": coll["n_positions"],
        "note": "winner = argmax final-pass weight in the LAST region; "
                "one dominant dot across all types = no specialization",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    coll = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last", need=("weights",),
                   seq_len=seq_len, tau0=r["tau0"], max_chars=max_chars,
                   prepared=prepared)
    summary = compute(coll, r["ubm"])
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "win_map", summary)
    return summary
