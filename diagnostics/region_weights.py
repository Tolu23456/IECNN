"""Diagnostic 7 — region weights by token type (spec 03 §8 / 08 §6.4).

Mean cross-region weight omega per coarse token class. Healthy sign: early
regions up-weighted on spelling-hard tokens (capitalized, long words), later
regions on common text.
"""
from __future__ import annotations

from .common import collect, default_text, load_run, save_json, token_types


def compute(coll: dict, ubm) -> dict:
    omega = coll["omega"].float()                          # [M, R]
    types = token_types(ubm, coll["xs"])
    n_regions = omega.shape[1]
    by_type: dict[str, list] = {}
    counts: dict[str, int] = {}
    for t, om in zip(types, omega.tolist()):
        by_type.setdefault(t, [0.0] * n_regions)
        counts[t] = counts.get(t, 0) + 1
        for r in range(n_regions):
            by_type[t][r] += om[r]
    table = {t: [round(v / counts[t], 5) for v in vals]
             for t, vals in sorted(by_type.items())}
    argmax_hist: dict[str, dict] = {}
    am = omega.argmax(-1).tolist()
    for t, a in zip(types, am):
        argmax_hist.setdefault(t, {})
        argmax_hist[t][str(a)] = argmax_hist[t].get(str(a), 0) + 1
    return {
        "mean_omega_by_token_type": table,
        "counts_by_token_type": counts,
        "dominant_region_by_token_type":
            {t: max(h, key=h.get) for t, h in argmax_hist.items()},
        "n_regions": n_regions,
        "n_positions": coll["n_positions"],
        "note": "omega = cross-region weights (D5); row order = region 1..R",
    }


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=8000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    coll = collect(r["model"], r["ubm"], text or default_text(),
                   region="last", passes="last", need=("omega",),
                   seq_len=seq_len, tau0=r["tau0"], max_chars=max_chars,
                   prepared=prepared)
    summary = compute(coll, r["ubm"])
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    save_json(out_dir, "region_weights", summary)
    return summary
