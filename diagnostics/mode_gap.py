"""Diagnostic 6 — mode gap (spec 04 §5 / 08 §6.5). Delegates to the real
implementation in train.evaluate (already reported by every eval); kept as a
module so diagnose-all covers the whole §6 list in one run."""
from __future__ import annotations

from train.evaluate import mode_gap

from .common import default_text, load_run, save_json


def run(checkpoint_path, config_path=None, out_dir="results/diagnostics",
        *, vocab=None, text=None, max_chars=20000, seq_len=None,
        prepared=None) -> dict:
    r = load_run(checkpoint_path, config_path, vocab)
    txt = text or default_text()
    if text is None and max_chars:
        txt = txt[:max_chars]
    mg = mode_gap(r["model"], r["ubm"], txt)
    summary = dict(mg)
    summary["ckpt_dir"] = r["ckpt_dir"]
    summary["best_step"] = r["best_step"]
    summary["n_chars"] = len(txt)
    summary["note"] = "hard top-m bpc minus soft bpc on the same text"
    save_json(out_dir, "mode_gap", summary)
    return summary
