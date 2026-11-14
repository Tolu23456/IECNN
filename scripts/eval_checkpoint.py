"""Re-score a trained checkpoint's BEST point on FULL validation (runner job).

Loads checkpoints/<dir>/checkpoint.pt (best_state preferred), rebuilds the
model from config_used.json, and evaluates:
  - full val bpc at the tau annealed to the BEST step (the pilot runs showed
    re-scoring at the final tau can read ~1.2 bpc worse — pure eval artifact),
  - per-pass bpc + weight entropy (evaluate_bpc_per_pass),
  - hard-mode bpc + mode gap at the same tau,
  - train-slice bpc (fit).
Writes JSON to --out. No training happens here; inference only.

Also exercises the vocab fingerprint guard: if the checkpoint records
vocab_sha256 and the provided vocab file hashes differently, this FAILS
LOUDLY (Tolu, 2026-11-13, item 2).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UBM import UBM                                   # noqa: E402
from UBM.tokenizer import fingerprint                 # noqa: E402
from train.train import build_model                   # noqa: E402
from train.schedules import temperature_at            # noqa: E402
from train.evaluate import (evaluate_bpc, evaluate_bpc_per_pass,   # noqa: E402
                            prepare_text)
from utils import load_run_meta                       # noqa: E402
import torch                                          # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True,
                    help="dir containing checkpoint.pt + summary.json + config_used.json")
    ap.add_argument("--vocab", default="UBM/vocab_v1.json")
    ap.add_argument("--data", default="data/val.txt")
    ap.add_argument("--train-slice-chars", type=int, default=100_000)
    ap.add_argument("--out", required=True)
    ap.add_argument("--config-fallback", default="",
                    help="repo config yaml to rebuild config_used.json from "
                         "when the artifact lacks it (timeout-killed chunk)")
    args = ap.parse_args()

    d = Path(args.ckpt_dir)
    ck = torch.load(d / "checkpoint.pt", map_location="cpu", weights_only=False)
    rec_sha = str(ck.get("vocab_sha256", ""))
    if rec_sha:
        got = fingerprint(args.vocab)
        if got != rec_sha:
            raise RuntimeError(
                f"VOCAB MISMATCH: checkpoint trained with sha {rec_sha[:16]}… "
                f"but '{args.vocab}' hashes to {got[:16]}… — refusing to evaluate")
        print(f"vocab fingerprint ok ({rec_sha[:16]}…)", flush=True)

    summary, config = load_run_meta(d, ck, args.config_fallback or None)
    ubm = UBM.load(args.vocab)
    vocab_size = len(ubm.bases)

    model = build_model(config, vocab_size)
    state = ck.get("best_state") or ck["model"]
    model.load_state_dict(state)
    model.eval()

    val_text = Path(args.data).read_text(encoding="utf-8")
    prepared = prepare_text(ubm, val_text)

    is_iecnn = config["model"]["kind"] == "iecnn"
    best_step = int(summary.get("best_step", ck.get("step", 0)))
    steps_target = int(summary.get("steps_target",
                                   config.get("training", {}).get("steps", 1)))
    cons = config.get("consensus", {})
    tau0 = temperature_at(best_step, steps_target,
                          int(cons.get("tau_warmup", 2000)),
                          float(cons.get("tau_max", 1.0)),
                          float(cons.get("tau_min", 0.2))) if is_iecnn else 1.0
    print(f"re-scoring best step {best_step} at tau0 {tau0:.4f}", flush=True)

    ev = evaluate_bpc_per_pass(model, ubm, val_text, prepared=prepared,
                               tau0=tau0) if is_iecnn else \
        {"passes": [evaluate_bpc(model, ubm, val_text, prepared=prepared)],
         "final": evaluate_bpc(model, ubm, val_text, prepared=prepared),
         "weight_entropy": None, "tau": [None]}
    hard = None
    if is_iecnn:
        hard = evaluate_bpc(model, ubm, val_text, prepared=prepared,
                            hard_mode=True, tau0=tau0)
    train_slice = Path(config.get("data", {}).get(
        "train_file", "data/train.txt")).read_text(encoding="utf-8")
    train_slice = train_slice[:max(args.train_slice_chars * 6, 512_000)]
    bpc_train = evaluate_bpc(model, ubm, train_slice, prepared=prepare_text(ubm, train_slice),
                             max_chars=args.train_slice_chars, tau0=tau0)

    out = {
        "ckpt_dir": str(d), "best_step": best_step, "tau0_used": round(tau0, 5),
        "best_full_val_bpc_at_best_tau": ev["final"],
        "per_pass_bpc": ev["passes"],
        "weight_entropy": ev.get("weight_entropy"),
        "hard_bpc_at_best_tau": hard,
        "mode_gap": None if hard is None else round(hard - ev["final"], 5),
        "bpc_train_slice": round(bpc_train, 5),
        "vocab_sha256": fingerprint(args.vocab),
    }
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
