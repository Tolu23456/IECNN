"""Effective sampling entropy per model, and variety at matched entropy.

`probe_generation`'s matched-entropy arm picks the grid temperature closest to
a target, which is enough to show that a variety gap SURVIVES matching, but not
how close the match actually was. This prints the entropies themselves, so the
claim "matched" can be checked instead of assumed, and the per-prompt distinct-n
at the two matched temperatures.

Usage:
  PYTHONPATH=. python3 scripts/probe_entropy.py ckptA.pt ckptB.pt
"""
from __future__ import annotations

import argparse
import os

import torch

from models.generate import generate_bytes
from scripts.probe_channels import T, VALSTRIDE, V, load_model
from scripts.probe_generation import _entropy_from_logits, distinct
from scripts.probe_generation import NPROMPT, PROMPT, SEEDS

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
WINDOWS = int(os.environ.get("PENT_WINDOWS", "8"))
TOPKS = [40]
TEMPS = [0.5, 0.6, 0.7, 0.85, 1.0, 1.2]
NSAMP = int(os.environ.get("PENT_N", "10"))
BYTES = int(os.environ.get("PGEN_BYTES", "240"))


def entropy(model, val, windows, device, temp, topk) -> float:
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            logits = model(x)["final_log_probs"].reshape(-1, V) / temp
            tot += float(_entropy_from_logits(logits, topk).sum())
            n += logits.shape[0]
    return tot / n


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    offs = [max(300, (i + 1) * (len(val) - PROMPT - 300) // (NPROMPT + 1)) + 300
            for i in range(NSAMP)]
    prompts = [val[o:o + PROMPT].unsqueeze(0) for o in offs]

    table = {}
    for path in args.ckpts:
        model, _ = load_model(path, device)
        name = os.path.basename(path)
        row = {}
        for topk in TOPKS:
            for t in TEMPS:
                e = entropy(model, val, WINDOWS, device, t, topk)
                row[(topk, t)] = e
                print(f"{name:<48} k{topk:<3} T{t:<5} H {e:.4f} nats", flush=True)
        table[name] = row

    # variety at each model's own closest-to-target temperature, on the SAME
    # prompts and seeds, target = the first model's T=0.7
    names = [os.path.basename(p) for p in args.ckpts]
    target = table[names[0]][(TOPKS[0], 0.7)]
    print(f"\ntarget {target:.4f} nats (from {names[0]} at T0.7)", flush=True)
    matched = {}
    for path, name in zip(args.ckpts, names):
        best = min(TEMPS, key=lambda t: abs(table[name][(TOPKS[0], t)] - target))
        matched[name] = (best, table[name][(TOPKS[0], best)])
        print(f"{name:<48} best T {best:.2f} -> H {matched[name][1]:.4f}"
              f"  (|dH| {abs(matched[name][1] - target):.4f})", flush=True)

    for path, name in zip(args.ckpts, names):
        model, _ = load_model(path, device)
        t = matched[name][0]
        ds = []
        for i, p in enumerate(prompts):
            b = generate_bytes(model, p, BYTES, mode="sample", temp=t,
                               topk=TOPKS[0], seed=SEEDS[i])
            ds.append(distinct(b, 2))
        m = sum(ds) / len(ds)
        sd = (sum((x - m) ** 2 for x in ds) / max(len(ds) - 1, 1)) ** 0.5
        print(f"{name:<48} T{t:.2f} d2 {m:.4f}+-{sd:.4f} (n={len(ds)})",
              flush=True)


if __name__ == "__main__":
    main()
