"""Does the checkpoint that loses on bpc produce better SAMPLED text?

A single 240-byte sample per arm is not evidence of anything, but it is a
falsifiable observation: rr2-c (1.8276 bpc) wrote recognisable wiki phrases
while tf-c (1.7123 bpc) wrote "cropstacious cropstacious". Two hypotheses
explain that without overturning bpc, and both are measurable on the saved
checkpoints:

  H1 SHARPNESS. The better model is more confident, so top-k40/T0.7 keeps
     re-drawing its own favourite tokens, and its sample looks repetitive
     while the worse-calibrated model's sample looks varied ("more phrases").
     Test: compare predictive entropy / top-1 accuracy / ECE, then resample
     with the temperature per model chosen so both have the SAME effective
     sampling entropy. If the difference survives that, it is not sharpness.

  H2 DRAW. Both distributions are near the byte marginal at this scale; a
     single trajectory from a 0.5M-parameter model is mostly the decoder's
     randomness. Test: N matched-seed samples per model, and report
     distinct-1/2/3, the longest repeated n-gram and sample-to-sample
     agreement. A "language-like" impression that does not survive 16 draws
     is a draw.

bpc stays the metric for the model (a proper scoring rule over the whole val
distribution); this probe only asks what the sampled channel can support.

Usage:
  PYTHONPATH=. python3 scripts/probe_generation.py ckptA.pt [ckptB.pt ...]
Reads the same corpus/val knobs as scripts/ab_small.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
from collections import Counter

import torch
import torch.nn.functional as F

from models.generate import generate_bytes
from scripts.probe_channels import T, VALSTRIDE, V, load_model

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", 5_000_000))
VALW = int(os.environ.get("ARM_VALW", "160"))
NSAMP = int(os.environ.get("PGEN_N", "12"))
NSAMP_B = int(os.environ.get("PGEN_BYTES", "240"))
PROMPT = int(os.environ.get("PGEN_PROMPT", "120"))
TEMP = float(os.environ.get("PGEN_TEMP", "0.7"))
TOPK = int(os.environ.get("PGEN_TOPK", "40"))
SEEDS = [1234 + i for i in range(NSAMP)]


# --------------------------------------------------------------------------
# teacher-forced statistics
# --------------------------------------------------------------------------
def teacher_stats(model, val: torch.Tensor, windows: int, device) -> dict:
    nll = ent = 0.0
    top1 = top5 = 0
    rank_sum = 0.0
    conf_sum = 0.0
    n = 0
    bins = [[0, 0.0, 0.0] for _ in range(10)]        # count, conf sum, acc sum
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            y = val[i + 1:i + 1 + T].unsqueeze(0).to(device).reshape(-1)
            lp = model(x)["final_log_probs"].reshape(-1, V)
            p = lp.exp()
            nll += float(F.nll_loss(lp, y, reduction="sum"))
            ent += float(-(p * lp).sum())
            conf = p.max(-1).values
            # rank of the TRUE byte: 0 = argmax. top-k accuracy is a threshold
            # on this, not a count of ties with the max.
            rank = (p > p.gather(-1, y[:, None])).sum(-1)
            hit = rank == 0
            top1 += int(hit.sum())
            top5 += int((rank < 5).sum())
            rank_sum += float(rank.sum())
            conf_sum += float(conf.sum())
            for b in range(10):
                lo, hi = b / 10, (b + 1) / 10
                m = (conf >= lo) & (conf < hi if b < 9 else conf <= hi)
                if m.any():
                    bins[b][0] += int(m.sum())
                    bins[b][1] += float(conf[m].sum())
                    bins[b][2] += float(hit[m].sum())
            n += int(y.numel())
    ece = sum(abs(cs / c - acc / c) * c for c, cs, acc in bins if c) / max(n, 1)
    return {"bpc": round(nll / n / math.log(2), 4),
            "entropy_nats": round(ent / n, 4),
            "top1": round(top1 / n, 4), "top5": round(top5 / n, 4),
            "mean_rank": round(rank_sum / n, 2),
            "mean_conf": round(conf_sum / n, 4), "ece": round(ece, 4)}


def sampling_entropy(model, val: torch.Tensor, windows: int, device,
                     temp: float, topk: int) -> float:
    """Mean entropy of the distribution top-k40/T<temp> actually samples from."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            logits = model(x)["final_log_probs"].reshape(-1, V) / max(temp, 1e-6)
            if 0 < topk < V:
                th = torch.topk(logits, topk, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < th, float("-inf"))
            lp = F.log_softmax(logits, dim=-1)
            tot += float(-(lp.exp() * lp).sum())
            n += lp.shape[0]
    return tot / n


def match_temp(model, val, windows, device, target: float) -> float:
    """Smallest temperature (grid, then 3 bisections) reaching `target` entropy."""
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.6, 3.4]
    lo, hi = grid[0], grid[-1]
    best = (abs(sampling_entropy(model, val, windows, device, lo, TOPK) - target), lo)
    for t in grid[1:]:
        d = abs(sampling_entropy(model, val, windows, device, t, TOPK) - target)
        if d < best[0]:
            best = (d, t)
    return best[1]


# --------------------------------------------------------------------------
# sample statistics
# --------------------------------------------------------------------------
def distinct(bs: bytes, n: int) -> float:
    grams = [bs[i:i + n] for i in range(len(bs) - n + 1)]
    return len(set(grams)) / max(len(grams), 1)


def longest_repeat(bs: bytes, n: int) -> int:
    """Most times any n-gram occurs in one sample (n=4: local loops)."""
    grams = [bs[i:i + n] for i in range(len(bs) - n + 1)]
    return max(Counter(grams).values()) if grams else 0


def sample_stats(model, prompt, device, temp: float, seeds=SEEDS) -> dict:
    samples = []
    ents = []
    for s in seeds:
        b = generate_bytes(model, prompt, NSAMP_B, mode="sample", temp=temp,
                           topk=TOPK, seed=s)
        samples.append(b)
    # effective entropy of the sampling distribution on the prompt (cheap proxy
    # for "how much randomness the decoder is injecting")
    with torch.no_grad():
        lp = model(prompt.to(device))["final_log_probs"][0, -1] / max(temp, 1e-6)
        if 0 < TOPK < V:
            th = torch.topk(lp, TOPK).values[-1:]
            lp = lp.masked_fill(lp < th, float("-inf"))
        lp = F.log_softmax(lp, -1)
        ents.append(float(-(lp.exp() * lp).sum()))
    return {"temp": round(temp, 3),
            "d1": round(sum(distinct(b, 1) for b in samples) / len(samples), 4),
            "d2": round(sum(distinct(b, 2) for b in samples) / len(samples), 4),
            "d3": round(sum(distinct(b, 3) for b in samples) / len(samples), 4),
            "max_rep4": max(longest_repeat(b, 4) for b in samples),
            "len_mean": round(sum(len(b) for b in samples) / len(samples), 1),
            "markup_per_sample": round(
                sum(b.count(b"[[") for b in samples) / len(samples), 2),
            "distinct_samples": len(set(samples)),
            "samples": [b.decode("latin-1") for b in samples]}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--windows", type=int, default=VALW)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    # the SAME unseen prompt for every model: no model gets an easier prefix
    prompt = val[VALSTRIDE * 5:VALSTRIDE * 5 + PROMPT].unsqueeze(0)

    out, models = {}, {}
    for path in args.ckpts:
        name = os.path.basename(path)
        model, ck = load_model(path, device)
        models[name] = model
        row = ck.get("row", {})
        st = teacher_stats(model, val, args.windows, device)
        print(f"\n=== {name}  (checkpoint row bpc {row.get('bpc_final')})",
              flush=True)
        print(f"  teacher  bpc {st['bpc']:.4f}  H {st['entropy_nats']:.3f} nats"
              f"  top1 {st['top1']:.3f}  top5 {st['top5']:.3f}"
              f"  rank {st['mean_rank']:.1f}  conf {st['mean_conf']:.3f}"
              f"  ECE {st['ece']:.4f}", flush=True)
        fixed = sample_stats(model, prompt, device, TEMP)
        print(f"  T{TEMP}/k{TOPK}  d1 {fixed['d1']:.3f} d2 {fixed['d2']:.3f}"
              f" d3 {fixed['d3']:.3f}  maxrep4 {fixed['max_rep4']}"
              f"  [[ {fixed['markup_per_sample']:.1f}/sample"
              f"  unique {fixed['distinct_samples']}/{NSAMP}", flush=True)
        out[name] = {"teacher": st, "fixed": {k: v for k, v in fixed.items()
                                              if k != "samples"}}

    # H1: match the effective sampling entropy across models, then resample.
    # If the "more phrases" impression is only sharper decoding, equalising
    # entropy removes it; if it survives, sharpness was not the explanation.
    names = list(out)
    if len(names) > 1:
        target = sampling_entropy(models[names[0]], val,
                                  min(8, args.windows), device, TEMP, TOPK)
        print(f"\n-- H1 matched sampling entropy (target {target:.3f} nats) --",
              flush=True)
        for name in names:
            model = models[name]
            t = match_temp(model, val, min(8, args.windows), device, target)
            m = sample_stats(model, prompt, device, t)
            print(f"  {name}  T {t:.2f}  d2 {m['d2']:.3f} d3 {m['d3']:.3f}"
                  f"  maxrep4 {m['max_rep4']}  [[ {m['markup_per_sample']:.1f}",
                  flush=True)
            out[name]["matched"] = {k: v for k, v in m.items() if k != "samples"}
            out[name]["matched_samples"] = m["samples"]

    print("\nJSON " + json.dumps({"prompt": prompt.reshape(-1).tolist()[:40],
                                  "stats": out}), flush=True)


if __name__ == "__main__":
    main()
