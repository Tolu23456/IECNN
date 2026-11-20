"""Does the checkpoint that loses on bpc produce better SAMPLED text?

One 240-byte sample per arm is not evidence, but the observation is
falsifiable on the saved checkpoints. Twenty DIFFERENT unseen prompts, the
SAME prompts and the SAME seeds for both models, so every comparison is a
matched pair and the statistic is a distribution over prompts, not one draw.

Three hypotheses, all cheap to test at inference:

  H1 SHARPNESS. The better model is more confident, so top-k40/T0.7 keeps
     re-drawing its own favourite tokens and looks repetitive, while the
     worse-calibrated model looks varied. Test: predictive entropy / top-1 /
     ECE, then resample with a per-model temperature chosen so both have the
     SAME effective sampling entropy. If the difference survives that, it is
     not sharpness.

  H2 DRAW. At this scale both distributions are near the byte marginal and a
     single trajectory is mostly decoder randomness. Test: 20 matched prompts
     per model, distinct-1/2/3, longest repeated 4-gram, markup rate, and the
     spread ACROSS prompts (sd), not just the mean.

  H3 SAMPLE vs MODEL. A sample can look more language-like while being less
     probable under the better model. Test: score every sample with BOTH
     models (teacher-forced bpc). If tf finds rr2's text more likely than its
     own, the "better sample" is a real distributional difference; if tf
     prefers its own, the judge preferred variety.

bpc stays the metric for the model (a proper scoring rule over the whole val
distribution); this probe asks what the sampled channel can support.

Usage:
  PYTHONPATH=. python3 scripts/probe_generation.py ckptA.pt [ckptB.pt ...]
Reads the same corpus/val knobs as scripts/ab_small.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import random
from collections import Counter

import torch
import torch.nn.functional as F

from models.generate import generate_bytes
from scripts.probe_channels import T, VALSTRIDE, V, load_model

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
NPROMPT = int(os.environ.get("PGEN_PROMPTS", "20"))
NSAMP_B = int(os.environ.get("PGEN_BYTES", "240"))
PROMPT = int(os.environ.get("PGEN_PROMPT", "120"))
TEMP = float(os.environ.get("PGEN_TEMP", "0.7"))
TOPK = int(os.environ.get("PGEN_TOPK", "40"))
MATCH_N = int(os.environ.get("PGEN_MATCH_N", "10"))
# one seed per prompt, identical across models: same prompt AND same uniform
# stream, so a difference cannot come from a lucky draw
SEEDS = [1234 + i for i in range(NPROMPT)]


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


def _entropy_from_logits(logits: torch.Tensor, topk: int) -> torch.Tensor:
    """Per-position entropy of the top-k-truncated softmax of `logits`.

    The tail must be removed with a FINITE mask. `masked_fill(..., -inf)` looks
    equivalent, but log_softmax then returns -inf for the tail and the entropy
    term is `0 * -inf` = nan, which propagated to a nan mean and made match_temp
    fall back to the coldest grid point (both models were sampled at T=0.30).
    Keeping exactly `topk` indices also removes the tie ambiguity a threshold
    has ("logits < th" can keep more than k entries).
    """
    if 0 < topk < logits.shape[-1]:
        idx = torch.topk(logits, topk, dim=-1).indices
        keep = torch.zeros_like(logits, dtype=torch.bool)
        keep.scatter_(-1, idx, True)
        logits = logits.masked_fill(~keep, -1e30)
    lp = F.log_softmax(logits, dim=-1)
    return -(lp.exp() * lp).sum(-1)


def sampling_entropy(model, val: torch.Tensor, windows: int, device,
                     temp: float, topk: int) -> float:
    """Mean entropy of the distribution top-k40/T<temp> actually samples from."""
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            logits = model(x)["final_log_probs"].reshape(-1, V) / max(temp, 1e-6)
            tot += float(_entropy_from_logits(logits, topk).sum())
            n += logits.shape[0]
    return tot / n


def match_temp(model, val, windows, device, target: float) -> float:
    """Grid temperature closest to `target` effective sampling entropy."""
    grid = [0.3, 0.4, 0.5, 0.6, 0.7, 0.85, 1.0, 1.2, 1.5, 2.0, 2.6, 3.4]
    best = (1e9, grid[0])
    for t in grid:
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


def _mean_sd(xs) -> tuple:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return round(m, 4), 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return round(m, 4), round(math.sqrt(var), 4)


def score_bytes(model, prompt: torch.Tensor, text: bytes, device) -> float:
    """Teacher-forced bpc of `text`, conditioned on the prompt it came from."""
    p = prompt.reshape(-1).tolist()
    t = list(text)
    # 16 bytes of context + 240 sample bytes = the trained T=256 exactly. The
    # transformer's position table is fixed at 256, so this cannot be "a bit
    # more context": 17 would raise inside the baseline's pos embedding.
    seq = p[-16:] + t
    x = torch.tensor(seq, dtype=torch.long, device=device)[None]
    with torch.no_grad():
        lp = model(x)["final_log_probs"][0, :-1]
        y = x[0, 1:]
        start = len(seq) - len(t) - 1     # first predicted sample byte
        nll = float(F.nll_loss(lp[start:], y[start:], reduction="sum"))
    return round(nll / len(t) / math.log(2), 4)


def sample_stats(model, prompts, device, temp: float) -> dict:
    per = []
    for i, p in enumerate(prompts):
        b = generate_bytes(model, p, NSAMP_B, mode="sample", temp=temp,
                           topk=TOPK, seed=SEEDS[i])
        per.append({"d1": distinct(b, 1), "d2": distinct(b, 2),
                    "d3": distinct(b, 3), "max_rep4": longest_repeat(b, 4),
                    "markup": b.count(b"[["), "text": b.decode("latin-1"),
                    "self_bpc": score_bytes(model, p, b, device)})
    agg = {}
    for key in ("d1", "d2", "d3", "markup", "self_bpc"):
        agg[key] = _mean_sd([r[key] for r in per])
    agg["max_rep4_mean"], agg["max_rep4_sd"] = _mean_sd([r["max_rep4"] for r in per])
    pooled = "".join(r["text"] for r in per).encode("latin-1")
    agg["pooled_d2"] = round(distinct(pooled, 2), 4)
    agg["pooled_d3"] = round(distinct(pooled, 3), 4)
    agg["temp"] = round(temp, 3)
    agg["samples"] = [{"prompt": i, "text": r["text"], "d2": round(r["d2"], 3),
                       "max_rep4": r["max_rep4"], "markup": r["markup"]}
                      for i, r in enumerate(per)]
    return agg


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--windows", type=int, default=VALW)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    # NPROMPT prompts spread over the whole dev slice, all unseen by training.
    # The SAME list is given to every model, in the same order.
    offs = [max(300, (i + 1) * (len(val) - PROMPT - 300) // (NPROMPT + 1)) + 300
            for i in range(NPROMPT)]
    prompts = [val[o:o + PROMPT].unsqueeze(0) for o in offs]

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
        fixed = sample_stats(model, prompts, device, TEMP)
        print(f"  T{TEMP}/k{TOPK}  {NPROMPT} prompts:"
              f"  d2 {fixed['d2'][0]:.3f}±{fixed['d2'][1]:.3f}"
              f"  d3 {fixed['d3'][0]:.3f}±{fixed['d3'][1]:.3f}"
              f"  maxrep4 {fixed['max_rep4_mean']:.2f}"
              f"  [[ {fixed['markup'][0]:.1f}"
              f"  self-bpc {fixed['self_bpc'][0]:.3f}", flush=True)
        out[name] = {"teacher": st, "fixed": fixed}

    names = list(out)

    # H3: score every sample with BOTH models. A sample that a judge likes but
    # the better model finds less likely than its own is a style preference,
    # not a distributional win.
    if len(names) > 1:
        print("\n-- H3 cross-scoring (bpc of each sample under each model) --",
              flush=True)
        texts = {n: [s["text"].encode("latin-1") for s in out[n]["fixed"]["samples"]]
                 for n in names}
        for owner in names:
            for scorer in names:
                bpcs = [score_bytes(models[scorer], prompts[i], t, device)
                        for i, t in enumerate(texts[owner])]
                m, sd = _mean_sd(bpcs)
                print(f"  {owner:<48} under {scorer:<48} {m:.3f}±{sd:.3f}",
                      flush=True)
                out[owner][f"scored_by_{scorer}"] = [round(b, 4) for b in bpcs]

    # H1: match the effective sampling entropy across models, then resample on
    # the same prompts. If "more phrases" is only sharper decoding, equalising
    # entropy removes it; if it survives, sharpness was not the explanation.
    if len(names) > 1:
        target = sampling_entropy(models[names[0]], val, min(8, args.windows),
                                  device, TEMP, TOPK)
        print(f"\n-- H1 matched sampling entropy (target {target:.3f} nats) --",
              flush=True)
        for name in names:
            t = match_temp(models[name], val, min(8, args.windows), device, target)
            m = sample_stats(models[name], prompts[:MATCH_N], device, t)
            print(f"  {name}  T {t:.2f}  d2 {m['d2'][0]:.3f}±{m['d2'][1]:.3f}"
                  f"  d3 {m['d3'][0]:.3f}  maxrep4 {m['max_rep4_mean']:.2f}"
                  f"  [[ {m['markup'][0]:.1f}", flush=True)
            out[name]["matched"] = {k: v for k, v in m.items() if k != "samples"}
            out[name]["matched_samples"] = [s["text"] for s in m["samples"]]

    # Blind pairs for a judge: same prompt index, order shuffled by a fixed
    # seed. `key` is printed separately so the pairs alone can be pasted.
    if len(names) == 2:
        rng = random.Random(20261120)
        pairs, key = [], []
        for i in range(NPROMPT):
            a = out[names[0]]["fixed"]["samples"][i]["text"]
            b = out[names[1]]["fixed"]["samples"][i]["text"]
            flip = rng.random() < 0.5
            pairs.append({"prompt": i,
                          "first": b if flip else a,
                          "second": a if flip else b})
            key.append({"prompt": i,
                        "first": names[1] if flip else names[0],
                        "second": names[0] if flip else names[1]})
        print("\nBLIND PAIRS " + json.dumps({"pairs": pairs,
                                             "prompt_text": [
                                                 p.reshape(-1).tolist()[:60]
                                                 for p in prompts]}, indent=1),
              flush=True)
        print("\nKEY " + json.dumps(key), flush=True)

    print("\nJSON " + json.dumps({"stats": out}), flush=True)


if __name__ == "__main__":
    main()
