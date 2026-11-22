"""Does the mechanism behave on the text the model itself writes?

The batch results say which arm has the best bpc. They do not say that a
mechanism is what a *decoder* uses. This takes one trained checkpoint, changes
ONE thing at inference with every weight fixed, and decodes the same prompt
with the same seed, so a difference in the bytes is attributable to that one
change rather than to a different arm:

    full     the checkpoint as trained
    off      copy gate forced shut (the mixture is the base distribution)
    slope0   slope b = 0, gate level a kept (the scalar-gate twin)
    scram    same gate, same coverage, pointer replaced by a random earlier
             byte -- the "any perturbation" control

Three questions, each with the measurement that can fail:

  Q1 INDUCTION. A repeated random pattern is the copy channel's best case: the
     k-gram recurs with a long match extension, so a working induction circuit
     should continue the pattern. Greedy decode under each condition, and
     report the per-byte match rate, the leading correct run, and the
     teacher-forced bpc of the true continuation (which does not depend on
     where greedy happens to diverge). If `full` does not beat `off`, the
     channel is not an induction circuit at decode time whatever its bpc says.

  Q2 NATURAL. The same protocol on real val passages, so the random-pattern
     result cannot be dismissed as an artificial construction. This is the
     hard case and is expected to be much weaker; it is here to show the size
     of the gap rather than to be won.

  Q3 DOTS. The model writes 96 bytes, then those bytes are scored
     teacher-forced under the full mixture, each dot dropped, and each dot
     alone. The same table is computed on real val text at the same
     checkpoint. This asks whether the committee's complementarity (drop-one
     costs 0.34-4.60 bpc on real text) survives on the distribution the model
     actually visits when decoding -- the regime that matters for generation
     and for any self-training loop.

Generation is not used as a quality argument here. The ledger closed that lane
(20-pair blind judge 11-9, p=0.82); samples are printed only so the ablations
can be read next to what they produced.

Usage:
  PYTHONPATH=. python3 scripts/probe_generation_ablation.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE like the harness.
"""
from __future__ import annotations

import argparse
import math
import time
import os
import random
from contextlib import contextmanager

import torch

from models.generate import generate_bytes
from scripts.probe_channels import T, V, load_model

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))

N_PAT = int(os.environ.get("GA_PATTERNS", "8"))
PAT_LEN = int(os.environ.get("GA_PAT_LEN", "16"))
N_NAT = int(os.environ.get("GA_NATURAL", "4"))
NAT_CTX = int(os.environ.get("GA_NAT_CTX", "64"))
GEN_BYTES = int(os.environ.get("GA_BYTES", "16"))
DOT_BYTES = int(os.environ.get("GA_DOT_BYTES", "96"))
DOT_PROMPTS = int(os.environ.get("GA_DOT_PROMPTS", "4"))
SAMPLE_PROMPTS = int(os.environ.get("GA_SAMPLE_PROMPTS", "3"))
SAMPLE_BYTES = int(os.environ.get("GA_SAMPLE_BYTES", "80"))


# --------------------------------------------------------------------------
# inference-time conditions
# --------------------------------------------------------------------------
def _scrambled(real, seed: int):
    """Same gate, same validity, random earlier byte as the copy."""

    def fn(ids, with_extra: bool = False):
        out = real(ids, with_extra=True)
        copy_ids, valid, extra = out
        g = torch.Generator().manual_seed(seed)
        b, t = ids.shape
        pos = (torch.rand(b, t, generator=g) * t).long()
        pos = torch.minimum(pos, torch.arange(t) - 1).clamp_min(0)
        fake = ids.gather(1, pos.to(ids.device))
        return (fake, valid, extra) if with_extra else (fake, valid)

    return fn


@contextmanager
def condition(model, name: str, seed: int = 0):
    """Patch exactly one thing, then put it back."""
    r = model.regions[0] if getattr(model, "regions", None) else None
    saved = {}
    if r is not None and getattr(r, "sc_gate", None) is not None:
        if name == "off":
            saved["gate"] = r.sc_gate.detach().clone()
            r.sc_gate.data.fill_(-60.0)
        elif name == "slope0":
            if r.sc_slope is not None:
                saved["slope"] = r.sc_slope.detach().clone()
                r.sc_slope.data.zero_()
        elif name == "scram":
            # the vote path calls _suffix_vote, not _suffix_copy, so a
            # monkeypatch here would silently do nothing
            assert r.suffix_copy_vote == 0, "scram not implemented for vote"
            saved["fn"] = r._suffix_copy
            r._suffix_copy = _scrambled(r._suffix_copy, seed)
        elif name != "full":
            raise ValueError(name)
    elif name != "full":
        raise ValueError(f"{name} needs a copy channel")
    try:
        yield
    finally:
        # unconditional, and no `return` in here: returning from a finally
        # block would swallow an exception raised by the caller's body
        if "gate" in saved:
            r.sc_gate.data.copy_(saved["gate"])
        if "slope" in saved:
            r.sc_slope.data.copy_(saved["slope"])
        if "fn" in saved:
            r._suffix_copy = saved["fn"]


# --------------------------------------------------------------------------
# measurements
# --------------------------------------------------------------------------
def score_bpc(model, ids, start: int, kwargs: dict | None = None) -> float:
    """Teacher-forced bpc of positions >= start (predicting ids[start:])."""
    x = torch.tensor(list(ids), dtype=torch.long)[None]
    with torch.no_grad():
        lp = model(x, **(kwargs or {}))["final_log_probs"][0]
    lo, hi = start - 1, x.numel() - 1
    y = x[0, 1:]
    nll = -lp[lo:hi].gather(1, y[lo:hi, None]).mean().item()
    return nll / math.log(2)


def decode(model, prompt_ids, n: int, mode: str = "greedy",
           seed: int = 1234) -> bytes:
    p = torch.tensor(list(prompt_ids), dtype=torch.long)[None]
    return generate_bytes(model, p, n, mode=mode, temp=0.7, topk=40,
                          seed=seed)


def mean_sd(xs) -> tuple[float, float]:
    m = sum(xs) / len(xs)
    if len(xs) < 2:
        return m, 0.0
    var = sum((x - m) ** 2 for x in xs) / (len(xs) - 1)
    return m, math.sqrt(var)


def leading_run(gen: bytes, tgt: bytes) -> int:
    n = 0
    for a, b in zip(gen, tgt):
        if a != b:
            break
        n += 1
    return n


# --------------------------------------------------------------------------
# sections
# --------------------------------------------------------------------------
def induction_section(model, val, conds, full_grid: bool, device) -> None:
    rnd = random.Random(20261122)
    pats = [bytes(rnd.randrange(32, 127) for _ in range(PAT_LEN))
            for _ in range(N_PAT)]
    offsets = [300 + i * 977 for i in range(N_NAT)]
    naturals = [(val[o:o + NAT_CTX].tolist(),
                 val[o + NAT_CTX:o + NAT_CTX + GEN_BYTES].tolist())
                for o in offsets]

    print(f"\n-- Q1 induction: random printable pattern x2, greedy "
          f"{GEN_BYTES} bytes, {N_PAT} patterns --", flush=True)
    print(f"{'cond':<8}{'match':>8}{'match sd':>10}{'run':>7}{'tf bpc':>9}"
          f"{'tokens':>8}", flush=True)
    for name in conds:
        with condition(model, name, seed=7):
            hits, runs, bpcs = [], [], []
            for p in pats:
                prompt = p + p
                tgt = p[:GEN_BYTES]
                g = decode(model, prompt, GEN_BYTES, "greedy")
                hits.append(sum(a == b for a, b in zip(g, tgt)) / GEN_BYTES)
                runs.append(leading_run(g, tgt))
                bpcs.append(score_bpc(model, prompt + tgt, len(prompt)))
        m, sd = mean_sd(hits)
        print(f"{name:<8}{m:>8.3f}{sd:>10.3f}"
              f"{sum(runs) / len(runs):>7.2f}{sum(bpcs) / len(bpcs):>9.4f}"
              f"{len(hits) * GEN_BYTES:>8}", flush=True)

    if not full_grid:
        return
    print(f"\n-- Q2 natural: real val {NAT_CTX}B passage, greedy "
          f"{GEN_BYTES} bytes, {N_NAT} passages --", flush=True)
    print(f"{'cond':<8}{'match':>8}{'match sd':>10}{'run':>7}{'tf bpc':>9}"
          f"{'tokens':>8}", flush=True)
    for name in conds:
        with condition(model, name, seed=7):
            hits, runs, bpcs = [], [], []
            for ctx_ids, tgt_ids in naturals:
                tgt = bytes(tgt_ids)
                g = decode(model, ctx_ids, GEN_BYTES, "greedy")
                hits.append(sum(a == b for a, b in zip(g, tgt)) / GEN_BYTES)
                runs.append(leading_run(g, tgt))
                bpcs.append(score_bpc(model, ctx_ids + tgt_ids,
                                      len(ctx_ids)))
        m, sd = mean_sd(hits)
        print(f"{name:<8}{m:>8.3f}{sd:>10.3f}"
              f"{sum(runs) / len(runs):>7.2f}{sum(bpcs) / len(bpcs):>9.4f}"
              f"{len(hits) * GEN_BYTES:>8}", flush=True)


def dot_section(model, val, dot_conds, device) -> None:
    """Does complementarity survive on the model's own output?"""
    offsets = [700 + i * 1201 for i in range(DOT_PROMPTS)]
    prompts = [val[o:o + 48].tolist() for o in offsets]

    self_seqs, real_seqs = [], []
    for pr in prompts:
        g = decode(model, pr, DOT_BYTES, "sample", seed=99)
        self_seqs.append((pr + list(g), len(pr)))
    for k in range(DOT_PROMPTS):
        o = 4000 + k * T * VALSTRIDE
        real_seqs.append((val[o:o + T].tolist(), 1))

    print(f"\n-- Q3 dots: teacher-forced bpc of {DOT_PROMPTS} x {DOT_BYTES}B "
          f"SELF-generated vs REAL val text --", flush=True)
    print(f"{'cond':<10}{'self':>9}{'real':>9}{'self-real':>11}", flush=True)
    for label, kw, cond in dot_conds:
        with condition(model, cond):
            s = sum(score_bpc(model, ids, st, kw) for ids, st in self_seqs)
            s /= len(self_seqs)
            r = sum(score_bpc(model, ids, st, kw) for ids, st in real_seqs)
            r /= len(real_seqs)
        print(f"{label:<10}{s:>9.4f}{r:>9.4f}{s - r:>+11.4f}", flush=True)

    # which dot agrees with what the mixture emitted
    print(f"\n  per-dot agreement with the EMITTED byte (self text):",
          flush=True)
    heads = []
    for label, kw, cond in dot_conds:
        if "pool_only_dot" not in kw:
            continue
        agree = 0
        n = 0
        with condition(model, cond):
            for ids, st in self_seqs:
                x = torch.tensor(ids, dtype=torch.long)[None]
                with torch.no_grad():
                    lp = model(x, **kw)["final_log_probs"][0]
                lo, hi = st - 1, x.numel() - 1
                y = x[0, 1:]
                am = lp[lo:hi].argmax(-1)
                agree += int((am == y[lo:hi]).sum())
                n += hi - lo
        heads.append((label, agree / n))
    print("   " + "  ".join(f"{k} {v:.3f}" for k, v in heads), flush=True)


def samples(model, val, device) -> None:
    offsets = [9000 + i * 2411 for i in range(SAMPLE_PROMPTS)]
    print(f"\n-- samples: temp 0.7 / top-40 / seed 1234, {SAMPLE_BYTES}B --",
          flush=True)
    for cond in ("full", "off"):
        with condition(model, cond):
            for o in offsets:
                pr = val[o:o + 48].tolist()
                g = decode(model, pr, SAMPLE_BYTES, "sample", seed=1234)
                print(f"\n  [{cond}] prompt: "
                      f"{bytes(pr).decode('latin-1')!r}", flush=True)
                print(f"  [{cond}] text:   {g.decode('latin-1')!r}",
                      flush=True)


# --------------------------------------------------------------------------
def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--windows", type=int, default=160)
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    starts = [w * T * VALSTRIDE for w in range(args.windows)
              if w * T * VALSTRIDE + T + 1 <= val.numel()]
    print(f"val {N_VAL} bytes | {len(starts)}x{T} stride {VALSTRIDE} | "
          f"{N_PAT} patterns x {PAT_LEN}B | {N_NAT} passages", flush=True)

    dot_seen = False
    for path in args.ckpts:
        model, ck = load_model(path, device)
        row = ck.get("row", {})
        name = os.path.basename(path).replace(".pt", "")
        n_par = sum(p.numel() for p in model.parameters())
        print(f"\n=== {name}", flush=True)
        is_dot = getattr(model, "regions", None) is not None
        r = model.regions[0] if is_dot else None
        if is_dot and getattr(r, "sc_gate", None) is not None:
            slope = (float(r.sc_slope) if r.sc_slope is not None else None)
            print(f"  params {n_par} | gate a={float(r.sc_gate):+.3f} "
                  f"(sigmoid {float(torch.sigmoid(r.sc_gate)):.3f}) | "
                  f"slope b={'none' if slope is None else f'{slope:+.4f}'} | "
                  f"k={r.suffix_copy} vote={r.suffix_copy_vote}", flush=True)
        else:
            print(f"  params {n_par} | no copy channel", flush=True)
        print(f"  data_fp {row.get('data_fp', 'n/a')} | row bpc_final "
              f"{row.get('bpc_final', 'n/a')}", flush=True)

        # faithfulness: same weights + same windows must reproduce the last
        # raw marker. Mismatch means the probe is scoring something else.
        curve = row.get("curve") or []
        tot = sum(score_bpc(model, val[o:o + T + 1].tolist(), 1)
                  * T for o in starts)
        got = tot / (len(starts) * T)
        ref = curve[-1][1] if curve else float("nan")
        flag = "" if not curve or abs(got - ref) <= 0.02 else "  <-- WARN"
        print(f"  faithfulness: recomputed {got:.4f} | last raw marker "
              f"{ref:.4f}{flag}", flush=True)

        if is_dot and getattr(r, "sc_gate", None) is not None:
            conds = ["full", "off"]
            if r.sc_slope is not None:
                conds.append("slope0")
            if r.suffix_copy_vote == 0:
                conds.append("scram")
        else:
            conds = ["full"]
        # the natural-text arm is the expensive one, so it runs on the first
        # dot checkpoint only; the random-pattern grid runs on every seed
        full_grid = is_dot and not dot_seen
        dot_seen = dot_seen or is_dot
        t0 = time.time()
        induction_section(model, val, conds, full_grid=full_grid, device=device)
        print(f"  [induction done in {time.time() - t0:.0f}s]", flush=True)

        if is_dot:
            dot_conds = [("full", {}, "full")]
            for k in range(r.n_dots):
                dot_conds.append((f"drop{k}", {"ablate_dot": k}, "full"))
            for k in range(r.n_dots):
                dot_conds.append((f"only{k}", {"pool_only_dot": k}, "full"))
            dot_conds.append(("gate_off", {}, "off"))
            t0 = time.time()
            dot_section(model, val, dot_conds, device=device)
            print(f"  [dots done in {time.time() - t0:.0f}s]", flush=True)
            t0 = time.time()
            samples(model, val, device)
            print(f"  [samples done in {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
