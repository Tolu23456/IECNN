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
    dropk    dot k removed from the committee (only on the first checkpoint)

Four prompt families, because one of them is the falsifier:

  Q1  matched    a random printable pattern repeated twice. The copy channel's
                 best case: the k-gram recurs with a long match extension.
  Q1b perturbed  the SAME pattern with ONE byte of the second copy flipped, so
                 no k-gram matches and the extension is zero. If `full` still
                 reproduces the pattern here, the channel is not doing exact
                 match continuation and Q1 is not evidence about it.
  Q2  natural    real val passages, the hard case. Reported to size the gap
                 honestly, not to be won.
  Q3  dots       the model's own 96-byte output, scored teacher-forced under
                 the full mixture, each dot dropped, and each dot alone, next
                 to the same table on real text. Tests whether the committee's
                 complementarity survives on self-generated text -- the regime
                 any self-training loop would actually visit.

Generation is not used as a quality argument here; that lane is closed (20-pair
blind judge 11-9, p=0.82). Samples are printed only so the ablations can be
read next to what they produced.

Usage:
  PYTHONPATH=. python3 scripts/probe_generation_ablation.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE like the harness.
"""
from __future__ import annotations

import argparse
import math
import os
import random
import time
from contextlib import contextmanager

import torch

from models.generate import generate_bytes
from scripts.probe_channels import T, V, load_model

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))

N_PAT = int(os.environ.get("GA_PATTERNS", "24"))
PAT_LEN = int(os.environ.get("GA_PAT_LEN", "16"))
N_NAT = int(os.environ.get("GA_NATURAL", "12"))
NAT_CTX = int(os.environ.get("GA_NAT_CTX", "64"))
GEN_BYTES = int(os.environ.get("GA_BYTES", "16"))
DOT_BYTES = int(os.environ.get("GA_DOT_BYTES", "96"))
DOT_PROMPTS = int(os.environ.get("GA_DOT_PROMPTS", "8"))
SAMPLE_PROMPTS = int(os.environ.get("GA_SAMPLE_PROMPTS", "4"))
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
           seed: int = 1234, kwargs: dict | None = None) -> bytes:
    p = torch.tensor(list(prompt_ids), dtype=torch.long)[None]
    return generate_bytes(model, p, n, mode=mode, temp=0.7, topk=40,
                          seed=seed, model_kwargs=kwargs)


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


def val_bpc(model, val, starts) -> float:
    """Recompute the harness bpc. The transformer's position table is exactly
    T, so it is scored on T tokens (T-1 predictions) and the dot model on
    T+1 (T predictions); feeding T+1 to the transformer raises."""
    is_tf = not getattr(model, "regions", None)
    n = 0
    tot = 0.0
    for o in starts:
        span = T if is_tf else T + 1
        tot += score_bpc(model, val[o:o + span].tolist(), 1) * (span - 1)
        n += span - 1
    return tot / n


# --------------------------------------------------------------------------
# sections.  A condition is (label, state patch, forward kwargs).
# --------------------------------------------------------------------------
def _family_table(model, title: str, conds, prompts) -> dict:
    """prompts: list of (prompt_bytes_or_list, target_bytes)."""
    print(f"\n{title}", flush=True)
    print(f"{'cond':<8}{'match':>8}{'match sd':>10}{'run':>7}{'tf bpc':>9}"
          f"{'tokens':>8}", flush=True)
    out = {}
    for label, patch, kw in conds:
        with condition(model, patch, seed=7):
            hits, runs, bpcs = [], [], []
            for prompt, tgt in prompts:
                g = decode(model, prompt, len(tgt), "greedy", kwargs=kw)
                hits.append(sum(a == b for a, b in zip(g, tgt)) / len(tgt))
                runs.append(leading_run(g, tgt))
                bpcs.append(score_bpc(model, list(prompt) + list(tgt),
                                      len(prompt), kw))
        m, sd = mean_sd(hits)
        out[label] = {"match": m, "sd": sd,
                      "run": sum(runs) / len(runs),
                      "tf": sum(bpcs) / len(bpcs)}
        print(f"{label:<8}{m:>8.3f}{sd:>10.3f}"
              f"{out[label]['run']:>7.2f}{out[label]['tf']:>9.4f}"
              f"{len(hits) * len(prompts[0][1]):>8}", flush=True)
    return out


def induction_section(model, val, conds, natural_conds, full_grid: bool):
    rnd = random.Random(20261122)
    pats = [bytes(rnd.randrange(32, 127) for _ in range(PAT_LEN))
            for _ in range(N_PAT)]
    # Q1b: one flipped byte near the end of the second copy. No k-gram matches,
    # extension 0, so the channel must stay silent if it is exact matching.
    def shifted(b: int) -> bytes:
        return bytes([b + 1 if b < 126 else b - 1])

    pert = [(p + p[:-1] + shifted(p[-1]), p[:GEN_BYTES]) for p in pats]
    matched = [(p + p, p[:GEN_BYTES]) for p in pats]
    naturals = []
    for i in range(N_NAT):
        o = 300 + i * 977
        naturals.append((val[o:o + NAT_CTX],
                         val[o + NAT_CTX:o + NAT_CTX + GEN_BYTES]))

    q1 = _family_table(
        model, f"-- Q1 induction: random printable pattern x2, greedy "
        f"{GEN_BYTES} bytes, {N_PAT} patterns --", conds, matched)

    low = [c for c in conds if c[0] in ("full", "off", "slope0")]
    q1b = _family_table(
        model, f"-- Q1b falsifier: same pattern, LAST byte of copy 2 flipped "
        f"(no k-gram matches) --", low, pert)

    if not full_grid:
        return
    q2 = _family_table(
        model, f"-- Q2 natural: real val {NAT_CTX}B passage, greedy "
        f"{GEN_BYTES} bytes, {N_NAT} passages --", natural_conds, naturals)

    def d(a, b, k="tf"):
        return q1[a][k] - q1[b][k] if a in q1 and b in q1 else float("nan")

    print("\n  read:", flush=True)
    print(f"   channel value on matched patterns  full-off   "
          f"{d('full', 'off'):+8.4f} bpc", flush=True)
    print(f"   slope share                       off-slope0 "
          f"{d('off', 'slope0'):+8.4f} bpc", flush=True)
    print(f"   gate level share                  slope0-off "
          f"{d('slope0', 'off'):+8.4f} bpc", flush=True)
    if "full" in q1b and "off" in q1b:
        print(f"   after one flipped byte            full-off   "
              f"{q1b['full']['tf'] - q1b['off']['tf']:+8.4f} bpc "
              f"(was {d('full', 'off'):+.4f})", flush=True)
    if q2:
        print(f"   natural text                      full-off   "
              f"{q2['full']['tf'] - q2['off']['tf']:+8.4f} bpc", flush=True)


def dots(model, val, dot_conds, device) -> None:
    prompts = [val[700 + i * 1201:748 + i * 1201].tolist()
               for i in range(DOT_PROMPTS)]
    self_seqs = []
    for pr in prompts:
        g = decode(model, pr, DOT_BYTES, "sample", seed=99)
        self_seqs.append((pr + list(g), len(pr)))
    real_seqs = []
    for k in range(DOT_PROMPTS):
        o = 4000 + k * T * VALSTRIDE
        real_seqs.append((val[o:o + T].tolist(), 1))

    print(f"\n-- Q3 dots: teacher-forced bpc of {DOT_PROMPTS} x {DOT_BYTES}B "
          f"SELF-generated vs REAL val text --", flush=True)
    print(f"{'cond':<10}{'self':>9}{'real':>9}{'self-real':>11}", flush=True)
    for label, kw, pat in dot_conds:
        with condition(model, pat):
            s = sum(score_bpc(model, ids, st, kw) for ids, st in self_seqs)
            s /= len(self_seqs)
            r = sum(score_bpc(model, ids, st, kw) for ids, st in real_seqs)
            r /= len(real_seqs)
        print(f"{label:<10}{s:>9.4f}{r:>9.4f}{s - r:>+11.4f}", flush=True)

    print("\n  per-dot agreement with the EMITTED byte (self text):",
          flush=True)
    heads = []
    for label, kw, pat in dot_conds:
        if "pool_only_dot" not in kw:
            continue
        agree = n = 0
        with condition(model, pat):
            for ids, st in self_seqs:
                x = torch.tensor(ids, dtype=torch.long)[None]
                with torch.no_grad():
                    lp = model(x, **kw)["final_log_probs"][0]
                lo, hi = st - 1, x.numel() - 1
                y = x[0, 1:]
                agree += int((lp[lo:hi].argmax(-1) == y[lo:hi]).sum())
                n += hi - lo
        heads.append((label, agree / n))
    print("   " + "  ".join(f"{k} {v:.3f}" for k, v in heads), flush=True)


def samples(model, val) -> None:
    print(f"\n-- samples: temp 0.7 / top-40 / seed 1234, {SAMPLE_BYTES}B --",
          flush=True)
    for cond in ("full", "off"):
        with condition(model, cond):
            for i in range(SAMPLE_PROMPTS):
                o = 9000 + i * 2411
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
        print(f"\n=== {name}", flush=True)
        is_dot = getattr(model, "regions", None) is not None
        r = model.regions[0] if is_dot else None
        if is_dot and getattr(r, "sc_gate", None) is not None:
            slope = (float(r.sc_slope.detach())
                     if r.sc_slope is not None else None)
            print(f"  params {sum(p.numel() for p in model.parameters())} | "
                  f"gate a={float(r.sc_gate):+.3f} "
                  f"(sigmoid {float(torch.sigmoid(r.sc_gate)):.3f}) | "
                  f"slope b={'none' if slope is None else f'{slope:+.4f}'} | "
                  f"k={r.suffix_copy} vote={r.suffix_copy_vote}", flush=True)
        else:
            print(f"  params {sum(p.numel() for p in model.parameters())} | "
                  f"no copy channel", flush=True)
        print(f"  data_fp {row.get('data_fp', 'n/a')} | bpc_final "
              f"{row.get('bpc_final', 'n/a')}", flush=True)

        # Faithfulness: the same weights on the same windows must reproduce
        # bpc_final. The saved weights ARE the SWA average, which is why they
        # reproduce bpc_final and NOT the last raw marker (0.045-0.061 worse).
        got = val_bpc(model, val, starts)
        ref = row.get("bpc_final")
        curve = row.get("curve") or []
        flag = ("" if ref is None or abs(got - float(ref)) <= 0.01
                else "  <-- WARN")
        print(f"  faithfulness: recomputed {got:.4f} | bpc_final "
              f"{ref} | last raw marker "
              f"{curve[-1][1] if curve else float('nan'):.4f}{flag}",
              flush=True)

        has_copy = is_dot and getattr(r, "sc_gate", None) is not None
        conds = [("full", "full", {})]
        if has_copy:
            conds = [("full", "full", {}), ("off", "off", {})]
            if r.sc_slope is not None:
                conds.append(("slope0", "slope0", {}))
            if r.suffix_copy_vote == 0:
                conds.append(("scram", "scram", {}))

        full_grid = is_dot and not dot_seen
        if full_grid:
            # does induction need the committee? one row per dot, so the
            # log shows whether the essential dot is also the induction route
            conds = conds + [(f"drop{k}", "full", {"ablate_dot": k})
                             for k in range(r.n_dots)]
        dot_seen = dot_seen or is_dot

        t0 = time.time()
        induction_section(model, val, conds, conds[:3], full_grid=full_grid)
        print(f"  [induction done in {time.time() - t0:.0f}s]", flush=True)

        if is_dot:
            dot_conds = [("full", {}, "full")]
            for k in range(r.n_dots):
                dot_conds.append((f"drop{k}", {"ablate_dot": k}, "full"))
            for k in range(r.n_dots):
                dot_conds.append((f"only{k}", {"pool_only_dot": k}, "full"))
            dot_conds.append(("gate_off", {}, "off"))
            t0 = time.time()
            dots(model, val, dot_conds, device)
            print(f"  [dots done in {time.time() - t0:.0f}s]", flush=True)
            t0 = time.time()
            samples(model, val)
            print(f"  [samples done in {time.time() - t0:.0f}s]", flush=True)


if __name__ == "__main__":
    main()
