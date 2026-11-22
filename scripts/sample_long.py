"""Long continuations from a trained arm, with the structure measured per chunk.

80 bytes cannot show whether a model drifts or self-corrects, and eyeballing one
sample at a time is how a corpus-idiom generator gets mistaken for a writer. This
decodes LONG continuations (default 350 bytes, sliding 256-byte context) and
reports the same statistics in each 100-byte chunk, so "it holds up" and "it
falls apart after 100 bytes" are different, checkable outcomes.

It is still not a quality argument. The only powered quality comparison in the
ledger is a 20-pair blind judge at 11-9 (p=0.82), and sample quality has never
tracked the arm that wins on bpc. These are diagnostics: markup well-formedness,
sentence-final rate, exact repetition, and within-sample self-bpc drift.

One prompt is deliberately an ECHO: a 48-byte val segment concatenated with
itself, so a long exact match exists in the context. If the copy channel is a
real induction circuit it should keep writing that segment; if the channel is
off it should not. That is the generation-side induction test on natural text.

Usage:
  PYTHONPATH=. python3 scripts/sample_long.py ckpt.pt [...]
Reads ARM_VAL / ARM_NVAL / ARM_T. Knobs: SL_PROMPTS, SL_BYTES, SL_CTX,
SL_TEMP, SL_SEED.
"""
from __future__ import annotations

import argparse
import difflib
import math
import os
from collections import Counter

import torch

from models.generate import generate_bytes
from scripts.probe_channels import T, load_model

VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
N_PROMPT = int(os.environ.get("SL_PROMPTS", "4"))
N_BYTES = int(os.environ.get("SL_BYTES", "350"))
CTX = int(os.environ.get("SL_CTX", "256"))
TEMP = float(os.environ.get("SL_TEMP", "0.7"))
SEED = int(os.environ.get("SL_SEED", "1234"))
PROMPT_B = 48
CHUNK = 100


def chunks(bs: bytes, n: int = CHUNK) -> list[bytes]:
    return [bs[i:i + n] for i in range(0, len(bs), n)]


def distinct(bs: bytes, n: int) -> float:
    g = [bs[i:i + n] for i in range(len(bs) - n + 1)]
    return len(set(g)) / max(len(g), 1)


def longest_repeat(bs: bytes, n: int = 4) -> int:
    g = [bs[i:i + n] for i in range(len(bs) - n + 1)]
    return max(Counter(g).values()) if g else 0


def bpc_range(model, ids, a: int, b: int) -> float:
    """Teacher-forced bpc of the bytes at positions [a, b) of `ids`.

    Two rules, both learned the hard way:
      * only the last `CTX` tokens are fed, because that is the window the
        decoder actually conditioned on and because the transformer's position
        table is exactly T (feeding 398 tokens raised IndexError);
      * lp[t] predicts ids[t+1], so position p is scored by lp[p-1]; slicing
        [a, b) directly would score the wrong bytes.
    """
    ids = list(ids)
    cut = max(0, len(ids) - CTX)
    ids, a, b = ids[cut:], a - cut, b - cut
    a, b = max(a, 1), min(b, len(ids))
    if b - a < 1:
        return float("nan")
    x = torch.tensor(ids, dtype=torch.long)[None]
    with torch.no_grad():
        lp = model(x)["final_log_probs"][0]
    y = x[0, 1:]
    nll = -lp[a - 1:b - 1].gather(1, y[a - 1:b - 1, None]).mean().item()
    return nll / math.log(2)


def pair_lcs(a: bytes, b: bytes) -> int:
    """Longest common substring, in bytes, between two generations.

    A model that has collapsed onto an attractor will emit the SAME passage
    from two different prompts; a model that is still conditional on its prompt
    will not. Sample-level d2 cannot see this because it is computed within one
    sample.
    """
    m = difflib.SequenceMatcher(None, a, b, autojunk=False)
    return m.find_longest_match(0, len(a), 0, len(b)).size


def greedy_hit(gen: bytes, seg: bytes) -> str:
    """Positional matches and longest common substring vs the induced segment."""
    pos = sum(a == b for a, b in zip(gen, seg))
    return f"={pos}/{len(seg)}B lcs{pair_lcs(gen, seg)}B"


def structural(bs: bytes) -> dict:
    link = bs.count(b"[[")
    close = bs.count(b"]]")
    sent = sum(bs.count(c) for c in (b".", b"!", b"?"))
    return {"d2": round(distinct(bs, 2), 3),
            "rep4": longest_repeat(bs, 4),
            "link": link, "close": close,
            "sent": round(sent / max(len(bs) / 100, 1e-9), 2)}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    offsets = [1200 + i * 3517 for i in range(N_PROMPT)]
    prompts = [("val", val[o:o + PROMPT_B].tolist()) for o in offsets]
    echo_at = 3300
    seg = val[echo_at:echo_at + PROMPT_B].tolist()
    prompts.append(("echo", seg + seg))

    print(f"val {N_VAL} bytes | {len(prompts)} prompts ({PROMPT_B}B, 1 echo) | "
          f"{N_BYTES}B each | T{TEMP}/k40 seed {SEED} | ctx {CTX}", flush=True)

    for path in args.ckpts:
        model, ck = load_model(path, device)
        name = os.path.basename(path).replace(".pt", "")
        is_dot = getattr(model, "regions", None) is not None
        r = model.regions[0] if is_dot else None
        has_copy = is_dot and getattr(r, "sc_gate", None) is not None
        print(f"\n===== {name}  (bpc_final "
              f"{ck.get('row', {}).get('bpc_final')})", flush=True)

        conds = ["full", "off"] if has_copy else ["full"]
        saved = r.sc_gate.detach().clone() if has_copy else None
        cond_texts = {}
        for cond in conds:
            print(f"\n--- {name} [{cond}] ---", flush=True)
            if cond == "off":
                r.sc_gate.data.fill_(-60.0)
            cond_texts[cond] = []
            for label, pr in prompts:
                p = torch.tensor(pr, dtype=torch.long)[None]
                g = generate_bytes(model, p, N_BYTES, mode="sample",
                                   temp=TEMP, topk=40, ctx=CTX, seed=SEED)
                text = g.decode("latin-1")
                cond_texts[cond].append(g)
                st = structural(g)
                # drift: self-bpc of the first and last chunk of the SAMPLE
                seq = pr + list(g)
                b1 = bpc_range(model, seq, PROMPT_B, PROMPT_B + CHUNK)
                bN = bpc_range(model, seq, PROMPT_B + len(g) - CHUNK,
                               PROMPT_B + len(g))
                echo_hit = ""
                if label == "echo":
                    # Two reads, because a positional count is not the question.
                    # The prompt is seg+seg, so the induced continuation is seg
                    # itself; a model that merely re-samples similar prose can
                    # match ~40% of positions by chance, while only real
                    # retrieval produces a long common SUBSTRING. Greedy is
                    # also reported: the pattern result (Q1) was greedy, so
                    # this is the same test on natural text.
                    gg = generate_bytes(model, p, len(seg), mode="greedy",
                                        ctx=CTX, temp=TEMP, topk=40, seed=SEED)
                    echo_hit = (f" echo{greedy_hit(gg, seg)} greedy={gg[:len(seg)]!r}")
                print(f"\n[{label}] prompt: {bytes(pr).decode('latin-1')!r}",
                      flush=True)
                for i, ch in enumerate(chunks(g)):
                    print(f"  {i * CHUNK:>3}: {ch.decode('latin-1')!r}",
                          flush=True)
                print(f"  d2 {st['d2']} | max rep4 {st['rep4']} | links "
                      f"{st['link']}/{st['close']} | sentence/100B "
                      f"{st['sent']} | self-bpc {b1:.3f} -> {bN:.3f}"
                      f"{echo_hit}", flush=True)
            ts = cond_texts[cond]
            lcs = [pair_lcs(ts[i], ts[j]) for i in range(len(ts))
                   for j in range(i + 1, len(ts))]
            print(f"\n  [{cond}] pairwise longest common substring over "
                  f"{len(ts)} prompts: mean {sum(lcs) / len(lcs):.1f}B, "
                  f"max {max(lcs)}B of {N_BYTES}B", flush=True)
            if cond == "off":
                r.sc_gate.data.copy_(saved)


if __name__ == "__main__":
    main()
