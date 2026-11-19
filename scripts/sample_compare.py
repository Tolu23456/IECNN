"""Train two arms on the SAME data/budget, then generate from the SAME prompt.

The bpc tables say which model predicts the corpus better; they do not say
what the difference looks like. This script trains each arm with the exact
harness in `scripts/ab_small.py` (same optimizer, warmup, SWA tail, wall-clock
budget, seed) and then decodes the same held-out prompt twice: greedily, and
with temperature 0.7 / top-k 40 from a fixed generator seed.

The prompt is the first `SC_PROMPT` bytes of the VAL file, so neither model
has trained on it and the true continuation can be printed beside the two
generations. That is the whole point: a model that has the right conditional
distribution reproduces wiki-style English even when it gets the facts wrong;
a model that is stuck on n-gram statistics produces locally plausible words
in globally wrong order, and the two are easy to tell apart by eye.

Run (env is the same as ab_small's):
    ARM_TRAIN=... ARM_VAL=... ARM_NTRAIN=... ARM_NVAL=... \
    ARM_B=32 ARM_T=256 ARM_VALW=160 \
    python3 -u scripts/sample_compare.py 360 tf-swa,iecnn-...-w512

    SC_PROMPT=120 SC_LEN=400 SC_TEMP=0.7 SC_TOPK=40 SC_SEED=1234
"""
from __future__ import annotations

import importlib.util
import os
import sys

import torch
import torch.nn.functional as F

HERE = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, HERE)

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 120.0
ARMS = sys.argv[2].split(",") if len(sys.argv) > 2 else [
    "tf-swa", "iecnn-n3-x3-p1-ln-decay-d128-w512"]
PROMPT_N = int(os.environ.get("SC_PROMPT", "120"))
GEN_N = int(os.environ.get("SC_LEN", "400"))
TEMP = float(os.environ.get("SC_TEMP", "0.7"))
TOPK = int(os.environ.get("SC_TOPK", "40"))
SC_SEED = int(os.environ.get("SC_SEED", "1234"))

# ab_small parses its budget/arm list from sys.argv at import time, and loads
# the corpus then too. Give it those two arguments, then put ours back so the
# module's CLI cannot see this script's arguments.
_argv = sys.argv
sys.argv = ["ab_small.py", str(BUDGET)]
_spec = importlib.util.spec_from_file_location(
    "ab_small", os.path.join(HERE, "scripts", "ab_small.py"))
ab = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(ab)
sys.argv = _argv

DEV = ab.DEV
T = ab.T


def _txt(b: bytes) -> str:
    s = b.decode("utf-8", errors="replace")
    return s.replace("\r", " ").replace("\x00", ".")


@torch.no_grad()
def generate(model, prompt: torch.Tensor, n: int, mode: str,
             seed: int = SC_SEED) -> bytes:
    """Byte-level ancestral/greedy decode from `prompt` [1, p]."""
    model.eval()
    g = torch.Generator(device=prompt.device).manual_seed(seed)
    seq = prompt.clone()
    out = []
    for _ in range(n):
        lp = model(seq[:, -T:])["final_log_probs"][:, -1]      # [1, V]
        if mode == "greedy":
            nxt = lp.argmax(-1)
        else:
            logits = lp / max(TEMP, 1e-6)
            if 0 < TOPK < logits.shape[-1]:
                th = torch.topk(logits, TOPK, dim=-1).values[:, -1:]
                logits = logits.masked_fill(logits < th, float("-inf"))
            prob = F.softmax(logits, dim=-1)
            nxt = torch.multinomial(prob, 1, generator=g)
        out.append(int(nxt))
        seq = torch.cat([seq, nxt.reshape(1, 1)], dim=1)
    return bytes(out)


def main():
    va_path = os.environ.get("ARM_VAL", "data/toy_val.txt")
    with open(va_path, "rb") as f:
        head = f.read(PROMPT_N + GEN_N)
    if len(head) < PROMPT_N + 1:
        raise SystemExit(f"val file too short: {va_path}")
    prompt = torch.tensor(list(head[:PROMPT_N]), dtype=torch.long,
                          device=DEV).unsqueeze(0)
    truth = head[PROMPT_N:PROMPT_N + GEN_N]

    print(f"=== PROMPT ({PROMPT_N} bytes of the VAL split, seen by neither arm)",
          flush=True)
    print(_txt(head[:PROMPT_N]), flush=True)
    print("=== TRUE CONTINUATION", flush=True)
    print(_txt(truth), flush=True)

    gens = {}
    for name in ARMS:
        entry = ab.ARMS[name]
        kind, mk = entry[0], entry[1]
        opts = dict(entry[2]) if len(entry) > 2 else {}
        opts.pop("compile", None)
        cfg = mk()
        torch.manual_seed(opts.get("seed", ab.SEED))
        model = (ab.build_transformer_baseline(cfg, ab.V) if kind == "transformer"
                 else ab.build_iecnn(cfg, ab.V)).to(DEV)
        n_par = sum(p.numel() for p in model.parameters())
        print(f"--- training {name} ({n_par} params, {BUDGET:.0f}s)", flush=True)
        row = ab.run(name, kind, model, **opts)
        for mode in ("greedy", "sample"):
            b = generate(model, prompt, GEN_N, mode)
            gens[(name, mode)] = b
            agree = sum(x == y for x, y in zip(b, truth)) / max(len(b), 1)
            print(f"=== {name} [{mode}]  ({agree * 100:.1f}% bytes match truth)",
                  flush=True)
            print(_txt(b), flush=True)
        del model
        if DEV.type == "cuda":
            torch.cuda.empty_cache()

    names = [n for n in ARMS]
    for mode in ("greedy", "sample"):
        if len(names) >= 2:
            a, b = gens[(names[0], mode)], gens[(names[1], mode)]
            same = sum(x == y for x, y in zip(a, b)) / max(len(a), 1)
            print(f"=== {names[0]} vs {names[1]} [{mode}]: "
                  f"{same * 100:.1f}% of bytes identical", flush=True)
    print("SAMPLE DONE", flush=True)


if __name__ == "__main__":
    main()
