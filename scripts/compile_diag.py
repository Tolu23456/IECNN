"""Why does `torch.compile` speed up the transformer 25% and the dot model 0%?

Standalone diagnostic, no training. Builds one arm, starts `torch._dynamo` in
explain mode on a single forward, and prints

    * how many graphs the forward was split into (graph breaks),
    * the reason for every break,
    * how long an eager step and a compiled step actually take (20 steps each,
      so the comparison is not a 180 s protocol run).

The point is to separate "compile cannot help this model because it is already
FLOP-bound" from "compile cannot help because the region keeps breaking the
graph". Only the second one is fixable, and the fix is worth a lot: compile
buys the transformer a quarter of its step.

Env: ARM_NAME (default the wide-shallow frontier arm), DIAG_STEPS.
Run on Kaggle via scripts/kaggle_diag.py.
"""
from __future__ import annotations

import os
import sys
import time

ARM = os.environ.get("ARM_NAME", "iecnn-n3-x3-p1-ln-decay-d128-w512")
STEPS = int(os.environ.get("DIAG_STEPS", "20"))

# ab_small parses budget/arm list from argv at import; give it a small budget
# and no arms so it only loads the data and defines the registry.
sys.argv = [sys.argv[0], "5", ""]

import torch                                                    # noqa: E402

import scripts.ab_small as A                                    # noqa: E402
from models.iecnn import build_iecnn                            # noqa: E402
from train.train import build_model                             # noqa: E402


def make():
    kind, cfg_fn, _opts = A.ARMS[ARM]
    cfg = cfg_fn()
    m = (build_iecnn(cfg, A.V) if kind == "iecnn"
         else build_model(cfg, A.V)).to(A.DEV)
    return kind, m


def step_time(model, x, y, n=STEPS, warm=3):
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3)
    ts = []
    for i in range(n + warm):
        t0 = time.perf_counter()
        out = model(x)
        loss = torch.nn.functional.nll_loss(
            out["final_log_probs"].reshape(-1, A.V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        if A.DEV.type == "cuda":
            torch.cuda.synchronize()
        if i >= warm:
            ts.append(time.perf_counter() - t0)
    ts.sort()
    return sum(ts) / len(ts), ts[len(ts) // 2]


def main():
    kind, model = make()
    print(f"=== DIAG {ARM} ({kind}) device {A.DEV}", flush=True)
    g = torch.Generator().manual_seed(11)
    i = (torch.rand(1, generator=g) * (A.train.numel() - A.T - 1)).long()
    x = A.train[i:i + A.T].to(A.DEV)
    y = A.train[i + 1:i + 1 + A.T].to(A.DEV)

    # ---- 1. explain: how many graphs, and why the breaks ------------------
    try:
        from torch._dynamo import explain
        exp = explain(model)(x)
        txt = str(exp)
        print("--- dynamo explain ---", flush=True)
        print(txt[:4000], flush=True)
    except Exception as e:                                      # noqa: BLE001
        print(f"EXPLAIN FAIL {type(e).__name__}: {e}", flush=True)

    # ---- 2. does fullgraph even compile? (the honest break test) ---------
    for label, kw in (("fullgraph", {"fullgraph": True}),
                      ("default", {"fullgraph": False}),
                      ("reduce-overhead", {"mode": "reduce-overhead",
                                           "fullgraph": False})):
        try:
            cm = torch.compile(model, **kw)
            with torch.no_grad():
                cm(x)
            print(f"COMPILE {label}: ok", flush=True)
        except Exception as e:                                  # noqa: BLE001
            msg = str(e).replace("\n", " ")[:300]
            print(f"COMPILE {label}: FAIL {type(e).__name__}: {msg}", flush=True)

    # ---- 3. measured step time, eager vs compiled ------------------------
    eager_mean, eager_med = step_time(model, x, y)
    print(f"STEP eager          mean {eager_mean*1e3:.2f} ms  med "
          f"{eager_med*1e3:.2f} ms", flush=True)
    for label, kw in (("default", {}),
                      ("reduce-overhead", {"mode": "reduce-overhead"})):
        try:
            cm = torch.compile(model, **kw)
            mean, med = step_time(cm, x, y)
            print(f"STEP compile-{label:<14} mean {mean*1e3:.2f} ms  med "
                  f"{med*1e3:.2f} ms  ({eager_mean/mean:.2f}x)", flush=True)
        except Exception as e:                                  # noqa: BLE001
            print(f"STEP compile-{label}: FAIL {type(e).__name__}: "
                  f"{str(e)[:200]}", flush=True)


if __name__ == "__main__":
    main()
