"""Why does `torch.compile` die on the region model?

`scripts/compile_diag.py` reports that the frontier arm fails to compile with

    Unsupported: Length mismatch when unpacking object for UNPACK_SEQUENCE
    (SizeVariable(length=2) unpacked for `a, b, c = d`)

and (under `dynamo.explain`, which swallows tracebacks) with

    ValueError: not enough values to unpack (expected 3, got 2)

This script reproduces that on whatever device is present (CPU is free) and
prints FULL tracebacks, so the offending unpack can be found without burning
GPU minutes. It runs a tiny tensor (B=4, T=32) so it finishes in seconds, then
optionally one full-size step.

Env: ARM_NAME, CR_B, CR_T, CR_STEPS (0 = only the compile check).
"""
from __future__ import annotations

import os
import sys
import time
import traceback

sys.argv = [sys.argv[0], "5", ""]

import torch                                                    # noqa: E402

import scripts.ab_small as A                                    # noqa: E402
from models.iecnn import build_iecnn                            # noqa: E402
from train.train import build_model                             # noqa: E402

ARM = os.environ.get("ARM_NAME", "iecnn-n3-x3-p1-ln-decay-d128-w512")
B = int(os.environ.get("CR_B", "4"))
T = int(os.environ.get("CR_T", "32"))


def make():
    kind, cfg_fn, _opts = A.ARMS[ARM]
    cfg = cfg_fn()
    m = (build_iecnn(cfg, A.V) if kind == "iecnn"
         else build_model(cfg, A.V))
    return kind, m


def run(model, x, label):
    t0 = time.perf_counter()
    out = model(x)
    if A.DEV.type == "cuda":
        torch.cuda.synchronize()
    print(f"FORWARD {label}: ok {1e3*(time.perf_counter()-t0):.1f} ms",
          flush=True)
    return out


def main():
    kind, model = make()
    print(f"=== CR {ARM} ({kind}) device {A.DEV} B={B} T={T}", flush=True)
    x = torch.randint(0, A.V, (B, T), device=A.DEV)
    run(model, x, "eager")

    for label, kw in (("eager-backend",
                       {"backend": "eager", "fullgraph": True}),
                      ("default", {}),
                      ("fullgraph", {"fullgraph": True})):
        cm = torch.compile(model, **kw)
        try:
            with torch.no_grad():
                run(cm, x, f"compile-{label}")
            print(f"COMPILE {label}: ok", flush=True)
        except Exception:                                       # noqa: BLE001
            tb = traceback.format_exc()
            print(f"COMPILE {label}: FAIL", flush=True)
            print(tb[-6000:], flush=True)


if __name__ == "__main__":
    main()
