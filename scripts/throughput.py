"""Harness throughput probe: how many tokens/second can this 2-vCPU box do?

The toy protocol compares bpc at equal wall clock, so the harness's own
tokens/s is part of the score: a 20% throughput gain is ~20% more tokens
seen, which is worth more than most architecture tweaks at a 150 s budget.

Both architectures are probed at the SAME batch/threads (whatever wins is
then used for both arms — this is a property of the machine, not of IECNN).

  python3 scripts/throughput.py            # default grid
  ARM_BS=8,16 python3 scripts/throughput.py

Only the train step is timed (model build and the first 3 warmup steps are
excluded), so the numbers are directly comparable to ab_small's s/step.
"""
from __future__ import annotations

import os
import statistics
import sys
import time

import torch
import torch.nn.functional as F

sys.path.insert(0, "/content/IECNN")

from models.iecnn import build_iecnn          # noqa: E402
from models.baselines import build_transformer_baseline  # noqa: E402

T = 64
V = 256
STEPS = 25


def iecnn_cfg():
    return {"model": {
        "kind": "iecnn", "d": 128, "d_q": 16, "d_v": 32, "d_s": 64,
        "n_dots": 4, "n_regions": 1, "n_passes": 2, "spans": [16],
        "use_weights": True, "use_messages": False, "pooling": "log_linear",
        "head_mode": "hidden", "read_mode": "soft",
        "share_pass_params": True, "dynamic_query": False,
        "use_cross_region": False, "fast_dot_math": True,
        "cache_attention": True, "cell": "lug", "head_last_only": True,
        "per_dot_logits": False, "global_read": True,
    }, "consensus": {"rho": 0.0, "kappa": 0.7},
        "training": {"compile_consensus": False, "dot_dropout": 0.0},
        "data": {"seq_len": T}}


def tf_cfg():
    return {"model": {"kind": "transformer", "d_model": 96, "n_heads": 4,
                      "n_layers": 3},
            "data": {"seq_len": T}}


def timed(model, b: int) -> tuple[float, float]:
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (200_000,), generator=g)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    ts = []
    for i in range(STEPS + 3):
        j = torch.randint(0, data.numel() - T - 1, (b,), generator=g)
        x = torch.stack([data[k:k + T] for k in j])
        y = torch.stack([data[k + 1:k + 1 + T] for k in j])
        t0 = time.perf_counter()
        out = model(x)
        loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
        dt = time.perf_counter() - t0
        if i >= 3:
            ts.append(dt)
    per_step = statistics.median(ts)
    tok_s = b * T / per_step
    return per_step, tok_s


def main() -> None:
    bs = [int(v) for v in os.environ.get("ARM_BS", "8,16,32,64").split(",")]
    threads = [int(v) for v in os.environ.get("ARM_THREADS_LIST", "1,2").split(",")]
    print(f"{'arch':<8}{'B':>5}{'thr':>5}{'s/step':>9}{'tok/s':>10}"
          f"{'us/1k tok':>11}")
    for nthr in threads:
        torch.set_num_threads(nthr)
        for b in bs:
            for name, build in (("iecnn", lambda: build_iecnn(iecnn_cfg(), V)),
                                ("tf", lambda: build_transformer_baseline(tf_cfg(), V))):
                torch.manual_seed(0)
                m = build()
                m.train()
                per_step, tok_s = timed(m, b)
                print(f"{name:<8}{b:>5}{nthr:>5}{per_step:>9.4f}{tok_s:>10.0f}"
                      f"{1e6 / tok_s:>11.1f}", flush=True)


if __name__ == "__main__":
    main()
