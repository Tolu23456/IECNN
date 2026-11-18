"""Per-step cost of registry arms, with no training noise.

Why this exists: every verdict in DECISIONS.md is an equal-wall-clock verdict,
so a mechanism's bpc gain only counts if it is priced. The 2-vCPU toy punishes
added capacity, which makes its *bpc* signal useless for structural questions,
but its relative step cost is a faithful read of the MAC ratio (cell vs block
vs read) and costs a minute instead of a GPU batch.

Usage: PYTHONPATH=. python3 scripts/stepcost.py arm1,arm2 [steps]
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("TP_THREADS", "2")))

# ab_small parses sys.argv at import time, so hide ours while importing it.
_argv = sys.argv
sys.argv = [sys.argv[0], "1", ""]
import scripts.ab_small as A  # noqa: E402
sys.argv = _argv

DEV = torch.device(os.environ.get("TP_DEV", "cpu"))
V = 256
B = int(os.environ.get("TP_B", "8"))
T = int(os.environ.get("TP_T", "64"))
STEPS = int(sys.argv[2]) if len(sys.argv) > 2 else 12


def bench(name, steps):
    kind, cfg_fn, _opts = A.ARMS[name]
    cfg = cfg_fn()
    torch.manual_seed(0)
    if kind == "transformer":
        m = A.build_transformer_baseline(cfg, V).to(DEV)
    else:
        m = A.build_iecnn(cfg, V).to(DEV)
    n_par = sum(p.numel() for p in m.parameters())
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    x = torch.randint(0, V, (B, T), device=DEV)
    y = torch.randint(0, V, (B, T), device=DEV)

    def step():
        out = m(x)
        loss = torch.nn.functional.cross_entropy(
            out["final_log_probs"].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()

    step()                                        # warm allocator/autograd
    t0 = time.perf_counter()
    for _ in range(steps):
        step()
    return n_par, (time.perf_counter() - t0) / steps


if __name__ == "__main__":
    names = (sys.argv[1].split(",") if len(sys.argv) > 1
             else ["tf-swa"])
    base = None
    for nm in names:
        try:
            n_par, dt = bench(nm, STEPS)
        except Exception as e:                    # noqa: BLE001
            print(f"{nm:<34} FAILED {type(e).__name__}: {str(e)[:90]}",
                  flush=True)
            continue
        base = base or dt
        print(f"{nm:<34} {n_par:>8} params  {dt * 1e3:7.2f} ms/step"
              f"   x{dt / base:5.2f}", flush=True)
