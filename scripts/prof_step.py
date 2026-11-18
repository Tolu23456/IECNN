"""Per-op profile of ONE toy training step (B=8, T=64), both architectures.

Why: the analytic MAC audit and the wall clock disagree. iecnn-lr2-glob has
~3.5x fewer MACs/token than tf-d96-l3 yet its step is only ~5% cheaper, so
the cost has moved somewhere the audit does not look. This prints the top
ops by SELF CPU time, so the next optimisation targets a measured cost
instead of another MAC-count guess.

  python3 scripts/prof_step.py iecnn
  python3 scripts/prof_step.py tf
"""
from __future__ import annotations

import sys

sys.path.insert(0, "/content/IECNN")

import torch  # noqa: E402
import torch.nn.functional as F  # noqa: E402
from torch.profiler import ProfilerActivity, profile  # noqa: E402

from scripts.throughput import iecnn_cfg, tf_cfg  # noqa: E402
from models.baselines import build_transformer_baseline  # noqa: E402
from models.iecnn import build_iecnn  # noqa: E402

V = 256
B, T = 8, 64


def main() -> None:
    arch = sys.argv[1] if len(sys.argv) > 1 else "iecnn"
    torch.set_num_threads(2)
    torch.manual_seed(0)
    if arch == "tf":
        cfg = tf_cfg()
        model = build_transformer_baseline(cfg, V)
    else:
        cfg = iecnn_cfg()
        cfg["model"]["read_schedule"] = "split"
        model = build_iecnn(cfg, V)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    g = torch.Generator().manual_seed(0)
    data = torch.randint(0, V, (100_000,), generator=g)

    def step():
        j = torch.randint(0, data.numel() - T - 1, (B,), generator=g)
        x = torch.stack([data[k:k + T] for k in j])
        y = torch.stack([data[k + 1:k + 1 + T] for k in j])
        out = model(x)
        loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(4):                     # warmup (allocator + threads)
        step()
    with profile(activities=[ProfilerActivity.CPU],
                 record_shapes=False) as prof:
        for _ in range(5):
            step()
    print(f"=== {arch} top self-CPU ===")
    print(prof.key_averages().table(sort_by="self_cpu_time_total",
                                    row_limit=18))


if __name__ == "__main__":
    main()
