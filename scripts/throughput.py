"""Where does a toy step's time actually go: fixed per-step overhead, or work
per token?  Runs the two arms that matter at several batch sizes and reports
tokens/second, so we know whether raising B (a harness knob both arms share)
buys real throughput or only makes each step slower.

No corpus, no val: random bytes, same shapes as the 150 s harness.  Deliberately
short (a few seconds per point) — this decides whether a B sweep is worth a
real batch, so it must not cost one.
"""
import os
import sys
import time

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(2)

from models.baselines import build_transformer_baseline  # noqa: E402
from models.iecnn import build_iecnn                     # noqa: E402

V = 256
T = int(os.environ.get("TP_T", "64"))
DEV = torch.device("cpu")


def iecnn_cfg():
    return {
        "model": {
            "kind": "iecnn", "d": 96, "d_q": 16, "d_v": 32, "d_s": 48,
            "n_dots": 3, "n_regions": 1, "n_passes": 2, "spans": [8],
            "use_weights": True, "use_messages": False,
            "pooling": "log_linear", "head_mode": "hidden",
            "read_mode": "soft", "share_pass_params": True,
            "dynamic_query": False, "use_cross_region": False,
            "fast_dot_math": True, "cache_attention": True,
            "cell": "lug", "head_last_only": False, "per_dot_logits": False,
            "window_mode": "contiguous", "seclusion_topk": 0,
            "pool_first": False, "slot_slots": 0, "dot_dist_bias": False,
            "global_read": True, "read_schedule": "mix", "head_untied": False,
            "cell_ln": False, "share_cell": False, "query_from_input": False,
            "id_lags": False, "ngram_bias": 0, "head_rank": 0,
            "state_carry": False, "carry_chunk": 8, "global_linear": False,
        },
        "consensus": {"rho": 0.0, "kappa": 0.7},
        "training": {"compile_consensus": False, "dot_dropout": 0.0},
        "data": {"seq_len": T},
    }


def tf_cfg():
    return {"model": {"kind": "transformer", "d_model": 96, "n_heads": 4,
                      "n_layers": 3, "ngram_bias": 0},
            "data": {"seq_len": T}}


def bench(model, B, steps=12, compile_it=True):
    model = model.to(DEV).train()
    if compile_it:
        model = torch.compile(model)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    x = torch.randint(0, V, (B, T), device=DEV)
    y = torch.randint(0, V, (B, T), device=DEV)
    ce = torch.nn.functional.cross_entropy

    def one():
        out = model(x)
        loss = ce(out["final_log_probs"].reshape(-1, V), y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()

    for _ in range(3):          # warm-up (and compile warm-up, unmeasured)
        one()
    t0 = time.time()
    for _ in range(steps):
        one()
    dt = (time.time() - t0) / steps
    tok = B * T
    print(f"B={B:<3} {tok:>5} tok/step  {dt*1000:7.2f} ms/step  "
          f"{tok/dt:9.0f} tok/s  {dt/tok*1e6:6.1f} us/tok", flush=True)
    return dt, tok / dt


for name, mk in (("iecnn-d96", lambda: build_iecnn(iecnn_cfg(), V)),
                 ("tf-d96-l3", lambda: build_transformer_baseline(tf_cfg(), V))):
    print(f"--- {name} (T={T}, compiled, 2 threads)", flush=True)
    for B in (8, 16, 32, 64):
        torch.manual_seed(0)
        bench(mk(), B)
    print(flush=True)

# how much of a step is not proportional to B: fit t(B) = a + b*B on the
# iecnn curve and report the fixed share at B=8.
print("--- fixed-overhead estimate (iecnn)", flush=True)
ts = []
for B in (8, 32):
    torch.manual_seed(0)
    dt, _ = bench(build_iecnn(iecnn_cfg(), V), B, steps=16)
    ts.append((B, dt))
(b1, t1), (b2, t2) = ts
slope = (t2 - t1) / (b2 - b1)
fixed = t1 - slope * b1
print(f"t(B) = {fixed*1000:.2f} ms + {slope*1e6:.2f} us * B   "
      f"=> fixed share at B=8: {fixed/t1*100:.0f}%", flush=True)
