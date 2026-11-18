"""Reproduce the Inductor SplitScan failure on whatever device is present.

torch.compile of the IECNN dies in `_global_ctx`'s cumsum with
'TypeError: list indices must be integers or slices, not NoneType' on CUDA.
SplitScan lowering is device-independent code, so if a CPU Inductor build
reproduces it we can iterate on the workaround for free instead of burning
GPU minutes. Prints, for each variant, whether it compiled and its step time.
"""
import os
import sys
import time
import traceback

import torch

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
torch.set_num_threads(int(os.environ.get("TP_THREADS", "2")))

from models.iecnn import build_iecnn  # noqa: E402
import models.dot as dotmod          # noqa: E402

V, B, T = 256, 32, 256
DEV = torch.device(os.environ.get("TP_DEV", "cpu"))


def cfg():
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


def run(tag, compile_it, steps=6):
    torch.manual_seed(0)
    m = build_iecnn(cfg(), V).to(DEV).train()
    if compile_it:
        m = torch.compile(m)
    opt = torch.optim.AdamW(m.parameters(), lr=2e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    x = torch.randint(0, V, (B, T), device=DEV)
    y = torch.randint(0, V, (B, T), device=DEV)
    try:
        for _ in range(steps):
            out = m(x)
            loss = torch.nn.functional.cross_entropy(
                out["final_log_probs"].reshape(-1, V), y.reshape(-1))
            opt.zero_grad(set_to_none=True)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(m.parameters(), 1.0)
            opt.step()
        print(f"OK    {tag:<34} loss {loss.item():.4f}", flush=True)
        return True
    except Exception as e:                               # noqa: BLE001
        print(f"FAIL  {tag:<34} {type(e).__name__}: {str(e)[:110]}",
              flush=True)
        return False


print(f"device={DEV} torch={torch.__version__}", flush=True)
run("eager", False)

# variant A: as committed (cumsum along dim=1)
run("compiled / cumsum dim=1", True)

# variant B: scan on the trailing dim, everything contiguous in [B,N,T]
orig = dotmod.DotPopulation._global_ctx


def gctx_trailing(self, full_k, full_v, round_idx, q_add=None, cache=None):
    b, t, _ = full_v.shape
    q = self._sel(round_idx, self.q)
    sc = torch.matmul(full_k, q.t()).permute(0, 2, 1).contiguous()  # [B,N,T]
    sc = sc / (self.d_q ** 0.5)
    e = torch.exp(sc - sc.amax(dim=-1, keepdim=True))
    cum_e = torch.cumsum(e, dim=-1)
    vv = full_v.transpose(1, 2).contiguous().unsqueeze(1)           # [B,1,dv,T]
    ev = e.unsqueeze(2) * vv
    cum_ev = torch.cumsum(ev, dim=-1)
    g = cum_ev / cum_e.unsqueeze(2).clamp_min(1e-9)
    return g.permute(0, 3, 1, 2).contiguous()


dotmod.DotPopulation._global_ctx = gctx_trailing
run("compiled / cumsum trailing dim", True)

# variant C: run only the scan eagerly (graph break), compile the rest
import torch._dynamo as _dynamo                            # noqa: E402

_dynamo.reset()
_bare = dotmod.DotPopulation._global_ctx


@_dynamo.disable
def _scan_eager(self, full_k, full_v, round_idx, q_add=None, cache=None):
    return _bare(self, full_k, full_v, round_idx, q_add=q_add, cache=cache)


dotmod.DotPopulation._global_ctx = _scan_eager
run("compiled / scan graph-break", True)

dotmod.DotPopulation._global_ctx = orig
print("REPRO DONE", flush=True)
