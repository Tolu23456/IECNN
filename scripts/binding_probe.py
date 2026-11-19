"""Does the dot model have INSTANCE BINDING, or only type-level statistics?

The audit says the three dots settle onto fixed lags (0 / ~2 / ~5 tokens back)
and that one of them reads the current slot at 99.6% of positions. That is the
signature of a read whose query is a per-dot CONSTANT (`dots.q`, dot.py:192):
with a constant query the scores can only ask "how much does the token at this
fixed offset look like my pattern", so the best the loss can do is assign each
dot an offset. It can therefore learn TYPE-level statistics ("what usually
follows a token like X at lag 3") but not INSTANCE binding ("the value paired
with THIS key, wherever it happens to be").

This is a synthetic test of exactly that difference, with no corpus involved:

    sequence = [k1 v1 k2 v2 ... kK vK, q]        keys unique within an example
    predict  = the value paired with q, at the position after q

Chance is log2(32) = 5.0 bpc. The partner sits at a uniformly random distance
from q, so no fixed lag can find it; the model must match q's identity and copy
its neighbour. A transformer does this with a copy/induction circuit; a
static-query bag cannot, and `dynamic_query` (q = q0 + s_n W_u, i.e. the query
is a function of the dot's own recurrent state) is the minimal change that lets
it.

Three arms, identical budget and data:
    tf          d96/4h/3l
    iecnn       the 360 s frontier shape (static query)
    iecnn-dyn   the same with dynamic_query=True

Reported separately: bpc on the BINDING position, and bpc on the ordinary
"predict v right after its k" positions (a fixed-lag control both should solve).
If tf < iecnn on binding but equal on the control, the ceiling is retrieval, not
optimisation; if iecnn-dyn closes the binding gap, the root cause is proven.

Run: python3 scripts/binding_probe.py [steps]
CPU-only, no data files.
"""
from __future__ import annotations

import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch                                                    # noqa: E402
import torch.nn.functional as F                                 # noqa: E402

from models.iecnn import build_iecnn                            # noqa: E402
from train.train import build_model                             # noqa: E402

STEPS = int(sys.argv[1]) if len(sys.argv) > 1 else 800
K = 8                     # key/value pairs per example
NKEY = NVAL = 32
V = NKEY + NVAL           # 0..31 keys, 32..63 values
T = 2 * K + 1             # keys+values, then the query token
B = 64
DEV = torch.device("cpu")
torch.set_num_threads(int(os.environ.get("BP_THREADS", "4")))


def make_batch(n, g):
    """x [n, T]: pairs then the query. y: next token, with the last target the
    value paired with the query (the only position that needs binding)."""
    keys = torch.stack([torch.randperm(NKEY, generator=g)[:K] for _ in range(n)])
    vals = torch.randint(0, NVAL, (n, K), generator=g) + NKEY
    pairs = torch.stack([keys, vals], dim=-1).reshape(n, 2 * K)
    which = torch.randint(0, K, (n,), generator=g)
    q = keys[torch.arange(n), which]
    tgt = vals[torch.arange(n), which]
    x = torch.cat([pairs, q[:, None]], dim=1)
    y = torch.cat([x[:, 1:], tgt[:, None]], dim=1)
    return x, y


def cfg_iecnn(dynamic=False, passes=1, carry=False):
    return {
        "model": {
            "kind": "iecnn", "d": 128, "d_q": 16, "d_v": 32, "d_s": 96,
            "n_dots": 3, "n_regions": 1, "n_passes": passes, "spans": [8],
            "use_weights": True, "use_messages": False,
            "pooling": "log_linear", "head_mode": "hidden",
            "read_mode": "soft", "share_pass_params": True,
            "dynamic_query": dynamic, "use_cross_region": False,
            "fast_dot_math": True, "cache_attention": True,
            "cell": "lug", "head_last_only": True, "per_dot_logits": False,
            "window_mode": "contiguous", "seclusion_topk": 0,
            "pool_first": False, "slot_slots": 0, "dot_dist_bias": False,
            "global_read": True, "read_schedule": "mix", "head_untied": False,
            "cell_ln": False, "share_cell": False,
            "query_from_input": False, "id_lags": False, "ngram_bias": 0,
            "head_rank": 0, "state_carry": carry, "carry_chunk": 8,
            "global_linear": False, "head_post": 512,
            "head_post_layers": 3, "interleave": False, "read_ln": True,
            "head_ln": True, "read_heads": 1, "global_decay": True,
            "global_decay_k": 1, "post_share": "none",
        },
        "consensus": {"rho": 0.0, "kappa": 0.7},
        "training": {"compile_consensus": False, "dot_dropout": 0.0},
        "data": {"seq_len": T},
    }


def cfg_tf():
    return {"model": {"kind": "transformer", "d_model": 96, "n_heads": 4,
                      "n_layers": 3},
            "data": {"seq_len": T}}


ARMS = {
    "tf":          dict(tf=True),
    "iecnn":       dict(),                                  # the frontier: static q, 1 pass
    "iecnn-p2":    dict(passes=2),                          # depth only
    "iecnn-dyn-p2": dict(passes=2, dynamic=True),            # q sees pass-1 state
}


def build(name):
    kw = ARMS[name]
    if kw.get("tf"):
        return build_model(cfg_tf(), V)
    return build_iecnn(cfg_iecnn(dynamic=kw.get("dynamic", False),
                                 passes=kw.get("passes", 1),
                                 carry=kw.get("carry", False)), V)


@torch.no_grad()
def evaluate(model, g, n=512):
    """bpc at the binding position, and at the ordinary pair values."""
    model.eval()
    x, y = make_batch(n, g)
    lp = model(x)["final_log_probs"]                        # [n, T, V]
    bind = F.nll_loss(lp[:, -1], y[:, -1], reduction="mean").item()
    # control: every even position 1,3,..,2K-1 predicts a value from its key
    idx = torch.arange(1, 2 * K, 2)
    loc = F.nll_loss(lp[:, idx].reshape(-1, V), y[:, idx].reshape(-1),
                     reduction="mean").item()
    model.train()
    return bind / 0.693147, loc / 0.693147


def run(name, steps=STEPS, lr=2e-3):
    torch.manual_seed(0)
    model = build(name).to(DEV).train()
    n_par = sum(p.numel() for p in model.parameters())
    opt = torch.optim.AdamW(model.parameters(), lr=lr, betas=(0.9, 0.95),
                            weight_decay=0.01)
    g = torch.Generator().manual_seed(123)
    t0 = time.perf_counter()
    curve = []
    for i in range(steps + 1):
        x, y = make_batch(B, g)
        lp = model(x)["final_log_probs"]
        # Ordinary next-token loss everywhere (so the control is a real
        # control) plus a weight on the binding position, which is 1 of 17
        # positions and would otherwise be ignored.
        loss = (F.nll_loss(lp.reshape(-1, V), y.reshape(-1))
                + 4.0 * F.nll_loss(lp[:, -1], y[:, -1]))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        if i % 250 == 0:
            b, l = evaluate(model, g)
            curve.append((i, round(b, 3), round(l, 3)))
            print(f"  {name:<10} step {i:>5}  bind {b:6.3f}  control {l:6.3f}",
                  flush=True)
    el = time.perf_counter() - t0
    b, l = evaluate(model, g)
    print(f"RESULT {name:<10} params {n_par:>8} bind {b:6.3f} control {l:6.3f} "
          f"chance {torch.log2(torch.tensor(float(NVAL))).item():.3f} "
          f"s/step {el/(steps+1):.4f}  curve {curve}", flush=True)
    return b


if __name__ == "__main__":
    print(f"=== BINDING PROBE steps={STEPS} K={K} T={T} V={V} B={B} "
          f"device={DEV} threads={torch.get_num_threads()}", flush=True)
    out = {}
    for name, lr in (("tf", 1e-3), ("iecnn", 2e-3), ("iecnn-p2", 2e-3),
                     ("iecnn-dyn-p2", 2e-3)):
        out[name] = run(name, lr=lr)
    print("PROBE DONE " + " ".join(f"{k}={v:.3f}" for k, v in out.items()),
          flush=True)
