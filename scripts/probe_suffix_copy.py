"""Is the suffix_copy gain from MATCHING, or just from adding a peaked prior?

A copy channel can improve bpc for a boring reason: mixing any peaked
distribution into a flat one lowers cross-entropy at init (the "safety net"
effect), and a learned gate can then keep whatever is left. That would be a
dead end that looks like a win. The distinguishing test is the pointer: keep
the mixture, the gate and every parameter identical, and only break the
MATCH -- replace each pointer with a uniformly random earlier position.

    true       pointers as trained            (the arm)
    scrambled  same distribution, wrong match  (the control)
    off        gate forced shut               (the baseline)

If the arm is only a peaked prior, scrambled ~= true and both are better than
off. If it is really induction, true << scrambled ~= off.

Reported overall and per induction bucket (the same buckets as
scripts/probe_error_by_distance.py), plus the learned gate, so "it helps" and
"it helps where the band is" are separate claims.

Usage:
  PYTHONPATH=. python3 scripts/probe_suffix_copy.py ckpt.pt
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE like ab_small.py.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline
from scripts.probe_error_by_distance import (IND_BUCKETS, induction_support,
                                             copy_distance)

V = 256
T = int(os.environ.get("ARM_T", "256"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", 5_000_000))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "1"))


def window_starts(val_len: int, n: int):
    out = []
    for w in range(n):
        i = w * T * VALSTRIDE
        if i + T + 1 > val_len:
            break
        out.append(i)
    return out


@torch.no_grad()
def per_position_nll(model, val, starts, device):
    out = []
    for i in starts:
        x = val[i:i + T].unsqueeze(0).to(device)
        y = val[i + 1:i + 1 + T].unsqueeze(0).to(device)
        lp = model(x)["final_log_probs"].reshape(-1, V)
        out.append(F.nll_loss(lp, y.reshape(-1), reduction="none"))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--device", default="cpu")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    device = torch.device(args.device)

    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    model = build_iecnn(ck["cfg"], V).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    r = model.regions[0]
    if getattr(r, "sc_gate", None) is None:
        raise SystemExit("checkpoint has no suffix_copy channel")
    gate = float(torch.sigmoid(r.sc_gate))
    print(f"{os.path.basename(args.ckpt)}: gate {gate:.4f} "
          f"(sigmoid {float(r.sc_gate):+.3f}), row bpc "
          f"{ck.get('row', {}).get('bpc_final', 'n/a')}", flush=True)

    raw = np.frombuffer(open(VAL_PATH, "rb").read(N_VAL), dtype=np.uint8)
    val = torch.from_numpy(raw.astype(np.int64))
    starts = window_starts(raw.shape[0], VALW)
    tgt = np.concatenate([np.arange(i + 1, i + 1 + T) for i in starts])
    pref = raw[: int(tgt[-1]) + 1]
    ind, _, _ = induction_support(pref)
    ind = ind[tgt]

    real = r._suffix_copy

    def scrambled(ids):
        """Same mixture, same gate, but the pointer is a random earlier
        position -- the copy prior survives, the induction does not."""
        copy_ids, valid = real(ids)
        g = torch.Generator(device="cpu").manual_seed(args.seed)
        b, t = ids.shape
        # uniform in [0, t-1] per position, so the pointer is always observed
        pos = (torch.rand(b, t, generator=g) * t).long()
        pos = torch.minimum(pos, torch.arange(t) - 1).clamp_min(0)
        return ids.gather(1, pos.to(ids.device)), valid

    runs = {}
    for name, fn, gate_val in (("true", real, None),
                               ("scrambled", scrambled, None),
                               ("off", real, -60.0)):
        r._suffix_copy = fn
        if gate_val is not None:
            saved = r.sc_gate.data.clone()
            r.sc_gate.data.fill_(gate_val)
        nll = per_position_nll(model, val, starts, device).numpy()
        if gate_val is not None:
            r.sc_gate.data.copy_(saved)
        runs[name] = nll
        print(f"  {name:<10}{nll.mean() / math.log(2):.4f} bpc", flush=True)
    r._suffix_copy = real

    print(f"\n{'bucket':>18}{'n':>8}{'true':>9}{'scram':>9}{'off':>9}"
          f"{'true-scram':>12}{'true-off':>10}")
    for b, name in enumerate(IND_BUCKETS):
        m = ind == b
        if not m.any():
            continue
        line = f"{name:>18}{int(m.sum()):>8}"
        for k in ("true", "scrambled", "off"):
            line += f"{runs[k][m].mean() / math.log(2):>9.4f}"
        line += (f"{(runs['true'][m].mean() - runs['scrambled'][m].mean()) / math.log(2):>+12.4f}"
                 f"{(runs['true'][m].mean() - runs['off'][m].mean()) / math.log(2):>+10.4f}")
        print(line, flush=True)

    print(f"\n  overall: true - scrambled = "
          f"{(runs['true'].mean() - runs['scrambled'].mean()) / math.log(2):+.4f}"
          f" bpc | true - off = "
          f"{(runs['true'].mean() - runs['off'].mean()) / math.log(2):+.4f} bpc",
          flush=True)


if __name__ == "__main__":
    main()
