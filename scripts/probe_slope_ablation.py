"""Causal control for the match-extension slope, on the TRAINED model.

The two-seed 900 s batch shows `sc3e` beats `sc3` by 0.0120, but a batch-level
difference and a mechanism are not the same claim. This holds every parameter of
a trained `sc3e` checkpoint fixed and varies only the slope coefficient b at
eval, on the same 160 windows the harness uses:

    gate = sigmoid(a + b * extension) * valid

  * b = 0 is the arm's own scalar-gate twin (same weights, same gate level a);
  * b = learned is the checkpoint as trained (must reproduce its bpc -- the
    faithfulness check);
  * a sweep over b, and a 2-D sweep over (a, b), says whether the learned slope
    is in the right place or the gain is really coming from gate level.

The base distribution p is obtained from the SAME checkpoint with the gate
forced shut (a = -50, b = 0), where the mixture is the identity, so the analytic
recomposition below is exact rather than an approximation of the model.

The channel (copy, valid, extension) is taken from the model's own
`_suffix_copy`, not reimplemented, so the probe cannot drift from the mechanism.

Usage: PYTHONPATH=. python3 scripts/probe_slope_ablation.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE.
"""
from __future__ import annotations

import argparse
import math
import os

import torch
import torch.nn.functional as F

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline

V = 256
T = int(os.environ.get("ARM_T", "256"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))


def bpc_from(p_true, mass_true, valid, a, b, extra, cap):
    """Analytic bpc of the trained mixture for a given (a, b)."""
    if b is None:
        logit = torch.full_like(extra, a, dtype=torch.float32)
    else:
        logit = a + b * extra.clamp(max=cap).to(torch.float32)
    g = torch.sigmoid(logit) * valid.to(torch.float32)
    mix = (1 - g) * p_true + g * mass_true
    # the model logs a mixture clamped at 1e-9; match it exactly so the
    # faithfulness check below compares the same quantity
    return float(-torch.log2(mix.clamp_min(1e-9)).mean())


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
    starts = [w * T * VALSTRIDE for w in range(VALW)
              if w * T * VALSTRIDE + T + 1 <= val.numel()]
    print(f"val {len(starts)}x{T} = {len(starts) * T} positions | "
          f"stride {VALSTRIDE} | {len(raw)} bytes", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        name = os.path.basename(path).replace(".pt", "")
        r = model.regions[0] if getattr(model, "regions", None) else None
        if r is None or getattr(r, "sc_gate", None) is None:
            print(f"\n{name}\n  no copy channel (base only)", flush=True)
            continue

        a = float(r.sc_gate.detach())
        b = (float(r.sc_slope.detach()) if r.sc_slope is not None else None)
        cap = int(r.sc_extra_cap)
        print(f"\n{name}\n  learned gate a={a:.4f}  slope b="
              f"{'none (scalar arm)' if b is None else f'{b:.4f}'}", flush=True)

        # channel straight from the model
        copies, valids, extras = [], [], []
        with torch.no_grad():
            for i in starts:
                x = val[i:i + T].unsqueeze(0).to(device)
                c, v, e = r._suffix_copy(x, with_extra=True)
                copies.append(c[0]); valids.append(v[0]); extras.append(e[0])
        copy = torch.cat(copies)
        valid = torch.cat(valids)
        extra = torch.cat(extras).to(torch.int64)
        y = torch.cat([val[i + 1:i + 1 + T] for i in starts]).to(device)
        print(f"  channel coverage {100 * valid.float().mean():.1f}% | copy "
              f"accuracy {100 * (copy == y)[valid].float().mean():.1f}% | "
              f"extra mean {float(extra[valid].float().mean()):.2f}", flush=True)

        # reference: the checkpoint as trained
        with torch.no_grad():
            lp = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                            ["final_log_probs"][0] for i in starts])
        model_bpc = float(F.nll_loss(lp, y, reduction="mean") / math.log(2))

        # base p: same weights, gate forced shut (identity mixture)
        old_gate, old_slope = r.sc_gate.detach().clone(), None
        if r.sc_slope is not None:
            old_slope = r.sc_slope.detach().clone()
            r.sc_slope.data.zero_()
        r.sc_gate.data.fill_(-50.0)
        with torch.no_grad():
            p = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                           ["final_log_probs"][0].exp() for i in starts])
        r.sc_gate.data.copy_(old_gate)
        if old_slope is not None:
            r.sc_slope.data.copy_(old_slope)

        p_true = p.gather(1, y.unsqueeze(1)).squeeze(1)
        mass = (copy == y).to(torch.float32)
        shut = bpc_from(p_true, mass, valid, -50.0, 0.0, extra, cap)
        recon = bpc_from(p_true, mass, valid, a, b, extra, cap)
        # faithfulness: the analytic recomposition must reproduce the model
        assert abs(recon - model_bpc) < 2e-3, (recon, model_bpc)
        print(f"  bpc: model {model_bpc:.4f} | analytic {recon:.4f} | "
              f"gate shut {shut:.4f}  (channel value {shut - recon:+.4f})",
              flush=True)

        if b is not None:
            ablate = bpc_from(p_true, mass, valid, a, 0.0, extra, cap)
            print(f"  slope ablated (b=0, same a): {ablate:.4f} | "
                  f"slope delivers {ablate - recon:+.4f}", flush=True)
            print(f"  {'b':>7}{'bpc':>10}{'vs learned':>12}", flush=True)
            best = (b, recon)
            for bb in (-2.0, -1.0, -0.5, -0.25, 0.0, 0.25, 0.5, 0.75,
                       1.0, 1.5, 2.0, 3.0):
                v = bpc_from(p_true, mass, valid, a, bb, extra, cap)
                print(f"  {bb:>7.2f}{v:>10.4f}{v - recon:>+12.4f}", flush=True)
                if v < best[1]:
                    best = (bb, v)
            # 2-D: is the learned gate LEVEL right, or is the slope standing in
            # for a level the arm failed to set?
            for aa in (-2.0, -1.0, 0.0, a, 1.0, 2.0):
                row = []
                for bb in (-1.0, 0.0, 0.5, b, 1.0, 2.0):
                    row.append(f"{bpc_from(p_true, mass, valid, aa, bb, extra, cap):.4f}")
                print(f"  a={aa:>6.2f}  b=-1,0,.5,learned,1,2 -> "
                      + " ".join(row), flush=True)

        print(f"  {'extra':>7}{'n':>7}{'copy acc':>10}{'gate':>8}", flush=True)
        for e in range(cap + 1):
            m = valid & (extra == e)
            if not bool(m.any()):
                continue
            gt = torch.sigmoid(torch.tensor(a + (0.0 if b is None else b * e)))
            print(f"  {e:>7}{int(m.sum()):>7}"
                  f"{100 * float((copy == y)[m].float().mean()):>9.1f}%"
                  f"{float(gt):>8.3f}", flush=True)


if __name__ == "__main__":
    main()
