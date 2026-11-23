"""Do PARALLEL small-k copy channels beat one k=3 channel?

The k sweep showed the tradeoff the user named: the more context a match
requires, the more precise it is and the fewer positions it covers (k=2: 44.7%
coverage / 41.7% right; k=3: 23.3% / 63.0%; k=4: 15.8% / 71.2%). The obvious
move is to run several channels at once and let the mixture use each where it
is informative: coverage becomes the UNION, and precision is bought per-channel.

That is NOT the same as `id_lags` / `k_lag` (closed lanes): those shift the
key or the query alignment of ONE channel. This runs several k values as
independent channels with independent weights, which was never measured.

Mixing is a proper convex mixture, not sequential gates (the order of
sequential mixes would otherwise become a free parameter):

    p_final = (1 - sum w_i) * p + sum_i w_i * q_i        w_i >= 0, sum <= 0.7

Each q_i is one-hot on the follower of the most recent exact i-gram match, so
q_i is only non-zero where that i-gram occurred earlier (`valid_i`).

Reported: each channel's own best constant weight (the single-channel ceiling),
the best JOINT weights over a 3-D grid, and the increment of the joint optimum
over the best single channel. A joint gain that does not clear +0.003 over
0.0563 is not worth an arm: it is below the campaign's resolution and would
have to be earned back through training dynamics.

Usage: PYTHONPATH=. python3 scripts/probe_copy_multik.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_EXTRA_CAP.
"""
from __future__ import annotations

import argparse
import math
import os

import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline
from scripts.probe_copy_k import suffix_k, bpc

V = 256
T = int(os.environ.get("ARM_T", "256"))
CAP = int(os.environ.get("COPY_EXTRA_CAP", "8"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
KS = (2, 3, 4)
W2 = [i * 0.02 for i in range(0, 11)]        # 0 .. 0.20
W3 = [i * 0.05 for i in range(0, 11)]        # 0 .. 0.50
W4 = [i * 0.05 for i in range(0, 9)]         # 0 .. 0.40


def joint(p_true, mass, valid, wts) -> float:
    """Convex mixture with per-position validity.

    A channel contributes w_i * valid_i, so where an i-gram has no earlier
    occurrence the channel contributes nothing and the base keeps that mass.
    The first version forgot the `* valid` and instead zeroed the mass, which
    made weight 0 look optimal everywhere -- a plausible-looking null produced
    by a bug, so the validity is now explicit in the formula.
    """
    share = wts[0] * valid[0] + wts[1] * valid[1] + wts[2] * valid[2]
    mix = (1 - share) * p_true
    for w, m, v in zip(wts, mass, valid):
        mix = mix + (w * v) * m
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
    y = torch.cat([val[i + 1:i + 1 + T] for i in starts]).to(device)
    print(f"val {len(starts)}x{T} = {len(starts) * T} positions | stride "
          f"{VALSTRIDE} | k {KS}", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        name = os.path.basename(path).replace(".pt", "")
        r = model.regions[0] if getattr(model, "regions", None) else None
        print(f"\n===== {name}", flush=True)
        if r is None or getattr(r, "sc_gate", None) is None:
            print("  no copy channel (base only)", flush=True)
            continue

        saved_a = r.sc_gate.detach().clone()
        saved_b = r.sc_slope.detach().clone() if r.sc_slope is not None else None
        if r.sc_slope is not None:
            r.sc_slope.data.zero_()
        r.sc_gate.data.fill_(-50.0)
        with torch.no_grad():
            p = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                           ["final_log_probs"][0].exp() for i in starts])
        r.sc_gate.data.copy_(saved_a)
        if saved_b is not None:
            r.sc_slope.data.copy_(saved_b)
        p_true = p.gather(1, y.unsqueeze(1)).squeeze(1)

        mass, valid, cov = [], [], {}
        for k in KS:
            cp, va = [], []
            with torch.no_grad():
                for i in starts:
                    c, v, _e = suffix_k(val[i:i + T].unsqueeze(0).to(device),
                                        k, CAP)
                    cp.append(c[0]); va.append(v[0])
            c, v = torch.cat(cp), torch.cat(va)
            mass.append((c == y).to(torch.float32))
            valid.append(v.to(torch.float32))
            cov[k] = (float(v.float().mean()), float((c == y)[v].float().mean()))

        print(f"  {'k':>3}{'cover':>8}{'acc':>7}{'own best w':>12}"
              f"{'vs k3':>10}{'joint w':>10}", flush=True)
        own = []
        for k, m, v in zip(KS, mass, valid):
            best = min(((bpc(p_true, m, g * v), g)
                        for g in [i / 40 for i in range(0, 33)]))
            own.append(best)
        best_joint = (math.inf, (0.0, 0.0, 0.0))
        for w2 in W2:
            for w3 in W3:
                if w2 + w3 > 0.8:
                    continue
                for w4 in W4:
                    if w2 + w3 + w4 > 0.8:
                        continue
                    j = joint(p_true, mass, valid, (w2, w3, w4))
                    if j < best_joint[0]:
                        best_joint = (j, (w2, w3, w4))
        for k, (b, g) in zip(KS, own):
            cvg, acc = cov[k]
            print(f"  {k:>3}{100 * cvg:>7.1f}%{100 * acc:>6.1f}%"
                  f"{g:>12.3f}{b - own[1][0]:>+10.4f}"
                  f"{best_joint[1][KS.index(k)]:>10.2f}", flush=True)
        print(f"  best single (k=3) {own[1][0]:.4f} at w={own[1][1]:.3f} | "
              f"joint {best_joint[0]:.4f} at w{k}= "
              f"{tuple(round(w, 2) for w in best_joint[1])} | "
              f"joint over single {own[1][0] - best_joint[0]:+.4f}", flush=True)


if __name__ == "__main__":
    main()
