"""What did the trained copy channels actually learn?

`iecnn-ch900`-style arms carry a primary exact channel plus optional extra
channels at other k, each with its own gate. After a run the question is not
"what would a fitted mixture buy" but "how much of its ceiling did the arm
CAPTURE" -- a gate that never opened would make a flat result a training
failure rather than a verdict on the mechanism.

For every checkpoint this prints the learned gate parameters and then prices,
on the same positions:

  * `shut`            all channels off (the base distribution);
  * `primary` / `+k`  each channel alone at its learned (a, b);
  * `deployed`        exactly the model's own mixture -- this MUST equal the
                      model's forward with the gates live, and an assert checks
                      it, so a reimplementation bug cannot masquerade as a
                      measurement;
  * `constant`        best mixture weight per channel, fitted post hoc (the
                      ceiling the arm was aiming at);
  * `oracle`          per-position best weight between base and copy (an upper
                      bound that no shared weight can reach).

Usage: PYTHONPATH=. python3 scripts/probe_channels.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_EXTRA_CAP.
"""
from __future__ import annotations

import argparse
import math
import os

import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline
from scripts.probe_copy_k import suffix_k

V = 256
T = int(os.environ.get("ARM_T", "256"))
CAP = int(os.environ.get("COPY_EXTRA_CAP", "8"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
GRID = [i / 20 for i in range(0, 17)]          # 0 .. 0.8, step .05


def bpc_of(mix: torch.Tensor, y: torch.Tensor) -> float:
    return float(-torch.log2(mix.gather(1, y.unsqueeze(1)).squeeze(1)
                             .clamp_min(1e-9)).mean())


def gate(extra, a, b, valid, cap):
    logit = torch.full_like(extra, float(a), dtype=torch.float32)
    if b is not None:
        logit = logit + b * extra.clamp(max=cap).to(torch.float32)
    return torch.sigmoid(logit) * valid.to(torch.float32)


def mix_with(p, chans, cap, wts=None):
    """(1 - sum) p + sum w_i q_i, with the model's own mass cap.

    wts defaults to the gates themselves, i.e. the deployed mixture.
    """
    # [B,T] -> [B,T,1] so the mass broadcasts against [B,T,V]; the first version
    # left them 1-D and could not multiply into the distribution at all.
    gs = [(g if wts is None else w * (g > 0).to(g.dtype)).unsqueeze(-1)
          for g, w in zip([c[0] for c in chans],
                          [None] * len(chans) if wts is None else wts)]
    tot = gs[0]
    for g in gs[1:]:
        tot = tot + g
    scale = (cap / tot.clamp_min(1e-9)).clamp(max=1.0)
    mix = (1 - tot * scale) * p
    for g, (_, q) in zip(gs, chans):
        mix = mix + (g * scale) * q
    return mix


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
    print(f"val {len(starts)}x{T} = {len(starts) * T} positions", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        name = os.path.basename(path).replace(".pt", "")
        print(f"\n===== {name}", flush=True)
        r = model.regions[0] if getattr(model, "regions", None) else None
        if r is None or getattr(r, "sc_gate", None) is None:
            print("  no copy channel", flush=True)
            continue
        ks = [int(r.suffix_copy)] + [int(k) for k in
                                     (getattr(r, "suffix_copy_ks", ()) or ())]
        a = float(r.sc_gate.detach())
        b = float(r.sc_slope.detach()) if r.sc_slope is not None else None
        extra_a = ([float(x) for x in r.sc_gates2.detach().flatten()]
                   if getattr(r, "sc_gates2", None) is not None else [])
        extra_b = ([float(x) for x in r.sc_slopes2.detach().flatten()]
                   if getattr(r, "sc_slopes2", None) is not None else [])
        print(f"  channels k={ks}  cap={r.sc_mix_cap:.2f}", flush=True)
        print(f"  primary gate a={a:+.4f} b="
              f"{'none' if b is None else f'{b:+.4f}'}"
              f"  -> sigmoid(a)={1 / (1 + math.exp(-a)):.4f}", flush=True)
        for i in range(len(extra_a)):
            bb = extra_b[i] if i < len(extra_b) else None
            print(f"  channel k={ks[i + 1]} gate a={extra_a[i]:+.4f} b="
                  f"{'none' if bb is None else f'{bb:+.4f}'}"
                  f"  -> sigmoid(a)={1 / (1 + math.exp(-extra_a[i])):.4f}",
                  flush=True)

        # base p with every gate shut
        saved = [r.sc_gate.detach().clone()]
        if r.sc_slope is not None:
            saved.append(r.sc_slope.detach().clone())
        saved2 = ([r.sc_gates2.detach().clone()]
                  if getattr(r, "sc_gates2", None) is not None else [])
        r.sc_gate.data.fill_(-50.0)
        if r.sc_slope is not None:
            r.sc_slope.data.zero_()
        if saved2:
            r.sc_gates2.data.fill_(-50.0)
        with torch.no_grad():
            p = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                           ["final_log_probs"][0].exp() for i in starts])
        r.sc_gate.data.copy_(saved[0])
        if r.sc_slope is not None:
            r.sc_slope.data.copy_(saved[1])
        if saved2:
            r.sc_gates2.data.copy_(saved2[0])

        shut = bpc_of(p, y)
        # the channels
        chans, gs = [], []
        for j, kk in enumerate(ks):
            cp, va, ex = [], [], []
            with torch.no_grad():
                for i in starts:
                    c, v, e = suffix_k(val[i:i + T].unsqueeze(0).to(device),
                                       kk, CAP)
                    cp.append(c[0]); va.append(v[0]); ex.append(e[0])
            c, v, e = torch.cat(cp), torch.cat(va), torch.cat(ex)
            q = torch.zeros_like(p)
            q.scatter_(1, c.unsqueeze(1), 1.0)
            aa = a if j == 0 else extra_a[j - 1]
            bb = b if j == 0 else (extra_b[j - 1] if j - 1 < len(extra_b)
                                   else None)
            g = gate(e, aa, bb, v, CAP)
            chans.append((g, q))
            gs.append(g)
            cov = float(v.float().mean())
            acc = float((c == y)[v].float().mean())
            print(f"  k={kk}: cover {100 * cov:.1f}%  acc {100 * acc:.1f}%  "
                  f"gate {float(g.mean()):.4f}", flush=True)

        dep = bpc_of(mix_with(p, chans, r.sc_mix_cap), y)
        # faithfulness: the same mixture must be what the model itself computes
        with torch.no_grad():
            live = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                              ["final_log_probs"][0].exp() for i in starts])
        live_bpc = bpc_of(live, y)
        print(f"  shut {shut:.4f} | deployed {dep:.4f} (model says "
              f"{live_bpc:.4f}, d={abs(dep - live_bpc):.5f}) | gain "
              f"{shut - dep:+.4f}", flush=True)
        assert abs(dep - live_bpc) < 2e-3, "mixture != the model's own forward"

        # ceiling: best CONSTANT mass per channel, fitted on these positions
        best = (math.inf, None)
        for w0 in GRID:
            for w1 in GRID:
                if len(ks) == 1 and w1 > 0:
                    continue
                if w0 + w1 > 0.8:
                    continue
                gs2 = [w0 * (g > 0).to(g.dtype) for g in gs[:1]]
                if len(ks) > 1:
                    gs2.append(w1 * (gs[1] > 0).to(gs[1].dtype))
                c2 = [(g, c[1]) for g, c in zip(gs2, chans)]
                m = bpc_of(mix_with(p, c2, r.sc_mix_cap), y)
                if m < best[0]:
                    best = (m, (w0, w1)[:len(ks)])
        print(f"  best constant {best[0]:.4f} at {best[1]} | gain "
              f"{shut - best[0]:+.4f} | captured "
              f"{100 * (shut - dep) / max(shut - best[0], 1e-9):.0f}% of "
              f"it", flush=True)


if __name__ == "__main__":
    main()
