"""What would a GLOBAL copy index be worth on top of a trained base model?

`probe_history` showed the 256-byte window is the binding constraint, not k:

    windowed k=3   23.3% cover  63.0% right  -> 14.7% of positions copied right
    global   k=4   96.9% cover  55.4% right  -> 53.7%

That is a match set 3.6x larger in correct copies, and its extension is exactly
the longest-match signal, so one channel subsumes the k family. But accuracy at
full coverage is only 55%, so the value depends entirely on whether a gate can
tell the good matches from the bad.

This prices that against a TRAINED checkpoint's own base distribution p, with
the checkpoint's copy channel shut. For each candidate index it fits the same
two-parameter gate the model uses, sigmoid(a + b*extension), on one parity half
of the windows and scores the other half, both folds. Also reports the
per-position ORACLE weight (an upper bound no gate can beat) and the accuracy
by extension bucket, which is the gate's actual input.

Usage:
  PYTHONPATH=. python3 scripts/probe_global_copy.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_VAL_OFF / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE and
ARM_HIST_MB. The val file must contain ARM_HIST_MB of history before ARM_VAL_OFF.
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline

V = 256
T = int(os.environ.get("ARM_T", "256"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
VAL_OFF = int(os.environ.get("ARM_VAL_OFF", "0"))
N_VAL = int(os.environ.get("ARM_NVAL", "2_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
HIST_MB = int(os.environ.get("ARM_HIST_MB", "4"))
KS = tuple(int(s) for s in os.environ.get("ARM_GKS", "4,6").split(","))
CAP = 16
GRID_A = [x / 4 for x in range(-28, 9)]        # -7 .. 2
GRID_B = [x / 10 for x in range(0, 16)]        #  0 .. 1.5


def index_one(blob: bytes, k: int, want) -> tuple[np.ndarray, np.ndarray]:
    """Causal global last-occurrence of the k-byte context ending at t.

    Returns (src, ext) at the `want` positions, where src is the most recent
    earlier occurrence (-1 if none) and ext is the bytes agreeing BEFORE the
    k-gram, capped. One sequential pass; the table is global, not windowed.
    """
    last: dict[bytes, int] = {}
    src = np.full(len(want), -1, dtype=np.int64)
    ext = np.zeros(len(want), dtype=np.int64)
    where = {t: i for i, t in enumerate(want)}
    for t in range(k - 1, len(blob) - 1):
        ctx = blob[t - k + 1:t + 1]
        s = last.get(ctx, -1)
        last[ctx] = t
        i = where.get(t)
        if i is None or s < 0:
            continue
        src[i] = s
        e = 0
        while e < CAP:
            a, b = t - k - e, s - k - e
            if a < 0 or b < 0 or blob[a] != blob[b]:
                break
            e += 1
        ext[i] = e
    return src, ext


def bpc_vec(pm, correct, ext, valid, a, b):
    """bpc of (1-g)p + g*q using only per-position p(true), q==true and ext."""
    g = torch.sigmoid(a + b * ext.clamp(max=CAP).to(torch.float32)) * valid
    mix = (1 - g) * pm + g * correct.to(pm.dtype)
    return float(-torch.log2(mix.clamp_min(1e-9)).mean())


def best_fit(pm, correct, ext, valid, idx):
    """Grid-fit (a, b) on `idx`, return (a, b) minimising bpc there."""
    best = (math.inf, 0.0, 0.0)
    for a in GRID_A:
        for b in GRID_B:
            v = bpc_vec(pm[idx], correct[idx], ext[idx], valid[idx], a, b)
            if v < best[0]:
                best = (v, a, b)
    return best


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    with open(VAL_PATH, "rb") as f:
        f.seek(max(0, VAL_OFF - HIST_MB * 1_000_000))
        blob = f.read(HIST_MB * 1_000_000 + N_VAL)
    seed = len(blob) - N_VAL
    starts = [seed + w * T * VALSTRIDE for w in range(VALW)
              if seed + w * T * VALSTRIDE + T + 1 <= len(blob)]
    want = [s + i for s in starts for i in range(T)]
    n = len(want)
    print(f"history {seed} B + val {N_VAL} B; {len(starts)}x{T} = {n} "
          f"positions", flush=True)

    # targets and the model's own window slice
    val = torch.frombuffer(bytearray(blob), dtype=torch.uint8).to(torch.int64)
    y = torch.cat([val[s + 1:s + 1 + T] for s in starts]).to(device)

    ck = torch.load(args.ckpts[0], map_location="cpu", weights_only=False)
    build = (build_transformer_baseline if ck["kind"] == "transformer"
             else build_iecnn)
    model = build(ck["cfg"], V).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    r = model.regions[0] if getattr(model, "regions", None) else None
    print(f"checkpoint {os.path.basename(args.ckpts[0])}  "
          f"channels k={getattr(r, 'suffix_copy', None)}", flush=True)

    def forward_from(start_list):
        with torch.no_grad():
            return torch.cat([model(val[s:s + T].unsqueeze(0).to(device))
                              ["final_log_probs"][0].exp()
                              for s in start_list])

    # base distribution: every gate shut (the checkpoint's own copy channel off)
    saved = None
    if r is not None and getattr(r, "sc_gate", None) is not None:
        saved = (r.sc_gate.detach().clone(),
                 r.sc_slope.detach().clone() if r.sc_slope is not None else None)
        r.sc_gate.data.fill_(-50.0)
        if r.sc_slope is not None:
            r.sc_slope.data.zero_()
    p = forward_from(starts)
    if r is not None and getattr(r, "sc_gate", None) is not None:
        r.sc_gate.data.copy_(saved[0])
        if r.sc_slope is not None and saved[1] is not None:
            r.sc_slope.data.copy_(saved[1])
    live = forward_from(starts)

    def bpc_of(pr):
        return float(-torch.log2(pr.gather(1, y.unsqueeze(1)).squeeze(1)
                                 .clamp_min(1e-9)).mean())

    shut_bpc = bpc_of(p)
    live_bpc = bpc_of(live)
    pm = p.gather(1, y.unsqueeze(1)).squeeze(1).to(torch.float32)
    print(f"  shut {shut_bpc:.4f} | deployed {live_bpc:.4f} "
          f"(gain {shut_bpc - live_bpc:+.4f})", flush=True)

    folds = [torch.tensor([i for i in range(n) if (i // T) % 2 == f])
             for f in (0, 1)]

    for k in KS:
        src, ext = index_one(blob, k, want)
        bn = np.frombuffer(blob, dtype=np.uint8)
        wn = np.asarray(want, dtype=np.int64)
        ok = src >= 0
        copied = np.where(ok, bn[np.clip(src + 1, 0, len(blob) - 1)], 0)
        valid = torch.from_numpy(ok).to(device)
        # position t predicts t+1 -- comparing to bn[t] is the bug that read
        # as 5.5% accuracy on the first run
        correct = torch.from_numpy(ok & (copied == bn[np.clip(wn + 1, 0,
                                                             len(blob) - 1)]))
        ext_t = torch.from_numpy(ext).to(device).to(torch.float32)
        out = []
        for f in (0, 1):
            fit, score = folds[f], folds[1 - f]
            _v, a, b = best_fit(pm, correct, ext_t, valid, fit)
            out.append((bpc_vec(pm[score], correct[score], ext_t[score],
                                valid[score], a, b), a, b))
        print(f"\n  GLOBAL k={k}: cover {100 * float(valid.float().mean()):.1f}%"
              f"  acc {100 * float(correct[valid].float().mean()):.1f}%"
              f"  (deployed: 23.3% / 63.0%)", flush=True)
        for f, (sc, a, b) in enumerate(out):
            print(f"    fold {f}: held-out {sc:.4f} (a={a:+.2f} b={b:+.2f})"
                  f"  d_vs_shut {sc - shut_bpc:+.4f}"
                  f"  d_vs_deployed {sc - live_bpc:+.4f}", flush=True)
        # accuracy by extension bucket -- the gate's input must separate them
        print("    ext bucket: ", end="", flush=True)
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 7), (8, CAP)):
            m = valid & (ext_t >= lo) & (ext_t <= hi if lo != hi else
                                         ext_t == lo)
            if bool(m.any()):
                print(f"[{lo}-{hi}] {100 * float(correct[m].float().mean()):.0f}%"
                      f" n={int(m.sum())}", end="  ", flush=True)
        print("", flush=True)
        # oracle: per-position best weight between base and this copy
        orc = float(-torch.log2(torch.maximum(pm, correct.to(pm.dtype))
                                .clamp_min(1e-9)).mean())
        print(f"    per-position oracle {orc:.4f}"
              f"  (d_vs_deployed {orc - live_bpc:+.4f})", flush=True)


if __name__ == "__main__":
    main()
