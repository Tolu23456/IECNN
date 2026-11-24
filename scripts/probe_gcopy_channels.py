"""What did a trained GLOBAL copy arm actually learn?

`probe_gate_channels.py` does this for the windowed channels. The global arm is
different in one decisive way: coverage is no longer the constraint (96.9% vs
23.3%), so the questions are

  * did the gate OPEN, and to what level (a) and extension slope (b)?
  * `shut` vs `deployed`: what is the channel worth inside the trained model,
    i.e. with co-adaptation, not against a frozen base?
  * accuracy by extension bucket -- the gate's own input must separate good
    matches from bad ones, or the mechanism is a constant-weight prior;
  * the per-position ORACLE (max of base and copy probability): the ceiling a
    perfect gate would reach, so "how much of the ceiling was captured" has a
    meaning.

It rebuilds the index from the corpus with the same k/cap/history convention as
training (val blob = train[-HIST:] + val), and it asserts that the rebuilt val
byte at a probe position matches a brute-force search -- an alignment bug here
would silently score the wrong pointer and look like a mechanism result.

Usage:
  PYTHONPATH=. python3 scripts/probe_gcopy_channels.py ckpt.pt [...]
Reads ARM_TRAIN / ARM_VAL / ARM_NTRAIN / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE /
ARM_GCOPY_HIST / ARM_GCOPY_K (defaults: the canonical 90 MB / 5 MB split).
"""
from __future__ import annotations

import argparse
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.iecnn import build_iecnn
from scripts.build_copy_index import build

V = 256
T = int(os.environ.get("ARM_T", "256"))
TRAIN_PATH = os.environ.get("ARM_TRAIN", "data/toy_train.txt")
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_TRAIN = int(os.environ.get("ARM_NTRAIN", "90_000_000"))
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
HIST = int(os.environ.get("ARM_GCOPY_HIST", "4_000_000"))
K = int(os.environ.get("ARM_GCOPY_K", "4"))
CAP = 16
CHUNK = 20


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)

    tr = np.fromfile(TRAIN_PATH, dtype=np.uint8, count=N_TRAIN)
    va = np.fromfile(VAL_PATH, dtype=np.uint8, count=N_VAL)
    hist = min(HIST, N_TRAIN)
    blob = np.concatenate([tr[N_TRAIN - hist:N_TRAIN], va])
    del tr, va

    starts = [hist + w * T * VALSTRIDE for w in range(VALW)
              if hist + w * T * VALSTRIDE + T + 1 <= len(blob)]
    n_pos = len(starts) * T
    print(f"val {len(starts)}x{T} = {n_pos} positions, hist {hist}, "
          f"k {K}", flush=True)

    cb, ce = build(blob, K, CAP)
    cb_t = torch.from_numpy(cb)
    ce_t = torch.from_numpy(ce)

    # alignment: one position must match a brute-force search over the blob
    i = 1000
    t = hist + i
    ctx = blob[t - K + 1:t + 1].tobytes()
    s = None
    for j in range(K - 1, t):
        if blob[j - K + 1:j + 1].tobytes() == ctx:
            s = j
    if s is not None:
        assert int(cb[t]) == int(blob[s + 1]), "val index is misaligned"
    print(f"align check ok (pos {t}: src {s})", flush=True)

    x_all = torch.stack([torch.from_numpy(blob[o:o + T].copy())
                         for o in starts]).long()
    y_all = torch.stack([torch.from_numpy(blob[o + 1:o + 1 + T].copy())
                         for o in starts]).long()
    cpy_b = torch.stack([cb_t[o:o + T] for o in starts])
    cpy_e = torch.stack([ce_t[o:o + T] for o in starts])
    valid = cpy_e != 255
    ext = cpy_e.long().clamp(max=CAP)
    true_byte = y_all
    correct_cpu = valid & (cpy_b == y_all)

    print(f"\ncoverage {100 * float(valid.float().mean()):.1f}%  "
          f"copy-right {100 * float(((cpy_b == y_all) & valid).float().mean()):.2f}% "
          f"of all positions", flush=True)
    for ckpt in args.ckpts:
        ck = torch.load(ckpt, map_location="cpu", weights_only=False)
        model = build_iecnn(ck["cfg"], V).to(dev)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        r = model.regions[0]
        a = float(r.sc_gate) if r.sc_gate is not None else float("nan")
        b = (float(r.sc_slope) if getattr(r, "sc_slope", None) is not None
             else 0.0)
        print(f"\n===== {os.path.basename(ckpt)}\n"
              f"  gate a={a:+.4f} b={b:+.4f}  sigmoid(a)={torch.sigmoid(torch.tensor(a)):.4f}"
              f"  global={getattr(r, 'suffix_copy_global', None)} "
              f"k={getattr(r, 'suffix_copy', None)}", flush=True)

        shut, live = [], []
        with torch.no_grad():
            for c0 in range(0, len(starts), CHUNK):
                sl = slice(c0, c0 + CHUNK)
                lp0 = model(x_all[sl].to(dev), copy=None)["final_log_probs"]
                lp1 = model(x_all[sl].to(dev),
                            copy=(cpy_b[sl].to(dev), cpy_e[sl].to(dev))
                            )["final_log_probs"]
                shut.append(lp0)
                live.append(lp1)
        shut = torch.cat(shut)
        live = torch.cat(live)

        def bpc(lp):
            return float(-lp.gather(2, true_byte.unsqueeze(-1).to(dev))
                         .clamp_min(1e-9).log2().mean())

        print(f"  shut {bpc(shut):.4f} | deployed {bpc(live):.4f} "
              f"(gain {bpc(shut) - bpc(live):+.4f})", flush=True)

        p = shut.exp()
        pm = p.gather(2, true_byte.unsqueeze(-1).to(dev)).squeeze(-1)
        correct = (cpy_b.to(dev) == true_byte.to(dev))
        orc = float(-torch.maximum(pm, correct.to(pm.dtype))
                    .clamp_min(1e-9).log2().mean())
        print(f"  per-position oracle {orc:.4f} "
              f"(channel captured {bpc(shut) - bpc(live):.4f} of "
              f"{bpc(shut) - orc:+.4f})", flush=True)

        print(f"  {'ext':>6}{'share':>8}{'acc':>8}{'gate':>8}", flush=True)
        ext8 = ext.float().clamp(max=8)
        g_all = torch.sigmoid(a + b * ext8) * valid.float()
        for lo, hi in ((0, 0), (1, 1), (2, 3), (4, 7), (8, CAP)):
            m = valid & (ext >= lo) & (ext <= hi)
            n = int(m.sum())
            if not n:
                continue
            acc = float(correct_cpu[m].float().mean())
            gmean = float(g_all[m].mean())
            print(f"  {lo:>3}-{hi:<2}{100 * n / (len(cpy_b) * T):>7.1f}%"
                  f"{100 * acc:>7.1f}%{gmean:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
