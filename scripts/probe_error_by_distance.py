"""WHERE is the remaining gap? Bucket the per-position loss by copy distance.

The horizon arms (tf-swa-w3/w8/w32) ask the question from the transformer side:
how much does the baseline lose when its exact access is shortened? This asks it
from the DATA side, on the two models we already have at 4.25B tokens:

    for every val position, how far back is the most recent occurrence of the
    byte that has to be predicted?

That distance is computable from the corpus alone (an oracle feature -- this is
analysis, not a model input), and it separates two stories that predict
different things:

  * if tf's advantage is concentrated on positions whose answer is a DISTANT
    repeat (d > 8, beyond the dot model's exact window), then the gap is the
    read horizon, and `x5-w16` / `x5-w32` are the cheap fix;
  * if tf wins uniformly, including on positions where the answer is the very
    next byte's repeat (d = 1, comfortably inside every read), then the gap is
    not horizon at all -- it is per-token capacity / interaction, and read work
    is the wrong spend.

A second axis: INDUCTION SUPPORT. If the last 3 bytes also occurred earlier,
the byte that followed that earlier occurrence is a copy candidate -- exactly
what an induction head computes. Positions where that guess is right and FAR
away are reachable only by exact long-range copying, so the gap there is the
part of the deficit that read horizon can explain.

(A "longest seen suffix" axis was tried first and thrown away: 95% of enwik8
positions have some long match, so it put four fifths of the corpus in one
bucket and separated nothing.)

Usage (CPU is fine, ~20k positions x 2 models):
  PYTHONPATH=. python3 scripts/probe_error_by_distance.py ckpt_a.pt ckpt_b.pt

Reads the same ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE knobs as
scripts/ab_small.py so the val slice matches the run being audited.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import numpy as np
import torch
import torch.nn.functional as F

from models.baselines import build_transformer_baseline
from models.iecnn import build_iecnn

V = 256
T = int(os.environ.get("ARM_T", "256"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", 5_000_000))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "1"))
MAXD = 1 << 14                      # clamp the copy-distance search window

DIST_EDGES = [0, 1, 2, 4, 8, 16, 64, 256, MAXD + 1]
# induction buckets, in reporting order (not a numeric axis)
IND_BUCKETS = ["no earlier ctx", "wrong guess d<=8", "wrong guess d>8",
               "right d<=8", "right d 9-64", "right d>64"]


def load_model(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, kind = ck["cfg"], ck["kind"]
    torch.manual_seed(0)
    build = build_transformer_baseline if kind == "transformer" else build_iecnn
    model = build(cfg, V).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def window_starts(val_len: int, n: int):
    """Same window enumeration as ab_small/ probe_population, so the slice is
    the one the reported bpc is computed on."""
    out = []
    for w in range(n):
        i = w * T * VALSTRIDE
        if i + T + 1 > val_len:
            break
        out.append(i)
    return out


def copy_distance(raw: np.ndarray) -> np.ndarray:
    """For each position p, distance back to the previous occurrence of raw[p].

    d = p - s where s is the largest index < p with raw[s] == raw[p]; 0 if the
    byte had never occurred before (so 0 is the "no evidence at all" bin).
    One pass, O(n), via last-seen-per-byte.
    """
    n = raw.shape[0]
    out = np.zeros(n, dtype=np.int32)
    last = np.full(256, -1, dtype=np.int64)
    for p in range(n):
        b = int(raw[p])
        s = last[b]
        if s >= 0:
            d = p - int(s)
            out[p] = d if d <= MAXD else MAXD
        last[b] = p
    return out


def induction_support(raw: np.ndarray, k: int = 3):
    """The induction test, computed causally: if the last k bytes before the
    target also occurred earlier, what followed that earlier occurrence is a
    candidate answer. Returns (bucket, distance, guess_is_right).

    This is the computation an induction head performs (match the context, copy
    what followed), and it is available to any model that can attend exactly to
    where that context occurred. So the gap on the "right, far" buckets is the
    part of the deficit that exact long-range access can explain -- which is why
    this is a better second axis than "longest seen suffix" (which saturates:
    95% of enwik8 positions have SOME long match, so it separates nothing).

    Buckets: no earlier occurrence of the context / guess wrong (near, far) /
    guess right at distance <=8, 9-64, >64.
    """
    n = raw.shape[0]
    nb = len(IND_BUCKETS)
    out = np.zeros(n, dtype=np.int8)
    dist = np.zeros(n, dtype=np.int32)
    right = np.zeros(n, dtype=bool)
    seen = {}
    get = seen.get
    h = 0
    mask = (1 << 64) - 1
    pow256 = 256 ** (k - 1)
    # rolling integer hash of the k bytes ending at p-1
    for p in range(1, n):
        b = int(raw[p - 1])
        if p - 1 >= k:
            h = ((h * 256 + b) - int(raw[p - 1 - k]) * pow256 * 256) & mask
        else:
            h = (h * 256 + b) & mask
        if p - 1 >= k - 1:                       # a full k-gram exists
            e = get(h, -1)                       # earlier occurrence END index
            if e >= 0 and e + 1 < p:             # its continuation is causal
                c = e + 1                        # the byte that followed it
                d = p - c
                ok = int(raw[c]) == int(raw[p])
                dist[p] = d if d <= MAXD else MAXD
                right[p] = ok
                if ok:
                    out[p] = 3 if d <= 8 else (4 if d <= 64 else 5)
                else:
                    out[p] = 1 if d <= 8 else 2
            # record AFTER the lookup, and only full k-grams: storing the
            # 1- and 2-byte prefix hashes of the first positions would let a
            # short gram collide with a full one
            seen[h] = p - 1
    return out, dist, right


def bucket(values: np.ndarray, edges: list) -> np.ndarray:
    return np.digitize(values, edges[1:-1], right=False)


def label(edges: list, i: int) -> str:
    lo, hi = edges[i], edges[i + 1] - 1
    if lo == 0:
        return "0 (unseen)"
    if hi >= edges[-1] - 1:
        return f"{lo}+"
    if lo == hi:
        return str(lo)
    return f"{lo}-{hi}"


@torch.no_grad()
def per_position_nll(model, val: torch.Tensor, starts, device) -> torch.Tensor:
    """Nats per position, aligned so element j is the loss on val[start+j+1]."""
    out = []
    for i in starts:
        x = val[i:i + T].unsqueeze(0).to(device)
        y = val[i + 1:i + 1 + T].unsqueeze(0).to(device)
        lp = model(x)["final_log_probs"].reshape(-1, V)
        out.append(F.nll_loss(lp, y.reshape(-1), reduction="none"))
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = np.frombuffer(open(VAL_PATH, "rb").read(N_VAL), dtype=np.uint8)
    val = torch.from_numpy(raw.astype(np.int64))
    starts = window_starts(raw.shape[0], VALW)
    if not starts:
        raise SystemExit("no windows: check ARM_VAL / ARM_VALW / ARM_VALSTRIDE")
    n_pos = len(starts) * T
    print(f"val {raw.shape[0]} bytes | {len(starts)} windows x {T} | "
          f"{n_pos} positions", flush=True)

    # the analysis positions are the TARGET indices, i.e. start+j+1
    tgt_idx = np.concatenate([np.arange(i + 1, i + 1 + T) for i in starts])
    # corpus features are causal: they may only use bytes strictly before the
    # target, which is exactly what these functions take (+1 so the last target
    # itself is addressable in the returned array)
    pref = raw[: int(tgt_idx[-1]) + 1]
    dist = copy_distance(pref)[tgt_idx]
    print(f"copy distance done ({dist.mean():.1f} mean, "
          f"{100 * (dist == 0).mean():.1f}% never-seen)", flush=True)
    ind, ind_d, ind_ok = induction_support(pref)
    ind, ind_d, ind_ok = ind[tgt_idx], ind_d[tgt_idx], ind_ok[tgt_idx]
    cover = float((ind_ok).mean())
    print(f"induction done: {100 * cover:.1f}% of positions have a correct "
          f"copy guess available", flush=True)
    db = bucket(dist, DIST_EDGES)

    rows = {}
    for path in args.ckpts:
        model, ck = load_model(path, device)
        nll = per_position_nll(model, val, starts, device).numpy()
        name = os.path.basename(path).replace(".pt", "")
        rows[name] = nll
        print(f"{name:<44} {nll.mean() / math.log(2):.4f} bpc "
              f"(row said {ck.get('row', {}).get('bpc_final', 'n/a')})",
              flush=True)

    names = list(rows)
    base = names[0]

    print("\n=== by COPY DISTANCE of the target byte (positions, then bpc) ===")
    hdr = f"{'d':>10}{'n':>8}" + "".join(f"{n[:18]:>20}" for n in names)
    if len(names) > 1:
        hdr += f"{'delta':>9}"
    print(hdr, flush=True)
    for b in range(len(DIST_EDGES) - 1):
        m = db == b
        if not m.any():
            continue
        line = f"{label(DIST_EDGES, b):>10}{int(m.sum()):>8}"
        for n in names:
            line += f"{rows[n][m].mean() / math.log(2):>20.4f}"
        if len(names) > 1:
            line += f"{(rows[base][m].mean() - rows[names[1]][m].mean()) / math.log(2):>+9.4f}"
        print(line, flush=True)

    print("\n=== by INDUCTION SUPPORT (copy the byte that followed the last "
          "occurrence of the 3-byte context) ===")
    print(f"{'bucket':>18}{'n':>8}" + "".join(f"{n[:18]:>20}" for n in names)
          + (f"{'delta':>9}" if len(names) > 1 else ""), flush=True)
    for b, name in enumerate(IND_BUCKETS):
        m = ind == b
        if not m.any():
            continue
        line = f"{name:>18}{int(m.sum()):>8}"
        for n in names:
            line += f"{rows[n][m].mean() / math.log(2):>20.4f}"
        if len(names) > 1:
            line += f"{(rows[base][m].mean() - rows[names[1]][m].mean()) / math.log(2):>+9.4f}"
        print(line, flush=True)

    if len(names) > 1:
        far_ok = ind == 5
        if far_ok.any():
            share = far_ok.mean()
            g = ((rows[base][far_ok].mean() - rows[names[1]][far_ok].mean())
                 * share)
            print(f"\n  positions reachable ONLY by exact long-range copying "
                  f"(right, d>64): {100 * share:.1f}% of positions, "
                  f"{g / math.log(2):+.4f} bpc of the gap", flush=True)

    if len(names) > 1:
        # a single decomposition number: what fraction of the total gap would
        # remain if the models agreed everywhere the exact window already
        # reaches (d <= 8)?
        # "inside the exact window" means the answer byte occurred within the
        # last 8 positions. d = 0 is "never occurred", which is NOT inside it.
        near = (dist >= 1) & (dist <= 8)
        tot = rows[base].mean() - rows[names[1]].mean()
        far_gap = ((rows[base][~near].mean() - rows[names[1]][~near].mean())
                   * (~near).mean())
        near_gap = ((rows[base][near].mean() - rows[names[1]][near].mean())
                    * near.mean())
        print(f"\ngap decomposition ({base} - {names[1]}): total "
              f"{tot / math.log(2):+.4f} bpc | inside d<=8: "
              f"{near_gap / math.log(2):+.4f} ({100 * near_gap / tot:.0f}%) | "
              f"beyond: {far_gap / math.log(2):+.4f} "
              f"({100 * far_gap / tot:.0f}%)", flush=True)

    out = os.environ.get("DIST_OUT")
    if out:
        with open(out, "w") as f:
            json.dump({"positions": n_pos, "windows": len(starts),
                       "names": names,
                       "bpc": {n: float(rows[n].mean() / math.log(2)) for n in names},
                       "dist_edges": DIST_EDGES,
                       "dist_hist": np.bincount(db, minlength=len(DIST_EDGES) - 1).tolist(),
                       "ind_buckets": IND_BUCKETS,
                       "ind_hist": np.bincount(ind, minlength=len(IND_BUCKETS)).tolist(),
                       "ind_coverage": cover},
                      f, indent=1)
        print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
