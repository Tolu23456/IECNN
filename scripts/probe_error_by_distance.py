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

A second, independent axis: how much context is actually available at that
position, measured as the length of the longest suffix of the prefix that has
occurred before (0 = the 1-gram is new, k = the last k bytes were seen earlier).
Positions with a long available match are exactly where an induction-style read
pays; bucket drift along this axis says which model exploits repetition better.

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
CONT_EDGES = [0, 1, 2, 4, 8, 16, 32, 1 << 30]


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


def context_availability(raw: np.ndarray) -> np.ndarray:
    """Longest k such that the k bytes ending at p-1 occurred as a contiguous
    block somewhere strictly earlier. 0 = the immediately preceding byte is
    itself new (nothing at all to copy from).

    Implemented with an incremental rolling hash per k (h_k = h_{k-1} * 256 +
    byte) rather than slicing: slicing a numpy array per position allocates
    ~n*k objects and dominates the runtime otherwise. Hashes are exact integers,
    and collisions can only make the result pessimistic by a hair (they would
    claim an earlier occurrence that is not one), so the bucket labels stay
    honest: this is a bucketing feature, not a model input.
    """
    n = raw.shape[0]
    out = np.zeros(n, dtype=np.int8)
    kmax = 6
    seen = [dict() for _ in range(kmax + 1)]
    prev = [0] * (kmax + 1)
    get = [d.get for d in seen]
    for p in range(1, n):
        b = int(raw[p - 1])
        best = 0
        cur = [0] * (kmax + 1)
        cur[1] = b
        for k in range(2, kmax + 1):
            cur[k] = prev[k - 1] * 256 + b
        for k in range(1, min(kmax, p) + 1):
            h = cur[k]
            if get[k](h, -1) >= 0:
                best = k
            seen[k][h] = p
        prev = cur
        out[p] = best
    return out


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
    cont = context_availability(pref)[tgt_idx]
    print(f"context availability done ({cont.mean():.2f} mean k)", flush=True)
    db = bucket(dist, DIST_EDGES)
    cb = bucket(cont, CONT_EDGES)

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

    print("\n=== by CONTEXT AVAILABILITY (longest k already seen) ===")
    print(hdr, flush=True)
    for b in range(len(CONT_EDGES) - 1):
        m = cb == b
        if not m.any():
            continue
        line = f"{label(CONT_EDGES, b):>10}{int(m.sum()):>8}"
        for n in names:
            line += f"{rows[n][m].mean() / math.log(2):>20.4f}"
        if len(names) > 1:
            line += f"{(rows[base][m].mean() - rows[names[1]][m].mean()) / math.log(2):>+9.4f}"
        print(line, flush=True)

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
                       "dist_edges": DIST_EDGES, "cont_edges": CONT_EDGES,
                       "dist_hist": np.bincount(db, minlength=len(DIST_EDGES) - 1).tolist(),
                       "cont_hist": np.bincount(cb, minlength=len(CONT_EDGES) - 1).tolist()},
                      f, indent=1)
        print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
