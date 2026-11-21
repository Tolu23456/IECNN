"""What can a scalar-gated 3-gram copy channel possibly be worth, offline?

The screen costs GPU time and the result is one seed. This computes the CEILING
from checkpoints we already have, with no training: for every val position take
the model's own distribution p and the byte that followed the most recent
earlier occurrence of the 3-byte context, then evaluate

    bpc(g) = mean -log2( (1-g) p[target] + g * [copy == target] )

over a grid of constant gates g, plus the per-position oracle (open the gate
exactly where the copy is right). That bounds what ONE learned scalar can buy on
a trained model, and the oracle bounds the channel itself.

Read it as a pre-flight:
  * if max_g bpc(g) is below the seed noise (~0.01-0.02), the arm cannot pay and
    the GPU time is better spent elsewhere;
  * if the oracle is large but the best constant g is small, the channel needs a
    per-position gate -- which is the next design, not a wider constant;
  * the same computation with a RANDOM earlier pointer prices the false positive:
    a peaked but wrong prior must not look like a win.

Usage: PYTHONPATH=. python3 scripts/probe_copy_ceiling.py ckpt.pt [ckpt2.pt ...]
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
from scripts.probe_error_by_distance import IND_BUCKETS, induction_support

V = 256
T = int(os.environ.get("ARM_T", "256"))
K = int(os.environ.get("COPY_K", "3"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "1"))
GRID = [i / 100 for i in range(0, 100, 2)] + [0.99]


def window_starts(val_len: int, n: int):
    out = []
    for w in range(n):
        i = w * T * VALSTRIDE
        if i + T + 1 > val_len:
            break
        out.append(i)
    return out


def copy_positions(raw: np.ndarray, k: int):
    """copy[i] = byte that followed the most recent earlier occurrence of the k
    bytes ending at i-1 (the context available when predicting raw[i]); valid[i]
    says whether such an occurrence exists. Exactly the model's channel."""
    n = raw.shape[0]
    copy = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    seen = {}
    get = seen.get
    for p in range(1, n):
        h = int.from_bytes(raw[p - k:p].tobytes(), "little") if p >= k else -1
        if p >= k:
            e = get(h, -1)
            if e >= 0 and e + 1 < p:
                copy[p] = int(raw[e + 1])
                valid[p] = True
            seen[h] = p - 1
    return copy, valid


@torch.no_grad()
def probs_per_position(model, val, starts, device):
    out = []
    for i in starts:
        x = val[i:i + T].unsqueeze(0).to(device)
        lp = model(x)["final_log_probs"].reshape(-1, V)
        out.append(lp.exp())
    return torch.cat(out)


def mixture_bpc(p_true, copy_right, valid, gates):
    """bpc of (1-g)p + g[copy], with the gate multiplied by validity: there is
    no pointer at all where the context never occurred before."""
    rows = []
    for g in gates:
        gv = g * valid.float()
        t = (1 - gv) * p_true + gv * copy_right.float()
        rows.append((g, float(-torch.log2(t.clamp_min(1e-12)).mean())))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = np.frombuffer(open(VAL_PATH, "rb").read(N_VAL), dtype=np.uint8)
    val = torch.from_numpy(raw.astype(np.int64))
    starts = window_starts(raw.shape[0], VALW)
    tgt = np.concatenate([np.arange(i + 1, i + 1 + T) for i in starts])
    copy, valid = copy_positions(raw, K)
    # the same induction buckets the gap was measured in, so "the channel helps"
    # and "the channel helps where the gap is" stay separate claims
    pref = raw[: int(tgt[-1]) + 1]
    ind, ind_d, _ = induction_support(pref)
    ind, ind_d = ind[tgt], ind_d[tgt]
    cp = torch.from_numpy(copy[tgt])
    vv = torch.from_numpy(valid[tgt])
    y = val[torch.from_numpy(tgt)]
    right = (cp == y) & vv
    print(f"val {len(starts)}x{T} = {len(tgt)} positions | 3-gram copy valid "
          f"{100 * vv.float().mean():.1f}% | correct where valid "
          f"{100 * right.sum() / max(vv.sum(), 1):.1f}%", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        p = probs_per_position(model, val, starts, device)
        p_true = p.gather(1, y.unsqueeze(1)).squeeze(1)
        base = float(-torch.log2(p_true).mean())
        rows = mixture_bpc(p_true, right, vv, GRID)
        gbest, bbest = min(rows, key=lambda r: r[1])
        oracle = float(-torch.log2(torch.where(right, torch.ones_like(p_true),
                                               p_true).clamp_min(1e-12)).mean())
        # the same mixture with a random earlier pointer: the false-positive price
        g = torch.Generator().manual_seed(0)
        n = len(tgt)
        # a uniformly random earlier byte for each position (from the corpus)
        rand_src = (torch.rand(n, generator=g) * torch.from_numpy(tgt)).long()
        rand_src = torch.minimum(rand_src, torch.from_numpy(tgt) - 1)
        cp_rand = val.gather(0, rand_src.clamp_min(0))
        right_rand = cp_rand == y
        rows_rand = mixture_bpc(p_true, right_rand, vv, GRID)
        gb_r, bb_r = min(rows_rand, key=lambda r: r[1])
        name = os.path.basename(path).replace(".pt", "")
        print(f"\n{name}", flush=True)
        print(f"  base {base:.4f} bpc | best constant gate {gbest:.2f} -> "
              f"{bbest:.4f} ({base - bbest:+.4f}) | per-position oracle "
              f"{oracle:.4f} ({base - oracle:+.4f})", flush=True)
        print(f"  random pointer: best g {gb_r:.2f} -> {bb_r:.4f} "
              f"({base - bb_r:+.4f})  <- the false-positive price", flush=True)
        line = "  gate:  " + "  ".join(f"{g:.2f}:{b:.3f}" for g, b in rows[::5])
        print(line, flush=True)

        # per-bucket: the best constant gate *within* the bucket and the oracle.
        # A bucket where the best gate is large is where a distance-conditioned
        # gate would spend its weight, i.e. this prices the next design.
        print(f"  {'bucket':>18}{'n':>7}{'base':>8}{'best':>8}{'gain':>8}"
              f"{'g*':>6}{'oracle':>8}", flush=True)
        for bidx, bname in enumerate(IND_BUCKETS):
            m = torch.from_numpy(ind == bidx)
            if not bool(m.any()):
                continue
            pt, rt, vv_b = p_true[m], right[m], vv[m]
            br = mixture_bpc(pt, rt, vv_b, GRID)
            gb, bb = min(br, key=lambda r: r[1])
            boa = float(-torch.log2(torch.where(rt, torch.ones_like(pt),
                                                pt).clamp_min(1e-12)).mean())
            print(f"  {bname:>18}{int(m.sum()):>7}"
                  f"{float(-torch.log2(pt).mean()):>8.4f}{bb:>8.4f}"
                  f"{float(-torch.log2(pt).mean()) - bb:>+8.4f}"
                  f"{gb:>6.2f}{boa:>8.4f}", flush=True)


if __name__ == "__main__":
    main()
