"""Which OBSERVABLE signal tells a copy gate that a match is trustworthy?

The suffix_copy audit shows the scalar gate's cost: it fires on ~37% of valid
positions and is wrong about a third of the time, so on "wrong guess d>8" it is
0.076 bpc WORSE than shutting the gate. A gate that could see correctness would
be worth +0.45 on this model; distance and entropy recover almost none of that.

This prices a different, still-observable signal: HOW FAR THE MATCH EXTENDS
BEYOND THE k-GRAM. If 6 bytes agree instead of 3, the copy is much more likely
right, and the model can see that at prediction time with no extra state.

Measured with 2-fold out-of-fold gates over several feature sets, on a
checkpoint that never trained with the channel (so p is not adapted and the
post-hoc gate is meaningful), plus the diagnostic that decides the whole thing:

    correctness rate by match extension

If that is flat, no gate built on it can work and the follow-up is closed before
any training. Reported for both checkpoints because the adapted one is the
control for "does p absorb the channel".

Usage: PYTHONPATH=. python3 scripts/probe_copy_gate.py ckpt.pt [ckpt2.pt ...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_K.
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
K = int(os.environ.get("COPY_K", "3"))
EXTRA_CAP = int(os.environ.get("COPY_EXTRA_CAP", "8"))
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


@torch.no_grad()
def probs_per_position(model, val, starts, device):
    out = []
    for i in starts:
        x = val[i:i + T].unsqueeze(0).to(device)
        lp = model(x)["final_log_probs"].reshape(-1, V)
        out.append(lp.exp())
    return torch.cat(out)


def window_channel(w: np.ndarray):
    """Per-position copy info for one window.

    Returns copy, valid, dist, extra where position p is the target (predicting
    w[p]) and the pointer is the most recent earlier completion of the k bytes
    ending at p-1 inside this window -- the same rule the model runs.
    """
    n = w.shape[0]
    copy = np.zeros(n, dtype=np.int64)
    valid = np.zeros(n, dtype=bool)
    dist = np.zeros(n, dtype=np.float32)
    extra = np.zeros(n, dtype=np.int16)
    seen = {}
    for p in range(1, n):
        if p >= K:
            key = w[p - K:p].tobytes()
            e = seen.get(key, -1)
            if e >= 0 and e + 1 < p:
                # w[p] is the TARGET byte (window offset p-1, because offset 0
                # is the context for the first target), so every output array is
                # indexed by p-1. Getting this wrong shifts the whole channel by
                # one and reports ~8% correct instead of the verified ~63%.
                tgt = p - 1
                copy[tgt] = w[e + 1]
                valid[tgt] = True
                dist[tgt] = p - (e + 1)
                # how many bytes agree BEFORE the k-gram: t = K, K+1, ...
                x = 0
                while x < EXTRA_CAP:
                    t = K + x
                    if p - 1 - t < 0 or e - t < 0:
                        break
                    if w[e - t] != w[p - 1 - t]:
                        break
                    x += 1
                extra[tgt] = x
            seen[key] = p - 1
    return copy, valid, dist, extra


def mixture(p_true, right, valid, gate):
    gv = gate * valid.float()
    return (1 - gv) * p_true + gv * right.float()


def bpc_of(p_true, right, valid, gate):
    return float(-torch.log2(mixture(p_true, right, valid, gate).clamp_min(1e-12)).mean())


def best_const(p_true, right, valid):
    rows = [(g, bpc_of(p_true, right, valid, torch.full_like(p_true, g)))
            for g in GRID]
    return min(rows, key=lambda r: r[1])


def cv_gate(p_true, right, valid, groups, seed=0):
    """2-fold out-of-fold constant gate per group. No target information."""
    gate = torch.zeros_like(p_true)
    perm = torch.randperm(len(p_true),
                          generator=torch.Generator().manual_seed(seed))
    for fit_idx, ev_idx in ((perm[: len(perm) // 2], perm[len(perm) // 2:]),
                            (perm[len(perm) // 2:], perm[: len(perm) // 2])):
        fit = torch.zeros(len(p_true), dtype=torch.bool)
        fit[fit_idx] = True
        for gidx in torch.unique(groups):
            m_f = (groups == gidx) & fit
            m_e = (groups == gidx) & ~fit
            if not bool(m_f.any()) or not bool(m_e.any()):
                continue
            gsel, _ = best_const(p_true[m_f], right[m_f], valid[m_f])
            gate[m_e] = gsel
    return gate


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    # Guard against the off-by-one that this probe shipped with once: the
    # pointer array is indexed by TARGET offset (p-1), and getting that wrong
    # shifts the channel by one and reports ~8% correct instead of ~63%.
    probe = np.array([1, 2, 3, 4, 1, 2, 3, 9], dtype=np.uint8)
    pc, pv, pd, pe = window_channel(probe)
    assert not pv[:3].any(), "a target with an incomplete context matched"
    assert bool(pv[6]) and int(pc[6]) == 4 and abs(float(pd[6]) - 4.0) < 1e-6, \
        (pc, pv, pd)
    assert not bool(pv[7]), "9,3,2 never occurred earlier and must not match"
    assert int(pe[6]) == 0, "the extension past the k-gram is zero here"
    print("window_channel self-check ok (pointer at target 6 -> byte 4, d=4)",
          flush=True)

    raw = np.frombuffer(open(VAL_PATH, "rb").read(N_VAL), dtype=np.uint8)
    val = torch.from_numpy(raw.astype(np.int64))
    starts = window_starts(raw.shape[0], VALW)
    tgt = np.concatenate([np.arange(i + 1, i + 1 + T) for i in starts])
    y = val[torch.from_numpy(tgt)]

    cp = np.zeros(len(tgt), dtype=np.int64)
    vv = np.zeros(len(tgt), dtype=bool)
    dd = np.zeros(len(tgt), dtype=np.float32)
    ee = np.zeros(len(tgt), dtype=np.int16)
    at = 0
    for i in starts:
        c, v, d, e = window_channel(raw[i:i + T])
        cp[at:at + T], vv[at:at + T] = c, v
        dd[at:at + T], ee[at:at + T] = d, e
        at += T
    copy = torch.from_numpy(cp)
    valid = torch.from_numpy(vv)
    right = (copy == y) & valid
    print(f"val {len(starts)}x{T} = {len(tgt)} positions | k={K} copy valid "
          f"{100 * valid.float().mean():.1f}% | correct where valid "
          f"{100 * right.sum() / max(int(valid.sum()), 1):.1f}%", flush=True)

    # the diagnostic: does match extension predict correctness at all?
    print(f"\n{'extra bytes':>12}{'n valid':>9}{'% of valid':>11}"
          f"{'correct':>9}{'mean d':>8}", flush=True)
    for x in range(EXTRA_CAP + 1):
        m = valid & (torch.from_numpy(ee) == x)
        if not bool(m.any()):
            continue
        print(f"{x:>12}{int(m.sum()):>9}{100 * m.sum() / valid.sum():>10.1f}%"
              f"{100 * right[m].float().mean():>8.1f}%"
              f"{dd[m.numpy()].mean():>8.1f}", flush=True)

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
        g0, b0 = best_const(p_true, right, valid)
        H = -(p * p.clamp_min(1e-12).log()).sum(-1)
        q = torch.quantile(H, torch.tensor([1 / 3, 2 / 3]))
        ebin = torch.bucketize(H, q)
        dbin = torch.bucketize(torch.from_numpy(dd), torch.tensor([4., 16., 65.]))
        xbin = torch.from_numpy(ee).clamp_max(4)
        name = os.path.basename(path).replace(".pt", "")
        print(f"\n{name}\n  base {base:.4f} | best constant {g0:.2f} -> "
              f"{b0:.4f} ({base - b0:+.4f})", flush=True)
        for label, groups in (
                ("d", dbin),
                ("d x H", dbin * 3 + ebin),
                ("extra", xbin),
                ("extra x H", xbin * 3 + ebin),
                ("extra x d x H", xbin * 9 + dbin * 3 + ebin)):
            gate = cv_gate(p_true, right, valid, groups)
            b = bpc_of(p_true, right, valid, gate)
            print(f"  {label:<16}{b:.4f}  ({base - b:+.4f})", flush=True)

        # per-extension gate and what it would be worth if it were perfect
        print(f"  {'extra':>7}{'n':>7}{'g*':>7}{'bpc':>9}{'gain':>9}", flush=True)
        for x in range(EXTRA_CAP + 1):
            m = valid & (torch.from_numpy(ee) == x)
            if not bool(m.any()):
                continue
            gx, bx = best_const(p_true[m], right[m], valid[m])
            bm = float(-torch.log2(p_true[m]).mean())
            print(f"  {x:>7}{int(m.sum()):>7}{gx:>7.2f}{bx:>9.4f}"
                  f"{bm - bx:>+9.4f}", flush=True)


if __name__ == "__main__":
    main()
