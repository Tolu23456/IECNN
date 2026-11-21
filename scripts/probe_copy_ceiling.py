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
        # the IMPLEMENTATION only sees matches inside its T-token window, while
        # this probe searches the whole slice. Restrict to d < T and re-fit, or
        # the ceiling is an over-estimate of what the arm can actually use.
        dwin = torch.from_numpy(ind_d) < T
        vw = vv & dwin
        rows_w = mixture_bpc(p_true, right, vw, GRID)
        gw, bw = min(rows_w, key=lambda r: r[1])
        print(f"  window-limited (d < {T}): valid {100 * vw.float().mean():.1f}%"
              f" | best g {gw:.2f} -> {bw:.4f} ({base - bw:+.4f})", flush=True)
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

        # ---- the IMPLEMENTED channel, not the corpus one --------------------
        # Run the model's own _suffix_copy on the same windows and evaluate the
        # same mixture. This must reproduce the window-limited row (valid ~37%,
        # gain ~+0.087 on x5); if it does not, the arm and the analysis are
        # measuring different pointers and the ceiling does not apply.
        if ck["kind"] == "iecnn":
            r = model.regions[0]
            if getattr(r, "sc_gate", None) is None:
                r.suffix_copy = K
            ip = torch.zeros(len(tgt), dtype=torch.int64)
            iv = torch.zeros(len(tgt), dtype=torch.bool)
            at = 0
            for i in starts:
                ci, vi = r._suffix_copy(val[i:i + T].unsqueeze(0))
                ip[at:at + T] = ci[0]
                iv[at:at + T] = vi[0]
                at += T
            iright = (ip == y) & iv
            rows_i = mixture_bpc(p_true, iright, iv, GRID)
            gi, bi = min(rows_i, key=lambda r: r[1])
            print(f"  MODEL channel: valid {100 * iv.float().mean():.1f}%"
                  f" | correct {100 * iright.sum() / max(int(iv.sum()), 1):.1f}%"
                  f" | best g {gi:.2f} -> {bi:.4f} ({base - bi:+.4f})",
                  flush=True)

            # ---- is a distance-conditioned gate worth building at all? ------
            # The corpus feature-gate number used the corpus pointers and the
            # whole-slice distance. Price the SAME design against the channel
            # the arm actually runs: only in-window matches, in-window distance.
            # If this is within noise of the scalar, the scalar captures the
            # channel and the follow-up is not worth a build.
            dist = torch.zeros(len(tgt), dtype=torch.float32)
            at = 0
            for i in starts:
                w = raw[i:i + T]
                seen = {}
                for p_ in range(1, T):
                    if p_ >= K:
                        key = w[p_ - K:p_].tobytes()
                        e = seen.get(key, -1)
                        if e >= 0 and e + 1 < p_:
                            # recency of the SOURCE byte, in tokens back
                            dist[at + p_] = p_ - (e + 1)
                        seen[key] = p_ - 1
                at += T
            dbin = torch.bucketize(dist, torch.tensor([4.0, 16.0, 65.0]))
            H = -(p * p.clamp_min(1e-12).log()).sum(-1)
            q = torch.quantile(H, torch.tensor([1 / 3, 2 / 3]))
            ebin = torch.bucketize(H, q)
            grp_i = dbin * 3 + ebin
            perm = torch.randperm(len(p_true),
                                  generator=torch.Generator().manual_seed(0))
            for name, gg in (("dist only", dbin), ("dist x ent", grp_i)):
                gate = torch.zeros_like(p_true)
                for fit_idx, ev_idx in (
                        (perm[: len(perm) // 2], perm[len(perm) // 2:]),
                        (perm[len(perm) // 2:], perm[: len(perm) // 2])):
                    fit = torch.zeros(len(p_true), dtype=torch.bool)
                    fit[fit_idx] = True
                    for gidx in torch.unique(gg):
                        m_f = (gg == gidx) & fit
                        m_e = (gg == gidx) & ~fit
                        if not bool(m_f.any()) or not bool(m_e.any()):
                            continue
                        gsel, _ = min(mixture_bpc(p_true[m_f], iright[m_f],
                                                  iv[m_f], GRID),
                                      key=lambda r: r[1])
                        gate[m_e] = gsel
                gv = gate * iv.float()
                cv = float(-torch.log2(((1 - gv) * p_true
                                        + gv * iright.float()).clamp_min(1e-12)
                                       ).mean())
                print(f"  MODEL channel, 2-fold gate ({name}): {cv:.4f} bpc "
                      f"({base - cv:+.4f})", flush=True)

        # ---- what an OBSERVABLE-feature gate could recover: 2-fold CV --------
        # Features a gate can actually see: the match distance and the model's
        # own uncertainty. Fit a constant gate per (distance bucket, entropy
        # tercile) on one half of the positions and score the other half, both
        # ways, so the number is not fitted on the data it is quoted on.
        d = torch.from_numpy(ind_d)
        dbin = torch.bucketize(d.float(), torch.tensor([9.0, 65.0, 257.0]))
        H = -(p * p.clamp_min(1e-12).log()).sum(-1)
        q = torch.quantile(H, torch.tensor([1 / 3, 2 / 3]))
        ebin = torch.bucketize(H, q)
        grp = dbin * 3 + ebin
        perm = torch.randperm(len(p_true),
                              generator=torch.Generator().manual_seed(0))
        gate = torch.zeros_like(p_true)
        for fit_idx, ev_idx in ((perm[: len(perm) // 2], perm[len(perm) // 2:]),
                                (perm[len(perm) // 2:], perm[: len(perm) // 2])):
            fit = torch.zeros(len(p_true), dtype=torch.bool)
            fit[fit_idx] = True
            for gidx in torch.unique(grp):
                m_fit = (grp == gidx) & fit
                m_ev = (grp == gidx) & ~fit
                if not bool(m_fit.any()) or not bool(m_ev.any()):
                    continue
                gsel, _ = min(mixture_bpc(p_true[m_fit], right[m_fit],
                                          vv[m_fit], GRID),
                              key=lambda r: r[1])
                gate[m_ev] = gsel
        gv = gate * vv.float()
        cv = float(-torch.log2(((1 - gv) * p_true
                                + gv * right.float()).clamp_min(1e-12)).mean())
        print(f"  2-fold feature gate (distance x entropy): {cv:.4f} bpc "
              f"({base - cv:+.4f})", flush=True)

        # per-bucket: the best constant gate *within* the bucket and the oracle.
        # A bucket where the best gate is large is where a distance-conditioned
        # gate would spend its weight, i.e. this prices the next design.
        print(f"  {'bucket':>18}{'n':>7}{'base':>8}{'best':>8}{'gain':>8}"
              f"{'g*':>6}{'oracle':>8}", flush=True)
        # The table below is an ORACLE decomposition: the buckets are defined
        # by "the guess is right", which is the target, not a feature. It says
        # where the value is, not what a gate can find. The honest version is
        # the 2-fold fit further down, which only sees observable features.
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
