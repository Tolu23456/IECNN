"""Should the copy channel VOTE over several earlier matches instead of the last?

The `suffix_copy` audit (scripts/probe_suffix_copy.py) prices the channel's one
real weakness: the scalar gate cannot tell a genuine repeat from a k-byte
coincidence, so on positions whose single pointer is wrong it is 0.076 bpc WORSE
than shutting the gate. About 37% of its firings are wrong.

A k-gram with several earlier occurrences gives several candidate next bytes.
Where they agree the copy is much more trustworthy; where they disagree the
channel can abstain or hedge. That is a materially different channel (a
distribution over candidates, not one pointer), so it is priced here before
anything is built.

Per window this finds up to M earlier occurrences of the k-gram, then derives:
    single   the current channel: most recent match, one source
    strict2  last two occurrences, valid only if they agree
    voteM    empirical distribution over up to M candidates; prediction is the
             majority, ties to the most recent; needs >= 2 to vote
    extpick  the occurrence whose match extends furthest back

Reports coverage, accuracy, the deciding diagnostic (when the single pointer is
WRONG, how often does a vote fix it?), and an out-of-fold gate price per channel
(2 folds grouped by extension x agreement x entropy, no target information), so
the number is comparable to the +0.0560 scalar / +0.0661 extension rows.

Run it on a checkpoint that never trained the channel: there the post-hoc gate
is meaningful, because p has not absorbed the channel.

Usage: PYTHONPATH=. python3 scripts/probe_copy_vote.py ckpt.pt [ckpt.pt ...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_K / VOTE_M.
"""
from __future__ import annotations

import argparse
import collections
import os

import numpy as np
import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline

V = 256
T = int(os.environ.get("ARM_T", "256"))
K = int(os.environ.get("COPY_K", "3"))
M = int(os.environ.get("VOTE_M", "3"))
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


def window_candidates(w: np.ndarray):
    """Up to M earlier occurrences of the k-gram for every output position.

    Output arrays are indexed by the OUTPUT position t (0..n-2): the context is
    the k bytes ending at t, all observed, and the target is w[t+1]. That is the
    model's convention (`_suffix_copy`: pointer s < t, copy w[s+1]), so a
    pointer never reads at or after its target.

    CVAL[t, :CN[t]] are the candidate bytes, OLDEST FIRST (so [-1] is the most
    recent, which is what the current channel copies); EXT[t, :CN[t]] are the
    corresponding match extensions (bytes agreeing BEFORE the k-gram).
    """
    n = w.shape[0]
    CVAL = np.full((n, M), -1, dtype=np.int16)
    EXT = np.full((n, M), -1, dtype=np.int16)
    CN = np.zeros(n, dtype=np.int8)
    dist = np.zeros(n, dtype=np.float32)
    extra = np.zeros(n, dtype=np.int16)
    agree = np.zeros(n, dtype=np.float32)
    prev: dict[bytes, list[int]] = {}

    for p in range(1, n):
        if p < K:
            continue
        key = w[p - K:p].tobytes()
        ends = prev.get(key)
        tgt = p - 1
        if ends:
            picks = [s for s in ends if s < tgt][-M:]
            if picks:
                vals = [int(w[s + 1]) for s in picks]
                CN[tgt] = len(vals)
                CVAL[tgt, :len(vals)] = vals
                dist[tgt] = tgt - (picks[-1] + 1)
                exts = []
                for s in picks:
                    x = 0
                    while x < EXTRA_CAP:
                        d = K + x
                        if tgt - d < 0 or s - d < 0:
                            break
                        if w[s - d] != w[tgt - d]:
                            break
                        x += 1
                    exts.append(x)
                EXT[tgt, :len(vals)] = exts
                extra[tgt] = exts[-1]
                if len(vals) >= 2:
                    agree[tgt] = 1.0 if vals[-1] == vals[-2] else 0.0
        prev.setdefault(key, []).append(p - 1)
        if len(prev[key]) > 8:
            prev[key] = prev[key][-8:]
    return CVAL, EXT, CN, dist, extra, agree


def derive(CVAL, EXT, CN):
    """Prediction, validity and mass-on-truth for each channel."""
    n = CVAL.shape[0]
    out = {}
    cnt_ok = np.zeros(n, dtype=np.float64)

    # single: the most recent occurrence
    single_pred = CVAL[:, -1].astype(np.int64)
    single_valid = CN >= 1
    out["single"] = (single_pred, single_valid)

    # strict2: requires two occurrences that agree
    s2_valid = (CN >= 2) & (CVAL[:, -1] == CVAL[:, -2])
    out["strict2"] = (CVAL[:, -1].astype(np.int64), s2_valid)

    # voteM: majority of up to M, ties to the most recent
    vote_pred = np.zeros(n, dtype=np.int64)
    vote_valid = CN >= 2
    for i in range(n):
        c = int(CN[i])
        if c < 2:
            continue
        vals = [int(v) for v in CVAL[i, :c]]
        cnt = collections.Counter(vals)
        top = max(cnt.values())
        winners = [v for v in cnt if cnt[v] == top]
        vote_pred[i] = vals[-1] if vals[-1] in winners else winners[-1]
    out["voteM"] = (vote_pred, vote_valid)

    # extpick: the occurrence whose match extends furthest back
    ext_pred = np.zeros(n, dtype=np.int64)
    ext_valid = CN >= 1
    for i in range(n):
        c = int(CN[i])
        if c < 1:
            continue
        best = max(range(c), key=lambda j: (int(EXT[i, j]), j))
        ext_pred[i] = int(CVAL[i, best])
    out["extpick"] = (ext_pred, ext_valid)
    return out


def mass_on_truth(channel, CVAL, CN, EXT, y):
    """q[true byte] for each channel: a real distribution for voteM."""
    n = CVAL.shape[0]
    idx = np.arange(M)[None, :]
    if channel == "single":
        return ((CVAL[:, -1] == y) & (CN >= 1)).astype(np.float32)
    if channel == "strict2":
        v = (CN >= 2) & (CVAL[:, -1] == CVAL[:, -2])
        return (v & (CVAL[:, -1] == y)).astype(np.float32)
    if channel == "voteM":
        hit = ((CVAL == y[:, None]) & (idx < CN[:, None])).sum(1)
        return (hit / np.maximum(CN, 1)).astype(np.float32)
    if channel == "extpick":
        pred, valid = derive(CVAL, EXT, CN)["extpick"]
        return (valid & (pred == y)).astype(np.float32)
    raise KeyError(channel)


def self_check() -> None:
    # occurrences of 4,5,6 ending at 2, 6, 10; followers 8, 7, 7.
    w = np.array([4, 5, 6, 8, 4, 5, 6, 7, 4, 5, 6, 7, 9, 1], dtype=np.uint8)
    CVAL, EXT, CN, dist, extra, agree = window_candidates(w)
    t = 10
    assert (w[8:11] == w[4:7]).all() and (w[8:11] == w[0:3]).all(), "setup"
    assert int(CN[t]) == 2, CN[t]
    assert list(CVAL[t, :2]) == [8, 7], CVAL[t]
    assert int(CVAL[t, -1]) == 7, "most recent follower must be the single copy"
    pred, valid = derive(CVAL, EXT, CN)["strict2"]
    assert not bool(valid[t]), "8 vs 7 must abstain under strict2"
    vp, vv = derive(CVAL, EXT, CN)["voteM"]
    assert bool(vv[t]) and int(vp[t]) == 7, "tie must break to the most recent"
    y = np.full(len(w), -1)
    y[t] = 7
    m = mass_on_truth("voteM", CVAL, CN, EXT, y)
    assert abs(float(m[t]) - 0.5) < 1e-6, m[t]
    # agreeing followers: strict2 fires, voteM puts full mass on truth
    w2 = np.array([4, 5, 6, 7, 4, 5, 6, 7, 4, 5, 6, 7, 9, 1], dtype=np.uint8)
    C2, E2, N2, *_ = window_candidates(w2)
    _, v2 = derive(C2, E2, N2)["strict2"]
    assert bool(v2[t]), "identical followers must fire"
    y2 = np.full(len(w2), -1); y2[t] = 7
    assert abs(float(mass_on_truth("voteM", C2, N2, E2, y2)[t]) - 1.0) < 1e-6
    print("self-check ok (candidates [8,7]: strict2 abstains, voteM ties to 7 "
          "with mass 1/2; agreeing case fires with mass 1)", flush=True)


def mixture(p_true, mass_true, valid, gate):
    gv = gate * valid.float()
    return (1 - gv) * p_true + gv * mass_true


def bpc_of(p_true, mass_true, valid, gate):
    out = mixture(p_true, mass_true, valid, gate).clamp_min(1e-12)
    return float(-torch.log2(out).mean())


def best_const(p_true, mass_true, valid):
    rows = [(g, bpc_of(p_true, mass_true, valid, torch.full_like(p_true, g)))
            for g in GRID]
    return min(rows, key=lambda r: r[1])


def cv_gate(p_true, mass_true, valid, groups, seed=0):
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
            gsel, _ = best_const(p_true[m_f], mass_true[m_f], valid[m_f])
            gate[m_e] = gsel
    return gate


@torch.no_grad()
def probs_per_position(model, val, starts, device):
    out = []
    for i in starts:
        x = val[i:i + T].unsqueeze(0).to(device)
        lp = model(x)["final_log_probs"].reshape(-1, V)
        out.append(lp.exp())
    return torch.cat(out)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)
    self_check()

    raw = np.frombuffer(open(VAL_PATH, "rb").read(N_VAL), dtype=np.uint8)
    val = torch.from_numpy(raw.astype(np.int64))
    starts = window_starts(raw.shape[0], VALW)
    tgt_idx = np.concatenate([np.arange(i + 1, i + 1 + T) for i in starts])
    y = val[torch.from_numpy(tgt_idx)]
    yn = y.numpy()
    N = len(tgt_idx)

    CVAL = np.full((N, M), -1, dtype=np.int16)
    EXT = np.full((N, M), -1, dtype=np.int16)
    CN = np.zeros(N, dtype=np.int8)
    DIST = np.zeros(N, dtype=np.float32)
    EXTRA = np.zeros(N, dtype=np.int16)
    AGREE = np.zeros(N, dtype=np.float32)
    at = 0
    for i in starts:
        c, e, n, di, ex, ag = window_candidates(raw[i:i + T])
        CVAL[at:at + T], EXT[at:at + T], CN[at:at + T] = c, e, n
        DIST[at:at + T], EXTRA[at:at + T], AGREE[at:at + T] = di, ex, ag
        at += T
    print(f"val {len(starts)}x{T} = {N} positions | k={K} M={M}", flush=True)

    ch = derive(CVAL, EXT, CN)
    single_pred, single_valid = ch["single"]
    single_ok = single_valid & (single_pred == yn)
    print(f"\nsingle-pointer coverage {100 * single_valid.mean():.1f}% | "
          f"accuracy {100 * single_ok.sum() / max(int(single_valid.sum()), 1):.1f}%",
          flush=True)
    print(f"\n{'channel':<10}{'cover':>8}{'acc':>8}{'fixes':>7}{'breaks':>7}"
          f"{'net':>7}{'massT':>8}")
    sw = single_valid & ~single_ok
    for k in ("single", "strict2", "voteM", "extpick"):
        pred, valid = ch[k]
        ok = valid & (pred == yn)
        m = mass_on_truth(k, CVAL, CN, EXT, yn)
        fixes = int((sw & ok).sum())
        breaks = int((single_ok & ~ok).sum())
        acc = 100 * ok.sum() / max(int(valid.sum()), 1)
        mt = float(m[valid].mean()) if bool(valid.any()) else float("nan")
        print(f"{k:<10}{100 * valid.mean():>7.1f}%{acc:>7.1f}%{fixes:>7}{breaks:>7}"
              f"{fixes - breaks:>+7}{mt:>8.3f}", flush=True)
    print("  fixes/breaks are counted against the single pointer; massT is the "
          "mean mass the channel puts on the true byte where it is valid",
          flush=True)

    print(f"\noccurrences per matched position: "
          + " ".join(f"{m}:{100 * (CN == m).mean():.1f}%"
                     for m in range(1, M + 1)), flush=True)
    two = single_valid & (CN >= 2)
    if bool(two.any()):
        print(f"where >=2 occurrences exist (n={int(two.sum())}), single "
              f"accuracy {100 * (single_ok & two).sum() / max(int(two.sum()), 1):.1f}%, "
              f"voteM accuracy "
              f"{100 * (ch['voteM'][1] & (ch['voteM'][0] == yn)).sum() / max(int(two.sum()), 1):.1f}%",
              flush=True)

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
        H = -(p * p.clamp_min(1e-12).log()).sum(-1)
        q = torch.quantile(H, torch.tensor([1 / 3, 2 / 3]))
        ebin = torch.bucketize(H, q)
        xbin = torch.from_numpy(EXTRA).clamp_max(4)
        abin = torch.from_numpy(AGREE).to(torch.int64)
        name = os.path.basename(path).replace(".pt", "")
        print(f"\n{name}\n  base {base:.4f}", flush=True)
        for k in ("single", "strict2", "voteM", "extpick"):
            mt = torch.from_numpy(mass_on_truth(k, CVAL, CN, EXT, yn))
            v = torch.from_numpy(ch[k][1])
            if not bool(v.any()):
                print(f"  {k:<10} (never valid)", flush=True)
                continue
            g0, b0 = best_const(p_true, mt, v)
            groups = xbin * 3 + ebin
            if k != "single":
                groups = groups * 2 + abin
            gate = cv_gate(p_true, mt, v, groups)
            b = bpc_of(p_true, mt, v, gate)
            print(f"  {k:<10} const {g0:.2f} -> {b0:.4f} ({base - b0:+.4f})"
                  f" | oof grp {b:.4f} ({base - b:+.4f})", flush=True)

        # the decisive diagnostic: fixes on the positions the single pointer
        # gets wrong, which is exactly where the audit found the 0.076 tax
        swt = torch.from_numpy(sw)
        print(f"  {'on single-wrong (n=' + str(int(sw.sum())) + ')':<26}"
              + "".join(f"{k:>10}" for k in ("single", "strict2", "voteM", "extpick")),
              flush=True)
        nowok = {}
        massw = {}
        for k in ("single", "strict2", "voteM", "extpick"):
            pred, valid = ch[k]
            nowok[k] = 100 * int((valid & (pred == yn) & sw).sum()) / max(int(sw.sum()), 1)
            mw = mass_on_truth(k, CVAL, CN, EXT, yn)[sw]
            massw[k] = float(np.mean(mw)) if mw.size else float("nan")
        print(f"  {'  now correct':<26}" + "".join(f"{nowok[k]:>9.1f}%" for k in nowok),
              flush=True)
        print(f"  {'  mean mass on truth':<26}" + "".join(f"{massw[k]:>10.3f}" for k in massw),
              flush=True)


if __name__ == "__main__":
    main()
