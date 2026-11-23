"""Is the coverage/accuracy tradeoff escapable, or is the copy channel broken?

The measured frontier for an EXACT k-gram copy is a strict tradeoff:

    k=2  44.7% coverage / 41.7% right
    k=3  23.3% / 63.0%        <- the deployed channel
    k=4  15.8% / 71.2%
    k=6   9.1% / 77.7%

Coverage is a property of the MATCH SET; accuracy is a property of the
ESTIMATOR. No estimator can lift coverage above the k=2 set's 44.7%, and every
k>=3 set is a SUBSET of the k=2 set. So "higher coverage AND higher accuracy
than k=3" has exactly one possible route: get 63%+ right on a subset covering
more than 23.3% of positions, i.e. on the best ~52% of the k=2 set.

That is a pure corpus question, decided here model-free:

  * `recent`  the deployed estimator -- the follower of the MOST RECENT earlier
              occurrence, one sample, one hot;
  * `top`     the argmax of the empirical follower DISTRIBUTION over all
              earlier occurrences in the window (many samples), which is the
              estimator the channel does NOT use;
  * `share`   top follower's share of those samples -- an observable confidence
              signal, analogous to (and richer than) match extension;
  * ranked    sort the k=2 positions by confidence (share, then sample count)
              and report `recent`/`top` accuracy for every coverage level. This
              is the best any confidence-based selector can do at k=2, so if it
              does not clear 63% above 23.3% then the tradeoff is not escapable
              from below, and that is a proof rather than an opinion.

Reads ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE (the same protocol as every
other probe).
"""
from __future__ import annotations

import os
from collections import Counter

T = 256
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
KS = (2, 3, 4, 5, 6)


def scan(win: bytes, k: int):
    """Per position whose k-context occurred earlier in the window:

        (target, recent_follower, n_samples, top_follower, top_count)

    Built incrementally, so a hit only ever sees occurrences at s < t: the
    counts are causal and cannot leak the answer through a later match.
    """
    cnt: dict[bytes, Counter] = {}
    last: dict[bytes, int] = {}
    out = []
    for t in range(k - 1, T):
        ctx = win[t - k + 1:t + 1]
        c = cnt.get(ctx)
        if c:
            top, topn = max(c.items(), key=lambda kv: (kv[1], -kv[0]))
            out.append((win[t + 1], last[ctx], sum(c.values()), top, topn))
        # the occurrence ending at t becomes available for later queries only
        fol = win[t + 1]
        if c is None:
            cnt[ctx] = Counter({fol: 1})
        else:
            c[fol] += 1
        last[ctx] = fol
    return out


def main() -> None:
    raw = open(VAL_PATH, "rb").read(N_VAL)
    starts = [w * T * VALSTRIDE for w in range(VALW)
              if w * T * VALSTRIDE + T + 1 <= len(raw)]
    wins = [raw[i:i + T + 1] for i in starts]
    total = len(wins) * T
    print(f"val {len(wins)}x{T} = {total} positions", flush=True)

    k2 = []
    print(f"\n{'k':>3}{'cover':>8}{'recent':>8}{'top':>8}"
          f"{'mean n':>8}{'n>=2':>8}{'top>=2':>9}{'agree':>8}", flush=True)
    for k in KS:
        hits = []
        for w in wins:
            hits.extend(scan(w, k))
        n = max(len(hits), 1)
        rec = top = many = many_ok = agree = ns = 0
        for y, recent, cnt_n, topf, topn in hits:
            rec += recent == y
            top += topf == y
            ns += cnt_n
            if cnt_n >= 2:
                many += 1
                many_ok += topf == y
                agree += recent == topf
                if k == 2:
                    k2.append((topn / cnt_n, cnt_n, topf == y, recent == y))
        print(f"{k:>3}{100 * len(hits) / total:>7.1f}%{100 * rec / n:>7.1f}%"
              f"{100 * top / n:>7.1f}%{ns / n:>8.2f}"
              f"{100 * many / total:>7.1f}%"
              f"{100 * many_ok / max(many, 1):>8.1f}%"
              f"{100 * agree / max(many, 1):>7.1f}%", flush=True)

    # ---- the decisive part: rank k=2 by confidence, sweep coverage ----------
    # Confident first: higher follower share, then more samples behind it.
    k2.sort(key=lambda r: (-r[0], -r[1]))
    print(f"\n  k=2 confidence-ranked selector (n={len(k2)} hits = "
          f"{100 * len(k2) / total:.1f}% of positions)", flush=True)
    print(f"  {'cover':>8}{'top':>8}{'recent':>9}{'n>=2':>8}", flush=True)
    for cov in (0.05, 0.10, 0.15, 0.20, 0.233, 0.25, 0.30, 0.35, 0.447):
        want = min(int(cov * total), len(k2))
        if want <= 0:
            continue
        sub = k2[:want]
        tacc = sum(r[2] for r in sub) / len(sub)
        racc = sum(r[3] for r in sub) / len(sub)
        nmany = sum(1 for r in sub if r[1] >= 2) / len(sub)
        print(f"  {100 * want / total:>7.1f}%{100 * tacc:>7.1f}%"
              f"{100 * racc:>8.1f}%{100 * nmany:>7.1f}%", flush=True)
    print(f"\n  k=3 reference: 23.3% cover, 63.0% right (recent)", flush=True)
    print("  verdict: if `top` at 23.3% cover is <= 63.0%, no confidence "
          "selector escapes the tradeoff from k=2 downward", flush=True)


if __name__ == "__main__":
    main()
