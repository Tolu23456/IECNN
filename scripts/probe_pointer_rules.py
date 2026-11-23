"""Which pointer rule should the SINGLE copy channel use?

Adding a second opinion costs ~2% of step time whether it is a second
`[B,T,T]` compare (`sc3e4`) or a selection over the existing scan (`sc3ev4`),
and both buy ~0.005 bpc per token -- a wash at equal wall clock. So the
question is not "add k=4" but "should the one pointer follow a different rule?".

The trained `sc3e4` mixture already answers what the model prefers: with both
channels live, k=3's extension ramp collapses (b=+0.0028) and k=4 takes the
mass (b=+0.2558). That says the k=4 pointer should be PRIMARY, with k=3 as
support, which is a replacement rule with no extra mixture mass and no extra
parameters.

Rules priced here, on the same match set (k=3, coverage is identical by
construction, 23.3%), over the same val windows as every other probe:

  recent   the deployed pointer: most recent earlier occurrence;
  maxext   `sc3ev`: most recent among those whose extension is maximal
           (1/k on a tie -- this reports the argmax-with-recent-tiebreak
           accuracy, and the fractional/hedged accuracy side by side);
  ext1     k=4's pointer: most recent occurrence whose extension is >= 1;
           fires on 15.8%, so it is not a candidate for the single channel on
           its own -- reported for reference;
  fallback THE CANDIDATE: ext1 if it exists, else maxext. Coverage stays the
           deployed 23.3%, the pointer is the longer match where one exists,
           and the rule is a few tiny ops over tensors the scan already built.

Also reported: the accuracy of a fixed hedge between the two (what `sc3ev4`
spends step time on) so the replacement can be compared against the additive
form it is meant to replace.

Reads ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE.
"""
from __future__ import annotations

import os

T = 256
K = 3
M = 3                       # the vote channel's candidate width
CAP = 8                     # extension cap, matching sc_extra_cap
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))


def ext(win: bytes, s: int, t: int) -> int:
    """Bytes agreeing BEFORE the k-gram, capped. Same definition as the model:
    the k-gram matches by construction, so compare x[t-K-j] with x[s-K-j]."""
    e = 0
    while e < CAP:
        i, j = t - K - e, s - K - e
        if i < 0 or j < 0 or win[i] != win[j]:
            break
        e += 1
    return e


def main() -> None:
    raw = open(VAL_PATH, "rb").read(N_VAL)
    starts = [w * T * VALSTRIDE for w in range(VALW)
              if w * T * VALSTRIDE + T + 1 <= len(raw)]
    wins = [raw[i:i + T + 1] for i in starts]
    total = len(wins) * T
    print(f"val {len(wins)}x{T} = {total} positions", flush=True)

    n = 0
    ok_recent = ok_max = ok_hedge = ok_ext1 = ok_fb = 0
    n_ext1 = n_fb_from_max = 0
    for win in wins:
        seen: dict[bytes, list[int]] = {}
        for t in range(K - 1, T):
            ctx = win[t - K + 1:t + 1]
            occ = seen.get(ctx)
            if occ:
                y = win[t + 1]
                recent = occ[-1]
                top = occ[-M:]                     # the M most recent
                es = [ext(win, s, t) for s in top]
                best = max(es)
                # candidates are most-recent-first in `top` after reversing
                cand = list(zip(top[::-1], es[::-1]))
                pick_max = next(s for s, e in cand if e == best)
                ties = [s for s, e in cand if e == best]
                hedge = 1.0 if y in [win[s + 1] for s in ties] else 0.0
                pick_ext1 = next((s for s, e in cand if e >= 1), None)
                if pick_ext1 is None:
                    pick_fb = pick_max
                else:
                    pick_fb = pick_ext1
                n += 1
                ok_recent += win[recent + 1] == y
                ok_max += win[pick_max + 1] == y
                ok_hedge += hedge
                if pick_ext1 is not None:
                    n_ext1 += 1
                    ok_ext1 += win[pick_ext1 + 1] == y
                else:
                    n_fb_from_max += 1
                ok_fb += win[pick_fb + 1] == y
            seen.setdefault(ctx, []).append(t)

    def line(name, ok, denom):
        print(f"  {name:<10}{100 * denom / total:>9.1f}%"
              f"{100 * ok / max(denom, 1):>10.1f}%", flush=True)

    print(f"\n  {'rule':<10}{'cover':>10}{'acc':>11}", flush=True)
    line("recent", ok_recent, n)
    line("maxext", ok_max, n)
    line("hedge", round(ok_hedge), n)
    line("ext1", ok_ext1, n_ext1)
    line("fallback", ok_fb, n)
    print(f"\n  ext1 exists on {100 * n_ext1 / max(n, 1):.1f}% of covered "
          f"positions; fallback uses maxext on the other "
          f"{100 * n_fb_from_max / max(n, 1):.1f}%", flush=True)
    print("  verdict: replace the pointer only if `fallback` beats `maxext` "
          "by more than seed noise at the SAME 23.3% coverage", flush=True)


if __name__ == "__main__":
    main()
