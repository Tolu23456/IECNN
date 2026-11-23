"""Does a longer copy index buy coverage at usable accuracy?

The frontier probe showed the 256-byte window caps what the copy channel can do:
k=3 covers 23.3% at 63.0%, k=2 covers 44.7% at 41.7%, and no estimator on those
match sets clears both. But coverage is not a property of k alone -- it is a
property of HOW MUCH HISTORY the index can see. In 256 bytes a 3-gram often has
not occurred yet; in 5 MB almost everything has.

So this streams the val region ONCE with a global last-occurrence table per k
and records, for every evaluated position, (distance to the most recent earlier
occurrence, whether its follower was right). Every history length is then a
filter on `distance`, which means one pass prices all of them:

    coverage(H) = fraction of positions whose most recent match is within H
    accuracy(H) = fraction of those whose copied byte is correct

The history is seeded with the last few MB of TRAIN (never evaluated), so early
val windows are not starved -- this is the index a model with persistent state
would have, not one that starts at the window boundary.

Reads ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE, plus ARM_HIST_MB.
"""
from __future__ import annotations

import os

T = 256
KS = (3, 4, 6, 8, 12)
HIST = (256, 1_024, 4_096, 16_384, 65_536, 262_144, 1_048_576, 1 << 40)
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
VAL_OFF = int(os.environ.get("ARM_VAL_OFF", "0"))
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
HIST_MB = int(os.environ.get("ARM_HIST_MB", "5"))


def main() -> None:
    with open(VAL_PATH, "rb") as f:
        f.seek(max(0, VAL_OFF - HIST_MB * 1_000_000))
        raw = f.read(HIST_MB * 1_000_000 + N_VAL)
    seed = len(raw) - N_VAL              # raw[seed:] is the val region
    if seed < 0:
        raise SystemExit("val slice smaller than the requested history")
    starts = [seed + w * T * VALSTRIDE for w in range(VALW)
              if seed + w * T * VALSTRIDE + T + 1 <= len(raw)]
    # cap the evaluated region so a short slice still works
    starts = [s for s in starts if s + T + 1 <= len(raw)][:48]
    ev = bytearray(len(raw))
    for s in starts:
        ev[s:s + T] = b"\x01" * T
    n_eval = sum(ev)
    print(f"history {seed} B + val {len(raw) - seed} B; "
          f"{len(starts)}x{T} = {n_eval} evaluated positions", flush=True)

    for k in KS:
        last: dict[bytes, int] = {}
        rec: list[tuple[int, bool]] = []
        for t in range(k - 1, len(raw) - 1):
            ctx = raw[t - k + 1:t + 1]
            s = last.get(ctx)
            last[ctx] = t
            if s is not None and ev[t]:
                rec.append((t - s, raw[s + 1] == raw[t + 1]))
        print(f"\n  k={k}  (table entries {len(last)})", flush=True)
        print(f"  {'history':>10}{'cover':>9}{'acc':>8}{'copy-right':>12}",
              flush=True)
        for h in HIST:
            sub = [ok for d, ok in rec if d <= h]
            if not sub:
                continue
            cov = len(sub) / n_eval
            acc = sum(sub) / len(sub)
            label = "all" if h > (1 << 39) else f"{h // 1000}k" if h >= 1000 \
                else str(h)
            print(f"  {label:>10}{100 * cov:>8.1f}%{100 * acc:>7.1f}%"
                  f"{100 * cov * acc:>11.1f}%", flush=True)
        del last

    print("\n  reference (windowed, k=3): 23.3% cover at 63.0% -> 14.7% "
          "copied right", flush=True)
    print("  verdict: a longer index pays only if `copy-right` rises above "
          "14.7% at a k whose accuracy the gate can trust", flush=True)


if __name__ == "__main__":
    main()
