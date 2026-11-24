"""Build a GLOBAL copy index for a byte corpus: one pass, vectorised.

For every position t, the copy channel needs the most recent EARLIER occurrence
of the k-byte context ending at t, the byte after it, and how many bytes agree
before the context (the extension). The windowed channel recomputes this with a
`[B,T,T]` compare every step; here it is precomputed once per corpus, so a
training step only gathers two arrays.

Outputs two uint8 arrays of the same length as the corpus:

  copy_byte[i]   the byte to copy at position i (0 where invalid)
  copy_ext[i]    bytes agreeing before the k-gram, capped at CAP;
                 the sentinel `255` means "no earlier occurrence"

`CAP < 255` always holds, so validity needs no third array.

The build is a sort, not a Python loop: hash every k-gram, stable-sort, and the
previous element inside a run of equal hashes is the previous occurrence.
Hashes are exact for k <= 8 (k*8 <= 64 bits), so there are no collisions to
worry about; longer k needs the rolling-hash form below.

Usage:
  python3 scripts/build_copy_index.py corpus.txt out_prefix --k 4 [--cap 16]
Writes <out_prefix>.byte.npy and <out_prefix>.ext.npy.
"""
from __future__ import annotations

import argparse

import numpy as np

CAP = 16


def _hashes(b: np.ndarray, k: int) -> np.ndarray:
    """Exact k-gram hash ending at each position (positions < k-1 are unused)."""
    if k > 8:
        raise SystemExit("k > 8 would need a rolling hash; keep it exact")
    n = b.size
    h = np.zeros(n, dtype=np.uint64)
    for j in range(k):
        # h[t] = sum_j b[t-k+1+j] << 8j, so every term has length n-k+1
        h[k - 1:] |= b[j:n - k + 1 + j].astype(np.uint64) << np.uint64(8 * j)
    return h


def build(raw, k: int, cap: int = CAP):
    b = (raw if isinstance(raw, np.ndarray)
         else np.frombuffer(raw, dtype=np.uint8))
    n = b.size
    h = _hashes(b, k)
    # positions before k-1 have no full context: sort only the rest, so they
    # stay invalid instead of joining a run of the real hash above
    order = np.argsort(h[k - 1:], kind="stable").astype(np.int32) + (k - 1)
    hs = h[order]
    src = np.full(n, -1, dtype=np.int32)
    if order.size > 1:
        same = hs[1:] == hs[:-1]
        src[order[1:]] = np.where(same, order[:-1], -1)
    del h, hs, order

    valid = src >= 0
    copy_byte = np.zeros(n, dtype=np.uint8)
    copy_byte[valid] = b[np.clip(src[valid].astype(np.int64) + 1, 0, n - 1)]

    # extension: walk back one byte at a time over the still-live positions.
    # Indexing only the live set keeps peak memory at a few n-sized uint8/bool
    # arrays instead of cap full int64 temporaries (90 MB corpus, cap 8).
    ext = np.full(n, 255, dtype=np.uint8)
    e = np.zeros(n, dtype=np.uint8)
    live = valid.copy()
    for m in range(cap):
        lo = k + m
        can = live & (src >= lo)
        can[:lo] = False
        tt = np.nonzero(can)[0]
        if tt.size == 0:
            break
        keep = b[tt - lo] == b[src[tt].astype(np.int64) - lo]
        e[tt[keep]] += 1
        nxt = np.zeros(n, dtype=bool)
        nxt[tt[keep]] = True
        live = nxt
    ext[valid] = e[valid]
    return copy_byte, ext


def _brute(raw: bytes, k: int, cap: int = CAP):
    """Reference implementation: O(n^2), for the self-check only."""
    n = len(raw)
    cb = np.zeros(n, dtype=np.uint8)
    ext = np.full(n, 255, dtype=np.uint8)
    for t in range(k - 1, n):
        best = -1
        for s in range(k - 1, t):
            if raw[s - k + 1:s + 1] == raw[t - k + 1:t + 1]:
                best = s
        if best >= 0:
            cb[t] = raw[best + 1]
            e = 0
            while e < cap:
                i, j = t - k - e, best - k - e
                if i < 0 or j < 0 or raw[i] != raw[j]:
                    break
                e += 1
            ext[t] = e
    return cb, ext


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("corpus")
    ap.add_argument("out_prefix")
    ap.add_argument("--k", type=int, default=4)
    ap.add_argument("--cap", type=int, default=CAP)
    ap.add_argument("--check", type=int, default=0,
                    help="also verify against brute force on the first N bytes")
    args = ap.parse_args()

    raw = open(args.corpus, "rb").read()
    cb, ext = build(raw, args.k, args.cap)
    if args.check:
        r = raw[:args.check]
        b_cb, b_ext = _brute(r, args.k, args.cap)
        assert np.array_equal(cb[:args.check], b_cb), "copy_byte mismatch"
        assert np.array_equal(ext[:args.check], b_ext), "extension mismatch"
        print(f"self-check ok on {args.check} bytes", flush=True)
    np.save(args.out_prefix + ".byte.npy", cb)
    np.save(args.out_prefix + ".ext.npy", ext)
    cov = float((ext != 255).mean())
    acc = float((cb[ext != 255] == np.frombuffer(raw, np.uint8)[ext != 255])
                .mean()) if cov else 0.0
    print(f"k={args.k} cap={args.cap} n={cb.size} coverage {100 * cov:.1f}% "
          f"byte-identity {100 * acc:.1f}%", flush=True)


if __name__ == "__main__":
    main()
