"""Analytic MAC audit: what each architecture actually computes per token.

Pure arithmetic (no torch, no training) so it can run anywhere and be
checked line by line. Multiply-accumulate counts per token position at a
given config. Used to justify/refute speed claims before measuring.

    python3 scripts/flops_audit.py
"""
from __future__ import annotations

import json
import sys

# (name, config) — same knobs as scripts/ab_small.py
CONFIGS = {
    "tf-d96-l3":   dict(kind="tf", d=96,  layers=3,  ctx=64, V=256),
    "tf-d128-l4":  dict(kind="tf", d=128, layers=4,  ctx=64, V=256),
    "iecnn-gru":   dict(kind="ie", d=128, d_s=128, d_q=16, d_v=32, N=8, P=3,
                        w=16, V=256, cell="gru", lean=False),
    "iecnn-lug":   dict(kind="ie", d=128, d_s=128, d_q=16, d_v=32, N=8, P=3,
                        w=16, V=256, cell="lug", lean=False),
    "iecnn-lean":  dict(kind="ie", d=128, d_s=128, d_q=16, d_v=32, N=8, P=3,
                        w=16, V=256, cell="gru", lean=True),
    "iecnn-lite":  dict(kind="ie", d=128, d_s=128, d_q=16, d_v=32, N=8, P=3,
                        w=16, V=256, cell="lug", lean=True, nologit=True),
    "iecnn-lite-ds64": dict(kind="ie", d=128, d_s=64, d_q=16, d_v=32, N=8, P=3,
                            w=16, V=256, cell="lug", lean=True, nologit=True),
    "iecnn-lite-p2n4": dict(kind="ie", d=128, d_s=128, d_q=16, d_v=32, N=4, P=2,
                            w=16, V=256, cell="lug", lean=True, nologit=True),
    # param-matched with tf-d96-l3 (~390k): d_s = 64
    "iecnn-gru-ds64": dict(kind="ie", d=128, d_s=64, d_q=16, d_v=32, N=8, P=3,
                           w=16, V=256, cell="gru", lean=False),
    "iecnn-lite-ds64": dict(kind="ie", d=128, d_s=64, d_q=16, d_v=32, N=8, P=3,
                            w=16, V=256, cell="lug", lean=True, nologit=True),
    "iecnn-lite-p2n4-ds64": dict(kind="ie", d=128, d_s=64, d_q=16, d_v=32,
                                 N=4, P=2, w=16, V=256, cell="lug", lean=True, nologit=True),
    # the production config (base.yaml): N=8, P=3, d_s=128, window 32, V=512
    "iecnn-gru-prod": dict(kind="ie", d=128, d_s=128, d_q=32, d_v=64, N=8, P=3,
                           w=32, V=512, cell="gru", lean=False),
    "iecnn-lite-prod": dict(kind="ie", d=128, d_s=128, d_q=32, d_v=64, N=8, P=3,
                            w=32, V=512, cell="lug", lean=True),
    "iecnn-lite-prod-ds64": dict(kind="ie", d=128, d_s=64, d_q=32, d_v=64, N=8,
                                 P=2, w=32, V=512, cell="lug", lean=True),
    # ---- the arms that actually run today (90/5 protocol, T=256) ----------
    # tf reference: d96 / L3 / T256 -- the anchor every session re-runs
    "tf-d96-l3-T256": dict(kind="tf", d=96, layers=3, ctx=256, V=256),
    # rr2-c: the arm that made reread pay
    "x3-rr2": dict(kind="ie", d=128, d_s=96, d_q=16, d_v=32, N=3, P=1, w=8,
                   V=256, cell="lug", lean=True, nologit=True,
                   reread=2, global_read=True, head_post=512,
                   head_post_layers=3),
    # x5+rr2-c: the frontier (head capacity paid 0.046)
    "x5-rr2": dict(kind="ie", d=128, d_s=96, d_q=16, d_v=32, N=3, P=1, w=8,
                   V=256, cell="lug", lean=True, nologit=True,
                   reread=2, global_read=True, head_post=512,
                   head_post_layers=5),
    # the staged width arms, to price the second doubling BEFORE paying for it
    "x5w768-rr2": dict(kind="ie", d=128, d_s=96, d_q=16, d_v=32, N=3, P=1, w=8,
                       V=256, cell="lug", lean=True, nologit=True,
                       reread=2, global_read=True, head_post=768,
                       head_post_layers=5),
    "x5w1024-rr2": dict(kind="ie", d=128, d_s=96, d_q=16, d_v=32, N=3, P=1, w=8,
                        V=256, cell="lug", lean=True, nologit=True,
                        reread=2, global_read=True, head_post=1024,
                        head_post_layers=5),
    # the read alternative: a 16-slot window instead of 8, at x5 width
    "x5-w16-rr2": dict(kind="ie", d=128, d_s=96, d_q=16, d_v=32, N=3, P=1, w=16,
                       V=256, cell="lug", lean=True, nologit=True,
                       reread=2, global_read=True, head_post=512,
                       head_post_layers=5),
}


def mac_tf(c: dict) -> dict:
    d, L, T, V = c["d"], c["layers"], c["ctx"], c["V"]
    # per layer, per position: qkv+out projections, attention scores+mix
    # (causal average: T/2 keys), MLP 4d->d->4d
    proj = 4 * d * d
    attn = (T / 2) * d + (T / 2) * d
    mlp = 8 * d * d
    body = L * (proj + attn + mlp)
    head = d * V
    return {"body": body, "head": head, "total": body + head}


def mac_ie(c: dict) -> dict:
    d, d_s, d_q, d_v = c["d"], c["d_s"], c["d_q"], c["d_v"]
    N, P, w, V = c["N"], c["P"], c["w"], c["V"]
    cell = c["cell"]
    d_in = d_v + d_s + 3          # ctx + pooled opinion + err feature
    # read (cached across passes by cache_attention): scores + value mix
    read = N * w * d_q + N * w * d_v
    # the SHARED key/value projections that build the window memory: every
    # position goes through both matrices once per region-forward (they are
    # shared by all N dots, hence no N factor)
    wkv = d * d_q + d * d_v
    # cell, every pass
    if cell == "gru":
        cell_mac = N * (d_in * 3 * d_s + d_s * 3 * d_s)
    else:  # lug: one input GEMM of width 2*d_s, diagonal recurrence
        cell_mac = N * (d_in * 2 * d_s)
    cell_tot = P * cell_mac
    # gamma/trust readouts, every pass
    scal = P * N * d_s * 2
    # e_state projection: every pass, or last pass only when lean
    e_state = (1 if c["lean"] else P) * N * d_s * d
    # region head (pooled shared head): every pass, or last only when lean
    head = (1 if c["lean"] else P) * d * V
    # per-dot logits: last pass only, and only when an auxiliary loss needs
    # them (the toy/lean arms train on plain CE -> 0)
    log_pi = 0 if c.get("nologit") else N * d * V
    # global decayed read: one elementwise cumsum over d_v per dot per token
    # (prefix-sum amortises to O(1) per position) plus the decay multiply
    glob = N * d_v * 2 if c.get("global_read") else 0
    # head_post blocks: u = LN(h); h = h + W2 gelu(W1 u), i.e. 2*d*width each.
    # This is the per-position half of a transformer block and, at the widths
    # the frontier arms use, it dominates the whole model -- which is exactly
    # why it has to be in the audit before pricing another doubling.
    n_post = c.get("head_post_layers", 0) if c.get("head_post") else 0
    post = n_post * 2 * d * c.get("head_post", 0)
    # reread sites: q,k,v,o projections on d, then an s-slot score+mix. The
    # params are shared across sites but the compute is not.
    rr_q = c.get("reread_dq", 16)
    rr_v = c.get("reread_dv", 32)
    rr_span = c.get("reread_span", 8)
    reread = c.get("reread", 0) * (3 * d * rr_q + d * rr_v
                                   + rr_span * (rr_q + rr_v))
    parts = {"wkv": wkv, "read": read, "cell": cell_tot, "scalars": scal,
             "e_state": e_state, "global": glob, "post": post,
             "reread": reread, "head": head, "per_dot_logits": log_pi}
    parts["total"] = sum(parts.values())
    return parts


def main():
    rows = {}
    for name, c in CONFIGS.items():
        rows[name] = mac_tf(c) if c["kind"] == "tf" else mac_ie(c)
    base = rows["iecnn-gru"]["total"]
    ref = rows["tf-d96-l3-T256"]["total"]
    print(f"{'arm':<22}{'MACs/token':>14}{'cheaper than':>14}"
          f"{'x tf-anchor':>13}")
    for n, r in rows.items():
        # "cheaper than" = how many of this arm fit in one iecnn-gru token;
        # "x tf-anchor" = multiples of the T=256 transformer anchor's compute,
        # which is the only comparable column across the two architectures
        print(f"{n:<22}{r['total']:>14,}{base / r['total']:>13.2f}x"
              f"{r['total'] / ref:>12.2f}x")
    print("\nbreakdown vs the tf anchor it is measured against:")
    for n, r in rows.items():
        if "cell" in r:
            print(f"  {n:<22} {r['total'] / ref:5.2f}x tf   "
                  + "  ".join(f"{k} {v / r['total'] * 100:4.1f}%"
                              for k, v in r.items() if k != "total"))
    print("\nbreakdown (MACs/token):")
    for n, r in rows.items():
        if "cell" in r:
            sh = {k: f"{v / r['total'] * 100:.1f}%" for k, v in r.items()
                  if k != "total"}
            print(f"  {n:<22} {sh}")
    print("\nJSON " + json.dumps(rows))


if __name__ == "__main__":
    sys.exit(main())
