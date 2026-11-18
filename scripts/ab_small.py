"""Arm registry: equal-wall-clock A/B of IECNN variants vs transformers.

Every arm gets the same data bytes, batch, context, optimizer and wall-clock
budget on the same 2-vCPU box. Reports params, steps, tokens seen, s/step,
val bpc at each mark, val bpc when the arm finishes, and bpc interpolated at
the token count EVERY arm reached (equal-token comparison), plus a pointer
diversity audit for IECNN arms.

Run:  PYTHONPATH=. python3 -u scripts/ab_small.py [budget_s] [arm,arm,...]
"""
from __future__ import annotations

import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F

from models.baselines import build_transformer_baseline
from models.iecnn import build_iecnn
from models.memory import region1_rows

BUDGET = float(sys.argv[1]) if len(sys.argv) > 1 else 150.0
ONLY = sys.argv[2].split(",") if len(sys.argv) > 2 else None
# Throughput knobs (both arms always get the same value: this is a property
# of the machine/harness, never of one architecture).
B = int(os.environ.get("ARM_B", 8))
torch.set_num_threads(int(os.environ.get("ARM_THREADS", 2)))
T = 64
V = 256                      # byte-level toy vocab
MARK_S = 30.0


# --------------------------------------------------------------------------
# config builders
# --------------------------------------------------------------------------
def iecnn(cell="gru", lean=False, n_passes=3, n_dots=8, d_s=128, d=128,
          spans=(16,), read_mode="soft", head_mode="hidden",
          per_dot_logits=True, window_mode="contiguous", window_reach=None,
          window_local=None, seclusion_topk=0, pool_first=False,
          slot_slots=0, dist_bias=False, dynamic_query=False,
          global_read=False, read_schedule="mix", head_untied=False,
          cell_ln=False, share_cell=False):
    m = {
        "kind": "iecnn", "d": d, "d_q": 16, "d_v": 32, "d_s": d_s,
        "n_dots": n_dots, "n_regions": 1, "n_passes": n_passes,
        "spans": list(spans), "use_weights": True, "use_messages": False,
        "pooling": "log_linear", "head_mode": head_mode,
        "read_mode": read_mode, "share_pass_params": True,
        "dynamic_query": dynamic_query, "use_cross_region": False,
        "fast_dot_math": True, "cache_attention": True,
        "cell": cell, "head_last_only": lean,
        "per_dot_logits": per_dot_logits,
        "window_mode": window_mode,
        "seclusion_topk": seclusion_topk,
        "pool_first": pool_first,
        "slot_slots": slot_slots,
        "dot_dist_bias": dist_bias,
        "global_read": global_read,
        "read_schedule": read_schedule,
        "head_untied": head_untied,
        "cell_ln": cell_ln,
        "share_cell": share_cell,
    }
    if window_reach is not None:
        m["window_reach"] = window_reach
    if window_local is not None:
        m["window_local"] = window_local
    return {"model": m,
            "consensus": {"rho": 0.0, "kappa": 0.7},
            "training": {"compile_consensus": False, "dot_dropout": 0.0},
            "data": {"seq_len": T}}


def tf(d_model=96, n_heads=4, n_layers=3):
    return {"model": {"kind": "transformer", "d_model": d_model,
                      "n_heads": n_heads, "n_layers": n_layers},
            "data": {"seq_len": T}}


ARMS = {
    # the baseline family we must beat on BOTH bpc and speed
    "tf-d96-l3":  ("transformer", lambda: tf(96, 4, 3)),
    "tf-d128-l4": ("transformer", lambda: tf(128, 4, 4)),
    # current champion lineage (GRU dot + per-pass shared head + per-dot logits)
    "iecnn-gru":  ("iecnn", lambda: iecnn("gru", False)),
    # lean cell only
    "iecnn-lug":  ("iecnn", lambda: iecnn("lug", False)),
    # last-pass-only head only
    "iecnn-lean": ("iecnn", lambda: iecnn("gru", True)),
    # no per-dot vocab logits (nothing in the toy training consumes them)
    "iecnn-nologit": ("iecnn", lambda: iecnn("gru", False, per_dot_logits=False)),
    # the full lean package: LUG + last-pass head/e_state + no per-dot logits
    "iecnn-lite": ("iecnn", lambda: iecnn("lug", True, per_dot_logits=False)),
    # smaller committee / fewer passes / smaller state
    "iecnn-lite-p2": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                             per_dot_logits=False)),
    "iecnn-lite-n4": ("iecnn", lambda: iecnn("lug", True, n_dots=4,
                                             per_dot_logits=False)),
    "iecnn-lite-ds64": ("iecnn", lambda: iecnn("lug", True, d_s=64,
                                               per_dot_logits=False)),
    "iecnn-lite-p2n4": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                               n_dots=4, per_dot_logits=False)),
    "iecnn-lite-fast": ("iecnn", lambda: iecnn("lug", True, n_passes=2,
                                               n_dots=4, d_s=64, d=96,
                                               per_dot_logits=False)),
    # seclusion on top of lite
    "iecnn-lite-sec": ("iecnn", lambda: iecnn("lug", True, per_dot_logits=False,
                                              read_mode="secluded")),
    # seclusion v2: hard read + top-k straight-through gradient
    "iecnn-lite-sec-k4": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, read_mode="secluded",
        seclusion_topk=4)),
    # long-range time axis at ZERO extra compute: same 16 slots spread
    # quadratically from 0 to window_reach (toy context is 64, so 64 = full)
    "iecnn-lite-q64": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, window_mode="quad",
        window_reach=64)),
    "iecnn-lite-q32": ("iecnn", lambda: iecnn(
        "lug", True, per_dot_logits=False, window_mode="quad",
        window_reach=32)),
    "iecnn-lite-p2n4-q256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, per_dot_logits=False,
        window_mode="quad", window_reach=256)),
    # --- param-matched with tf-d96-l3 (~390k): d_s = 64 -------------------
    "iecnn-gru-ds64": ("iecnn", lambda: iecnn("gru", False, d_s=64)),
    "iecnn-lite-ds64": ("iecnn", lambda: iecnn("lug", True, d_s=64,
                                               per_dot_logits=False)),
    "iecnn-lite-ds64-q64": ("iecnn", lambda: iecnn(
        "lug", True, d_s=64, per_dot_logits=False, window_mode="quad",
        window_reach=64)),
    "iecnn-lite-ds64-q64-sec": ("iecnn", lambda: iecnn(
        "lug", True, d_s=64, per_dot_logits=False, window_mode="quad",
        window_reach=64, read_mode="secluded", seclusion_topk=4)),
    "iecnn-lite-p2n4-ds64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False)),
    "iecnn-lite-p2n4-ds64-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="quad", window_reach=64)),
    # hybrid multi-scale: 6 dense local slots + geometric far slots
    "iecnn-lite-p2n4-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="hybrid", window_reach=64)),
    "iecnn-lite-p2n4-h32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="hybrid", window_reach=32)),
    "iecnn-lite-p2n4-ds80-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=80, per_dot_logits=False,
        window_mode="hybrid", window_reach=64)),
    "iecnn-lite-p2n4-ds80": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=80, per_dot_logits=False)),
    "iecnn-lite-p3n4-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=3, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="hybrid", window_reach=64)),
    "iecnn-lite-p2n6-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=6, d_s=64, per_dot_logits=False,
        window_mode="hybrid", window_reach=64)),
    # --- capacity via d (cheap MACs: only the pooled head + e_state widen) --
    "iecnn-lite-p2n4-d192": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, d=192,
        per_dot_logits=False)),
    "iecnn-lite-p2n4-d256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, d=256,
        per_dot_logits=False)),
    "iecnn-lite-p2n4-d256-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, d=256,
        per_dot_logits=False, window_mode="hybrid", window_reach=64)),
    "iecnn-lite-p2n6-d192-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=6, d_s=64, d=192,
        per_dot_logits=False, window_mode="hybrid", window_reach=64)),
    "tf-d96-l6": ("transformer", lambda: tf(96, 4, 6)),
    # --- pool_first: shared output projection, pool the states first ------
    "iecnn-p2n4-d256-pool": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, d=256,
        per_dot_logits=False, pool_first=True)),
    "iecnn-p2n8-d256-pool": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, d=256,
        per_dot_logits=False, pool_first=True)),
    "iecnn-p2n8-d384-pool": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, d=384,
        per_dot_logits=False, pool_first=True)),
    "iecnn-p2n8-d512-pool": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, d=512,
        per_dot_logits=False, pool_first=True)),
    # --- temporal carry (slot table) on the winning small shape ----------
    "iecnn-p2n4-lr2e3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 2e-3}),
    "iecnn-p2n4-lr3e3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 3e-3}),
    "iecnn-p2n4-lr3e4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 3e-4}),
    "iecnn-p2n4-slot4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        slot_slots=4)),
    "iecnn-p2n4-h64-slot4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        slot_slots=4, window_mode="hybrid", window_reach=64)),
    # --- real per-pass depth + distance-specialised dots ----------------
    # dynamic_query: the read is cached and shared across passes when false,
    # so passes 2..P re-read the SAME ctx. True makes the query state-
    # dependent -> every pass asks the memory a different question.
    # cost: recomputing N*w*d_q scores + N*w*d_v mix per pass (~2% MACs).
    "iecnn-p2n4-dyn": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dynamic_query=True)),
    # dist_bias: one learned scalar per (dot, slot) -> dots specialise by
    # distance instead of collapsing onto the same token (52-70% all-same).
    # cost: N*w additions per position. Effectively zero.
    "iecnn-p2n4-bias": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dist_bias=True)),
    "iecnn-p2n4-dyn-bias": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dynamic_query=True, dist_bias=True)),
    "iecnn-p2n4-bias-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dist_bias=True, window_mode="hybrid", window_reach=64)),
    # --- schedule: time-based cosine over the fixed budget (free) --------
    "iecnn-p2n4-cos": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"sched": "cos"}),
    "tf-d96-l3-cos": ("transformer", lambda: tf(96, 4, 3), {"sched": "cos"}),
    # --- bands: each pass reads a DIFFERENT part of the sequence ---------
    # pass 1 dense local 0..15, pass 2 geometric 16..64 (P=2);
    # P=3 splits the far range into two log bands.
    "iecnn-p2n4-bands": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="bands", window_reach=64)),
    "iecnn-p3n4-bands": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=3, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="bands", window_reach=64)),
    "iecnn-p2n4-bands-dyn": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="bands", window_reach=64, dynamic_query=True)),
    # --- lr 2e-3 (the winner) x the new mechanisms ----------------------
    "iecnn-lr2-bands": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="bands", window_reach=64), {"lr": 2e-3}),
    "iecnn-lr2-dyn": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dynamic_query=True), {"lr": 2e-3}),
    "iecnn-lr2-bias": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dist_bias=True), {"lr": 2e-3}),
    "iecnn-lr2-dyn-bias": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dynamic_query=True, dist_bias=True), {"lr": 2e-3}),
    "iecnn-lr2-bias-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        dist_bias=True, window_mode="hybrid", window_reach=64),
        {"lr": 2e-3}),
    "iecnn-lr2-cos": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 2e-3, "sched": "cos"}),
    "iecnn-lr2-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        window_mode="quad", window_reach=64), {"lr": 2e-3}),
    # --- fairness controls: give the transformer the SAME lr/schedule ----
    # --- O(T) global causal read (prefix-sum attention) on top of the
    # window read, at lr 2e-3. This is the transformer's own trick (every
    # position sees its whole past) at ~1/10 the cost.
    "iecnn-lr2-plain": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 2e-3}),
    # P=1 + global read: one cell pass (half the cell cost) but full causal
    # context, which is exactly what the 2nd pass was trying to provide.
    "iecnn-lr2-g1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 2e-3}),
    "iecnn-lr2-g1-n8": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=8, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 2e-3}),
    "iecnn-lr2-g1-s8": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3}),
    "iecnn-lr2-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 2e-3}),
    # wider local window is cheap at P=1 (read is ~3% of a pass)
    "iecnn-lr2-g1-s32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(32,), window_mode="hybrid",
        window_reach=64), {"lr": 2e-3}),
    # --- cheapest committee: N and d_s are the two cost knobs left ------
    # Every 10% off the step is worth ~10% more tokens at equal wall clock,
    # which is exactly the size of the bpc gap we are trying to open.
    "iecnn-lr2-g8-n3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-ds48": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n4-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    # --- ONE shared cell, N specialised reads ---------------------------
    # The cell is the expensive part and it is 4 tiny GEMMs (one per dot).
    # Sharing it across dots makes it ONE wide GEMM (~4x the rows, same
    # FLOPs, better shape) and drops 3/4 of the cell parameters. Dots keep
    # their own queries, read mix and output projection, so specialisation
    # moves to the read side. Bonus: N no longer scales the cell, so a
    # bigger committee is nearly free.
    "iecnn-lr2-g8-scell": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), share_cell=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-scell-n8": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), share_cell=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-scell-n8-s16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, per_dot_logits=False,
        global_read=True, share_cell=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- capacity / normalisation, at ~0 extra MACs ---------------------
    "iecnn-lr2-g8-untied": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), head_untied=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-ln": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), cell_ln=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-ln-untied": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), cell_ln=True, head_untied=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-ds96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=96, per_dot_logits=False,
        global_read=True, spans=(8,), cell_ln=True, head_untied=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- per-pass read schedule (zero extra reads) -----------------------
    # Both reads are cached separately, so passes 2..P used to receive a
    # BIT-IDENTICAL ctx: "depth" was nominal. g_glob now has one row per
    # pass, init pass 1 local / pass 2 global. Same reads, same GEMMs.
    "iecnn-lr2-split-s8-swa99": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), read_schedule="split"),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-split-s8-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), read_schedule="split"),
        {"lr": 2e-3, "swa": 0.995}),
    "iecnn-lr2-split-s16-swa99": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, read_schedule="split"),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-split3-s8-swa99": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=3, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), read_schedule="split"),
        {"lr": 2e-3, "swa": 0.99}),
    # --- lr x SWA x window span, on top of the O(T) global read ----------
    # SWA is worth 0.06-0.08 bpc and turns the "constant lr is pure noise at
    # the end" problem into an asset: more lr = more noise = more to average.
    "iecnn-lr3-glob-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 3e-3, "swa": 0.995}),
    "iecnn-lr3-glob-s8-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 3e-3, "swa": 0.995}),
    "iecnn-lr2-glob-s8-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.995}),
    # longer averaging window = less variance (free)
    "iecnn-lr2-glob-s8-swa99": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr3-g1-s8-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 3e-3, "swa": 0.995}),
    # --- tail weight averaging (free: no extra training steps) ----------
    "tf-swa": ("transformer", lambda: tf(96, 4, 3), {"swa": 0.995}),
    "iecnn-lr2-plain-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False),
        {"lr": 2e-3, "swa": 0.995}),
    "iecnn-lr2-glob-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 2e-3, "swa": 0.995}),
    "iecnn-lr2-g1-s8-swa": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.995}),
    "iecnn-lr2-glob-s8": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3}),
    "iecnn-lr2-g1-ds96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, per_dot_logits=False,
        global_read=True), {"lr": 2e-3}),
    "iecnn-lr2-glob-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, window_mode="hybrid", window_reach=64),
        {"lr": 2e-3}),
    "iecnn-lr2-glob": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 2e-3}),
    "iecnn-lr2-glob-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, window_mode="quad", window_reach=64),
        {"lr": 2e-3}),
    "iecnn-lr2-glob-n6": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=6, d_s=64, per_dot_logits=False,
        global_read=True), {"lr": 2e-3}),
    "tf-lr2": ("transformer", lambda: tf(96, 4, 3), {"lr": 2e-3}),
    "tf-lr2-cos": ("transformer", lambda: tf(96, 4, 3),
                   {"lr": 2e-3, "sched": "cos"}),
    "tf-d96-l3-ctl": ("transformer", lambda: tf(96, 4, 3)),
    "iecnn-p2n8-bands": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, per_dot_logits=False,
        window_mode="bands", window_reach=64)),
    "iecnn-p2n8-d384-pool-h64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=8, d_s=64, d=384,
        per_dot_logits=False, pool_first=True, window_mode="hybrid",
        window_reach=64)),
}


# --------------------------------------------------------------------------
# data
# --------------------------------------------------------------------------
def load(path: str, n: int) -> torch.Tensor:
    raw = open(path, "rb").read(n)
    return torch.frombuffer(raw, dtype=torch.uint8).long()


train = load("data/toy_train.txt", 4_000_000)
val = load("data/toy_val.txt", 60_000)
print(f"tokens train={train.numel()} val={val.numel()} | B={B} T={T} V={V} "
      f"| budget {BUDGET:.0f}s/arm", flush=True)


_NT = None


def batches(seed: int = 0):
    g = torch.Generator().manual_seed(seed)
    span = train.numel() - T - 1
    while True:
        # torch.randint over a 4M range measured at 1.19 ms/step — 3.4% of a
        # whole 150 s toy step, paid identically by both arms and buying
        # nothing. torch.rand + scale is the same uniform draw, ~50x cheaper.
        i = (torch.rand(B, generator=g) * span).long()
        x = torch.stack([train[j:j + T] for j in i])
        y = torch.stack([train[j + 1:j + 1 + T] for j in i])
        yield x, y


@torch.no_grad()
def val_bpc(model, windows: int = 48) -> float:
    model.eval()
    tot, n = 0.0, 0
    for w in range(windows):
        i = w * T
        x = val[i:i + T].unsqueeze(0)
        y = val[i + 1:i + 1 + T].unsqueeze(0)
        if x.shape[1] < T:
            break
        lp = model(x)["final_log_probs"]
        tot += F.nll_loss(lp.reshape(-1, V), y.reshape(-1),
                          reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n / math.log(2)


@torch.no_grad()
def pointer_audit(model) -> dict:
    """How the N dots pick tokens: distinct picks + distance distribution."""
    r = model.regions[0]
    g = torch.Generator().manual_seed(7)
    i = torch.randint(0, train.numel() - T - 1, (8,), generator=g)
    x = torch.stack([train[j:j + T] for j in i])
    rows = region1_rows(model.embed(x))
    wind_k, _wv, valid = r._window_memory(rows, x.device)
    q = r.dots._sel(0, r.dots.q)
    scores = torch.einsum("bfwj,nj->bfnw", wind_k, q) / math.sqrt(r.d_q)
    scores = scores.masked_fill(~valid[None, :, None, :], float("-inf"))
    idx = scores.argmax(-1)
    distinct = torch.tensor([len(set(row.tolist()))
                             for row in idx.reshape(-1, idx.shape[-1])])
    w = wind_k.shape[2]
    dist = (w - 1 - idx).float()
    return {
        "distinct_per_pos": round(distinct.float().mean().item(), 2),
        "all_same": round((distinct == 1).float().mean().item(), 3),
        "mean_dist_back": round(dist.float().mean().item(), 2),
        "frac_window_end": round((idx == (w - 1)).float().mean().item(), 3),
    }


SEED = int(os.environ.get("ARM_SEED", "0"))
BASE_LR = float(os.environ.get("ARM_LR", "1e-3"))


def run(name: str, kind: str, model, lr: float = None,
        seed: int = None, sched: str = None, swa: float = 0.0) -> dict:
    lr0 = BASE_LR if lr is None else float(lr)
    sd = SEED if seed is None else int(seed)
    torch.manual_seed(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    n_par = sum(p.numel() for p in model.parameters())
    gen = batches(sd)
    sched = sched or ""
    # Tail weight averaging (SWA/EMA). With a constant lr the last steps
    # bounce around the minimum; averaging the tail is a free variance
    # reduction (one extra eval, no extra training). Active only in the
    # last 55% of the budget, where the weights are already worth keeping.
    ema = None
    if swa:
        ema = [p.detach().clone() for p in model.parameters()]
    t0 = time.time()
    step = 0
    marks = []
    next_mark = MARK_S
    while time.time() - t0 < BUDGET:
        x, y = next(gen)
        warm = min(1.0, (step + 1) / 100.0)
        if sched == "cos":
            # time-based cosine over the FIXED wall-clock budget: the arm
            # cannot know its step count in advance, but it knows the clock.
            frac = min(1.0, (time.time() - t0) / BUDGET)
            warm = warm * 0.5 * (1.0 + math.cos(math.pi * frac))
        cur_lr = lr0 * warm
        for grp in opt.param_groups:
            grp["lr"] = cur_lr
        out = model(x)
        loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
        opt.step()
        step += 1
        if ema is not None and (time.time() - t0) > 0.45 * BUDGET:
            with torch.no_grad():
                torch._foreach_mul_(ema, swa)
                torch._foreach_add_(ema, list(model.parameters()),
                                    alpha=1.0 - swa)
        el = time.time() - t0
        if el >= next_mark:
            marks.append([int(step * B * T), round(val_bpc(model), 4), step])
            next_mark += MARK_S
    el = time.time() - t0
    if ema is not None:
        with torch.no_grad():
            for p, a in zip(model.parameters(), ema):
                p.copy_(a)
    final = round(val_bpc(model), 4)
    row = {"arm": name, "kind": kind, "seed": sd, "lr": lr0, "sched": sched,
           "swa": swa,
           "params": n_par, "steps": step,
           "tokens": step * B * T, "s_per_step": round(el / max(step, 1), 4),
           "s_per_1k_tokens": round(el / max(step * B * T / 1000, 1e-9), 3),
           "curve": marks, "bpc_final": final}
    if kind == "iecnn":
        row["pointers"] = pointer_audit(model)
    print("RESULT " + json.dumps(row), flush=True)
    return row


def bpc_at(marks, tokens: int):
    """Linear interpolation of the (tokens, bpc) curve at a token count."""
    pts = [(m[0], m[1]) for m in marks]
    if not pts or tokens < pts[0][0]:
        return None
    if tokens >= pts[-1][0]:
        return pts[-1][1]
    for (t0_, b0), (t1_, b1) in zip(pts, pts[1:]):
        if t0_ <= tokens <= t1_:
            f = (tokens - t0_) / max(t1_ - t0_, 1)
            return round(b0 + f * (b1 - b0), 4)
    return pts[-1][1]


def main():
    names = ONLY or list(ARMS)
    rows = []
    if os.environ.get("ARM_DRY") == "1":
        # Smoke-test EVERY arm (config builds + one fwd/bwd on a real batch)
        # before any long run is launched. A NameError in one arm used to
        # burn a whole batch: the log showed nothing but a traceback.
        ok = 0
        for name in names:
            entry = ARMS[name]
            kind, mk = entry[0], entry[1]
            torch.manual_seed(0)
            try:
                model = (build_transformer_baseline(mk(), V) if kind == "transformer"
                         else build_iecnn(mk(), V))
                x, y = next(batches(0))
                out = model(x)
                loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                                       y.reshape(-1))
                loss.backward()
                print(f"DRY OK   {name:<28}"
                      f"{sum(p.numel() for p in model.parameters()):>9} params"
                      f"  loss {loss.item():.3f}", flush=True)
                ok += 1
            except Exception as e:  # noqa: BLE001 - report and keep going
                print(f"DRY FAIL {name:<28}{type(e).__name__}: {e}", flush=True)
        print(f"DRY {ok}/{len(names)} arms ok", flush=True)
        return
    for name in names:
        entry = ARMS[name]
        kind, mk = entry[0], entry[1]
        opts = entry[2] if len(entry) > 2 else {}
        torch.manual_seed(opts.get("seed", SEED))
        cfg = mk()
        model = (build_transformer_baseline(cfg, V) if kind == "transformer"
                 else build_iecnn(cfg, V))
        print(f"--- {name} ({sum(p.numel() for p in model.parameters())} params"
              f"{', lr ' + str(opts['lr']) if 'lr' in opts else ''})", flush=True)
        rows.append(run(name, kind, model, **opts))

    common_tok = min(r["tokens"] for r in rows)
    print(f"\n=== SUMMARY (equal wall-clock {BUDGET:.0f}s/arm, "
          f"common {common_tok/1e6:.2f}M tokens) ===", flush=True)
    hdr = f"{'arm':<18}{'params':>8}{'steps':>7}{'tokens':>9}{'s/step':>9}" \
          f"{'s/1k tok':>10}{'bpc_fin':>9}{'bpc@eqtok':>11}"
    print(hdr, flush=True)
    for r in rows:
        be = bpc_at(r["curve"], common_tok)
        print(f"{r['arm']:<18}{r['params']:>8}{r['steps']:>7}"
              f"{r['tokens']/1e6:>8.2f}M{r['s_per_step']:>9.4f}"
              f"{r['s_per_1k_tokens']:>10.2f}{r['bpc_final']:>9.4f}"
              f"{('%.4f' % be) if be is not None else '   -':>11}", flush=True)
        if "pointers" in r:
            print(f"{'':<18}pointers {r['pointers']}", flush=True)
    print("\nJSON " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
