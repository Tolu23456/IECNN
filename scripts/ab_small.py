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
# Context, corpus and val sampling are harness knobs (both arms always get
# the same value). Defaults reproduce the 2-vCPU toy exactly.
T = int(os.environ.get("ARM_T", 64))
TRAIN_PATH = os.environ.get("ARM_TRAIN", "data/toy_train.txt")
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_TRAIN = int(os.environ.get("ARM_NTRAIN", 4_000_000))
N_VAL = int(os.environ.get("ARM_NVAL", 60_000))
VALW = int(os.environ.get("ARM_VALW", 160))
# val windows are contiguous from the start by default (the toy val is only
# 60 KB). On a multi-MB val set set ARM_VALSTRIDE>1 to spread the windows
# over the whole file instead of scoring only its first 40 KB.
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", 1))
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
          cell_ln=False, share_cell=False, query_from_input=False,
          id_lags=False, ngram_bias=0, head_rank=0,
          state_carry=False, carry_chunk=8, gla=False,
          global_exact=False, head_skip=False, break_scan=False,
          d_q=16, d_v=32, head_post=0, head_post_layers=1,
          interleave=False, read_ln=False, head_ln=False, read_heads=1,
          global_decay=False, post_share="none", global_decay_k=1,
          gate_agree=False):
    m = {
        "kind": "iecnn", "d": d, "d_q": d_q, "d_v": d_v, "d_s": d_s,
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
        "gate_agree": gate_agree,
        "slot_slots": slot_slots,
        "dot_dist_bias": dist_bias,
        "global_read": global_read,
        "read_schedule": read_schedule,
        "head_untied": head_untied,
        "cell_ln": cell_ln,
        "share_cell": share_cell,
        "query_from_input": query_from_input,
        "id_lags": id_lags,
        "ngram_bias": ngram_bias,
        "head_rank": head_rank,
        "state_carry": state_carry,
        "carry_chunk": carry_chunk,
        "global_linear": gla,
        "global_exact": global_exact,
        "head_skip": head_skip,
        "break_scan": break_scan,
        "head_post": head_post,
        "head_post_layers": head_post_layers,
        "interleave": interleave,
        "read_ln": read_ln,
        "head_ln": head_ln,
        "read_heads": read_heads,
        "global_decay": global_decay,
        "global_decay_k": global_decay_k,
        "post_share": post_share,
    }
    if window_reach is not None:
        m["window_reach"] = window_reach
    if window_local is not None:
        m["window_local"] = window_local
    return {"model": m,
            "consensus": {"rho": 0.0, "kappa": 0.7},
            "training": {"compile_consensus": False, "dot_dropout": 0.0},
            "data": {"seq_len": T}}


def tf(d_model=96, n_heads=4, n_layers=3, ngram_bias=0):
    return {"model": {"kind": "transformer", "d_model": d_model,
                      "n_heads": n_heads, "n_layers": n_layers,
                      "ngram_bias": ngram_bias},
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
    # --- content-dependent query (the one thing attention has) ----------
    # q_t = q_n + W x_t on the WINDOW read only (the global read keeps its
    # static query so the O(T) prefix sum survives). W zero-init.
    "iecnn-lr2-g8-qin": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, spans=(8,), query_from_input=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-qin-s16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=64, per_dot_logits=False,
        global_read=True, query_from_input=True),
        {"lr": 2e-3, "swa": 0.99}),
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
    # --- cost frontier at equal wall clock (compiled) -------------------
    # torch.compile fuses the elementwise cell graph on CPU: measured
    # -24% s/step for IECNN vs -5% for the transformer (GEMM-bound), so the
    # cheap-config frontier must be re-mapped with compilation ON.
    "iecnn-lr2-g8-n3-ds48-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n2-ds48": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=32, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-d96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    # --- head width d: the biggest remaining GEMM (2026-11-18) ----------
    # logits = h[d] @ E[d,V]; at d=128 that is 16.8M MACs/step, comparable to
    # the whole cell. d also sets the memory-row width. Compiled 150 s run
    # showed d=128 DEGRADING after ~2M tokens (4.06 -> 4.19) while d=96 kept
    # falling (4.29 -> 4.05): the wide head is not just expensive, it is
    # worse-conditioned at this budget.
    "iecnn-lr2-g8-n3-ds48-d80": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=80, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-d64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=64, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds64-d96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=64, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n4-ds48-d96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-d96-b32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99, "batch": 32}),
    # lr / swa retune on the d96 shape (150 s now buys ~7700 steps, 35% more
    # than when lr 2e-3 was chosen at ~5100 steps).
    "iecnn-g8-d96-lr15": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 1.5e-3, "swa": 0.99}),
    "iecnn-g8-d96-lr25": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2.5e-3, "swa": 0.99}),
    "iecnn-g8-d96-swa995": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.995}),
    "iecnn-lr2-g8-n3-ds48-d96-r48": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), head_rank=48),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-d96-r24": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), head_rank=24),
        {"lr": 2e-3, "swa": 0.99}),
    # --- content-addressed read + exact identity, on the d96 shape ------
    # The d96 control reads with a STATIC query: "a dot always asks the same
    # question of every window", so it can only average the window, never
    # address it by content. query_from_input gives each dot a query built
    # from the CURRENT token (q_t = q_n + W x_t, W zero-init) so slot scores
    # become "how much does this past token look like the one I am holding".
    # Cost: one [d,d_q] GEMM, no extra reads, still causal.
    # id/ng are the batch-19 mechanisms retested on the current corpus split
    # (the old split was book-disjoint; this one is same-source, so identity
    # copying is worth more here). Same flags, same code.
    "iecnn-d96-qin": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), query_from_input=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d96-id": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), id_lags=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d96-ng": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), ngram_bias=32),
        {"lr": 2e-3, "swa": 0.99}),
    # --- content-addressed LINEAR-attention global read (zero new params)
    # The static-query prefix sum can only weight the prefix with a FIXED
    # kernel; the transformer can address it by content. Linear attention
    # gives that at O(T) and with no parameters: ctx = phi(q)^T S / phi(q)^T z
    # with S = cumsum(phi(k) v^T). The old read is its phi==1 special case.
    # gla      : phi(q) from the existing static per-dot query.
    # gla-q    : phi(q) also carries the existing w_qi content term, so the
    #            query depends on the current token (the only piece qin was
    #            missing: _global_ctx used to ignore q_add entirely).
    "iecnn-d96-gla": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), gla=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d96-gla-q": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), gla=True, query_from_input=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the two zero-parameter mechanisms the mechanism audit asked for ---
    # res: skip the current token's embedding past the cell and into the head.
    # Today E[x_t] can only reach the logits through a softmax over 8 slots, a
    # gated tanh and a 3-way weight; the transformer gets h = h + f(h) twice
    # per block for free. 0 params, 0 time.
    "iecnn-d96-res": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), head_skip=True),
        {"lr": 2e-3, "swa": 0.99}),
    # gx: the STRONG form of the far read — exact causal softmax over every
    # past key, with a query that depends on the current token. `gla` was the
    # linear-attention approximation of this and lost; the static prefix sum
    # is the q-independent degenerate case. This is the only arm that tests
    # "content-addressed far read" as such.
    "iecnn-d96-gx": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), global_exact=True,
        query_from_input=True), {"lr": 2e-3, "swa": 0.99}),
    # --- capacity at GPU scale (2026-11-18) -----------------------------
    # On the 2-vCPU box every capacity knob lost because it bought param-
    # eters with wall clock (the cell is elementwise-bound). On a GPU the
    # step is launch-bound, so the same parameters are nearly free and the
    # only question left is whether the architecture can USE them.
    # d192/ds128/N4 ~ 0.5M params; d256/ds256/N6 ~ 1.6M; matched tf below.
    "iecnn-d192-ds128-n4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=128, d=192, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    # the same arm with each zero-parameter mechanism grafted on, so the GPU
    # comparison is mechanism-for-mechanism at the capacity point that won.
    "iecnn-d192-res": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=128, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_skip=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-gx": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=128, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), global_exact=True,
        query_from_input=True), {"lr": 2e-3, "swa": 0.99}),
    # bracket the capacity sweet spot: d192 ran out of tokens at 48M, d96 was
    # under-capacity at 72k params, so ~200k params may be the fixed point.
    "iecnn-d128-ds96-n4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    # --- widening the READ instead of the model -------------------------
    # Every capacity sweep so far widened d and d_s, which are the expensive
    # axes: d enters the head (d*vocab) and the cell input (d*d_s), both
    # quadratic. The read itself is only d_q=16 / d_v=32 wide and its cost is
    # linear in d_v (a [T,N,d_v] gather and one bmm), so this is capacity that
    # costs tokens at a third of the rate. At d192 the context is 32 dims
    # summarising a 128-dim state, which looks like the obvious starvation.
    "iecnn-d192-dv64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=128, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), d_v=64), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-dv96": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=4, d_s=128, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), d_v=96), {"lr": 2e-3, "swa": 0.99}),
    # --- the missing block: per-position MLP after pooling ----------------
    # The dots' read is ~10k MAC/token against attention's ~98k, but the cell
    # spends ~412k/token at d192 (N=4 x 179-dim input projection + 128->192
    # output projection) against ~172k for an entire tf block, and there is no
    # per-position MLP. These two arms spend the read's saving where the
    # transformer puts 2/3 of its parameters: LN + residual + GELU MLP after
    # the population is pooled. Cheap dots (n_dots=2, d_s=64) keep the total
    # cost near tf's, so it is a cost-matched architectural test, not a
    # "throw parameters at it" test.
    "iecnn-d128-n2-pmlp": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=64, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384),
        {"lr": 2e-3, "swa": 0.99}),
    # The block won on BOTH axes (-0.095 bpc AND -35% step time), so the next
    # question is how much further the same trade goes: fewer/cheaper dots,
    # a wider block, more blocks, a wider model.
    "iecnn-d192-n1-pmlp": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=1, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-w768": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=768),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x2": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=2),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d256-n2-pmlp": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=256, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512),
        {"lr": 2e-3, "swa": 0.99}),
    # Depth is where the block pays: x2 beat both a wider block and a wider
    # model at the same wall clock. Keep going, and test whether the block
    # makes the SECOND PASS redundant -- each pass re-runs the whole cell, so
    # dropping it nearly halves the step and doubles the token budget for a
    # model whose per-position work now lives in the block.
    "iecnn-d192-n2-pmlp-x3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=3),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x2-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=2),
        {"lr": 2e-3, "swa": 0.99}),
    # The two wins combine: one pass was better than two (2.1909 vs 2.2108)
    # and cheaper, and depth saturates at three blocks. These are the
    # candidates for the best single arm under the equal-wall-clock rule.
    "iecnn-d192-n2-pmlp-x3-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=3),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x4-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    # one pass frees ~40% of the step, so spend it on the population: with a
    # single pass the cell is run once, and N=3 may now be affordable.
    "iecnn-d192-n3-pmlp-x3-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=3),
        {"lr": 2e-3, "swa": 0.99}),
    # --- interleave: read -> block -> read -> block ------------------------
    # A transformer re-MIXES positions at every layer (attention -> MLP ->
    # attention); this model mixes once and then stacks MLPs, which is exactly
    # why block depth saturated (x4 == x3). With `interleave`, each pass reads
    # the rows as refined by the previous pass's block, so depth adds new
    # MIXED information instead of re-processing the same gathered context.
    # The read is cheap (~10k MAC/token vs attention's ~98k); the cost is one
    # more cell pass, and the cell is now the small part of the step.
    "iecnn-d192-n2-pmlp-il": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=1,
        interleave=True), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x2-il": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=2,
        interleave=True), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x2-il-p3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=3, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=2,
        interleave=True), {"lr": 2e-3, "swa": 0.99}),
    # Depth had NOT saturated at x4 with one pass (2.1035 < 2.1571 < 2.1918),
    # and N=3 paid too (2.1022 at x3). Push both, and test interleave at the
    # population that works.
    "iecnn-d192-n3-pmlp-x4-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n2-pmlp-x6-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=6),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d192-n3-pmlp-x2-il": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=2,
        interleave=True), {"lr": 2e-3, "swa": 0.99}),
    # --- the two LayerNorms a transformer has and this model never did ------
    # read_ln: normalise rows before w_k/w_v. Today the read sees raw
    # embeddings, and any mechanism that writes back into `rows` (interleave,
    # deep stacks) makes that stream's scale depth-dependent.
    # head_ln: final LN before the tied output matrix (transformer ln_f).
    "iecnn-n3-x4-p1-ln": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True), {"lr": 2e-3, "swa": 0.99}),
    # the ordered receptive field: span 8 -> 16 (4x the ordered range for ~4%
    # of the read's negligible cost). Was pruned on the 2-vCPU toy, where any
    # added fitting power is punished and both arms memorise.
    "iecnn-n3-x4-p1-s16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(16,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    # depth and population both still pay at x4/N3, so push each once more.
    "iecnn-n3-x5-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n4-x4-p1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    # Cost levers at the x4/N3 shape that won. Blocks dominate the parameter
    # count (2*d*384 each), so a narrower block is the cheapest way to buy
    # step time; d_s is the cell's quadratic axis; and the untied head is the
    # one handicap that has only ever been tested on the invalid toy arena.
    "iecnn-n3-x4-p1-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ds64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=64, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-untied": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        head_untied=True), {"lr": 2e-3, "swa": 0.99}),
    # --- multi-head read: more softmax PATTERNS, not more width -------------
    # Each dot reads its window with ONE query today, so it cannot attend to
    # the previous token and to the token six back at the same time; a
    # transformer gets 4 patterns per layer for free. Heads split the existing
    # d_v into H slices, so total context width, total value MACs and the ctx
    # size are unchanged and the only new parameters are (H-1)*d_q per dot
    # (48 for N=3 at H=2). H=1 is bit-identical to the current read.
    "iecnn-n3-x4-p1-h2": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_heads=2), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-h4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_heads=4), {"lr": 2e-3, "swa": 0.99}),
    # --- recency in the prefix read ----------------------------------------
    # The global read is an unordered bag: e_s = exp(q.k_s) has no distance
    # term, so the token 10 back and the token 200 back are indistinguishable
    # to it. One learned per-dot geometric decay per step fixes that for
    # N extra scalars and no extra MACs (gamma^-s multiply before the scan).
    "iecnn-n3-x4-p1-decay": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        global_decay=True), {"lr": 2e-3, "swa": 0.99}),
    # --- the LN pair won (-0.038 bpc at equal wall clock), so re-ask the
    # structural questions WITH it: does depth go further now, is the smaller
    # population enough now, and do the two new read mechanisms pay on top?
    "iecnn-n3-x5-p1-ln": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n2-x4-p1-ln": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=2, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-h2": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, read_heads=2), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the two LN-batch wins, combined and shrunk ------------------------
    # `decay` was the best arm in its session (2.0521 vs tf 2.0745) and `x5`
    # stopped saturating once the LN pair was there (2.0574 vs 2.0671), so
    # combine them; then pay for the depth by making the block stack cheaper
    # (w256 cuts 148k params, d128 cuts 235k) to test whether the win survives
    # below the transformer's 408k parameter count.
    "iecnn-n3-x5-p1-ln-decay": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-d128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # d128 beat tf per step AND per token at 1.37x its parameters, while w256
    # at d192 was the worst arm in the same session. Both cut ~200k, so the
    # axis is not the parameter count: it is the block's EXPANSION RATIO
    # (w/d), 3.0 at d128-w384 vs 1.33 at d192-w256. These arms push the
    # winning stream width instead: depth at d128, and d160 as a third point.
    "iecnn-n3-x5-p1-ln-decay-d128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x6-p1-ln-decay-d128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x5-p1-ln-decay-d160": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=160, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-h2": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, read_heads=2),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the parameter frontier -------------------------------------------
    # Width has never beaten depth at equal parameters (2.2005 two blocks at
    # w384 < 2.2349 one block at w768), and a block costs 384*w + 960 params,
    # so the efficient direction is NARROW AND DEEP: at w256 six blocks cost
    # the same 786k as four w384 blocks, and at w192 eight blocks do. These
    # are the points that decide whether the 593k block stack can be paid for
    # with depth instead of width, and the d128 pairs test whether the stream
    # itself can be narrowed.
    "iecnn-n3-x5-p1-ln-decay-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x6-p1-ln-decay-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x8-p1-ln-decay-w192": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=192, head_post_layers=8,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x5-p1-ln-decay-d128-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- is the compressed ladder a PRIOR problem or a CAPACITY problem? ---
    # Audit of the d128 frontier: the three dots settle at 0 / 0.8 / 2.8 tokens
    # back and two of them read with JS 0.015 nats, i.e. the scale ladder that
    # d192 gets for free is compressed at d128. dist_bias is ONE learned scalar
    # per (dot, slot) added to that dot's read score -- 24 parameters, zero
    # measurable compute -- so a dot can price a distance instead of having to
    # discover it through the query. It was pruned early on the OLD config,
    # where the diagnosis was "every dot picks the same slot"; that diagnosis
    # is now confirmed on the frontier shape and the fix is free, so it is
    # re-tested where the failure actually is. If the horizons spread and bpc
    # falls, the compression was a prior problem; if nothing moves, it is
    # capacity (the d_s funnel) and no bias can fix it.
    "iecnn-n3-x5-p1-ln-decay-d128-db": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True, dist_bias=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- fewer-but-wider blocks: the only remaining way to be MUCH faster ---
    # Blocks are 74% of the step and cost 2*d*w per layer, so the frontier has
    # a constant-compute line: x5/w384 = 491k MAC/token per position, x4/w448 =
    # 459k, x3/w512 = 393k. If the ratio sweep says w>3d pays, then trading two
    # blocks for a wider MLP is a straight 20% off the step at equal nominal
    # capacity, which is the only shape of change that can make this model
    # MASSIVELY faster rather than 16% faster.
    "iecnn-n3-x4-p1-ln-decay-d128-w512": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # two layers at the same MAC budget as three-wider: x2/w768 = 2*128*768*2
    # = 393k MAC/token, identical to x3/w512 and 20% under x5/w384. This is the
    # depth-vs-width trade AT CONSTANT COMPUTE, which is the only form of the
    # question that matters for wall clock.
    "iecnn-n3-x2-p1-ln-decay-d128-w768": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=768, head_post_layers=2,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x3-p1-ln-decay-d128-w512": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- where does WIDTH turn over? (wide-shallow is now the frontier) -----
    # x3/w512 beat x5/w384 on both currencies; w/d there is 4.0 and the width
    # axis has never turned over. w640 (5.0) and w768 (6.0) both stay at or
    # under x5/w384's MAC count (491k: w640 = 491k, w768 = 590k), so if the
    # monotone trend continues the frontier keeps moving without paying.
    "iecnn-n3-x3-p1-ln-decay-d128-w640": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=640, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x3-p1-ln-decay-d128-w768": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=768, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # do the two surviving wins COMPOSE? The read is 0.4% of the step, so span
    # 12 on top of the wide-shallow block should be nearly free if both are real.
    "iecnn-n3-x3-p1-ln-decay-d128-w512-s12": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(12,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- compile the frontier, three ways --------------------------------
    # Compile bought the transformer 25% and the dot model 0%. These arms ask
    # whether that is fixable engineering: default compile, CUDA graphs
    # (reduce-overhead; shapes are static), and compile with the prefix-scan
    # eager boundary removed. Same arms as tf so the harness is like-for-like.
    "iecnn-n3-x3-p1-ln-decay-d128-w512-c": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "compile": True}),
    "iecnn-n3-x3-p1-ln-decay-d128-w512-cro": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "compile": "reduce-overhead"}),
    "iecnn-n3-x3-p1-ln-decay-d128-w512-cns": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "compile": "noscan"}),
    # --- a compile-friendly read: prefix SUM -> masked MATMUL --------------
    # The one graph break the region was forced to keep is an eager boundary
    # around the prefix scan (Inductor's SplitScan lowering crashed on CUDA).
    # `global_exact` computes the same static-query far read as a lower-
    # triangular masked matmul instead: no scan, nothing for Inductor to
    # refuse, and with a STATIC query it is the same function up to numerics,
    # so a bpc difference would be a bug. Cost is T*d_v*N MAC per token
    # (~24k at T=256, d_v=32, N=3 = ~3% of the wide-shallow block), which
    # compile would repay many times over if it works.
    "iecnn-n3-x3-p1-ln-decay-d128-w512-gx": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True, global_exact=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x3-p1-ln-decay-d128-w512-gx-c": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True, global_exact=True),
        {"lr": 2e-3, "swa": 0.99, "compile": True}),
    "tf-swa-cro": ("transformer", lambda: tf(96, 4, 3),
                   {"swa": 0.995, "compile": "reduce-overhead"}),
    # --- the cheap end of the frontier: is x3 enough? ----------------------
    # x4 (559k, 0.0212 s/step) is 16% faster than tf; x5 adds 0.004 bpc for
    # 14% more step time. x3 has never been run at d128, and it is the arm
    # that would matter most for "massively faster": if x3 keeps within noise
    # of tf on bpc it is a ~25-30% faster model at equal quality.
    "iecnn-n3-x3-p1-ln-decay-d128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=3,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the expansion ratio ABOVE 3, at fixed d ----------------------------
    # Every w/d data point so far is on the low side: 1.33 (d192-w256, bad),
    # 2.0 (d192-w384 / d128-w256, bad), 2.4 (d160-w384, bad), 3.0 (d128-w384,
    # best). The ratio has only ever been tested where it was too SMALL, so
    # "3.0 is optimal" is unsupported — 3.0 may be the edge of a curve that
    # keeps rising, or the peak. These arms hold d=128 fixed and move w only,
    # so the ratio axis is isolated: w320 (2.5), w448 (3.5), w512 (4.0). If
    # 3.5 or 4.0 wins, the block wants a wider MLP and the frontier moves
    # there; if 2.5 holds its own, the ratio is a plateau and parameters are
    # better spent on depth.
    "iecnn-n3-x5-p1-ln-decay-d128-w320": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=320, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x5-p1-ln-decay-d128-w448": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=448, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x5-p1-ln-decay-d128-w512": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=512, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- eager vs torch.compile, in one session -----------------------------
    # The throughput batch showed tokens are the only currency that buys bpc,
    # and it killed batch size as a lever. Compile is the other way to buy
    # tokens: it is the only remaining harness knob. It has to be measured
    # in-session against an eager twin because its warmup comes out of the
    # same 180 s, so the question is whether the steady-state win repays the
    # trace. ARM_BREAK_SCAN=1 makes the prefix scan an eager graph boundary,
    # which is what made the CUDA Inductor crash disappear.
    "tf-swa-c": ("transformer", lambda: tf(96, 4, 3),
                 {"swa": 0.995, "compile": True}),
    "iecnn-n3-x5-p1-ln-decay-d128-c": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "compile": True}),
    # --- the far dot may be capped by the WINDOW, not by the data -----------
    # Measured horizons on the winning arm are 0.00 / 2.96 / 5.87 tokens back
    # out of an 8-slot contiguous window: the far dot sits one slot from the
    # edge. Widening the window is the cheapest possible fix — the read is
    # ~0.4% of the step, so 16 slots cost ~1.5k MAC/token against 800k, and
    # it needs no new code, no new parameters and no new gradient path. If the
    # horizons spread (0/5/11) and bpc drops, the structure was
    # under-provisioned; if nothing moves, 8 slots already cover the scale the
    # data has. `d_bias` (a free per-dot per-slot score table) is deliberately
    # NOT retried here: it was pruned once, and the audit shows the horizons
    # are already learnt without it.
    "iecnn-n3-x5-p1-ln-decay-d128-s12": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(12,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x5-p1-ln-decay-d128-s16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(16,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # seed-1 twins: span 12 and N=4-span-16 both came back inside one-seed noise
    # (0.024 is the transformer's own spread), so neither is a result yet. The
    # seed-0 twins are in the same session, so the comparison is paired.
    "iecnn-n3-x5-p1-ln-decay-d128-s12-s1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(12,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "seed": 1}),
    "iecnn-n4-x5-p1-ln-decay-d128-s16-s1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(16,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "seed": 1}),
    "iecnn-n4-x5-p1-ln-decay-d128-s16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(16,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- more horizons, now that we know the dots CHOOSE a horizon ----------
    # The audit found the three dots sitting at 0 / ~3 / ~6 tokens back out of
    # an 8-slot window, reading JS 0.20-0.27 and agreeing on 6-10% of
    # predictions. That is a multi-scale receptive field nobody designed, and
    # 5.87 back is close to the window edge, so the far dot may be capped by
    # the SPEC rather than by the data. With a quad window (8 slots spread
    # geometrically to reach) four dots can cover four real scales — 0, a few,
    # tens, and tens-plus — for the same read compute. N=4 lost before
    # (span 16, contiguous), but there the four horizons could only be 0/1/2/3;
    # these arms test whether N=4 was starved of SCALES rather than of dots.
    "iecnn-n4-x4-p1-ln-decay-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True,
        window_mode="quad", window_reach=64), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n4-x4-p1-ln-decay-q128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=4, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True,
        window_mode="quad", window_reach=128), {"lr": 2e-3, "swa": 0.99}),
    # --- the gate is blind to the committee (gate_agree) --------------------
    # gamma is a scalar read of a dot's OWN state, so the gate cannot express
    # "this dot disagrees with the other two". Audit finding: the gate is
    # near-uniform (0.29/0.37/0.34 vs 0.333), i.e. the pool is close to a
    # plain mean of three specialists. gate_agree adds one number per dot:
    # cosine(state_n, mean_m state_m) at the same position. Zero-initialised,
    # N scalars, ~0.4% of the step. If a real gate is worth anything, this is
    # the cheapest information it is currently missing.
    "iecnn-n3-x5-p1-ln-decay-d128-ga": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True, gate_agree=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-ga": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, gate_agree=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- SEED REPLICATION of the headline claim -----------------------------
    # The d128 family beats tf by 0.007 (x4) / 0.018 (x5) in one session. The
    # seed noise on a single run is 0.02-0.05, so that is not yet evidence;
    # the ledger's own rule says a claim needs >=2 seeds or a margin above the
    # noise. These are the SAME configs with seed 1, and the seed-0 twins run
    # in the same session, so the comparison is paired and drift-free.
    "tf-swa-s1": ("transformer", lambda: tf(96, 4, 3),
                  {"swa": 0.995, "seed": 1}),
    "iecnn-n3-x5-p1-ln-decay-d128-s1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "seed": 1}),
    "iecnn-n3-x4-p1-ln-decay-d128-s1": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "seed": 1}),
    # --- pool_first on the frontier: is 7% of the step free? ----------------
    # The dot->stream readout (`p_out`) costs N*d_s*d MACs/token (55k at
    # d192/d_s96/N3 = 7% of the step) and materialises [B,T,N,d] before the
    # pool. Sharing ONE projection and pooling the d_s states first turns the
    # sum_n w_n (s_n W_n) into (sum_n w_n s_n) W: same head, 1/N the MACs and
    # 1/N the parameters, but only IF the per-dot W_n can be replaced by one
    # W. The ledger calls this an "exact prune"; that is true of the algebra,
    # not of the model (the per-dot matrices differ), so it gets measured
    # here rather than assumed. A win means 7% of the step is free; a loss
    # means the per-dot readout is doing real work, like the block stack.
    "iecnn-n3-x5-p1-ln-decay-d128-pf": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True, pool_first=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-pf": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, pool_first=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the winning SHAPE at a smaller d, preserving w/d = 3 ---------------
    # The parameter axis that has ever paid is the block EXPANSION RATIO
    # w/d (3.0 at d128-w384 wins; 1.33 at d192-w256 loses), not the raw
    # count. That makes a prediction: the same ratio at a narrower stream with
    # MORE layers should keep the quality while cutting the step time, because
    # blocks cost 2*d*w per layer and 2*d*w at (96,288) is 44% cheaper than at
    # (128,384). Same three dots (the read is 0.4% of the step, so the cell's
    # d_s is the only fixed cost). These arms test the ratio invariant rather
    # than adding a mechanism: if (96,288) tracks (128,384) at equal depth,
    # the frontier gets cheaper; if it loses, the ratio was a coincidence of
    # two points and d=128 stays the floor.
    "iecnn-n3-x6-p1-ln-decay-d96-w288": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=288, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x8-p1-ln-decay-d96-w288": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=288, head_post_layers=8,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x6-p1-ln-decay-d112-w336": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=112, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=336, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- THROUGHPUT SWEEP: is the step launch-bound or FLOP-bound? ----------
    # Every arm so far has been measured at B=32, T=256 (8192 tok/step). On a
    # T4 with eager execution the step is a few hundred kernel launches, and
    # the launch cost does not grow with batch size while the arithmetic does
    # — so if the step is launch-bound, doubling B buys nearly free tokens and
    # the same 180 s buys a third more DATA (bpc at fixed wall clock falls).
    # The transformer gets the same treatment: this is a harness axis, not an
    # architecture claim, so the anchor is re-measured at each B in-session.
    # lr is deliberately NOT retuned per B (fairness: same optimiser for both
    # arms); a B win with the same lr is a pure throughput win.
    "tf-swa-b64": ("transformer", lambda: tf(96, 4, 3),
                   {"swa": 0.995, "batch": 64}),
    "tf-swa-b128": ("transformer", lambda: tf(96, 4, 3),
                    {"swa": 0.995, "batch": 128}),
    "iecnn-n3-x4-p1-ln-decay-b64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "batch": 64}),
    "iecnn-n3-x4-p1-ln-decay-b128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "batch": 128}),
    "iecnn-n3-x5-p1-ln-decay-d128-b64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "batch": 64}),
    "iecnn-n3-x5-p1-ln-decay-d128-b128": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=5,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99, "batch": 128}),
    "iecnn-n3-x6-p1-ln-decay-d128-w256": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=128, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=256, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    # --- shared block weights: depth of compute, one copy of the parameters --
    # The stack is 74% of the model, and it is four near-identical MLPs. One
    # block applied four times is a universal-transformer stack: same step
    # cost as x4, 3/4 fewer block parameters (795k -> 350k, BELOW the
    # transformer's 408k). "up" shares only the feature basis and keeps a
    # readout per block.
    "iecnn-n3-x4-p1-ln-decay-shareall": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, post_share="all"),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x6-p1-ln-decay-shareall": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=6,
        read_ln=True, head_ln=True, global_decay=True, post_share="all"),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-shareup": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, post_share="up"),
        {"lr": 2e-3, "swa": 0.99}),
    # --- K timescales instead of one --------------------------------------
    # One decay rate is a guess about the corpus; a mixture is the model
    # choosing its own recency kernel. K reads cost K extra prefix scans
    # (~1/100 of a block each) and N*(K-1) parameters.
    "iecnn-n3-x4-p1-ln-decay-k3": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, global_decay_k=3),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-k5": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True, global_decay_k=5),
        {"lr": 2e-3, "swa": 0.99}),
    # --- the cheap half of "regions": several receptive fields, ONE trunk ----
    # A second Region module today means a second dot population AND a second
    # post stack: n_regions=2 is 2x the step for a depth-like gain, so it can
    # never pay. The read is 0.4% of the step, though (2.8k MAC/token against
    # 590k for the block stack), so more VIEWS over the same trunk are nearly
    # free. The cheapest version needs no new code at all: `quad`/`bands` put
    # the same 8 slots on a geometric axis reaching window_reach, so the read
    # sees 64 tokens back for exactly the compute of 8. Both were pruned on
    # the 2-vCPU toy, where added fitting power is punished by construction.
    "iecnn-n3-x4-p1-ln-decay-q32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True,
        window_mode="quad", window_reach=32), {"lr": 2e-3, "swa": 0.99}),
    "iecnn-n3-x4-p1-ln-decay-q64": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True,
        window_mode="quad", window_reach=64), {"lr": 2e-3, "swa": 0.99}),
    # --- the dot cell, which no GPU arm has ever varied ----------------------
    # lug (a diagonal recurrence gain: one timescale per state dim) was chosen
    # for cost, never measured against the gru it replaced.
    "iecnn-n3-x4-p1-ln-decay-gru": ("iecnn", lambda: iecnn(
        "gru", True, n_passes=1, n_dots=3, d_s=96, d=192, per_dot_logits=False,
        global_read=True, spans=(8,), head_post=384, head_post_layers=4,
        read_ln=True, head_ln=True, global_decay=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d256-ds256-n6": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=6, d_s=256, d=256, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99}),
    "tf-d192-l6": ("transformer", lambda: tf(192, 4, 6), {"swa": 0.995}),
    "tf-d128-l4": ("transformer", lambda: tf(128, 4, 4), {"swa": 0.995}),
    # --- cross-position state carry (2026-11-18) ------------------------
    # The cell has always RESET to the learned s0 at every position, so the
    # dots are a per-position committee and all temporal structure lives in
    # the fixed read window. A toy measurement put the reset at ~0.42 bpc
    # (reset 4.21 vs carry 3.79), the largest single lever measured so far.
    # state_carry turns the cell into an exact scan over the sequence (gate
    # loses its state term -> affine recurrence -> two-level scan, no
    # division). lr variants because a 64-step recurrence changes the
    # effective gradient scale; c16 doubles the chunk depth of the scan.
    "iecnn-d96-carry": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), state_carry=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-d96-carry-lr15": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), state_carry=True),
        {"lr": 1.5e-3, "swa": 0.99}),
    "iecnn-d96-carry-lr25": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), state_carry=True),
        {"lr": 2.5e-3, "swa": 0.99}),
    "iecnn-d96-carry-c16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, d=96, per_dot_logits=False,
        global_read=True, spans=(8,), state_carry=True, carry_chunk=16),
        {"lr": 2e-3, "swa": 0.99}),
    # --- op-shape: rows per step at equal wall clock (2026-11-18) -------
    # Measured throughput (scripts/throughput.py) is 15.5k tok/s at B=8 but
    # 21k at B=32 for the same model: bigger steps are cheaper per token.
    # Wall clock is fixed, so a wider step buys tokens; the question is
    # whether the loss of update frequency costs more than the tokens gain.
    "tf-swa-b32": ("transformer", lambda: tf(96, 4, 3),
                   {"swa": 0.995, "batch": 32}),
    "iecnn-lr2-g8-n3-ds48-b16": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99, "batch": 16}),
    "iecnn-lr2-g8-n3-ds48-b32": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 2e-3, "swa": 0.99, "batch": 32}),
    "iecnn-lr2-g8-n3-ds48-b32-lr4": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,)), {"lr": 4e-3, "swa": 0.99, "batch": 32}),
    # --- exact-identity pathways (2026-11-18) ---------------------------
    # The read is a POOLED window: it carries what the neighbourhood means,
    # not which tokens are in it. Order-2 byte entropy on the val set is
    # 3.68 bpc while these models sit at ~4.0, so exact short-range identity
    # is the largest cheap signal still on the table.
    #   id : dot n is injected with the exact token at lag n+1 (one GEMM,
    #        no extra cell width: the code is added to the value channel).
    #   ng : a rank-32 factorised order-1 prior on the head logits
    #        (lookup + one GEMM, 16k params).
    "iecnn-lr2-g8-n3-ds48-id": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,), id_lags=True),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-ng": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,), ngram_bias=32),
        {"lr": 2e-3, "swa": 0.99}),
    "iecnn-lr2-g8-n3-ds48-id-ng": ("iecnn", lambda: iecnn(
        "lug", True, n_passes=2, n_dots=3, d_s=48, per_dot_logits=False,
        global_read=True, spans=(8,), id_lags=True, ngram_bias=32),
        {"lr": 2e-3, "swa": 0.99}),
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
    # fairness control: the transformer gets the identical rank-32 exact
    # order-1 prior, so "IECNN + prior" is never compared against a
    # handicapped baseline.
    "tf-swa-ng": ("transformer", lambda: tf(96, 4, 3, ngram_bias=32),
                  {"swa": 0.995}),
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
    # The apples-to-apples control. The headline gap has always compared an
    # IECNN tuned to lr 2e-3 WITH tail averaging against a transformer at
    # lr 1e-3 with none — two knobs that were only ever tuned on the IECNN
    # side. This arm gives the transformer the identical pair, so the gap can
    # no longer be explained by the optimiser.
    "tf-lr2-swa99": ("transformer", lambda: tf(96, 4, 3),
                     {"lr": 2e-3, "swa": 0.99}),
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


def _fingerprint(path: str, n: int) -> str:
    """md5 (first 10 hex) of the bytes actually used — hashed as RAW BYTES
    so it matches `md5sum`, not as an int64 tensor.

    The toy files on the lab VM were silently regenerated on 2026-11-18
    11:09 (book-disjoint split -> same-source split). Every arm moved 0.72
    bpc at once, which looked like a mechanism win until the corpus was
    hashed. Absolute bpc is only comparable between runs with the SAME
    fingerprint; the in-session control arm still governs every claim.
    """
    import hashlib
    return hashlib.md5(open(path, "rb").read(n)).hexdigest()[:10]


train = load(TRAIN_PATH, N_TRAIN)
val = load(VAL_PATH, N_VAL)
# Device: the same harness runs on the 2-vCPU lab VM and on a GPU runner.
# Everything that depends on the machine (threads, device) is a harness knob,
# never an arm knob, so both arms in a session always share it.
DEV = torch.device(os.environ.get("ARM_DEV")
                   or ("cuda" if torch.cuda.is_available() else "cpu"))
DATA_FP = f"{_fingerprint(TRAIN_PATH, N_TRAIN)}/{_fingerprint(VAL_PATH, N_VAL)}"
print(f"train={TRAIN_PATH} val={VAL_PATH} | tokens train={train.numel()} "
      f"val={val.numel()} | B={B} T={T} valw={VALW} stride={VALSTRIDE} V={V} "
      f"| device {DEV} | budget {BUDGET:.0f}s/arm | data_fp {DATA_FP}",
      flush=True)


_NT = None


def batches(seed: int = 0, n: int = None):
    n = B if n is None else int(n)
    g = torch.Generator().manual_seed(seed)
    span = train.numel() - T - 1
    while True:
        # torch.randint over a 4M range measured at 1.19 ms/step — 3.4% of a
        # whole 150 s toy step, paid identically by both arms and buying
        # nothing. torch.rand + scale is the same uniform draw, ~50x cheaper.
        i = (torch.rand(n, generator=g) * span).long()
        x = torch.stack([train[j:j + T] for j in i]).to(DEV, non_blocking=True)
        y = torch.stack([train[j + 1:j + 1 + T] for j in i]).to(DEV, non_blocking=True)
        yield x, y


@torch.no_grad()
def val_bpc(model, windows: int = None) -> float:
    windows = VALW if windows is None else windows
    model.eval()
    tot, n = 0.0, 0
    for w in range(windows):
        i = w * T * VALSTRIDE
        x = val[i:i + T].unsqueeze(0).to(DEV)
        y = val[i + 1:i + 1 + T].unsqueeze(0).to(DEV)
        if x.shape[1] < T:
            break
        lp = model(x)["final_log_probs"]
        tot += F.nll_loss(lp.reshape(-1, V), y.reshape(-1),
                          reduction="sum").item()
        n += y.numel()
    model.train()
    return tot / n / math.log(2)


@torch.no_grad()
def train_bpc(model, n_batches: int = 48) -> float:
    """In-sample bpc on fresh draws from the training split.

    The whole strategy turned on an unmeasured assumption: that the IECNN
    plateau at ~3.29 is a CAPACITY limit (train bpc also stuck) rather than
    overfitting (train bpc far below val). Every capacity knob was pruned on
    wall-clock grounds, so if the model is actually overfitting then the GPU
    capacity lane is the wrong direction entirely and cheap regularisation is
    the right one. One extra forward pass per batch settles it at ~1 s/arm.
    """
    model.eval()
    gen = batches(seed=1234)
    tot, n = 0.0, 0
    for _ in range(n_batches):
        x, y = next(gen)
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
    x = torch.stack([train[j:j + T] for j in i]).to(DEV)
    rows = region1_rows(model.embed(x))
    wind_k, _wv, valid = r._window_memory(rows, x.device)
    q = r.dots._sel(0, r.dots.q)
    scores = torch.einsum("bfwj,nj->bfnw", wind_k, q) / math.sqrt(r.d_q)
    scores = scores.masked_fill(~valid[None, :, None, :], float("-inf"))
    idx = scores.argmax(-1)                                   # [B,T,N]
    distinct = torch.tensor([len(set(row.tolist()))
                             for row in idx.reshape(-1, idx.shape[-1])])
    w = wind_k.shape[2]
    # REAL slot -> distance map, from the same offsets the read used. With
    # window_mode quad/bands the slots are not consecutive, so "w-1-idx" would
    # silently report a fictional distance for those arms.
    offs = r._window_offsets(x.device, 0)
    dist = (-offs)[idx].clamp_min(0).float()                   # [B,T,N]
    p = scores.softmax(-1)
    ent = -(p.clamp_min(1e-12).log() * p).sum(-1)              # [B,T,N]
    n = idx.shape[-1]
    return {
        "distinct_per_pos": round(distinct.float().mean().item(), 2),
        "all_same": round((distinct == 1).float().mean().item(), 3),
        "mean_dist_back": round(dist.float().mean().item(), 2),
        "frac_window_end": round((idx == (w - 1)).float().mean().item(), 3),
        # per-dot: WHO looks where. One dot parked on distance 0 while the
        # others look back is division of labour; every dot at distance 0 is
        # the clone failure mode, and it is visible in bpc too.
        "dot_dist_back": [round(v, 2)
                          for v in dist.reshape(-1, n).mean(0).tolist()],
        "dot_frac_end": [round(v, 3) for v in
                         (idx == (w - 1)).reshape(-1, n).float().mean(0).tolist()],
        "dot_read_entropy": [round(v, 3)
                             for v in ent.reshape(-1, n).mean(0).tolist()],
    }


SEED = int(os.environ.get("ARM_SEED", "0"))
BASE_LR = float(os.environ.get("ARM_LR", "1e-3"))


def run(name: str, kind: str, model, lr: float = None,
        seed: int = None, sched: str = None, swa: float = 0.0,
        eval_model=None, batch: int = None) -> dict:
    lr0 = BASE_LR if lr is None else float(lr)
    sd = SEED if seed is None else int(seed)
    torch.manual_seed(sd)
    opt = torch.optim.AdamW(model.parameters(), lr=1e-3, betas=(0.9, 0.95),
                            weight_decay=0.01)
    n_par = sum(p.numel() for p in model.parameters())
    em = eval_model if eval_model is not None else model
    # per-arm batch override: same wall clock, same data, same everything
    # except how many rows a step carries (op-shape experiment; the harness
    # default is still one batch for the whole machine).
    nb = B if batch is None else int(batch)
    gen = batches(sd, nb)
    sched = sched or ""
    # Tail weight averaging (SWA/EMA). With a constant lr the last steps
    # bounce around the minimum; averaging the tail is a free variance
    # reduction (one extra eval, no extra training). Active only in the
    # last 55% of the budget, where the weights are already worth keeping.
    ema = None
    if swa:
        ema = [p.detach().clone() for p in model.parameters()]
    # torch.compile: the first calls trace/autotune and are 1-2 orders of
    # magnitude slower than a steady-state step. They must be paid BEFORE the
    # wall-clock budget starts, otherwise the arm is timed on its compiler.
    warm_steps = int(os.environ.get("ARM_WARM", "10")) \
        if os.environ.get("ARM_COMPILE") == "1" else 0
    for _ in range(warm_steps):
        x, y = next(gen)
        out = model(x)
        loss = F.cross_entropy(out["final_log_probs"].reshape(-1, V),
                               y.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        opt.step()
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
            marks.append([int(step * nb * T), round(val_bpc(em), 4), step])
            next_mark += MARK_S
    el = time.time() - t0
    if ema is not None:
        with torch.no_grad():
            for p, a in zip(model.parameters(), ema):
                p.copy_(a)
    final = round(val_bpc(em), 4)
    tr_bpc = round(train_bpc(em), 4)
    row = {"arm": name, "kind": kind, "seed": sd, "lr": lr0, "sched": sched,
           "swa": swa, "data_fp": DATA_FP,
           "params": n_par, "steps": step,
           "tokens": step * nb * T, "s_per_step": round(el / max(step, 1), 4),
           "s_per_1k_tokens": round(el / max(step * nb * T / 1000, 1e-9), 3),
           "curve": marks, "bpc_final": final, "bpc_train": tr_bpc}
    if kind == "iecnn":
        row["pointers"] = pointer_audit(em)
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
        # Harness-level knob (never an arm knob): when torch.compile is on but
        # Inductor's CUDA SplitScan lowering cannot handle the prefix scan,
        # every IECNN arm gets the eager graph boundary around that one
        # function. Applied uniformly so a compile comparison stays fair.
        if (os.environ.get("ARM_BREAK_SCAN") == "1"
                and cfg.get("model", {}).get("kind") == "iecnn"):
            cfg["model"]["break_scan"] = True
        model = (build_transformer_baseline(cfg, V) if kind == "transformer"
                 else build_iecnn(cfg, V)).to(DEV)
        print(f"--- {name} ({sum(p.numel() for p in model.parameters())} params"
              f"{', lr ' + str(opts['lr']) if 'lr' in opts else ''})", flush=True)
        inner = model
        # Per-arm compile: the global env switch stays the harness default, but
        # an arm may opt in so compile-vs-eager can be measured in ONE session
        # (across two sessions it would be confounded by tf drift). Compile
        # pays only if its warmup, which comes out of the same 180 s budget, is
        # repaid by the faster steady-state step.
        cflag = opts.get("compile", os.environ.get("ARM_COMPILE") == "1")
        if cflag:
            # `compile` is either True, a torch.compile mode name
            # ("reduce-overhead" = CUDA graphs; static B/T here, so it should
            # apply), or "noscan" = compile WITHOUT the eager boundary around
            # the prefix scan, which is the one graph break the region was
            # forced to keep after Inductor's SplitScan lowering crashed.
            mode = None
            if isinstance(cflag, str):
                if cflag == "noscan":
                    cfg["model"]["break_scan"] = False
                else:
                    mode = cflag
            model = (torch.compile(model, mode=mode) if mode
                     else torch.compile(model))
        # `compile` is consumed HERE: passing it on to run() is a TypeError, and
        # that killed a whole session after the first arm had trained (the GPU
        # slot is the scarce resource, so knobs are stripped, not forwarded).
        run_opts = {k: v for k, v in opts.items() if k != "compile"}
        try:
            rows.append(run(name, kind, model, eval_model=inner, **run_opts))
        except Exception as e:  # noqa: BLE001
            # One broken arm must not cost the other arms their session: the
            # failure is printed loudly and the summary simply lacks the arm.
            print(f"ARM FAIL {name:<26}{type(e).__name__}: {e}", flush=True)

    common_tok = min(r["tokens"] for r in rows)
    print(f"\n=== SUMMARY (equal wall-clock {BUDGET:.0f}s/arm, "
          f"common {common_tok/1e6:.2f}M tokens) ===", flush=True)
    hdr = f"{'arm':<18}{'params':>8}{'steps':>7}{'tokens':>9}{'s/step':>9}" \
          f"{'s/1k tok':>10}{'bpc_fin':>9}{'bpc@eqtok':>11}{'bpc_tr':>8}{'gap':>7}"
    print(hdr, flush=True)
    for r in rows:
        be = bpc_at(r["curve"], common_tok)
        gap = r["bpc_train"] - r["bpc_final"]
        print(f"{r['arm']:<18}{r['params']:>8}{r['steps']:>7}"
              f"{r['tokens']/1e6:>8.2f}M{r['s_per_step']:>9.4f}"
              f"{r['s_per_1k_tokens']:>10.4f}{r['bpc_final']:>9.4f}"
              f"{('%.4f' % be) if be is not None else '   -':>11}"
              f"{r['bpc_train']:>8.4f}{gap:>+7.4f}", flush=True)
        if "pointers" in r:
            print(f"{'':<18}pointers {r['pointers']}", flush=True)
    print("\nJSON " + json.dumps(rows), flush=True)


if __name__ == "__main__":
    main()
