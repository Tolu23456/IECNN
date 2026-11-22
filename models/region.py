"""One region: N dots, P passes, weights, consensus, summary (spec 03).

The parameters are shared across passes (D6); the round signal, the carried
state, the pooled opinion and the error feature give each pass its behavior.
The consensus machinery lives in `consensus.py` (label-free, C2).
"""
from __future__ import annotations

import math
import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from models import consensus
from models.dot import DotPopulation
from models.memory import gather_windows

_COMPILED_CORE = None
_COMPILE_MODE = None


def _compiled_consensus_core(*args, mode=None):
    """Lazily compiled consensus_core (torch.compile/inductor). The whole
    per-pass [., N, V] chain — exp, leave-one-out, JS identity, scores,
    trust, floor-softmax, error features, matmul pooling — as one graph so
    inductor can fuse the bandwidth-bound elementwise chain. dynamic=False:
    shapes are fixed per config; recompiles only on (rare) shape changes.

    mode (GPU speed pack, 2026-11-15): None = default inductor; "cuda_graphs"
    wraps the compiled core in torch.cuda.graphs-style reduce-overhead
    (kills kernel-launch overhead — the suspected bottleneck on small-SM
    GPUs like the T4); "max-autotune-no-cudnn" = inductor autotuning.
    """
    global _COMPILED_CORE, _COMPILE_MODE
    if _COMPILED_CORE is None or _COMPILE_MODE != mode:
        kw = {}
        if mode == "cuda_graphs":
            kw["mode"] = "reduce-overhead"
        elif mode:
            kw["mode"] = mode
        _COMPILED_CORE = torch.compile(consensus.consensus_core,
                                       dynamic=False, fullgraph=False, **kw)
        _COMPILE_MODE = mode
    return _COMPILED_CORE(*args)


class Region(nn.Module):
    """Rung-building block. One region reads a fixed window of memory rows."""

    def __init__(self, region_index: int, config: dict, vocab_size: int):
        super().__init__()
        m = config["model"]
        d = int(m.get("d", 128)); d_q = int(m.get("d_q", 32))
        d_v = int(m.get("d_v", 64)); d_s = int(m.get("d_s", 128))
        self.index = region_index
        self.d = d; self.d_q = d_q; self.d_v = d_v; self.d_s = d_s
        self.span = int((m.get("spans") or [256])[region_index % len(m.get("spans") or [256])])
        self.n_dots = int(m.get("n_dots", 8))
        self.n_passes = int(m.get("n_passes", 3))
        self.use_weights = bool(m.get("use_weights", True))
        self.use_messages = bool(m.get("use_messages", False))
        self.pooling = m.get("pooling", "log_linear")
        self.rho = float(config.get("consensus", {}).get("rho", 0.5))
        self.kappa = float(config.get("consensus", {}).get("kappa", 0.7))
        # speed goal round 2: fuse the per-pass [., N, V] consensus chain
        # with torch.compile (training.compile_consensus: true). Default OFF
        # — eager stays the bit-for-bit reference path.
        self.compile_consensus = bool(config.get("training", {})
                                      .get("compile_consensus", False))
        # GPU speed pack (2026-11-15): torch.compile mode for the fused core
        # ("cuda_graphs" -> reduce-overhead, i.e. CUDA graphs, for
        # launch-bound small-SM GPUs; any other string passes to inductor).
        self.compile_mode = str(config.get("training", {})
                                .get("compile_mode", "")) or None
        # torch 2.14 removed "max-autotune-no-cudnn" (renamed to
        # "max-autotune-no-cudagraphs") — map it so configs work on both.
        if self.compile_mode == "max-autotune-no-cudnn":
            self.compile_mode = "max-autotune-no-cudagraphs"
        self.hard_top_m = int(m.get("hard_top_m", 4))
        self.dynamic_query = bool(m.get("dynamic_query", False))
        self.share_pass_params = bool(m.get("share_pass_params", True))
        self.dot_dropout = float(m.get("dot_dropout", 0.0))
        # Rung E11 (Tolu, 2026-11-17: "instead of every dot having its own
        # head they all share one head and only their state differs"):
        # "prob" (default) = the original per-dot vocab heads + consensus
        # pooling over [., N, V]; "hidden" = dots emit hidden states only,
        # a learned gate weights them, and ONE shared tied head reads the
        # pooled state. Prunes: per-dot logits on P-1 of P passes, the whole
        # JS/leave-one-out/verifier chain, and the [B,T,N,V] elementwise
        # traffic — with the committee read intact in hidden space.
        self.head_mode = str(m.get("head_mode", "prob"))

        # Rung E13 (Tolu's slot-table memory, 2026-11-17): model.slot_slots
        # > 0 gives every dot S private slots [slot_dim each] carried across
        # positions inside a forward (the dot's GRU state still resets every
        # position — clarification C1 stays; the TABLE is the persistence).
        # Gated detached writes; the read enters the GRU input.
        self.slot_slots = int(m.get("slot_slots", 0))
        self.slot_dim = int(m.get("slot_dim", 32))
        self.slot_decay = float(m.get("slot_decay", 0.5))

        # pool_first: ONE shared output projection + pool the d_s states
        # BEFORE projecting. Because the projection is linear and shared,
        #     sum_n w_n (s_n W) == (sum_n w_n s_n) W,
        # so this is numerically the same head for 1/N of the e_state MACs
        # and without ever materializing [B,T,N,d]. Must be set BEFORE the
        # DotPopulation is built (it decides p_out's shape).
        self.pool_first = bool(m.get("pool_first", False))
        # gate_agree: the gate's only view of a dot is gamma, a scalar read
        # of that dot's OWN state — it cannot see the committee. One number
        # per dot closing that hole: the cosine between the dot's state and
        # the population mean this position. A dot that disagrees with the
        # other two is either the one that found something or the one that
        # went wrong, and nothing in the current gate can express either.
        # Cost is N*d_s MAC-equivalents per token (~0.4% of the step) and the
        # scale starts at zero, so the model begins bit-identical to no-gate.
        self.gate_agree = bool(m.get("gate_agree", False))
        # head_untied: an independent output matrix [d, V] instead of the
        # tied embedding. Same compute per token (d*V either way), +d*V
        # parameters; the transformer baseline has always had one, so tying
        # was an unexamined handicap on our side of a like-for-like test.
        # query_from_input: the WINDOW read's query becomes q_n + W x_t, i.e.
        # content-dependent, which is the one thing a transformer's attention
        # has and our static-query read does not. W is zero-initialised, so
        # the model starts bit-identical to the static version and keeps the
        # term only if it pays. The GLOBAL read stays static on purpose: the
        # O(T) prefix-sum trick needs a position-independent query, and that
        # mechanism is worth more than the extra conditioning there.
        self.q_from_input = bool(m.get("query_from_input", False))
        self.w_qi = (nn.Parameter(torch.zeros(self.d, self.d_q))
                     if self.q_from_input else None)
        # q_shift: which row the per-position query is built from.
        #  0 = the current row (query_from_input as it always was: "address the
        #      prefix with what I am"), which is still TYPE-level — it cannot
        #      bind an instance, because the thing being matched is the token
        #      itself.
        #  1 = the PREVIOUS row. That is the induction circuit in one
        #      projection: address the prefix with the token just seen, so the
        #      read retrieves what FOLLOWED earlier occurrences of it. It is
        #      the cheapest known way to give the region an instance-binding
        #      read (one d x d_q GEMM, ~2k MAC/token per dot), and it is why
        #      `global_exact` is required alongside it: the O(T) prefix-sum
        #      read cannot use a position-dependent query at all.
        self.q_shift = int(m.get("q_shift", 0) or 0)
        # k_lag: the KEY stream is the row stream DELAYED by `k_lag`
        # positions, while the value stream is not. With k_lag=1 the key at s
        # encodes x_{s-1} and the value at s is x_s, so a query built from the
        # current row implements the induction rule directly:
        #
        #     "find an earlier occurrence of the token I just saw, and copy
        #      what followed it"          keys x_{s-1}, value x_s
        #
        # This is the exact circuit the audit says the frontier cannot form:
        # with unlagged keys the query matches x_s and the read returns x_s
        # itself, so the best a constant query can do is a fixed horizon
        # (type-level n-gram). Costs nothing — one shift, no parameter — and
        # is inert only in the sense that it needs a position-dependent query
        # to matter (query_from_input + global_exact for the global read; the
        # WINDOW read uses q_add already and needs no exactness).
        self.k_lag = int(m.get("k_lag", 0) or 0)
        # --- exact-token-identity pathways (2026-11-18) -------------------
        # The read is a POOLED window: it tells a dot what the neighbourhood
        # *means*, not which tokens are in it. Byte-level text is strongly
        # order-2 (val set: H0=4.82, H1=3.85, H2=3.68 bits/byte), so exact
        # short-range identity is the cheapest signal a softmax-pooled read
        # throws away. Two independent, near-zero-MAC paths:
        #   id_lags  : dot n gets rows[t-(n+1)] projected into its VALUE
        #              width and ADDED to the pooled ctx. No new cell input
        #              width, no extra cell GEMM — one [d -> N*id_dim] GEMM
        #              per forward (4.2M MACs at N=4,d=128,id=16) plus a
        #              shift/stack. Each dot is born with its own lag, so
        #              the committee is a multi-lag n-gram mixture.
        #   ngram_bias: a rank-r factorised order-1 term on the head logits,
        #              bias = E1[x_{t-1}] @ E2^T (2*V*r params, 4.2M MACs at
        #              r=32). Starts as a weak random bigram, not neutral:
        #              a product of two zero-initialised factors has zero
        #              gradient, so both factors must start non-zero.
        self.id_lags = bool(m.get("id_lags", False))
        self.id_dim = int(m.get("id_dim", 0)) or d_v
        self.w_id = (nn.Parameter(torch.zeros(self.d, self.n_dots * self.id_dim))
                     if self.id_lags else None)
        # one scalar per (pass, dot): how much weight to give the dot's
        # agreement with the population mean. Zero-init => the mechanism is
        # inert until it is trained, so an arm either keeps it or loses it on
        # merit and never because of an init accident.
        self.c_agree = (nn.Parameter(torch.zeros(self.n_dots))
                        if self.gate_agree else None)
        # suffix_copy: a DISCRETE induction channel. At position t (whose
        # output predicts x_{t+1}) take the k bytes ending at t, find the most
        # recent earlier position s < t where the same k-byte context ended,
        # and mix the byte that followed it, x_{s+1}, into the output
        # distribution with a learned gate.
        #
        # This is the one thing the measured gap asks for. On the 900 s
        # checkpoints, positions where the answer is exactly the continuation
        # of the last occurrence of this 3-byte context -- 9 to 64 bytes back
        # -- are 10.4% of the corpus and carry 115% of the frontier arm's
        # remaining gap; the band sits just outside an 8-slot window and no
        # width or depth buys it (rr2 -> x5 puts only 18% of its gain there).
        #
        # Deliberately NOT attention: no learned keys, no softmax, no query --
        # an exact integer match plus a pointer, which is the circuit the
        # binding probe proved the static-query read cannot express. Gate
        # starts closed (sigmoid(-8) ~ 3e-4), so the arm is numerically the
        # baseline until training opens it, and `ids=None` skips it entirely.
        self.suffix_copy = int(m.get("suffix_copy", 0))
        self.sc_gate = (nn.Parameter(torch.tensor(-8.0))
                        if self.suffix_copy > 0 else None)
        # Optional second gate parameter: gate = sigmoid(a + b * extension),
        # where `extension` is how many bytes agree BEFORE the k-gram. It is an
        # observable precision signal (49.4% of 3-gram matches with no extension
        # are right, ~70% with one, 87.4% with eight or more), so a gate can
        # spend its weight where the match is long. b starts at exactly 0, so
        # the arm is numerically identical to the scalar gate at init and has to
        # earn the slope on the val loss.
        self.sc_slope = (nn.Parameter(torch.zeros(()))
                         if (self.suffix_copy > 0
                             and m.get("suffix_copy_slope", False)) else None)
        self.sc_extra_cap = int(m.get("suffix_copy_extra_cap", 8))
        # suffix_copy_vote = M: instead of copying from the single most recent
        # occurrence of the k-gram, take up to M earlier occurrences and put
        # the copy mass on those whose match extends furthest back (ties share
        # the mass). Measured over 40,960 val positions: the single pointer is
        # 62.9% right at 23.1% coverage; max-extension selection is 63.7% and
        # voting among max-extension ties 64.3% at the SAME coverage, fixing
        # 7.5% of the single pointer's errors instead of 0%. Plain majority
        # voting is measured worse (65.5% at 9.0% coverage) because duplicate
        # k-grams are mostly the same coincidence, not independent evidence, so
        # only the extension-weighted form is implemented. The gate feature
        # becomes the MAX extension, which is the one the copy came from.
        self.suffix_copy_vote = (int(m.get("suffix_copy_vote", 0))
                                 if self.suffix_copy > 0 else 0)
        self.ngram_r = int(m.get("ngram_bias", 0))
        if self.ngram_r > 0:
            e1 = torch.empty(vocab_size, self.ngram_r)
            e2 = torch.empty(vocab_size, self.ngram_r)
            nn.init.normal_(e1, std=0.02)
            nn.init.normal_(e2, std=0.02)
            self.ng_e1 = nn.Parameter(e1)
            self.ng_e2 = nn.Parameter(e2)
        else:
            self.ng_e1 = None
            self.ng_e2 = None
        # head_rank: low-rank output head logits = (h A) B with A [d,r],
        # B [r,V]. The tied head costs d*V MACs/token (12.6M at d=96,V=256)
        # — as much as the whole cell. A rank-r bottleneck cuts that to
        # r*(d+V), e.g. 11.3M -> 5.9M at d=96/r=48. Not tied to the
        # embedding; the memory rows keep their own embedding.
        self.head_rank = int(m.get("head_rank", 0))
        if self.head_rank > 0 and not bool(m.get("head_untied", False)):
            a = torch.empty(self.d, self.head_rank)
            b = torch.empty(self.head_rank, vocab_size)
            nn.init.normal_(a, std=1.0 / math.sqrt(self.d))
            nn.init.normal_(b, std=0.02)
            self.w_head_a = nn.Parameter(a)
            self.w_head_b = nn.Parameter(b)
        else:
            self.head_rank = 0
            self.w_head_a = None
            self.w_head_b = None
        self.head_untied = bool(m.get("head_untied", False))
        if self.head_untied:
            w = torch.empty(self.d, vocab_size)
            nn.init.normal_(w, std=0.02)     # transformer-style small init
            self.w_head = nn.Parameter(w)
        else:
            self.w_head = None
        # global_read: every dot additionally reads its WHOLE prefix (causal)
        # through an O(T) prefix-sum attention — see DotPopulation.__init__.
        # It costs no T*T tensor and no per-position GEMM. Mixing g_glob lets
        # some dots stay local (bigram precision) and others read globally.
        self.global_read = bool(m.get("global_read", False))
        self.dots = DotPopulation(
            self.n_dots, d, d_q, d_v, d_s, vocab_size,
            message_dim=(16 if self.use_messages else 0),
            n_passes=self.n_passes, dynamic_query=self.dynamic_query,
            share_pass_params=self.share_pass_params,
            seed=1000 + region_index,
            fast_math=bool(m.get("fast_dot_math", True)),
            slot_slots=self.slot_slots, slot_dim=self.slot_dim,
            slot_decay=self.slot_decay,
            cell=str(m.get("cell", "gru")),
            share_out=self.pool_first,
            span=self.span,
            dist_bias=bool(m.get("dot_dist_bias", False)),
            global_read=self.global_read,
            read_schedule=str(m.get("read_schedule", "mix")),
            in_ln=bool(m.get("cell_ln", False)),
            share_cell=bool(m.get("share_cell", False)),
            carry_state=bool(m.get("state_carry", False)),
            carry_chunk=int(m.get("carry_chunk", 8)),
            gla=bool(m.get("global_linear", False)),
            global_exact=bool(m.get("global_exact", False)),
            break_scan=bool(m.get("break_scan", False)),
            read_heads=int(m.get("read_heads", 1)),
            input_res=bool(m.get("input_res", False)),
            no_lag0=bool(m.get("no_lag0", False)),
            global_decay=bool(m.get("global_decay", False)),
            global_decay_k=int(m.get("global_decay_k", 1)),
            scan_mode=str(m.get("scan_mode", "cumsum")))
        # Lean passes: in hidden mode the pooled shared head and the per-dot
        # e_state projection are only consumed on the LAST pass. With
        # head_last_only the earlier passes skip both (they still update the
        # dots' states and the weighting). Default false = unchanged.
        self.head_last_only = bool(m.get("head_last_only", False))
        # head_skip (head_skip=True): add the region's own rows (for region 1,
        # the token embedding itself) to the pooled state before the head.
        #
        # Nothing in the dot mechanism requires that E[x_t] reach the logits
        # only through a softmax over 8 window slots, a gated tanh and a 3-way
        # weight — but that is what happens today, and the transformer gets
        # the alternative for free (h = h + sublayer(h) twice per block, plus
        # an output matrix initialised from the embedding). This restores the
        # skip at zero parameters and zero measurable time. It is also the
        # cheapest possible test of "is the head information-starved?", which
        # is a different question from "is the read information-starved?".
        self.head_skip = bool(m.get("head_skip", False))
        # read_ln / head_ln: the two LayerNorms a transformer has and this
        # model never did (2*d params each, negligible time).
        #
        # read_ln normalises the row stream BEFORE it is projected to keys and
        # values. Today the read sees raw embeddings, and with `interleave`
        # (or a deep block stack feeding back into rows) it sees an unnormalised
        # sum whose scale depends on depth — exactly the pathology pre-LN exists
        # to prevent. head_ln is the final LN before the output matrix
        # (transformer ln_f): the tied head multiplies h by E^T, so the logit
        # scale is currently whatever the pooled dot state happens to be.
        self.read_ln = nn.LayerNorm(d) if m.get("read_ln", False) else None
        self.head_ln = nn.LayerNorm(d) if m.get("head_ln", False) else None
        # interleave: the transformer-block structure the dots were missing.
        # A transformer re-MIXES positions at every layer (attention -> MLP ->
        # attention); this model mixed once, then stacked MLPs, which is
        # exactly why block depth saturated (x4 == x3). With interleave, each
        # pass reads the rows as REFINED by the previous pass's block, so
        # depth adds new mixed information instead of re-processing the same
        # gathered context. Cheap: the read is ~10k MAC/token against
        # attention's ~98k.
        self.interleave = bool(m.get("interleave", False))
        # Per-dot vocab logits are consumed ONLY by the optional auxiliary
        # losses (oracle / boosted / decorrelation / confidence), and those
        # losses read exactly one number per dot: the log-prob of the true
        # token. With per_dot_logits: false the [B,T,N,V] tensor is never
        # materialized (no GEMM, no log-softmax, no big transpose).
        self.per_dot_logits = bool(m.get("per_dot_logits", True))
        self.dots.head_layout = str(m.get("head_layout", "shared_gemm"))
        # Rung E12: Tolu's seclusion mechanism — model.read_mode: "secluded"
        # makes every dot commit to exactly ONE token in its window (hard
        # top-1 read, straight-through backward). Default "soft" = unchanged.
        self.read_mode = str(m.get("read_mode", "soft"))
        self.dots.read_mode = self.read_mode
        self.dots.tau_seclusion = float(m.get("tau_seclusion", 1.0))
        self.dots.ste_topk = int(m.get("seclusion_topk", 0))
        # speed opt 7 (Tolu's "don't compute the same op twice"): with
        # share_pass_params + static queries the attention block is
        # bit-identical across passes; one fresh dict per region-forward
        # lets pass 1 compute it and later passes reuse ctx. Default ON;
        # model.cache_attention: false restores the recompute-every-pass
        # behavior (paired A/B knob).
        self.cache_attention = bool(m.get("cache_attention", True))
        self.w_k = nn.Linear(d, d_q, bias=False)
        self.w_v = nn.Linear(d, d_v, bias=False)
        # post_mlp (head_post: width, 0 = off): a per-position block AFTER the
        # population is pooled and BEFORE the tied head:
        #     u = LN(h_mix); h = h_mix + W2 gelu(W1 u)
        # This is the one major component of a transformer block the dots do
        # not have. The transformer spends ~2/3 of its parameters on exactly
        # this (d -> 4d -> d after attention), which is why it fits train 0.3
        # bpc lower and still generalises better; IECNN's pooling averages N
        # state vectors and hands the result straight to a tied head, so the
        # layer that does the per-position nonlinear work is missing. The
        # dot READ is much cheaper than attention (~10k vs ~98k MAC/token), so
        # this is where that saving gets reinvested.
        self.head_post = int(m.get("head_post", 0))
        n_post = int(m.get("head_post_layers", 1)) if self.head_post > 0 else 0
        # post_share: how many parameter copies the block stack keeps.
        #   "none" (default) — n_post independent blocks, 384*w + 960 each.
        #   "all"  — ONE block applied n_post times (a universal-transformer
        #            stack): depth of compute, cost of one block in parameters
        #            and in optimiser state.
        #   "up"   — one shared up-projection, per-block LN and down-projection
        #            (blocks share a feature basis but keep their own readout).
        # The stack is 74% of the model's parameters, and a block that is
        # applied twice does strictly more function composition than two
        # blocks that are never forced to agree, so if sharing is even close
        # it is the cheapest parameter win available.
        self.post_share = str(m.get("post_share", "none"))
        n_store = 1 if self.post_share == "all" else n_post
        self.post_blocks = nn.ModuleList()
        for _ in range(n_store):
            blk = nn.ModuleDict({
                "ln": nn.LayerNorm(d),
                "w2": nn.Linear(self.head_post, d)})
            if self.post_share != "up":
                blk["w1"] = nn.Linear(d, self.head_post)
            self.post_blocks.append(blk)
        self.post_w1 = (nn.Linear(d, self.head_post)
                        if self.post_share == "up" else None)
        self.post_repeat = n_post if self.post_share == "all" else 1
        # block-level re-read (reread = how many inter-block gaps get one).
        #
        # The head stack is a transformer block with the attention sublayer
        # deleted: pool -> MLP -> MLP -> MLP. A transformer gives EVERY layer
        # a fresh view of its neighbours, and the head has exactly one pooled
        # view of the prefix, so deeper blocks re-process the same numbers.
        # `interleave` tested restoring that at the dot read (a full cell pass
        # per gap) and lost: +0.10 bpc, +68% step. This is the cheap version
        # of the same hypothesis: a SINGLE-HEAD, SPAN-8 read of the block's
        # own output, added residually between blocks.
        #
        # Cost (d128, span 8, d_q 16, d_v 32): 2*128*16 (q,k) + 128*32 (v)
        # + 128*128 (out) + 2*8*(16+32) (the read itself) ~= 25k MAC/token,
        # i.e. ~6% of the 393k block stack and ~4% of the step, against the
        # ~100k of a full attention layer. Zero-initialised out projection:
        # the stack is unchanged at step 0 and the read has to earn its way
        # in, so a null result is "no gain", not "damaged training".
        self.reread = int(m.get("reread", 0))
        self.reread_span = int(m.get("reread_span", 8))
        # reread_spans: one exact horizon per reread site, e.g. (2, 8, 32).
        #
        # The default applies `reread_span` at EVERY site, so a 3-site stack
        # reads the same 8 slots three times -- the same scale the dots'
        # window already has. The stack then has exactly two views of the
        # past: window-8 and the global decayed bag, with nothing exact in
        # between. A ladder spends the same parameters (rr_* are shared
        # across sites) and ~1.2k extra MAC/token on a coarse-to-fine view:
        # site 0 reads 2 back, site 1 reads 8, site 2 reads 32. Empty (the
        # default) is bit-identical to uniform `reread_span`; a short ladder
        # repeats its last rung, so the flag can never run out of spans.
        _sp = tuple(int(x) for x in (m.get("reread_spans", ()) or ()))
        if not _sp:
            _sp = (self.reread_span,) * max(1, self.reread)
        if len(_sp) < self.reread:
            _sp = _sp + (_sp[-1],) * (self.reread - len(_sp))
        self.reread_spans = _sp
        self.reread_dq = int(m.get("reread_dq", 16))
        self.reread_dv = int(m.get("reread_dv", 32))
        if self.reread > 0:
            self.rr_ln = nn.LayerNorm(d)
            self.rr_q = nn.Linear(d, self.reread_dq, bias=False)
            self.rr_k = nn.Linear(d, self.reread_dq, bias=False)
            self.rr_v = nn.Linear(d, self.reread_dv, bias=False)
            self.rr_o = nn.Linear(self.reread_dv, d, bias=False)
            nn.init.zeros_(self.rr_o.weight)
        else:
            self.rr_ln = self.rr_q = self.rr_k = self.rr_v = self.rr_o = None
        # Time axis of the read window. "contiguous" (default) = the last
        # `span` tokens, one slot each. "quad" = the same number of slots
        # spread quadratically from distance 0 out to `window_reach`, so the
        # read reaches window_reach tokens back for the SAME compute (same
        # key/value gathers, same scores, same mix). Rationale: with a
        # contiguous window and span=16 the measured read collapsed to
        # distance 0.17 (86% of picks = the previous token) — the "memory"
        # was a bigram. A multi-scale axis keeps the cheap near slots and
        # adds far slots nobody has to pay extra for.
        self.window_mode = str(m.get("window_mode", "contiguous"))
        self.window_reach = int(m.get("window_reach", self.span))
        self.window_local = int(m.get("window_local", 6))
        emb_n = (self.span if self.window_mode == "contiguous"
                 else max(self.window_reach, self.span) + 1)
        self.rel_emb = nn.Embedding(emb_n, d_q)
        nn.init.normal_(self.rel_emb.weight, std=1.0 / math.sqrt(d_q))

        # learned scale/trust scalars per region (spec 04 section 3)
        self.alpha = nn.Parameter(torch.tensor(1.0))
        self.beta = nn.Parameter(torch.tensor(1.0))

        # E6 consensus-v2: verifier head replaces pure scalars with a learned
        # oracle-supervised trust judgment. Off (absent) = bit-identical E1.
        self.verifier = None
        if bool(m.get("verifier", False)):
            from models.verifier import VerifierHead
            self.verifier = VerifierHead(seed=2000 + region_index)
            self.v_scale = nn.Parameter(torch.tensor(0.05))

        # messages: write at pass p, read at pass p+1 (spec 05)
        self.messages = None
        if self.use_messages:
            from models.messages import MessageChannel
            self.messages = MessageChannel(
                d_s, 16, self.n_dots, radius=2,
                dropout=float(m.get("message_dropout", 0.2)))

    # ------------------------------------------------------------------
    def _window_offsets(self, device, pass_idx: int = 0) -> torch.Tensor:
        """Slot -> position offset. Slot span-1 is always the current token.

        "bands" (per-pass division of labour): pass 1 gets the DENSE local
        band 0..w-1 (what byte came before); passes 2..P get geometric
        bands that tile (w .. window_reach] in log space, one band per
        pass. Every pass therefore reads a DIFFERENT part of the sequence,
        which is what makes the extra passes real depth instead of a
        re-read of the same cached ctx. Same slot count per pass."""
        if self.window_mode == "contiguous":
            return torch.arange(self.span, device=device) - self.span + 1
        w = self.span
        reach = max(self.window_reach, w)
        if self.window_mode == "bands":
            if pass_idx <= 0:
                dist = torch.arange(w, device=device, dtype=torch.float32)
                return (-dist.round().long()).flip(0)
            # P-1 log-spaced bands tiling (w, reach]
            nb = max(self.n_passes - 1, 1)
            lo = float(max(w, 1)) ** ((nb - (pass_idx - 1)) / nb) * \
                 float(reach) ** ((pass_idx - 1) / nb)
            hi = float(max(w, 1)) ** ((nb - pass_idx) / nb) * \
                 float(reach) ** (pass_idx / nb)
            hi = max(hi, lo + 1.0)
            g = torch.exp(torch.linspace(math.log(lo), math.log(hi), w))
            dist = g.round().clamp(min=w).long()
            return (-dist).flip(0)
        if self.window_mode == "hybrid":
            # dense local band + geometric far band: the near slots keep full
            # resolution (the bigram/local signal dominates next-token
            # prediction) while the far slots reach `window_reach` for free.
            h = min(int(getattr(self, "window_local", 6)), w)
            parts = [torch.arange(h, device=device, dtype=torch.float32)]
            rem = w - h
            if rem > 0:
                g = torch.exp(torch.linspace(math.log(max(h, 1)),
                                             math.log(reach), rem))
                parts.append(g.round().clamp(min=h))
            dist = torch.cat(parts).round().long()[:w]
        else:  # "quad": quadratic spacing 0 .. reach
            i = torch.arange(w, device=device, dtype=torch.float32)
            dist = ((i / max(w - 1, 1)) ** 2 * reach).round().long()
            dist[0] = 0
        return (-dist).flip(0)        # ascending offsets, slot -1 = current

    def _query_add(self, r: torch.Tensor):
        """Per-position query term, optionally addressing with a LAGGED row.

        Zero-initialised `w_qi`, so every q_shift starts bit-identical to the
        static-query model and has to earn its way in.
        """
        if self.w_qi is None:
            return None
        if self.q_shift > 0:
            k = self.q_shift
            r = torch.cat([torch.zeros_like(r[:, :k]), r[:, :-k]], dim=1)
        return r.matmul(self.w_qi)

    def _reread_span_at(self, i: int) -> int:
        """Span for the i-th reread site (see reread_spans)."""
        if i < len(self.reread_spans):
            return self.reread_spans[i]
        return self.reread_spans[-1]

    def _reread(self, h: torch.Tensor, span=None) -> torch.Tensor:
        """One causal single-head read of the last `span` positions.

        Written as a gather (`unfold` over a left-padded stream) rather than
        `scaled_dot_product_attention` with a band mask: SDPA still computes
        the full T x T score matrix before masking, which would cost a full
        attention layer (98k MAC/token) for a read that only needs 8 slots
        (~1.5k). No padding values leak: slots before position 0 are masked
        with -inf.
        """
        b, t, _ = h.shape
        s = self.reread_span if span is None else int(span)
        u = self.rr_ln(h)
        q = self.rr_q(u)                                   # [B,T,dq]
        k = self.rr_k(u)                                   # [B,T,dq]
        v = self.rr_v(u)                                   # [B,T,dv]
        kp = F.pad(k.transpose(1, 2), (s - 1, 0)).unfold(-1, s, 1)
        vp = F.pad(v.transpose(1, 2), (s - 1, 0)).unfold(-1, s, 1)
        sc = torch.einsum("btd,bdts->bts", q, kp)
        sc = sc * (q.shape[-1] ** -0.5)
        i = torch.arange(t, device=h.device)
        off = torch.arange(s - 1, -1, -1, device=h.device)
        ok = (i[:, None] - off[None, :]) >= 0                # [T,s]
        sc = sc.masked_fill(~ok, float("-inf"))
        a = sc.softmax(-1)
        return self.rr_o(torch.einsum("bts,bdts->btd", a, vp))

    def _post(self, h: torch.Tensor) -> torch.Tensor:
        """Per-position residual blocks after pooling (identity when off).

        Each block is  u = LN(h); h = h + W2 gelu(W1 u), i.e. the per-position
        half of a transformer block. `head_post_layers` stacks them: the dots
        have already mixed the prefix, so this is where depth can act on the
        gathered context at O(1) cost in T.
        """
        if self.post_share == "all":
            blk = self.post_blocks[0]
            for i in range(self.post_repeat):
                h = h + blk["w2"](F.gelu(blk["w1"](blk["ln"](h))))
                if self.reread > i and h.dim() == 3:
                    h = h + self._reread(h, self._reread_span_at(i))
            return h
        for i, blk in enumerate(self.post_blocks):
            w1 = blk["w1"] if self.post_w1 is None else self.post_w1
            h = h + blk["w2"](F.gelu(w1(blk["ln"](h))))
            if self.reread > i and h.dim() == 3:
                h = h + self._reread(h, self._reread_span_at(i))
        return h

    def _head_logits(self, h: torch.Tensor, emb_table: torch.Tensor):
        """[.., d] -> [.., V]. Tied embedding (default), an untied head, or
        the low-rank bottleneck (h A) B."""
        if self.head_rank > 0:
            return h.matmul(self.w_head_a).matmul(self.w_head_b)
        if self.w_head is not None:
            return h.matmul(self.w_head)
        return h.matmul(emb_table.t())

    def _full_kv(self, rows: torch.Tensor):
        """Keys/values for the GLOBAL read: one row per position, no window.

        `k_lag` delays ONLY the key stream (see the comment in __init__), so
        key[s] and value[s] describe different positions — the causality is
        still exact (key[s] is x_{s-lag}, which is in the causal past).
        """
        if self.k_lag > 0:
            k = min(self.k_lag, rows.shape[1])
            rows_k = torch.cat([torch.zeros_like(rows[:, :k]), rows[:, :-k]],
                               dim=1)
        else:
            rows_k = rows
        return self.w_k(rows_k), self.w_v(rows)

    def _window_memory(self, rows: torch.Tensor, device, pass_idx: int = 0,
                       k_all: torch.Tensor | None = None,
                       v_all: torch.Tensor | None = None):
        """Rows [B,T,d] -> (wind_k [B,T,w,d_q], wind_v [B,T,w,d_v], valid [T,w]).

        `k_all`/`v_all` may be passed in when the caller already projected the
        rows (the global read needs exactly the same two projections as the
        window read — one GEMM pair serves both).
        """
        if k_all is None:
            k_all = self.w_k(rows)                   # [B, T, d_q]
        if v_all is None:
            v_all = self.w_v(rows)                   # [B, T, d_v]
        offs = self._window_offsets(device, pass_idx)
        # Python-level answer to "is this the plain sliding window?" so the
        # gather does not branch on tensor data (see gather_windows).
        plain = (self.window_mode == "contiguous"
                 or (self.window_mode == "bands" and pass_idx <= 0))
        wind_k_raw, valid = gather_windows(k_all, self.span, offs, plain)
        wind_v, _ = gather_windows(v_all, self.span, offs, plain)
        # slot i holds position t + offs[i]; relative distance = -offs[i]
        dist_idx = (-offs).clamp_min(0)
        rel = self.rel_emb(dist_idx)                 # [w, d_q], distance 0 at t
        wind_k = wind_k_raw + rel[None, None]
        return wind_k, wind_v, valid.to(device)

    # ------------------------------------------------------------------
    def _verifier_features(self, pi: torch.Tensor, gamma: torch.Tensor,
                           state: torch.Tensor, err: torch.Tensor,
                           w_prev: torch.Tensor | None,
                           log_b_prev: torch.Tensor | None,
                           p: int, device) -> torch.Tensor:
        """Per-dot features for the verifier head: [..., N, 9].

        All label-free and all already computed inside the pass loop (the
        oracle only ever TRAINS the head — see train/losses.py — it is not an
        input). Pass 1 uses neutral values for the not-yet-existing weight
        and region entropy. Leading dims are agnostic: callers pass either
        [B, T, N, ...] or a row chunk [c, N, ...].
        """
        from models.verifier import verifier_features
        n = gamma.shape[-1]
        top = pi.argmax(-1)                                  # [..., N]
        same = (top.unsqueeze(-1) == top.unsqueeze(-2)).float()
        agree = (same.sum(-1) - 1.0) / max(n - 1, 1)         # [..., N]
        state_norm = state.norm(dim=-1) / math.sqrt(self.d_s)
        if w_prev is None:
            w_feat = torch.full_like(gamma, 1.0 / n)
        else:
            w_feat = w_prev
        if log_b_prev is not None:
            h_region = consensus.entropy(log_b_prev)         # [...] leading dims
        else:
            h_region = torch.zeros(gamma.shape[:-1], device=device)
        pass_frac = torch.tensor(p / self.n_passes, device=device)
        return verifier_features(gamma, w_feat, agree, state_norm, err,
                                 h_region, pass_frac)

    # ------------------------------------------------------------------
    def _pass_consensus(self, log_pi_c, gamma_c, trust_term_c, trust_prev_c,
                        state_c, err_c, w_prev_c, log_b_prev_c, dot_mask,
                        p: int, tau_p: float, hard_mode: bool, floor: float):
        """One row-chunk of the per-pass consensus (see the chunking note in
        `forward`). Inputs/outputs carry a leading chunk dim instead of
        [B, T]: log_pi_c [c, N, V], gamma/trust_term/trust/w [c, N],
        state [c, N, d_s], err [c, N, 3], log_b_prev [c, V]. Returns
        (trust_c, w_c, log_b_c, v_out_c, err_next_c)."""
        pi = log_pi_c.exp()
        v_out = None
        if self.use_weights:
            # loo/JS are computed ONCE here and reused for the error feature
            # of the next pass (spec 04 = same numbers either way; computing
            # them twice cost ~30% of the forward). The _fast variant reuses
            # log_pi for the JS entropy terms (last-bit differences only).
            if (self.compile_consensus and self.n_dots >= 2
                    and self.pooling == "log_linear" and dot_mask is None):
                # fused path (torch.compile): the whole [c, N, V] chain in
                # one graph. tau/floor as 0-dim tensors so per-step annealed
                # floats are data, not recompile-triggering guards. Verifier
                # features stay eager (they need pi before the core); their
                # verdict folds in as v_scale * v_out (zeros when absent —
                # bit-exact x + 0 = x).
                dtype, device = log_pi_c.dtype, log_pi_c.device
                if self.verifier is not None:
                    feats = self._verifier_features(
                        pi, gamma_c, state_c, err_c, w_prev_c, log_b_prev_c,
                        p, device)
                    v_out = self.verifier(feats).squeeze(-1)     # [c, N]
                    v_scale_t, v_out_t = self.v_scale, v_out
                else:
                    v_scale_t = torch.zeros((), dtype=dtype, device=device)
                    v_out_t = torch.zeros(gamma_c.shape, dtype=dtype,
                                          device=device)
                tau_t = torch.tensor(tau_p, dtype=dtype, device=device)
                floor_t = torch.tensor(0.0 if hard_mode else floor,
                                       dtype=dtype, device=device)
                trust_in = trust_prev_c if trust_prev_c is not None \
                    else torch.zeros_like(gamma_c)
                trust_c, w_c, log_b_c, err_next = _compiled_consensus_core(
                    log_pi_c, gamma_c, trust_term_c, trust_in,
                    self.alpha, self.beta, v_scale_t, v_out_t,
                    self.rho, tau_t, floor_t, self.n_dots,
                    pi if self.verifier is not None else None,
                    mode=self.compile_mode)
                return trust_c, w_c, log_b_c, v_out, err_next
            scores, loo, js = consensus.compute_scores_fast(
                pi, log_pi_c, gamma_c, trust_term_c, self.alpha, self.beta,
                return_parts=True)
            if self.verifier is not None:
                # E6: the verifier's trust judgment joins the consensus
                # scores (scaled — it must EARN influence from small).
                feats = self._verifier_features(
                    pi, gamma_c, state_c, err_c, w_prev_c, log_b_prev_c,
                    p, log_pi_c.device)
                v_out = self.verifier(feats).squeeze(-1)         # [c, N]
                scores = scores + self.v_scale * v_out
            trust_c = consensus.update_trust(trust_prev_c, scores, self.rho)
            w_c = consensus.compute_weights(trust_c, tau_p,
                                            floor=0.0 if hard_mode else floor)
            err_next = consensus.error_features_from_parts(pi, loo, js)
        else:
            trust_c = None
            w_c = consensus.uniform_weights(self.n_dots, pi)   # rung E0
            if p < self.n_passes:      # only a LATER pass needs these numbers
                crowd = consensus.leave_one_out_mean(pi)
                js_own = consensus.js_from_parts_fast(pi, log_pi_c, crowd)
                err_next = consensus.error_features_from_parts(pi, crowd, js_own)
            else:
                err_next = None

        if dot_mask is not None:      # dot dropout: excluded dot gets 0
            w_c = w_c * dot_mask
            w_c = w_c / w_c.sum(-1, keepdim=True).clamp_min(1e-9)

        if self.pooling == "mixture":
            b_mix = consensus.pool_mixture(pi, w_c)
            log_b_c = (b_mix / b_mix.sum(-1, keepdim=True).clamp_min(1e-12)).clamp_min(1e-12).log()
        else:
            log_b_c = consensus.pool_log_linear(log_pi_c, w_c)
        return trust_c, w_c, log_b_c, v_out, err_next

    # ------------------------------------------------------------------
    def _pass_hidden(self, gamma_c, trust_term_c, trust_prev_c, dot_mask,
                     tau_p: float, floor: float):
        """One pass of the shared-head gate (hidden mode). All ops are
        [., N]-scale — no vocab-space tensor exists in this path. Weights
        come from the same learned scalar projections as the prob-space
        consensus (gamma / trust_term + trust accumulation), with the
        JS-disagreement and verifier terms dropped (the committee study
        bounded their value at ~0.047 bpc; they were the dominant cost).

        Returns (trust_c, w_c)."""
        scores = self.beta * gamma_c + trust_term_c
        trust_c = consensus.update_trust(trust_prev_c, scores, self.rho)
        t = trust_c.clamp(-20.0, 20.0)
        w = F.softmax(t / max(tau_p, 1e-6), dim=-1)
        if floor > 0.0:
            wf = w.clamp_min(floor)
            w = wf / wf.sum(-1, keepdim=True).clamp_min(1e-9)
        if dot_mask is not None:
            w = w * dot_mask
            w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
        return trust_c, w

    # ------------------------------------------------------------------
    def _forward_hidden(self, rows: torch.Tensor, emb_table: torch.Tensor,
                        tau0: float, hard_mode: bool, floor: float,
                        training: bool, ablate_dot: int = -1,
                        pool_only_dot: int = -1) -> dict:
        """Shared-head mode (rung E11): dots produce hidden states; a
        learned gate pools them; ONE tied head reads the pooled state.

        Passes 1..P-1 skip the per-dot vocab GEMM entirely; the LAST pass
        materializes per-dot logits once (the oracle/metric machinery and
        the boosted-dot losses consume [., N, V] there — train-time only
        cost, and a single GEMM + log-softmax instead of the old chain)."""
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        n = self.n_dots
        f = b * t

        def _read_inputs(r):
            """Projections the read needs from a row stream.

            Pulled out of the preamble so `interleave` can rebuild them from
            an UPDATED row stream between passes: the rows feed only the read
            (w_k/w_v), never the head, so refining them is safe.
            """
            if self.read_ln is not None:
                r = self.read_ln(r)
            fk = fv = ka = va = None
            if self.global_read:
                # ONE projection pair serves both the window gather and the
                # global read (they read the same keys/values).
                ka, va = self._full_kv(r)
                fk, fv = ka, va
            qa = self._query_add(r)
            ca = self._id_code(r) if self.w_id is not None else None
            return ka, va, fk, fv, qa, ca

        k_all, v_all, full_k, full_v, q_add, ctx_add = _read_inputs(rows)

        # Read windows. Normally every pass reads the SAME window, and the
        # attention cache is shared so the read is computed once (that is
        # exactly what made P>=2 "fake depth": passes 2..P re-used ctx).
        # window_mode "bands" gives each pass its own band (pass 1 local,
        # later passes far), so each pass must also have its OWN cache —
        # a cached ctx from a different band would be a different input.
        if self.interleave:
            # each pass reads the rows as they stood at the START of that
            # pass, so the window is built inside the loop and nothing may be
            # cached across passes.
            winds = None
            caches = [None] * self.n_passes
        elif self.window_mode == "bands":
            winds = [self._window_memory(rows, device, p, k_all, v_all)
                     for p in range(self.n_passes)]
            # each pass reads its own band, so a ctx cached by one pass is
            # never reusable by another: no cache at all.
            caches = [None] * self.n_passes
        else:
            one = self._window_memory(rows, device, 0, k_all, v_all)
            winds = [one] * self.n_passes
            # ONE shared dict: pass 1 fills ctx, passes 2..P hit the cache.
            # (Separate per-pass dicts would silently recompute the read P
            # times — same numbers, needless work.)
            shared = {} if self.cache_attention else None
            caches = [shared] * self.n_passes

        state = self.dots.initial_state((b, t))
        trust = None
        w_prev = None
        state_prev = None
        msg_read = None
        per_pass = []
        err = torch.zeros(b, t, n, 3, device=device, dtype=rows.dtype)
        e_state_last = None
        # e13: fresh (zeroed) slot table at the sequence boundary; it is
        # updated with a detached scan after every pass and read by the next
        slots = (self.dots.empty_slots(b, device, rows.dtype)
                 if self.dots.S > 0 else None)

        dot_mask = None
        if training and self.dot_dropout > 0 and torch.rand(()) < self.dot_dropout:
            k = int(torch.randint(0, n, ()))
            dot_mask = torch.ones(n, device=device, dtype=rows.dtype)
            dot_mask[k] = 0.0

        for p in range(1, self.n_passes + 1):
            if self.interleave:
                # the rows moved between passes, so this pass's read must be
                # rebuilt from the CURRENT rows (and cannot be cached).
                wind_k, wind_v, valid = self._window_memory(
                    rows, device, p - 1, k_all, v_all)
            else:
                wind_k, wind_v, valid = winds[p - 1]
            attn_cache = caches[p - 1]
            tau_p = tau0 * (self.kappa ** (p - 1))
            if p == 1:
                h_pool = torch.zeros(b, t, n, self.d_s, device=device,
                                     dtype=rows.dtype)
            else:
                pooled = torch.einsum("bfn,bfnd->bfd", w_prev, state_prev)
                h_pool = pooled[:, :, None, :].expand(b, t, n, self.d_s)
            last = (p == self.n_passes)
            lean = self.head_last_only and not last
            want_pi = last and self.per_dot_logits
            # pool_first needs no per-dot e_state unless the per-dot logits
            # are being materialized (they read the d-space states).
            want_e = (not lean) and (not self.pool_first or want_pi)
            out = self.dots(wind_k, wind_v, valid, h_pool, err,
                            round_idx=p - 1,
                            msg_in=msg_read if p > 1 else None, state=state,
                            emb_table=emb_table, attn_cache=attn_cache,
                            want_log_pi=want_pi, slot_table=slots,
                            want_e_state=want_e, full_k=full_k,
                            full_v=full_v, q_add=q_add, ctx_add=ctx_add)
            state_prev = out.state
            if slots is not None:
                # e13: detached scan — the table this pass wrote is read by
                # the next pass (one-pass-delay persistence, no BPTT)
                slots = self.dots.update_slots(state_prev, slots, p - 1)
            if last:
                e_state_last = out.e_state

            gam = out.gamma
            if self.c_agree is not None:
                # cos(s_n, mean_m s_m) per position: one number saying whether
                # this dot is with the committee or on its own. The mean is
                # over dots at the SAME position, so it is causal by
                # construction and needs no extra pass.
                mstate = out.state.mean(dim=2, keepdim=True)
                cos = torch.nn.functional.cosine_similarity(
                    out.state, mstate.expand_as(out.state), dim=-1)
                gam = gam + self.c_agree.view(1, 1, n) * cos
            tr_c, w_c = self._pass_hidden(
                gam, out.trust_term,
                trust.reshape(b, t, n) if trust is not None else None,
                dot_mask, tau_p, 0.0 if hard_mode else floor)
            if self.use_weights:
                trust = tr_c
            w = w_c
            if ablate_dot >= 0:
                # Diagnostic only (never used in training): drop one dot from
                # the committee and re-normalise, so the pool stays a convex
                # combination. Answers "what does this dot actually add?"
                # without changing the model or its parameters.
                keep = torch.ones_like(w)
                keep[..., ablate_dot] = 0.0
                w = w * keep
                w = w / w.sum(-1, keepdim=True).clamp_min(1e-9)
            if pool_only_dot >= 0:
                # Diagnostic only: force the pool to a one-hot so each dot is
                # judged ALONE through the TRAINED pool->head path. This is the
                # honest per-dot quality number, because per-dot logits are an
                # untrained side output in the arms where per_dot_logits is
                # off during training, so their own bpc says nothing.
                w = torch.zeros_like(w)
                w[..., pool_only_dot] = 1.0

            if self.interleave and not last:
                # The transformer-block step the dots were missing: a
                # transformer re-MIXES positions at every layer, while this
                # model mixed once and then stacked MLPs, which is why depth
                # saturated (x4 == x3). Pool what this pass found, run the
                # per-position block, and add it back into the row stream so
                # the NEXT pass reads a refined representation. The dot read
                # is ~10k MAC/token against attention's ~98k, so an extra
                # read is cheap; the cell is run once per pass either way.
                if self.pool_first:
                    s_pool = torch.einsum("bfn,bfnd->bfd", w, out.state)
                    upd = s_pool.reshape(f, self.d_s).matmul(
                        self.dots._sel(p - 1, self.dots.p_out))
                    upd = upd.reshape(b, t, self.d)
                else:
                    upd = torch.einsum(
                        "bfn,bfnd,ndv->bfv", w, out.state,
                        self.dots._sel(p - 1, self.dots.p_out))       # [B,T,d]
                rows = rows + self._post(upd)
                (k_all, v_all, full_k, full_v,
                 q_add, ctx_add) = _read_inputs(rows)

            # ONE shared head on the pooled hidden state (last pass only when
            # head_last_only is set: earlier passes exist to refine the dots'
            # states, not to emit a distribution)
            if lean:
                log_b = None
            else:
                if self.pool_first:
                    s_pool = torch.einsum("bfn,bfnd->bfd", w, out.state)
                    h_mix = s_pool.reshape(f, self.d_s).matmul(
                        self.dots._sel(p - 1, self.dots.p_out))     # [f, d]
                    h_mix = h_mix.reshape(b, t, self.d)
                else:
                    h_mix = torch.einsum("bfn,bfnd->bfd", w,
                                         out.e_state)               # [B, T, d]
                if self.head_skip:
                    h_mix = h_mix + rows
                h_mix = self._post(h_mix)
                if self.head_ln is not None:
                    h_mix = self.head_ln(h_mix)
                logits = self._head_logits(h_mix.reshape(f, self.d), emb_table)
                log_b = F.log_softmax(logits, dim=-1).reshape(b, t, -1)

            # hidden-space error features for the next pass (3 dims, cheap):
            # own weight, gate entropy, top weight
            w_ent = -(w * w.clamp_min(1e-12).log()).sum(-1, keepdim=True)
            w_top = w.max(-1, keepdim=True).values
            err = torch.cat([w.unsqueeze(-1), w_ent.unsqueeze(-1).expand(b, t, n, 1),
                             w_top.unsqueeze(-1).expand(b, t, n, 1)], dim=-1)

            per_pass.append({"weights": w, "log_b": log_b,
                             "gamma": out.gamma, "tau": tau_p,
                             "verifier": None})
            w_prev = w
            if self.messages is not None:
                msg_write = self.messages(out.state, training=training)
                msg_read = msg_write

        w_final = per_pass[-1]["weights"]
        summary = torch.einsum("bfn,bfnd->bfd", w_final, state_prev)

        if hard_mode and self.use_weights:
            w_hard = consensus.hard_top_m(w_final, self.hard_top_m)
            if self.pool_first:
                s_hard = torch.einsum("bfn,bfnd->bfd", w_hard, state_prev)
                h_mix = s_hard.reshape(f, self.d_s).matmul(
                    self.dots._sel(self.n_passes - 1, self.dots.p_out)
                ).reshape(b, t, self.d)
            else:
                h_mix = torch.einsum("bfn,bfnd->bfd", w_hard, e_state_last)
            if self.head_skip:
                h_mix = h_mix + rows
            h_mix = self._post(h_mix)
            if self.head_ln is not None:
                h_mix = self.head_ln(h_mix)
            logits = self._head_logits(h_mix.reshape(f, self.d), emb_table)
            log_final = F.log_softmax(logits, dim=-1).reshape(b, t, -1)
            w_out = w_hard
        else:
            log_final = per_pass[-1]["log_b"]
            w_out = w_final

        # per-dot logits, ONE GEMM, last pass only (oracle/metrics/losses)
        if self.per_dot_logits:
            e_f = e_state_last.reshape(f * n, self.d)
            log_pi = F.log_softmax(e_f.matmul(emb_table.t()),
                                   dim=-1).reshape(b, t, n, -1)
            per_pass[-1]["log_pi"] = log_pi

        return {"log_b": log_final,
                "region_summary": summary,
                "weights": w_out,
                "per_pass": per_pass,
                "state_final": state_prev}

    def _id_code(self, rows: torch.Tensor) -> torch.Tensor:
        """[B,T,d] -> [B,T,N,id_dim]: dot n sees the EXACT token at lag n+1.

        Region 1's rows ARE the token embeddings, so a time shift of the rows
        is a token shift — no token ids needed, and deeper regions get the
        same mechanism on their own row sequence. Position t reads rows[t-k-1]
        (zero where t-k-1 < 0): strictly causal, no wrap-around.
        """
        b, t, _ = rows.shape
        n = self.n_dots
        p = rows.matmul(self.w_id).reshape(b, t, n, self.id_dim)  # ONE GEMM
        p = F.pad(p, (0, 0, 0, 0, n, 0))                # n zero rows up front
        return torch.stack([p[:, n - k - 1: n - k - 1 + t, k]
                            for k in range(n)], dim=2)   # [B,T,N,id_dim]

    def _suffix_copy(self, ids: torch.Tensor, with_extra: bool = False):
        """(copy_ids [B,T], valid [B,T]) for the induction prior.

        ids[b, t] is the byte AT position t; the output at t predicts t+1, so
        the context is the k bytes ending at t -- all already observed -- and
        the copied byte x[s+1] is at most position t-1. Nothing at or after
        the target is read, so this is causal by construction.
        """
        k = self.suffix_copy
        b, t = ids.shape
        x = ids.to(torch.int64)
        zero = torch.zeros(b, 1, dtype=x.dtype, device=x.device)
        h = torch.zeros_like(x)
        for j in range(k):                      # exact for k <= 4 (< 2**32)
            # h[i] = sum_j x[i - j] * 256**j  ->  the k bytes ENDING at i,
            # i.e. x[i], x[i-1], ... (the context the output at i has seen)
            h = h + torch.cat((zero.expand(b, j), x), dim=1)[:, :t] \
                * (256 ** j)
        idx = torch.arange(t, dtype=torch.long, device=ids.device)
        full = idx >= (k - 1)                   # both contexts complete
        cand = (h.unsqueeze(2) == h.unsqueeze(1))
        cand = cand & full.view(1, 1, t) & full.view(1, t, 1)
        # s < t (an earlier occurrence) and s + 1 <= t (the copied byte is
        # observed), so the pointer never reaches the target
        cand = cand & (idx.view(1, 1, t) < idx.view(1, t, 1))
        pos = torch.where(cand, idx.view(1, 1, t).expand(b, t, t),
                          torch.full((), -1, dtype=torch.long,
                                     device=ids.device))
        s, _ = pos.max(dim=-1)                  # most recent match wins
        valid = s >= 0
        src = (s + 1).clamp_(min=0, max=t - 1)
        copy_ids = torch.where(valid, x.gather(1, src),
                               torch.zeros_like(x))
        if not with_extra:
            return copy_ids, valid
        # extension: how many bytes agree before the k-gram, capped. The match
        # is h[s]==h[t], i.e. x[s-j]==x[t-j] for j<k, so the first byte beyond
        # it compares x[t-k] with x[s-k]. Contiguous by construction: the
        # running mask requires every earlier byte to have matched.
        extra = torch.zeros_like(x)
        cap = self.sc_extra_cap
        for m in range(cap):
            ai = idx.view(1, t) - k - m
            si = s - k - m
            av = x.gather(1, ai.clamp_min(0).expand(b, t))
            sv = x.gather(1, si.clamp_min(0))
            ok = (ai >= 0) & (si >= 0) & (av == sv) & (extra == m)
            extra = extra + ok.to(extra.dtype)
        return copy_ids, valid, extra

    def _suffix_vote(self, ids: torch.Tensor, m: int):
        """Copy distribution over the max-EXTENSION occurrences of the k-gram.

        Returns (copy_ids [B,T,m], weights [B,T,m], valid [B,T], maxext [B,T]).
        `weights` is uniform over the candidates whose extension equals the
        maximum (1/k on a k-way tie), so a tie hedges instead of picking
        arbitrarily. `valid` is False only when the k-gram never occurred
        earlier, so coverage is identical to the single pointer's and any gain
        is precision on the same firing set. Causality is unchanged: every
        pointer s has s < t and copies x[s+1] with s+1 <= t.
        """
        k = self.suffix_copy
        b, t = ids.shape
        x = ids.to(torch.int64)
        zero = torch.zeros(b, 1, dtype=x.dtype, device=x.device)
        h = torch.zeros_like(x)
        for j in range(k):                      # same hash as _suffix_copy
            h = h + torch.cat((zero.expand(b, j), x), dim=1)[:, :t] \
                * (256 ** j)
        idx = torch.arange(t, dtype=torch.long, device=ids.device)
        full = idx >= (k - 1)
        cand = (h.unsqueeze(2) == h.unsqueeze(1))
        cand = cand & full.view(1, 1, t) & full.view(1, t, 1)
        cand = cand & (idx.view(1, 1, t) < idx.view(1, t, 1))
        pos = torch.where(cand, idx.view(1, 1, t).expand(b, t, t),
                          torch.full((), -1, dtype=torch.long,
                                     device=ids.device))
        # the m most recent occurrences, MOST RECENT FIRST: take the max, then
        # mask it out so the next max is the next most recent.
        ss = []
        for _ in range(m):
            s, _p = pos.max(dim=-1)
            ss.append(s)
            pos = pos.masked_fill(pos == s.unsqueeze(-1), -1)
        s = torch.stack(ss, dim=-1)             # [B,T,m], invalid = -1
        valid = s >= 0
        src = (s + 1).clamp_(min=0, max=t - 1)
        xe = x.unsqueeze(2).expand(b, t, m).contiguous()
        copy_ids = xe.gather(1, src)            # [B,T,m]
        # extension of each candidate: bytes agreeing BEFORE the k-gram
        extra = torch.zeros_like(s)
        for e in range(self.sc_extra_cap):
            ai = idx.view(1, t, 1) - k - e
            si = s - k - e
            av = xe.gather(1, ai.clamp_min(0).expand(b, t, m))
            sv = xe.gather(1, si.clamp_min(0))
            ok = (ai >= 0) & (si >= 0) & (av == sv) & (extra == e)
            extra = extra + ok.to(extra.dtype)
        exv = extra.masked_fill(~valid, -1)
        maxext, _p = exv.max(dim=-1)            # [B,T], -1 if none
        keep = valid & (exv == maxext.unsqueeze(-1))
        w = keep.to(extra.dtype) / keep.sum(-1, keepdim=True).clamp_min(1)
        return copy_ids, w, valid.any(dim=-1), maxext.clamp_min(0)

    # ------------------------------------------------------------------
    def forward(self, rows: torch.Tensor, emb_table: torch.Tensor,
                tau0: float = 1.0, hard_mode: bool = False,
                floor: float = 0.0, training: bool | None = None,
                ids: torch.Tensor | None = None,
                ablate_dot: int = -1, pool_only_dot: int = -1) -> dict:
        """rows [B, T, d] -> the region's consensus and diagnostics.

        floor: the per-dot weight floor eps (spec 04 section 3); callers hand
        0.1/N during training and 0 for the final phase / inference.
        ids: input token ids [B, T], only consumed by the ngram_bias term.
        """
        if training is None:
            training = self.training
        if self.head_mode == "hidden":
            out = self._forward_hidden(rows, emb_table, tau0, hard_mode,
                                       floor, training, ablate_dot=ablate_dot,
                                       pool_only_dot=pool_only_dot)
        else:
            out = self._forward_prob(rows, emb_table, tau0, hard_mode,
                                     floor, training)
        if self.ng_e1 is not None and ids is not None:
            # exact order-1 prior on the head logits; a lookup + one GEMM.
            prior = self.ng_e1[ids].matmul(self.ng_e2.t())      # [B,T,V]
            out["log_b"] = F.log_softmax(out["log_b"] + prior, dim=-1)
        if self.sc_gate is not None and ids is not None:
            # discrete induction: a probability MIXTURE, not a logit bias,
            # because the copy is itself a distribution. The gate starts ~3e-4
            # so the arm is the baseline at init and only earns weight on the
            # val loss. With `sc_slope` the gate is sigmoid(a + b*extension),
            # i.e. it can ask for more weight on a longer match; b=0 at init.
            p = out["log_b"].exp()
            if self.suffix_copy_vote > 0:
                # selection variant: distribute the copy over max-extension
                # occurrences; the gate sees the MAX extension.
                cm, wgt, valid, maxext = self._suffix_vote(
                    ids, self.suffix_copy_vote)
                q = torch.zeros_like(p)
                for j in range(cm.shape[-1]):
                    q = q + (wgt[..., j:j + 1].to(p.dtype)
                             * F.one_hot(cm[..., j], p.shape[-1]).to(p.dtype))
                logit = self.sc_gate
                if self.sc_slope is not None:
                    logit = (logit + self.sc_slope
                             * maxext.clamp(max=self.sc_extra_cap).to(
                                 self.sc_gate.dtype))
            else:
                if self.sc_slope is None:
                    copy_ids, valid = self._suffix_copy(ids)
                    logit = self.sc_gate
                else:
                    copy_ids, valid, extra = self._suffix_copy(ids,
                                                               with_extra=True)
                    logit = (self.sc_gate + self.sc_slope
                             * extra.clamp(max=self.sc_extra_cap).to(
                                 self.sc_gate.dtype))
                q = F.one_hot(copy_ids, p.shape[-1]).to(p.dtype)
            g = (torch.sigmoid(logit).unsqueeze(-1)
                 * valid.unsqueeze(-1).to(p.dtype))
            out["log_b"] = torch.log(((1 - g) * p + g * q).clamp_min(1e-9))
        return out

    def _forward_prob(self, rows: torch.Tensor, emb_table: torch.Tensor,
                      tau0: float, hard_mode: bool, floor: float,
                      training: bool) -> dict:
        if self.interleave:
            # only the fast path implements the between-pass re-read; the
            # prob-space path would silently read stale keys (and its
            # consensus machinery is unused by every arm anyway).
            raise NotImplementedError(
                "interleave is implemented in _forward_hidden only; set "
                "head_mode='hidden' (or fast_dot_math=True)")
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        n = self.n_dots

        full_k = full_v = k_all = v_all = None
        if self.global_read:
            k_all, v_all = self._full_kv(rows)
            full_k, full_v = k_all, v_all
        q_add = self._query_add(rows)
        # pass-invariant exact-identity code (one GEMM per forward)
        ctx_add = self._id_code(rows) if self.w_id is not None else None

        # same per-pass band machinery as _forward_hidden (bands mode gives
        # each pass its own window, so each needs its own attention cache)
        if self.window_mode == "bands":
            winds = [self._window_memory(rows, device, p, k_all, v_all)
                     for p in range(self.n_passes)]
            # each pass reads its own band, so a ctx cached by one pass is
            # never reusable by another: no cache at all.
            caches = [None] * self.n_passes
        else:
            one = self._window_memory(rows, device, 0, k_all, v_all)
            winds = [one] * self.n_passes
            # ONE shared dict: pass 1 fills ctx, passes 2..P hit the cache.
            # (Separate per-pass dicts would silently recompute the read P
            # times — same numbers, needless work.)
            shared = {} if self.cache_attention else None
            caches = [shared] * self.n_passes
        state = self.dots.initial_state((b, t))
        trust = None
        w_prev = None
        state_prev = None
        msg_read = None
        log_b_prev = None
        per_pass = []
        # e13: fresh slot table per forward, detached scan after each pass
        slots = (self.dots.empty_slots(b, device, rows.dtype)
                 if self.dots.S > 0 else None)

        # chunked consensus (speed opt 4): rows per consensus slice; 0 or
        # >= f disables chunking (single slice = the pre-opt-4 math exactly).
        # DEFAULT OFF: measured on the bench (results/bench_opt4_bf16.json,
        # vocab2/batch16/V=2048), chunk 64 was a 1.9x SLOWDOWN (10.13 vs
        # 5.43 s/step) — the [B,T,N,V] temporaries fit RAM, so slicing only
        # breaks GEMM batching. Keep the knob for memory-tight big-vocab
        # runs where the same block approaches GB scale.
        f_total = b * t
        chunk = f_total
        try:
            cz = int(os.environ.get("IECNN_CONS_CHUNK", "0"))
        except ValueError:
            cz = 0
        if 0 < cz < f_total:
            chunk = cz

        # dot dropout (spec table at E4: "5% of steps, one random dot masked
        # per region"): active only in training; the SAME dot is masked across
        # this region's passes so the region behaves as if the dot were absent.
        dot_mask = None
        if training and self.dot_dropout > 0 and torch.rand(()) < self.dot_dropout:
            k = int(torch.randint(0, n, ()))
            dot_mask = torch.ones(n, device=device, dtype=rows.dtype)
            dot_mask[k] = 0.0

        for p in range(1, self.n_passes + 1):
            wind_k, wind_v, valid = winds[p - 1]
            attn_cache = caches[p - 1]
            tau_p = tau0 * (self.kappa ** (p - 1))
            if p == 1:
                err = torch.zeros(b, t, n, 3, device=device, dtype=rows.dtype)
            if p == 1 or not self.use_weights:
                h_pool = torch.zeros(b, t, n, self.d_s, device=device, dtype=rows.dtype)
            else:
                pooled = torch.einsum("bfn,bfnd->bfd", w_prev, state_prev)  # [B,T,d_s]
                h_pool = pooled[:, :, None, :].expand(b, t, n, self.d_s)
                # ^ every dot reads the SAME pooled opinion (spec 01, input table)
            out = self.dots(wind_k, wind_v, valid, h_pool, err, round_idx=p - 1,
                            msg_in=msg_read if p > 1 else None, state=state,
                            emb_table=emb_table, attn_cache=attn_cache,
                            slot_table=slots, full_k=full_k, full_v=full_v,
                            q_add=q_add, ctx_add=ctx_add)
            state_prev = out.state
            if slots is not None:
                slots = self.dots.update_slots(state_prev, slots, p - 1)

            # ---- consensus + pooling, chunked over rows (speed opt 4) ----
            # The [., N, V] chain (exp, loo, JS, scores, weights, pool) is
            # memory-bound: whole-tensor it streams hundreds of MB through a
            # small L3 several times per pass. In `chunk`-row slices the
            # working set stays cache-hot and the big temporaries (loo, m,
            # logm) exist only per chunk. chunk == f reproduces the previous
            # single-slice math op-for-op (same ops, same order); only the
            # pooling GEMM's reassociation differs (last bit).
            f = b * t
            log_pi_f = out.log_pi.reshape(f, n, -1)          # contiguous views
            gamma_f = out.gamma.reshape(f, n)
            trust_term_f = out.trust_term.reshape(f, n)
            state_f = out.state.reshape(f, n, self.d_s)
            err_f = err.reshape(f, n, 3)
            w_prev_f = w_prev.reshape(f, n) if w_prev is not None else None
            trs, ws, lbs, vos, ens = [], [], [], [], []
            for i in range(0, f, chunk):
                sl = slice(i, min(i + chunk, f))
                tr_c, w_c, lb_c, v_c, en_c = self._pass_consensus(
                    log_pi_f[sl], gamma_f[sl], trust_term_f[sl],
                    trust[sl] if trust is not None else None,
                    state_f[sl], err_f[sl],
                    w_prev_f[sl] if w_prev_f is not None else None,
                    log_b_prev[sl] if log_b_prev is not None else None,
                    dot_mask, p, tau_p, hard_mode, floor)
                trs.append(tr_c); ws.append(w_c); lbs.append(lb_c)
                vos.append(v_c); ens.append(en_c)
            if self.use_weights:
                trust = torch.cat(trs, 0)
            w_f = torch.cat(ws, 0)
            log_b_f = torch.cat(lbs, 0)
            v_out = torch.cat(vos, 0) if self.verifier is not None else None
            # error features for the NEXT pass, computed now (chunk-wise) so
            # this pass's pi/loo never have to be kept whole
            if p < self.n_passes:
                err = torch.cat(ens, 0).reshape(b, t, n, 3)
            w = w_f.reshape(b, t, n)
            log_b = log_b_f.reshape(b, t, -1)
            per_pass.append({"log_pi": out.log_pi, "weights": w,
                             "log_b": log_b, "gamma": out.gamma, "tau": tau_p,
                             "verifier": (v_out.reshape(b, t, n)
                                          if v_out is not None else None)})
            log_b_prev = log_b_f
            w_prev = w
            if self.messages is not None:
                msg_write = self.messages(out.state, training=training)   # read at p+1
                msg_read = msg_write

        w_final = per_pass[-1]["weights"]
        summary = torch.einsum("bfn,bfnd->bfd", w_final, state_prev)  # [B, T, d_s]

        if hard_mode and self.use_weights:
            w_hard = consensus.hard_top_m(w_final, self.hard_top_m)
            log_final = consensus.pool_log_linear(per_pass[-1]["log_pi"], w_hard)
            w_out = w_hard
        else:
            log_final = per_pass[-1]["log_b"]   # [B, T, V] (log_b_prev is the
            w_out = w_final                     # chunk-layout [f, V] twin)

        return {"log_b": log_final,              # [B, T, V] region consensus
                "region_summary": summary,       # [B, T, d_s] for the next memory
                "weights": w_out,                # [B, T, N]
                "per_pass": per_pass,
                "state_final": state_prev}
