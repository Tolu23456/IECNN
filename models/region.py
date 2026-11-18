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
            share_cell=bool(m.get("share_cell", False)))
        # Lean passes: in hidden mode the pooled shared head and the per-dot
        # e_state projection are only consumed on the LAST pass. With
        # head_last_only the earlier passes skip both (they still update the
        # dots' states and the weighting). Default false = unchanged.
        self.head_last_only = bool(m.get("head_last_only", False))
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

    def _head_logits(self, h: torch.Tensor, emb_table: torch.Tensor):
        """[.., d] -> [.., V]. Tied embedding (default) or an untied head."""
        if self.w_head is not None:
            return h.matmul(self.w_head)
        return h.matmul(emb_table.t())

    def _full_kv(self, rows: torch.Tensor):
        """Keys/values for the GLOBAL read: one row per position, no window."""
        return self.w_k(rows), self.w_v(rows)

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
        wind_k_raw, valid = gather_windows(k_all, self.span, offs)
        wind_v, _ = gather_windows(v_all, self.span, offs)
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
                        training: bool) -> dict:
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

        full_k = full_v = k_all = v_all = None
        if self.global_read:
            # ONE projection pair serves both the window gather and the
            # global read (they read the same keys/values).
            k_all, v_all = self._full_kv(rows)
            full_k, full_v = k_all, v_all
        q_add = rows.matmul(self.w_qi) if self.w_qi is not None else None

        # Read windows. Normally every pass reads the SAME window, and the
        # attention cache is shared so the read is computed once (that is
        # exactly what made P>=2 "fake depth": passes 2..P re-used ctx).
        # window_mode "bands" gives each pass its own band (pass 1 local,
        # later passes far), so each pass must also have its OWN cache —
        # a cached ctx from a different band would be a different input.
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
                            full_v=full_v, q_add=q_add)
            state_prev = out.state
            if slots is not None:
                # e13: detached scan — the table this pass wrote is read by
                # the next pass (one-pass-delay persistence, no BPTT)
                slots = self.dots.update_slots(state_prev, slots, p - 1)
            if last:
                e_state_last = out.e_state

            tr_c, w_c = self._pass_hidden(
                out.gamma, out.trust_term,
                trust.reshape(b, t, n) if trust is not None else None,
                dot_mask, tau_p, 0.0 if hard_mode else floor)
            if self.use_weights:
                trust = tr_c
            w = w_c

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

    # ------------------------------------------------------------------
    def forward(self, rows: torch.Tensor, emb_table: torch.Tensor,
                tau0: float = 1.0, hard_mode: bool = False,
                floor: float = 0.0, training: bool | None = None) -> dict:
        """rows [B, T, d] -> the region's consensus and diagnostics.

        floor: the per-dot weight floor eps (spec 04 section 3); callers hand
        0.1/N during training and 0 for the final phase / inference.
        """
        if training is None:
            training = self.training
        if self.head_mode == "hidden":
            return self._forward_hidden(rows, emb_table, tau0, hard_mode,
                                        floor, training)
        device = rows.device
        b, t = rows.shape[0], rows.shape[1]
        n = self.n_dots

        full_k = full_v = k_all = v_all = None
        if self.global_read:
            k_all, v_all = self._full_kv(rows)
            full_k, full_v = k_all, v_all
        q_add = rows.matmul(self.w_qi) if self.w_qi is not None else None

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
                            q_add=q_add)
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
