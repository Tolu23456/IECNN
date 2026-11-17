"""A population of dots with separate weights (spec 01).

All N dots of a region are stored with a leading dot dimension N and computed
with batched matrix operations. The GRU cell is written by hand, because
built-in GRUs cannot hold separate weights per dot.

Per pass p, per dot k, per position t (b, f=batch shape B,T):
    q     = q_k                       (dynamic: q_k + U_k s_prev)   [d_q]
    K_t   = windowed memory keys (shared, built by Region)          [w, d_q]
    a     = softmax(K_t q / sqrt(d_q), causal window mask applied)  [w]
    c     = a V_t                                                   [d_v]
    u     = [c ; h_pool ; e ; r_p ; m_in ; slot_r]              [d_in]
    s'    = GRU_k(u, s)                                             [d_s]
    pi,gamma,trust from s' through the tied embedding E [V, d].

E13 slot table (Tolu, 2026-11-17): each dot also owns S private slots
[d_w each] carried ACROSS positions (dot states reset every position per
clarification C1 — the table is the persistence loophole). Writes are
gated and DETACHED (no gradient through time: near-zero training cost,
the fast-weights move validated in scripts/slot_toy.py — reset+table
recovered most of the ~0.4 bpc the reset costs at toy scale). The read
enters the GRU input; slots differentiate through per-slot gate
trajectories (different temporal mixtures of the dot's value stream).
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUND_DIM = 16          # learned pass-index embedding (spec 01)
ERR_DIM = 3             # error feature e (spec 01, section 5)


def gru_input_dim(d_v: int, d_s: int, message_dim: int = 0,
                  slot_read_dim: int = 0) -> int:
    """d_in = value + pooled opinion + error feature + round signal + message
    + slot read (e13: the dot's private carried table)."""
    return (d_v + d_s + ERR_DIM + ROUND_DIM + message_dim + slot_read_dim)


@dataclass
class DotOutputs:
    log_pi: torch.Tensor        # [B, T, N, V]
    gamma: torch.Tensor         # [B, T, N]
    trust_term: torch.Tensor    # [B, T, N]
    state: torch.Tensor         # [B, T, N, d_s]
    # shared-head mode (Tolu, 2026-11-17: "instead of every dot having its
    # own head they all share one head and only their state differs"):
    # per-dot pre-head hidden states [B, T, N, d]; None when log_pi was
    # materialized (the two are mutually exclusive outputs).
    e_state: torch.Tensor | None = None


def _orth_cols(feat: int, count: int, gen: torch.Generator) -> torch.Tensor:
    """[count, feat] rows that are (near-)orthogonal: distinct dots at init."""
    if count <= feat:
        a = torch.randn(feat, count, generator=gen)
        q_, _ = torch.linalg.qr(a)
        return q_.T
    return torch.randn(count, feat, generator=gen)


def _xavier(in_f: int, out_f: int, gen: torch.Generator) -> torch.Tensor:
    x = torch.empty(in_f, out_f)
    bound = math.sqrt(6.0 / max(in_f + out_f, 1))
    return x.uniform_(-bound, bound, generator=gen)


def _orth(feat: int, out_f: int, gen: torch.Generator) -> torch.Tensor:
    """[feat, out_f] with each of the 3 square blocks orthogonal (recurrent)."""
    if out_f % feat == 0:
        blocks = []
        for _ in range(out_f // feat):
            a = torch.randn(feat, feat, generator=gen)
            q_, _ = torch.linalg.qr(a)
            blocks.append(q_)
        return torch.cat(blocks, dim=1)
    x = torch.empty(feat, out_f)
    return x.normal_(std=1.0 / math.sqrt(feat), generator=gen)


class DotPopulation(nn.Module):
    """N dots in one region, all parameters stacked on a dot axis N.

    `forward` is given a pass index; with share_pass_params=False the pass
    selects its own parameter slice [p] (spec 03, D6). Otherwise parameters
    are shared across passes and the pass index only picks the round signal.
    """

    def __init__(self, n_dots, d, d_q, d_v, d_s, vocab_size, message_dim=0,
                 n_passes=3, dynamic_query=False, share_pass_params=True,
                 seed: int = 0, fast_math: bool = True,
                 slot_slots: int = 0, slot_dim: int = 32,
                 slot_decay: float = 0.5):
        super().__init__()
        self.N = n_dots
        self.d = d
        self.d_q = d_q
        self.d_v = d_v
        self.d_s = d_s
        self.d_m = message_dim
        self.P = n_passes
        self.dynamic_query = dynamic_query
        self.shared = share_pass_params
        self.fast_math = fast_math
        # speed opt 6: "shared_gemm" (default) = tied-head GEMM writes the
        # [f, N, V] layout directly (no permute copy of the [., N, V] head
        # output); "legacy" = the pre-opt-6 [N, f, V] + permute path, kept
        # for the paired A/B bench and as a fallback.
        self.head_layout = "shared_gemm"
        # Rung E12 (Tolu's seclusion mechanism, 2026-11-17): "every token
        # focuses on exactly one token ... since every token knows what they
        # entail they shout really loud so tokens get the ones they need
        # without having to search every other tokens first."
        # "soft" (default) = the original weighted average over the window;
        # "secluded" = each dot commits to exactly ONE token in its window
        # (hard top-1 on the read scores, straight-through backward). Set by
        # Region from model.read_mode; tau_seclusion anneals the soft term.
        self.read_mode = "soft"
        self.tau_seclusion = 1.0
        # E13 (Tolu's slot-table memory): S private slots per dot, dim d_w,
        # carried across positions inside one forward (zeroed at sequence
        # start). Parameter-lean: the write value is a per-dot d_s->d_w
        # projection shared across slots (slots differentiate via their own
        # gate trajectories = different temporal mixtures), the gate is
        # per-dot d_s->S, the read is per-slot-per-dim weights [N, S, d_w].
        # Writes run under no_grad -> grads reach only the read path.
        self.S = int(slot_slots)
        self.d_w = int(slot_dim)
        self.slot_decay = float(slot_decay)
        self.slot_read_dim = self.d_w if self.S > 0 else 0
        self.in_dim = gru_input_dim(d_v, d_s, message_dim, self.slot_read_dim)

        p_layers = 1 if share_pass_params else n_passes   # stacked per-pass depth
        g = torch.Generator().manual_seed(seed)

        self.q = nn.Parameter(torch.stack([_orth_cols(d_q, n_dots, g) for _ in range(p_layers)]))
        self.u_q = nn.Parameter(torch.stack(
            [torch.stack([_xavier(d_s, d_q, g) for _ in range(n_dots)]) for _ in range(p_layers)])) \
            if dynamic_query else None
        self.s0 = nn.Parameter(0.02 * torch.randn(p_layers, n_dots, d_s, generator=g))
        self.w_ih = nn.Parameter(torch.stack(
            [torch.stack([_xavier(self.in_dim, 3 * d_s, g) for _ in range(n_dots)]) for _ in range(p_layers)]))
        self.b_ih = nn.Parameter(torch.zeros(p_layers, n_dots, 3 * d_s))
        self.w_hh = nn.Parameter(torch.stack(
            [torch.stack([_orth(d_s, 3 * d_s, g) for _ in range(n_dots)]) for _ in range(p_layers)]))
        self.p_out = nn.Parameter(torch.stack(
            [torch.stack([_xavier(d_s, d, g) for _ in range(n_dots)]) for _ in range(p_layers)]))
        self.c_w = nn.Parameter(0.1 * torch.randn(p_layers, n_dots, d_s, generator=g))
        self.c_b = nn.Parameter(torch.zeros(p_layers, n_dots))
        self.g_w = nn.Parameter(0.1 * torch.randn(p_layers, n_dots, d_s, generator=g))
        self.g_b = nn.Parameter(torch.zeros(p_layers, n_dots))
        self.round_emb = nn.Embedding(max(n_passes, 1), ROUND_DIM)
        nn.init.normal_(self.round_emb.weight, std=0.02)

        # ---- e13 slot table parameters (Tolu) -----------------------------
        if self.S > 0:
            self.sv = nn.Parameter(torch.stack(      # write value d_s -> d_w
                [torch.stack([_xavier(d_s, self.d_w, g)
                              for _ in range(n_dots)])
                 for _ in range(p_layers)]))         # [p_layers, N, d_s, d_w]
            self.sg = nn.Parameter(0.1 * torch.randn(  # write gate d_s -> S
                p_layers, n_dots, d_s, self.S, generator=g))
            self.sgb = nn.Parameter(torch.zeros(p_layers, n_dots, self.S))
            self.sr = nn.Parameter(0.1 * torch.randn(  # read [N, S, d_w]
                p_layers, n_dots, self.S, self.d_w, generator=g))
        else:
            self.sv = self.sg = self.sgb = self.sr = None

    # ----- e13 slot table --------------------------------------------------
    def empty_slots(self, batch: int, device, dtype) -> torch.Tensor:
        """Fresh (zeroed) slot table [B, N, S, d_w] — the sequence boundary.

        Each forward starts with an empty table, matching how the windowed
        memory treats a new document chunk (clarification C1 stays true for
        the GRU state; ONLY the table persists, and only within a sequence).
        """
        return torch.zeros(batch, self.N, self.S, self.d_w,
                           device=device, dtype=dtype)

    def slot_context(self, slot_table: torch.Tensor | None,
                     round_idx: int) -> torch.Tensor | None:
        """Read the dot's own slots: [B, N, S, d_w] -> [B, N, d_w].

        Differentiable ONLY through the read weights (the table itself is
        built under no_grad), so training cost stays near zero.
        """
        if self.S == 0:
            return None
        return torch.einsum("bnsd,nsd->bnd", slot_table,
                            self._sel(round_idx, self.sr))

    def update_slots(self, state: torch.Tensor,
                     slot_table: torch.Tensor,
                     round_idx: int) -> torch.Tensor:
        """Gated DETACHED write, sequential over positions: after a pass.

        state [B, T, N, d_s] -> new [B, N, S, d_w].
        M <- dec*M + (1-dec) * sigmoid(gate) * tanh(value)
        Per-slot gates give each slot its own temporal mixture of the dot's
        value stream (that is how slots differentiate — no per-slot heads).
        """
        with torch.no_grad():
            b, t = state.shape[0], state.shape[1]
            v = torch.tanh(torch.einsum(
                "btnd,nde->btne", state, self._sel(round_idx, self.sv)))
            g = torch.sigmoid(torch.einsum(
                "btnd,nds->btns", state, self._sel(round_idx, self.sg))
                + self._sel(round_idx, self.sgb))
            m = slot_table
            dec = self.slot_decay
            for i in range(t):
                # gate [B,N,S,1] x value [B,N,1,d_w] -> [B,N,S,d_w]: the
                # write VALUE is shared across slots; the per-slot gate
                # trajectory is what makes slot contents differ over time
                m = (dec * m + (1.0 - dec) * g[:, i].unsqueeze(-1)
                     * v[:, i].unsqueeze(2))
            return m

    # ----- helpers ---------------------------------------------------------
    def _sel(self, pass_idx: int, t: torch.Tensor) -> torch.Tensor:
        """[p_layers, N, ...] -> current pass's slice (or layer 0 when shared)."""
        return t[0 if self.shared else pass_idx]

    def _secluded_ctx(self, scores: torch.Tensor, wv: torch.Tensor) -> torch.Tensor:
        """Tolu's seclusion mechanism: each dot commits to exactly ONE token
        in its window and reads only that token's value (rung E12).

        scores [f, N, w] (already sqrt(d_q)-scaled and validity-masked),
        wv [f, w, d_v]. Forward = the hard pick (argmax token's value, no
        average over the other w-1 tokens). Backward = the soft gradient:
        straight-through, so near-miss tokens keep learning their keys —
        the 'shouting' part of the mechanism (a token's key is what makes
        it findable; the soft term is what tells it how loud to shout).
        """
        idx = scores.argmax(-1)                                    # [f, N]
        ctx_hard = wv.gather(
            1, idx.unsqueeze(-1).expand(-1, -1, wv.shape[-1]))      # [f, N, d_v]
        attn_soft = torch.nan_to_num(
            F.softmax(scores / self.tau_seclusion, dim=-1), nan=0.0)
        ctx_soft = torch.matmul(attn_soft.unsqueeze(-2),
                                wv.unsqueeze(1)).squeeze(-2)       # [f, N, d_v]
        return ctx_hard + (ctx_soft - ctx_soft.detach())

    def initial_state(self, batch_shape) -> torch.Tensor:
        """Broadcast learned s0 to [*batch_shape, N, d_s] for pass 1.

        States reset every position (clarification C1), so pass 1 always
        starts from the learned s0, independent of history.
        """
        s0 = self._sel(0, self.s0)                                  # [N, d_s]
        return s0.expand(*batch_shape, self.N, self.d_s).contiguous()

    # ----- one pass --------------------------------------------------------
    def forward(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                round_idx: int, msg_in, state, emb_table,
                attn_cache: dict | None = None,
                want_log_pi: bool = True,
                slot_table: torch.Tensor | None = None) -> DotOutputs:
        """One pass for all dots at all positions simultaneously.

        wind_k    [B, T, w, d_q]   windowed, rel-distance-added keys
        wind_v    [B, T, w, d_v]   windowed values
        valid_mask [T, w] bool     True = row exists (t - j < w and j >= 0)
        h_pool    [B, T, N, d_s]   previous pass's pooled opinion (0 at pass 1)
        err_feat  [B, T, N, 3]     error feature (0 at pass 1)
        msg_in    [B, T, N, d_m]   or None
        state     [B, T, N, d_s]   s_prev
        emb_table [V, d]           tied embedding, shared by the whole model
        slot_table [B, N, S, d_w]  e13: the dot's carried private slots
                                   (None when slots are off)
        attn_cache dict           speed opt 7 (Tolu's "don't compute the same
                                  operation twice"): when shared pass params +
                                  static queries make the attention block
                                  IDENTICAL across passes, the caller passes
                                  a fresh dict per region-forward and pass 1
                                  stores ctx; later passes reuse it (gradients
                                  accumulate exactly as with recompute — the
                                  same shared nodes get the sum of all pass
                                  contributions). Only valid when the caller
                                  guarantees same wind_k/wind_v/valid + q.
        """
        if getattr(self, "fast_math", True):
            return self._forward_fast(wind_k, wind_v, valid_mask, h_pool,
                                      err_feat, round_idx, msg_in, state,
                                      emb_table, attn_cache, want_log_pi,
                                      slot_table)
        return self._forward_reference(wind_k, wind_v, valid_mask, h_pool,
                                       err_feat, round_idx, msg_in, state,
                                       emb_table, want_log_pi, slot_table)

    # the einsum-per-op version, kept for the equivalence test and as a
    # reference implementation of spec 01 (slower: einsum re-plans paths the
    # kernel cannot fuse into batched GEMMs)
    def _forward_reference(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                           round_idx: int, msg_in, state, emb_table,
                           want_log_pi: bool = True,
                           slot_table: torch.Tensor | None = None) -> DotOutputs:
        b, t, w = wind_k.shape[0], wind_k.shape[1], wind_k.shape[2]
        device = wind_k.device

        # ---- queries + reads ---------------------------------------------
        if self.dynamic_query:
            q_dyn = self._sel(round_idx, self.q).unsqueeze(0).unsqueeze(0) \
                + torch.einsum("bfnd,ndj->bfnj", state, self._sel(round_idx, self.u_q))
            scores = torch.einsum("bfwj,bfnj->bfnw", wind_k, q_dyn)     # read by query
        else:
            scores = torch.einsum("bfwj,nj->bfnw", wind_k, self._sel(round_idx, self.q))
        scores = scores / math.sqrt(self.d_q)
        if valid_mask is not None:
            scores = scores.masked_fill(~valid_mask[None, :, None, :], float("-inf"))
        if self.read_mode == "secluded":
            ctx = self._secluded_ctx(
                scores.reshape(b * t, self.N, w),
                wind_v.reshape(b * t, w, self.d_v)).reshape(
                b, t, self.N, self.d_v)
        else:
            attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)    # [B, T, N, w]
            ctx = torch.matmul(attn.unsqueeze(-2), wind_v.unsqueeze(-3)).squeeze(-2)  # [B, T, N, d_v]

        # ---- GRU step (all dots at once) ----------------------------------
        parts = [ctx, h_pool, err_feat,
                 self.round_emb.weight[round_idx].expand(b, t, self.N, ROUND_DIM)]
        if self.d_m > 0:
            parts.append(msg_in if msg_in is not None else
                         torch.zeros(b, t, self.N, self.d_m, device=device, dtype=ctx.dtype))
        if self.S > 0:
            # e13: the dot's carried slots read as ONE vector per dot, the
            # same for every position in this pass (the table updates
            # between passes — a one-pass-delay persistence channel)
            r = self.slot_context(slot_table, round_idx)            # [B, N, d_w]
            parts.append(r.unsqueeze(1).expand(b, t, self.N, self.d_w))
        x = torch.cat(parts, dim=-1)                                    # [B, T, N, d_in]

        in_gates = torch.einsum("bfni,niq->bfnq", x, self._sel(round_idx, self.w_ih))
        hid_gates = torch.einsum("bfnd,ndq->bfnq", state, self._sel(round_idx, self.w_hh))
        in_gates = in_gates + self._sel(round_idx, self.b_ih)
        iz, ir, inet = in_gates.chunk(3, dim=-1)
        hz, hr, hnet = hid_gates.chunk(3, dim=-1)
        z = torch.sigmoid(iz + hz)
        r = torch.sigmoid(ir + hr)
        n_t = torch.tanh(inet + r * hnet)                               # r-gated candidate
        s_new = (1.0 - z) * n_t + z * state

        # ---- heads --------------------------------------------------------
        e_state = torch.einsum("bfnd,ndm->bfnm", s_new, self._sel(round_idx, self.p_out))
        gamma = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.c_w)) \
            + self._sel(round_idx, self.c_b)
        trust = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.g_w)) \
            + self._sel(round_idx, self.g_b)
        if not want_log_pi:
            # shared-head mode: no per-dot vocab head — the region pools
            # e_state in hidden space and runs ONE shared head (Tolu).
            return DotOutputs(log_pi=None, gamma=gamma, trust_term=trust,
                              state=s_new, e_state=e_state)
        logits = torch.einsum("bfnm,vm->bfnv", e_state, emb_table)
        log_pi = F.log_softmax(logits, dim=-1)
        return DotOutputs(log_pi=log_pi, gamma=gamma, trust_term=trust, state=s_new,
                          e_state=e_state)

    # the optimized twin: same math, same op ORDER within each vector op, but
    # the three heavy contractions (GRU in-gates, GRU hidden gates, tied head)
    # run as plain batched GEMMs over a merged [B*T, N] layout. Differences
    # vs the reference are last-bit floating-point (summation order inside a
    # GEMM); pinned by test_pyi_fast_math_matches_reference.
    def _forward_fast(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                      round_idx: int, msg_in, state, emb_table,
                      attn_cache: dict | None = None,
                      want_log_pi: bool = True,
                      slot_table: torch.Tensor | None = None) -> DotOutputs:
        b, t, w = wind_k.shape[0], wind_k.shape[1], wind_k.shape[2]
        f = b * t
        n = self.N
        device = wind_k.device
        dt = wind_k.dtype

        wk = wind_k.reshape(f, w, self.d_q)                              # [f, w, d_q]
        wv = wind_v.reshape(f, w, self.d_v)                              # [f, 1?, w, d_v]

        # ---- queries + reads ---------------------------------------------
        # speed opt 7 (Tolu's "don't compute the same op twice"): with shared
        # pass params + static queries the whole attention block is
        # bit-identical on every pass — compute it ONCE per region-forward
        # (pass 1 stores ctx in the caller's fresh cache dict) and reuse it
        # on later passes. Gradients accumulate identically: the same shared
        # nodes receive the sum of all pass contributions, exactly like three
        # separate recomputations summed. Only valid when the caller feeds
        # one cache dict with the same wind_k/wind_v/valid + q (region
        # forward does; per-call dicts never leak across steps).
        cacheable = attn_cache is not None and not self.dynamic_query
        cache_hit = cacheable and "ctx" in attn_cache
        if cache_hit:
            ctx = attn_cache["ctx"]                                     # [f, N, d_v]
        else:
            if self.dynamic_query:
                s_f = state.reshape(f, n, self.d_s)
                q_dyn = torch.bmm(
                    s_f.transpose(0, 1),                                # [N, f, d_s]
                    self._sel(round_idx, self.u_q))                     # [N, d_s, d_q]
                q_dyn = (q_dyn.transpose(0, 1)                          # [f, N, d_q]
                         + self._sel(round_idx, self.q).unsqueeze(0))
                # read by query: scores[n, f?, w] per position: contract d_q
                scores = torch.einsum("fwj,fnj->fnw", wk, q_dyn)
            else:
                # one GEMM: [f*w, d_q] @ [d_q, N] -> [f, w, N]
                scores = torch.tensordot(wk, self._sel(round_idx, self.q),
                                         dims=([2], [1]))
                scores = scores.permute(0, 2, 1)                        # [f, N, w]
            scores = scores / math.sqrt(self.d_q)
            if valid_mask is not None:
                vm = valid_mask.unsqueeze(0).expand(b, t, w).reshape(f, 1, w)
                scores = scores.masked_fill(~vm, float("-inf"))         # [f, N, w]
            if self.read_mode == "secluded":
                ctx = self._secluded_ctx(scores, wv)                    # [f, N, d_v]
            else:
                attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)  # [f, N, w]
                ctx = torch.matmul(attn.unsqueeze(-2), wv.unsqueeze(1)).squeeze(-2)  # [f, N, d_v]
            if cacheable:
                attn_cache["ctx"] = ctx

        # ---- GRU step (batched GEMM per dot over merged [B*T]) ------------
        parts = [ctx, h_pool.reshape(f, n, self.d_s), err_feat.reshape(f, n, 3),
                 self.round_emb.weight[round_idx].expand(f, n, -1)]
        if self.d_m > 0:
            parts.append(msg_in.reshape(f, n, self.d_m) if msg_in is not None else
                         torch.zeros(f, n, self.d_m, device=device, dtype=dt))
        if self.S > 0:
            r = self.slot_context(slot_table, round_idx)            # [B, N, d_w]
            parts.append(r.unsqueeze(1).expand(b, t, n, self.d_w)
                         .reshape(f, n, self.d_w))
        x = torch.cat(parts, dim=-1)                                    # [f, N, d_in]

        w_ih_b = self._sel(round_idx, self.w_ih)                        # [N, in, 3h]
        in_gates = torch.bmm(x.transpose(0, 1), w_ih_b)                 # [N, f, 3h]
        s_f = state.reshape(f, n, self.d_s)
        hid_gates = torch.bmm(s_f.transpose(0, 1),                      # [N, f, d_s]
                              self._sel(round_idx, self.w_hh))          # [N, d_s, 3h]
        in_gates = (in_gates.transpose(0, 1)                            # [f, N, 3h]
                    + self._sel(round_idx, self.b_ih).unsqueeze(0))
        h3 = hid_gates.transpose(0, 1)                                  # [f, N, 3h] (view)
        iz, ir, inet = in_gates.chunk(3, dim=-1)
        hz, hr, hnet = h3.chunk(3, dim=-1)
        z = torch.sigmoid(iz + hz)
        r = torch.sigmoid(ir + hr)
        n_t = torch.tanh(inet + r * hnet)
        s_new = (1.0 - z) * n_t + z * s_f                               # [f, N, d_s]

        # ---- heads --------------------------------------------------------
        # speed opt 6 (Tolu's "don't move data around" hint): with
        # head_layout="shared_gemm" the tied-head GEMM writes the [f, N, V]
        # layout directly — the legacy path wrote [N, f, V] and then paid a
        # FULL permute+reshape copy of the [B, T, N, V] log-probs (~134 MB in
        # bf16 at the bench shape) on every region-pass, forward AND
        # backward. The small [., ., d_s]-scale transpose stays; the big
        # tensor is only ever written once, by the GEMM, in final layout.
        # Same math; only GEMM reassociation differs (last bits).
        e_state = torch.bmm(s_new.transpose(0, 1),                     # [N, f, d_s]
                            self._sel(round_idx, self.p_out))          # [N, d_s, d]
        gamma = torch.einsum("fnd,nd->fn", s_new, self._sel(round_idx, self.c_w)) \
            + self._sel(round_idx, self.c_b).unsqueeze(0)
        trust = torch.einsum("fnd,nd->fn", s_new, self._sel(round_idx, self.g_w)) \
            + self._sel(round_idx, self.g_b).unsqueeze(0)
        if not want_log_pi:
            # shared-head mode: no per-dot vocab head — the region pools
            # e_state in hidden space and runs ONE shared head (Tolu).
            e_state = e_state.permute(1, 0, 2).reshape(b, t, n, self.d)
            return DotOutputs(log_pi=None, gamma=gamma.reshape(b, t, n),
                              trust_term=trust.reshape(b, t, n),
                              state=s_new.reshape(b, t, n, self.d_s),
                              e_state=e_state)
        if self.head_layout == "shared_gemm":
            e_state_btnd = e_state.transpose(0, 1).reshape(f * n, self.d)  # small copy
            logits = e_state_btnd.matmul(emb_table.t())                # [f*N, V]
            log_pi = F.log_softmax(logits, dim=-1).reshape(b, t, n, -1)  # view only
            e_state_out = e_state_btnd.reshape(b, t, n, self.d)
        else:
            logits = torch.matmul(e_state, emb_table.t())              # [N, f, V]
            log_pi = F.log_softmax(logits, dim=-1).permute(1, 0, 2)    # [f, N, V]
            log_pi = log_pi.reshape(b, t, n, -1)                       # [B, T, N, V]
            e_state_out = e_state.permute(1, 0, 2).reshape(b, t, n, self.d)
        return DotOutputs(log_pi=log_pi,
                          gamma=gamma.reshape(b, t, n),
                          trust_term=trust.reshape(b, t, n),
                          state=s_new.reshape(b, t, n, self.d_s),
                          e_state=e_state_out)
