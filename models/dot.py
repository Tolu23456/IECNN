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


@torch.compiler.disable
def _global_ctx_eager(dots, full_k, full_v, round_idx, q_add=None,
                      cache=None):
    """Run the prefix scan as an eager graph break.

    Inductor's CUDA SplitScan lowering dies on the cumsum in `_global_ctx`
    ('TypeError: list indices must be integers or slices, not NoneType') on
    torch 2.11+cu130, whatever the axis or memory order; the CPU backend on
    2.14 compiles the identical code fine. This wrapper compiles everything
    around the read and leaves only the five-op scan eager, which costs one
    graph break and recovers `torch.compile` for the whole arm (~24% of a
    CPU step, and the d192 capacity arm is token-starved, so it matters).
    Opt-in via the `break_scan` config key; off by default so the compiled
    shape of every past measurement stays reproducible.
    """
    return dots._global_ctx(full_k, full_v, round_idx, q_add=q_add,
                            cache=cache)

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
                 span=16,
                 seed: int = 0, fast_math: bool = True,
                 slot_slots: int = 0, slot_dim: int = 32,
                 slot_decay: float = 0.5, cell: str = "gru",
                 share_out: bool = False, dist_bias: bool = False,
                 global_read: bool = False, read_schedule: str = "mix",
                 in_ln: bool = False, share_cell: bool = False,
                 carry_state: bool = False, carry_chunk: int = 8,
                 gla: bool = False, global_exact: bool = False,
                 break_scan: bool = False, read_heads: int = 1,
                 global_decay: bool = False, global_decay_k: int = 1):
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
        # Seclusion v2: the straight-through backward only needs the tokens
        # that NEARLY won. seclusion_topk=0 (default) keeps the original
        # full-window softmax; k>0 restricts it to the top-k scores, which
        # cuts the gradient path (and its [f,N,w,d_v] tensors) by w/k and
        # aims the learning signal exactly at the near-miss keys ("shout
        # louder" is a message for the runner-up, not for token 20 back).
        self.ste_topk = 0
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

        # Recurrence cell. "gru" (default) = the original 3-gate GRU; "lug"
        # = the Lean Update Gate (SRU-style): the only recurrent matrix is
        # DIAGONAL (u * s), so the cell costs one input GEMM of width 2*d_s
        # and NO hidden GEMM — ~2.5x fewer MACs than the GRU:
        #     z = sigmoid(x W_z + u * s + b_z)      (forget/keep gate)
        #     h = tanh(x W_h + b_h)                 (candidate)
        #     s' = (1 - z) h + z s
        # The committee still integrates over passes; only the integrator
        # changed. Selected by model.cell.
        self.cell = str(cell)
        # share_out: ONE output projection for all dots. Then
        #   sum_n w_n (s_n W)  ==  (sum_n w_n s_n) W,
        # so the region can pool the d_s states FIRST and run a single
        # d_s->d GEMM on the pooled vector: the per-dot e_state tensor and its
        # N-way GEMM disappear (region.pool_first turns this on).
        self.share_out = bool(share_out)

        p_layers = 1 if share_pass_params else n_passes   # stacked per-pass depth
        g = torch.Generator().manual_seed(seed)

        self.q = nn.Parameter(torch.stack([_orth_cols(d_q, n_dots, g) for _ in range(p_layers)]))
        # read_heads: several softmax PATTERNS per dot over the SAME window.
        # Each dot currently reads its window with one query, i.e. one softmax
        # over 8 slots, so a dot cannot simultaneously attend to (say) the
        # previous token and the token 6 back. A transformer gets 4 patterns
        # per layer for free. Heads split the existing d_v into H slices —
        # total context width, total value MACs and total ctx size are all
        # unchanged; the only new parameters are (H-1)*d_q per dot, and head 0
        # reuses `self.q`, so H=1 is bit-identical and adds nothing.
        # global_decay: the prefix read currently sums a bag of EVERY past
        # key with no distance term (e_s = exp(q.k_s) does not know how far
        # back s is), so "the token 10 back" and "the token 200 back" are
        # indistinguishable to it. One learned per-dot geometric decay per
        # step gives the bag recency:
        #     ctx_t = sum_{s<=t} e_s gamma^(t-s) v_s / sum_{s<=t} e_s gamma^(t-s)
        # which is still two cumsums: multiply by gamma^-s before the scan,
        # the common factor gamma^-t cancels in the ratio. N extra scalars.
        self.global_decay = bool(global_decay)
        # global_decay_k: how many timescales the decay is a convex mix of.
        # ONE geometric decay says "recency matters, at a fixed rate", but a
        # real sequence has several: the previous token, the current word, the
        # current section. K geometric reads combined by a per-dot softmax
        # cover all of them for still-O(T) work (K prefix scans, each ~1/100
        # of a post block) and N*(K-1) extra parameters. K=1 is exactly the
        # single decay, same code path, same numbers.
        self.global_decay_k = max(1, int(global_decay_k))
        if self.global_decay:
            # sigmoid^-1(0.99) so the read starts almost uniform and can only
            # sharpen if that pays
            shape = ((p_layers, n_dots) if self.global_decay_k == 1
                     else (p_layers, n_dots, self.global_decay_k))
            self.g_decay = nn.Parameter(torch.full(shape, 4.6))
        else:
            self.g_decay = None
        self.read_heads = max(1, int(read_heads))
        if self.read_heads > 1:
            self.q_extra = nn.Parameter(torch.stack([
                _orth_cols(d_q, n_dots * (self.read_heads - 1), g).view(
                    n_dots, self.read_heads - 1, d_q)
                for _ in range(p_layers)]))
        else:
            self.q_extra = None
        self.u_q = nn.Parameter(torch.stack(
            [torch.stack([_xavier(d_s, d_q, g) for _ in range(n_dots)]) for _ in range(p_layers)])) \
            if dynamic_query else None
        # dist_bias: ONE learned scalar per (dot, window slot) added to that
        # dot's read scores. Cost = N*w additions per position (~64 MACs,
        # i.e. free), and it lets dots SPECIALISE BY DISTANCE instead of all
        # collapsing onto the same token: the pointer audit measured 52-70%
        # of positions where all dots pick the same slot (mean pick distance
        # 0.11-0.28 back), i.e. an 8-dot committee acting as one bigram
        # detector. A per-dot distance prior is the cheapest possible fix:
        # one dot can price the previous token, another distance 8, another
        # distance 64, and the mixture weights arbitrate per position.
        self.d_bias = (nn.Parameter(torch.zeros(p_layers, n_dots, span))
                       if dist_bias else None)
        # global_read: add a CAUSAL read of the ENTIRE prefix to the windowed
        # read, at O(T) cost instead of O(T^2).
        #
        # Why this is cheap: a dot's query is STATIC (dynamic_query off), so
        # score(s) = q_n . k_s does not depend on the reading position t.
        # Full causal attention is therefore just a normalised PREFIX SUM:
        #     e_s     = exp(q_n . k_s)
        #     ctx_t   = sum_{s<=t} e_s v_s / sum_{s<=t} e_s
        # which is two cumsums — no T*T matrix, no extra GEMM per position.
        # Cost at T=64, N=4, d_v=32: ~16k MAC-equivalents vs 59k for the LUG
        # cell, i.e. the dots get the whole sequence for less than the cell.
        # g_glob mixes the two reads per dot, so dots can SPECIALISE: some
        # stay local (bigram precision), some read the whole prefix.
        self.global_read = bool(global_read)
        # read_schedule "split": one g_glob row PER PASS, initialised so pass 1
        # reads the LOCAL window and later passes read the GLOBAL prefix.
        #
        # This is the fix for "fake depth" that costs nothing: with the
        # attention cache both reads are already computed once and cached
        # separately (ctx, g_ctx), so passes 2..P were receiving a
        # BIT-IDENTICAL input and depth was nominal. Differing only in the
        # gate row makes each pass read a different VIEW of the same prefix
        # at zero extra reads and zero extra GEMMs (the mix is N*d_v adds).
        # Init +-2 gives 0.12 / 0.88 mixing, i.e. a strong but learnable
        # preference, so a pass can still soften it if that is better.
        self.read_schedule = str(read_schedule)
        # in_ln: LayerNorm over the concatenated cell input. Transformer
        # blocks normalise their input before every sublayer; the cell was
        # fed ctx/h_pool/err/round-emb raw, so at lr 2e-3 the input scale
        # drifted with the read magnitude. Cost: one pass over
        # [f, N, in_dim] (~1e5 elements, <0.3% of a step).
        # share_cell: the N dots share ONE update cell and differ only in
        # what they read. That turns N batched [f, in] @ [in, 2h] GEMMs into
        # ONE [f*N, in] @ [in, 2h] GEMM (4x the rows, same FLOPs, far better
        # shape) and cuts the cell's parameters by N. The dots still have
        # their own queries, their own read mix and their own output
        # projection, so specialisation happens on the READ side — which is
        # where the pointer audit says the committee actually differs.
        self.share_cell = bool(share_cell)
        # carry_state: the dot state RECURS ACROSS POSITIONS instead of
        # resetting at every position. Up to now the cell was applied
        # position-wise (state = learned s0 at every t, all passes), i.e. the
        # dots were a per-position MLP with a committee; the only temporal
        # structure came from the fixed read window and the global prefix-sum.
        # A toy measurement (reset 4.21 vs carry 3.79 bpc) says this reset is
        # worth ~0.4 bpc, which is larger than any mechanism kept so far.
        #
        # Making it exact AND cheap needs one change to the cell: the gate
        # loses its state term,  z = sigmoid(x W_z + b)  instead of
        # z = sigmoid(x W_z + u * s + b), because the latter is non-linear in
        # s and cannot be scanned. The cell then becomes an SRU-style
        # diagonal linear recurrence
        #     s_t = z_t * s_{t-1} + (1 - z_t) * tanh(x_t W_h + b_h)
        # which is affine in s and therefore has an EXACT parallel scan.
        # _carry_scan does it in two levels (carry_chunk positions composed
        # per chunk, then one short loop over chunk carries) with NO division
        # and no cumprod blow-up: within-chunk products have <= carry_chunk
        # factors, so the gate products stay in range.
        self.carry_state = bool(carry_state)
        self.carry_chunk = int(carry_chunk)
        # gla: replace the global read's STATIC-QUERY prefix-sum with
        # content-addressed LINEAR attention. This is the one thing the
        # transformer has that the dots did not: a query that depends on the
        # current token, so the read can say "find the past positions that
        # look like what I am holding" instead of "average the prefix with a
        # fixed weighting".
        #
        # The old read is the special case phi == 1 of
        #     S_t = sum_{s<=t} phi(k_s) v_s^T,  z_t = sum_{s<=t} phi(k_s)
        #     ctx_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)
        # with phi = elu(z / d_q^(1/4)) + 1. Both prefix sums are shared by
        # ALL dots (the keys are shared) and pass-invariant, so they are
        # computed once and cached with the window read. No O(T^2) tensor,
        # no new parameters of any kind (q is the existing per-dot query,
        # optionally plus the existing w_qi content term).
        #
        # Why this and not more parameters: five batches in a row have shown
        # that ANY mechanism that adds parameters or gradient paths costs
        # 0.06-0.12 bpc at this budget. This one adds zero.
        self.gla = bool(gla)
        # global_exact: replace the static-query prefix sum with a REAL
        # causal softmax over every past key, with the query allowed to depend
        # on the current token (q_add). Why this is not redundant with `gla`:
        # gla is the LINEAR-attention approximation of this read (phi = elu+1,
        # a [d_q,d_v] running state). It lost by +0.09 bpc, but a linear
        # feature map is exactly the weak form of content addressing, so that
        # result does not test the strong form. The static prefix sum is the
        # other degenerate case (q independent of t), where softmax over the
        # prefix is algebraically identical to a cumsum ratio — which makes
        # `global_exact` with a static query a free exactness test for the
        # whole family. Cost is O(T^2 * d_q) for the scores, but only for the
        # dots that take the global read, at d_q=16, so at T=64 it is a few
        # percent of a step; at T=256 it is the honest price of the test.
        self.global_exact = bool(global_exact)
        # break_scan: see _global_ctx_eager. CUDA-only Inductor workaround.
        self.break_scan = bool(break_scan)
        n_cell = 1 if self.share_cell else n_dots
        self.in_ln = (nn.LayerNorm(gru_input_dim(d_v, d_s, message_dim,
                                                 self.slot_read_dim))
                      if bool(in_ln) else None)
        if self.global_read:
            if self.read_schedule == "split":
                bias = torch.zeros(n_passes, n_dots)
                bias[0::2] = -2.0
                bias[1::2] = 2.0
                self.g_glob = nn.Parameter(bias)
            else:
                self.g_glob = nn.Parameter(torch.zeros(p_layers, n_dots))
        else:
            self.g_glob = None
        self.s0 = nn.Parameter(0.02 * torch.randn(p_layers, n_dots, d_s, generator=g))
        if self.cell == "lug":
            self.w_ih = None
            self.b_ih = None
            self.w_hh = None
            self.w_ih_lug = nn.Parameter(torch.stack(
                [torch.stack([_xavier(self.in_dim, 2 * d_s, g)
                              for _ in range(n_cell)]) for _ in range(p_layers)]))
            self.b_lug = nn.Parameter(torch.zeros(p_layers, n_cell, 2 * d_s))
            # diagonal recurrence gain, init 0.5 so the gate starts ~0.5
            self.u_lug = nn.Parameter(torch.full((p_layers, n_cell, d_s), 0.5))
        else:
            self.w_ih = nn.Parameter(torch.stack(
                [torch.stack([_xavier(self.in_dim, 3 * d_s, g) for _ in range(n_dots)]) for _ in range(p_layers)]))
            self.b_ih = nn.Parameter(torch.zeros(p_layers, n_dots, 3 * d_s))
            self.w_hh = nn.Parameter(torch.stack(
                [torch.stack([_orth(d_s, 3 * d_s, g) for _ in range(n_dots)]) for _ in range(p_layers)]))
            self.w_ih_lug = self.b_lug = self.u_lug = None
        if self.share_out:
            self.p_out = nn.Parameter(torch.stack(
                [_xavier(d_s, d, g) for _ in range(p_layers)]))   # [p, d_s, d]
        else:
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
        k = self.ste_topk if self.ste_topk > 0 else scores.shape[-1]
        k = min(k, scores.shape[-1])
        if k < scores.shape[-1]:
            topv, topi = scores.topk(k, dim=-1)                     # [f, N, k]
            attn_soft = torch.nan_to_num(
                F.softmax(topv / self.tau_seclusion, dim=-1), nan=0.0)
            wv_k = (wv.unsqueeze(1)
                    .expand(-1, scores.shape[1], -1, -1)
                    .gather(2, topi.unsqueeze(-1)
                            .expand(-1, -1, -1, wv.shape[-1])))     # [f, N, k, d_v]
            ctx_soft = (attn_soft.unsqueeze(-1) * wv_k).sum(-2)    # [f, N, d_v]
        else:
            attn_soft = torch.nan_to_num(
                F.softmax(scores / self.tau_seclusion, dim=-1), nan=0.0)
            ctx_soft = torch.matmul(attn_soft.unsqueeze(-2),
                                    wv.unsqueeze(1)).squeeze(-2)    # [f, N, d_v]
        return ctx_hard + (ctx_soft - ctx_soft.detach())

    # ----- exact cross-position state scan ---------------------------------
    def _carry_scan(self, a: torch.Tensor, bb: torch.Tensor,
                    s_init: torch.Tensor) -> torch.Tensor:
        """Exact s_t = a_t * s_{t-1} + b_t for t = 0..T-1, s_{-1} = s_init.

        Two-level scan: compose the affine map inside each chunk of C
        positions (C-1 steps over the whole chunk batch), then one short
        sequential pass over the K = T/C chunk carries, then broadcast each
        position's own A_i / B_i against its chunk's incoming state. Total
        sequential depth C-1 + K instead of T, and every step is a tiny
        elementwise op on [B, K, N, d_s].

        No division anywhere (unlike the usual cumprod/cumprod-ratio trick,
        which loses all precision once the running product underflows), and
        the products never span more than C gates.
        """
        b, t, n, d = a.shape
        c = max(1, min(int(self.carry_chunk), t))
        pad = (-t) % c
        if pad:
            a = F.pad(a, (0, 0, 0, 0, 0, pad))
            bb = F.pad(bb, (0, 0, 0, 0, 0, pad))
        t2 = t + pad
        k = t2 // c
        a = a.reshape(b, k, c, n, d)
        bb = bb.reshape(b, k, c, n, d)

        # s_i = A_i * s_in + B_i  (exact affine prefix inside the chunk)
        As = [a[:, :, 0]]
        Bs = [bb[:, :, 0]]
        for i in range(1, c):
            As.append(As[-1] * a[:, :, i])
            Bs.append(Bs[-1] * a[:, :, i] + bb[:, :, i])
        A = torch.stack(As, dim=2)                                  # [b,k,c,n,d]
        Bv = torch.stack(Bs, dim=2)

        # chunk carries: K sequential steps on [b,n,d]
        carry = s_init
        prev = []
        for j in range(k):
            prev.append(carry)
            carry = A[:, j, c - 1] * carry + Bv[:, j, c - 1]
        prev = torch.stack(prev, dim=1)                              # [b,k,n,d]

        s = A * prev.unsqueeze(2) + Bv                               # broadcast
        return s.reshape(b, t2, n, d)[:, :t]

    def initial_state(self, batch_shape) -> torch.Tensor:
        """Broadcast learned s0 to [*batch_shape, N, d_s] for pass 1.

        States reset every position (clarification C1), so pass 1 always
        starts from the learned s0, independent of history.
        """
        s0 = self._sel(0, self.s0)                                  # [N, d_s]
        return s0.expand(*batch_shape, self.N, self.d_s).contiguous()

    # ----- global causal read ----------------------------------------------
    def global_ctx(self, full_k, full_v, round_idx: int, q_add=None,
                   cache: dict | None = None) -> torch.Tensor:
        """Public entry point: honours `break_scan` (eager scan)."""
        if self.break_scan:
            return _global_ctx_eager(self, full_k, full_v, round_idx,
                                     q_add=q_add, cache=cache)
        return self._global_ctx(full_k, full_v, round_idx, q_add=q_add,
                                cache=cache)

    def _global_ctx(self, full_k, full_v, round_idx: int,
                    q_add=None, cache: dict | None = None) -> torch.Tensor:
        """Exact causal attention over the whole prefix at O(T) time.

        A dot's query is static, so score(s) = q_n.k_s is independent of the
        reading position t and full causal attention collapses to a
        normalised prefix sum:

            e_s   = exp(q_n . k_s)
            ctx_t = sum_{s<=t} e_s v_s / sum_{s<=t} e_s

        Two cumsums, no T*T tensor, no per-position GEMM. The cumsum IS the
        causal mask. Returns [B, T, N, d_v].

        Layout note: the value accumulation runs in [B, N, d_v, T] so the
        cumsum walks the CONTIGUOUS last axis; summing over dim=-2 of a
        [B, N, T, d_v] tensor walks a d_v-strided axis and is measurably
        slower for the same numbers.
        """
        if self.gla:
            return self._global_ctx_linear(full_k, full_v, round_idx,
                                           q_add=q_add, cache=cache)
        if self.global_exact:
            return self._global_ctx_exact(full_k, full_v, round_idx,
                                          q_add=q_add)
        b, t, _ = full_v.shape
        n = self.N
        # Keep the whole prefix scan in [B, T, ...] memory order. The obvious
        # [B, N, T] formulation gets there by permuting a contiguous [B,T,N]
        # bmm result, i.e. it hands downstream pointwise ops a (768, 1, 3)
        # strided buffer; torch.compile's Inductor then mis-codegens that
        # buffer on CUDA (size [32,3,256], stride [768,1,3]) and the whole
        # arm crashes at the first step. Scanning along dim=1 is the same
        # arithmetic with no permutes anywhere and returns the layout the
        # caller wants, so it is both faster to trace and no slower to run.
        sc = torch.matmul(full_k, self._sel(round_idx, self.q).t())
        sc = sc / math.sqrt(self.d_q)                                # [B, T, N]
        # Stabilise over TIME (dim=1), exactly as the [B,N,T] version did over
        # its last dim: a per-position constant cancels in the ratio either
        # way, but the time-axis max keeps the cumsum terms O(1) so float32
        # does not lose digits in the prefix sum.
        e = torch.exp(sc - sc.amax(dim=1, keepdim=True))
        if self.g_decay is not None and self.global_decay_k > 1:
            return self._global_ctx_mix(e, full_v, round_idx)
        if self.g_decay is not None:
            # ctx_t = sum_{s<=t} e_s gam^(t-s) v_s / sum_{s<=t} e_s gam^(t-s)
            # is still two prefix sums: weight term s by gam^(-s) before the
            # scan and the common gam^(-t) cancels in the ratio. Two ways to
            # break float32 here and both are real: gam^(-T) > 3e38 overflows
            # (gam < 0.71 at T=256), and folding by gam^((T-1)-s) instead
            # UNDERFLOWS the early terms to zero — at T=256 with gam<0.9 the
            # whole prefix sums to 0 and the read returns zeros. Keep the
            # weights >= 1 (folding by gam^(-s)) and bound the top end by
            # construction with a T-aware floor: gam >= exp(-60/T) means the
            # largest weight is gam^(-(T-1)) <= e^60 = 1e26, which is 12
            # decades inside float32's range and still sharp enough that
            # position 0 is e^-60 of the newest term, i.e. numerically absent.
            # At T=4096 that floor is 0.9855, which is the honest limit: a
            # sharper per-step decay than that has no reachable history.
            gam = torch.sigmoid(self._sel(round_idx, self.g_decay))
            gmin = math.exp(-60.0 / max(t - 1, 1))
            gam = gam.clamp(gmin, 0.99999).view(1, 1, n)
            inv = torch.exp((-torch.arange(t, device=e.device, dtype=e.dtype)
                             ).view(1, t, 1) * torch.log(gam))
            e = e * inv
        cum_e = torch.cumsum(e, dim=1)                               # [B, T, N]
        ev = e.unsqueeze(-1) * full_v.unsqueeze(2)                   # [B, T, N, d_v]
        cum_ev = torch.cumsum(ev, dim=1)                              # [B, T, N, d_v]
        return cum_ev / cum_e.unsqueeze(-1).clamp_min(1e-9)

    def _global_ctx_mix(self, e: torch.Tensor, full_v: torch.Tensor,
                       round_idx: int) -> torch.Tensor:
        """K geometric prefix reads combined by a per-dot softmax.

            ctx_t = S_k a_k * (S_{s<=t} e_s gam_k^(t-s) v_s
                               / S_{s<=t} e_s gam_k^(t-s)),  a = softmax(w)

        One geometric decay says "recency matters at a FIXED rate", but a
        sequence has several at once: the previous token, the current word,
        the current section. The mixture is a convex combination of the
        single-decay reads (the same two cumsums each, K times), so the model
        learns the recency KERNEL's shape per dot instead of one rate. With
        every gam_k equal the mixture is that one read, so K=1 and "K with
        tied gammas" are the same function.
        """
        n = e.shape[2]
        k = self.global_decay_k
        g = self._sel(round_idx, self.g_decay)                    # [N, K]
        mix = F.softmax(g, dim=-1)                                # [N, K]
        gmin = math.exp(-60.0 / max(e.shape[1] - 1, 1))
        gam = torch.sigmoid(g).clamp(gmin, 0.99999)               # [N, K]
        ar = torch.arange(e.shape[1], device=e.device,
                          dtype=e.dtype).view(1, -1, 1, 1)
        e_k = e.unsqueeze(-1) * torch.exp(
            -ar * torch.log(gam.view(1, 1, n, k)))                # [B,T,N,K]
        cum_e = torch.cumsum(e_k, dim=1)                          # [B,T,N,K]
        ev = e_k.unsqueeze(-1) * full_v.unsqueeze(2).unsqueeze(2)
        num = torch.cumsum(ev, dim=1) / cum_e.unsqueeze(-1).clamp_min(1e-9)
        return (num * mix.view(1, 1, n, k, 1)).sum(-2)            # [B,T,N,d_v]

    def _global_ctx_linear(self, full_k, full_v, round_idx: int,
                           q_add=None, cache: dict | None = None) -> torch.Tensor:
        """Content-addressed linear-attention read of the whole prefix.

            S_t = sum_{s<=t} phi(k_s) v_s^T      [B,T,d_q,d_v]
            z_t = sum_{s<=t} phi(k_s)            [B,T,d_q]
            ctx_t = phi(q_t)^T S_t / (phi(q_t)^T z_t)

        with phi(z) = elu(z / d_q^(1/4)) + 1 (positive features, the standard
        linear-attention map; the d_q^(1/4) keeps phi(q).phi(k) on the same
        scale as the softmax path's q.k/sqrt(d_q)).

        S and z do not depend on the query and the keys are shared by every
        dot, so they are computed ONCE per region-forward and cached: the
        per-pass and per-dot work is one [d_q] x [d_q, d_v] contraction
        (~512 MACs/token). Total cost is ~5% of the cell, and no parameter
        anywhere — the query is the existing per-dot q, optionally plus the
        existing content term from w_qi.
        """
        b, t, dq = full_k.shape
        n = self.N
        s = float(self.d_q) ** -0.25
        if cache is not None and "gla_S" in cache:
            S, Z = cache["gla_S"], cache["gla_Z"]
        else:
            phik = F.elu(full_k * s) + 1.0                       # [B,T,d_q]
            S = torch.cumsum(phik.unsqueeze(-1) * full_v.unsqueeze(2), dim=1)
            Z = torch.cumsum(phik, dim=1)                        # [B,T,d_q]
            if cache is not None:
                cache["gla_S"], cache["gla_Z"] = S, Z
        q0 = self._sel(round_idx, self.q)                        # [N,d_q]
        if q_add is not None:
            qv = q0.view(1, 1, n, dq) + q_add.reshape(b, t, 1, dq)
        else:
            qv = q0.view(1, 1, n, dq).expand(b, t, n, dq)
        phiq = F.elu(qv * s) + 1.0
        num = torch.einsum("btnd,btdv->btnv", phiq, S)           # [B,T,N,d_v]
        den = torch.einsum("btnd,btd->btn", phiq, Z).clamp_min(1e-6)
        return num / den.unsqueeze(-1)

    def _global_ctx_exact(self, full_k, full_v, round_idx: int,
                          q_add=None) -> torch.Tensor:
        """True causal softmax attention over the whole prefix, per dot.

        `_global_ctx` (prefix sum) is the same function with a query that
        cannot depend on t; `_global_ctx_linear` is its linear-attention
        approximation. This is the non-degenerate version:

            ctx[b,t,n,:] = sum_{s<=t} softmax_s(q[b,t,n].k[b,s]) v[b,s,:]

        with q[b,t,n] = q0_n (+ q_add[b,t,n] when the caller wants a query
        that depends on the current token). When q_add is None this is
        algebraically identical to the cumsum read, which is why the pair
        makes a free exactness test for the whole family.
        """
        b, t, dq = full_k.shape
        n = self.N
        q0 = self._sel(round_idx, self.q)                        # [N, d_q]
        if q_add is not None:
            qv = q0.view(1, 1, n, dq) + q_add.reshape(b, t, 1, dq)
        else:
            qv = q0.view(1, 1, n, dq).expand(b, t, n, dq)
        sc = torch.einsum("btnd,bsd->btns", qv, full_k)          # [B,T,N,T]
        sc = sc / math.sqrt(self.d_q)
        # causal mask reused across calls (it depends only on T)
        m = self._causal_mask(t, full_k.device, full_k.dtype)
        sc = sc.masked_fill(~m, float("-inf"))
        attn = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
        return torch.einsum("btns,bsv->btnv", attn, full_v)      # [B,T,N,d_v]

    def _causal_mask(self, t: int, device, dtype) -> torch.Tensor:
        cache = getattr(self, "_cmask", None)
        if cache is not None and cache.shape[0] == t and cache.device == device:
            return cache
        i = torch.arange(t, device=device)
        m = (i[:, None] >= i[None, :]).view(1, t, 1, t)
        self._cmask = m
        return m

    # ----- one pass --------------------------------------------------------
    def forward(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                round_idx: int, msg_in, state, emb_table,
                attn_cache: dict | None = None,
                want_log_pi: bool = True,
                slot_table: torch.Tensor | None = None,
                want_e_state: bool = True, full_k=None,
                full_v=None, q_add=None, ctx_add=None) -> DotOutputs:
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
                                      slot_table, want_e_state, full_k,
                                      full_v, q_add, ctx_add)
        return self._forward_reference(wind_k, wind_v, valid_mask, h_pool,
                                       err_feat, round_idx, msg_in, state,
                                       emb_table, want_log_pi, slot_table,
                                       want_e_state, full_k, full_v, q_add,
                                       ctx_add)

    # the einsum-per-op version, kept for the equivalence test and as a
    # reference implementation of spec 01 (slower: einsum re-plans paths the
    # kernel cannot fuse into batched GEMMs)
    def _forward_reference(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                           round_idx: int, msg_in, state, emb_table,
                           want_log_pi: bool = True,
                           slot_table: torch.Tensor | None = None,
                           want_e_state: bool = True, full_k=None,
                           full_v=None, q_add=None, ctx_add=None) -> DotOutputs:
        b, t, w = wind_k.shape[0], wind_k.shape[1], wind_k.shape[2]
        device = wind_k.device

        # ---- queries + reads ---------------------------------------------
        if self.dynamic_query:
            q_dyn = self._sel(round_idx, self.q).unsqueeze(0).unsqueeze(0) \
                + torch.einsum("bfnd,ndj->bfnj", state, self._sel(round_idx, self.u_q))
            scores = torch.einsum("bfwj,bfnj->bfnw", wind_k, q_dyn)     # read by query
        else:
            if q_add is not None:
                qa = (self._sel(round_idx, self.q).view(1, 1, n, self.d_q)
                      + q_add.unsqueeze(2))
                scores = torch.einsum("bfwj,bfnj->bfnw", wind_k, qa)
            else:
                scores = torch.einsum("bfwj,nj->bfnw", wind_k,
                                      self._sel(round_idx, self.q))
        scores = scores / math.sqrt(self.d_q)
        if self.d_bias is not None:
            scores = scores + self._sel(round_idx, self.d_bias).view(1, 1, n, w)
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

        # ---- global causal read (same O(T) prefix-sum as the fast path) ----
        if self.global_read and full_k is not None:
            g_ctx = self.global_ctx(full_k, full_v, round_idx,
                                    q_add=q_add)                     # [B, T, N, d_v]
            g_row = (self.g_glob[round_idx] if self.read_schedule == "split"
                     else self._sel(round_idx, self.g_glob))
            mix = torch.sigmoid(g_row).view(1, 1, n, 1)
            ctx = (1.0 - mix) * ctx + mix * g_ctx

        # exact-identity injection: dot n carries the token at lag n+1 into
        # its value channel (pass-invariant, so it is added after the cache
        # lookup and never written back into the cache).
        if ctx_add is not None:
            ctx = ctx + ctx_add.reshape(b, t, n, self.d_v)

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

        if self.cell == "lug":
            if self.share_cell:
                w_lug = self._sel(round_idx, self.w_ih_lug)[0]
                g2 = (x.reshape(b * t * self.N, self.in_dim).matmul(w_lug)
                      .reshape(b, t, self.N, 2 * self.d_s))
            else:
                g2 = torch.einsum("bfni,niq->bfnq", x,
                                  self._sel(round_idx, self.w_ih_lug))
            g2 = g2 + self._sel(round_idx, self.b_lug)
            iz, ih = g2.chunk(2, dim=-1)
            if self.carry_state:
                # reference: the same recurrence written as the naive
                # per-position loop (slow, obviously correct; used by the
                # equivalence test against _carry_scan).
                z = torch.sigmoid(iz)
                hh = torch.tanh(ih)
                s = state[:, 0]
                outs = []
                for tt in range(t):
                    s = (1.0 - z[:, tt]) * hh[:, tt] + z[:, tt] * s
                    outs.append(s)
                s_new = torch.stack(outs, dim=1)
            else:
                z = torch.sigmoid(
                    iz + self._sel(round_idx, self.u_lug) * state)
                s_new = (1.0 - z) * torch.tanh(ih) + z * state
        else:
            in_gates = torch.einsum("bfni,niq->bfnq", x, self._sel(round_idx, self.w_ih))
            hid_gates = torch.einsum("bfnd,ndq->bfnq", state, self._sel(round_idx, self.w_hh))
            in_gates = in_gates + self._sel(round_idx, self.b_ih)
            iz, ir, inet = in_gates.chunk(3, dim=-1)
            hz, hr, hnet = hid_gates.chunk(3, dim=-1)
            z = torch.sigmoid(iz + hz)
            r = torch.sigmoid(ir + hr)
            n_t = torch.tanh(inet + r * hnet)                           # r-gated candidate
            s_new = (1.0 - z) * n_t + z * state

        # ---- heads --------------------------------------------------------
        gamma = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.c_w)) \
            + self._sel(round_idx, self.c_b)
        trust = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.g_w)) \
            + self._sel(round_idx, self.g_b)
        if not want_log_pi and not want_e_state:
            # lean pass: neither the per-dot vocab head nor the pooled hidden
            # state is needed on this pass — skip p_out entirely.
            return DotOutputs(log_pi=None, gamma=gamma, trust_term=trust,
                              state=s_new, e_state=None)
        if self.share_out:
            e_state = torch.einsum("bfnd,dm->bfnm", s_new,
                                   self._sel(round_idx, self.p_out))
        else:
            e_state = torch.einsum("bfnd,ndm->bfnm", s_new,
                                   self._sel(round_idx, self.p_out))

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
                      slot_table: torch.Tensor | None = None,
                      want_e_state: bool = True, full_k=None,
                      full_v=None, q_add=None, ctx_add=None) -> DotOutputs:
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
        elif (self.q_extra is not None and not self.dynamic_query
              and q_add is None and self.read_mode != "secluded"):
            # multi-head read: H independent slot distributions per dot, each
            # over the SAME window, contracting its own d_v/H value slice.
            # Free in MACs and in context width; only (H-1)*d_q new params.
            q_all = torch.cat([self._sel(round_idx, self.q).unsqueeze(1),
                               self._sel(round_idx, self.q_extra)], dim=1)
            d_vh = self.d_v // self.read_heads
            wv_h = wv.view(f, w, self.read_heads, d_vh)
            vm = None
            if valid_mask is not None:
                vm = valid_mask.unsqueeze(0).expand(b, t, w).reshape(f, 1, w)
            heads = []
            for h in range(self.read_heads):
                sc = torch.tensordot(wk, q_all[:, h], dims=([2], [1]))
                sc = sc.permute(0, 2, 1) / math.sqrt(self.d_q)        # [f, N, w]
                if self.d_bias is not None:
                    sc = sc + self._sel(round_idx, self.d_bias).view(1, n, w)
                if vm is not None:
                    sc = sc.masked_fill(~vm, float("-inf"))
                a = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
                heads.append(torch.bmm(a, wv_h[:, :, h]))             # [f,N,d_vh]
            ctx = torch.cat(heads, dim=-1)                            # [f,N,d_v]
            if cacheable:
                attn_cache["ctx"] = ctx
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
                if q_add is not None:
                    # content-dependent query: q_t = q_n + W x_t. Zero-init
                    # of W, so this starts exactly as the static-query model
                    # and can only move away from it if it pays.
                    qa = (self._sel(round_idx, self.q).view(1, n, self.d_q)
                          + q_add.reshape(f, 1, self.d_q))
                    scores = torch.einsum("fwj,fnj->fnw", wk, qa)       # [f, N, w]
                else:
                    # one GEMM: [f*w, d_q] @ [d_q, N] -> [f, w, N]
                    scores = torch.tensordot(wk, self._sel(round_idx, self.q),
                                             dims=([2], [1]))
                    scores = scores.permute(0, 2, 1)                    # [f, N, w]
            scores = scores / math.sqrt(self.d_q)
            if self.d_bias is not None:
                scores = scores + self._sel(round_idx, self.d_bias).view(1, n, w)
            if valid_mask is not None:
                vm = valid_mask.unsqueeze(0).expand(b, t, w).reshape(f, 1, w)
                scores = scores.masked_fill(~vm, float("-inf"))         # [f, N, w]
            if self.read_mode == "secluded":
                ctx = self._secluded_ctx(scores, wv)                    # [f, N, d_v]
            else:
                attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)  # [f, N, w]
                # ONE bmm over f groups of [N, w] @ [w, d_v]. The obvious
                # attn.unsqueeze(-2) @ wv.unsqueeze(1) form makes torch
                # broadcast it into f*N separate [1, w] @ [w, d_v] GEMMs
                # (2048 of them at B=8/T=64/N=4) — measured as the single
                # largest bmm in the step profile.
                ctx = torch.bmm(attn, wv)                              # [f, N, d_v]
            if cacheable:
                attn_cache["ctx"] = ctx

        # ---- global causal read (O(T), see __init__) ----------------------
        if self.global_read and full_k is not None:
            # The read itself is PASS-INVARIANT (static query, shared params),
            # so cache it exactly like the window ctx: recomputing the two
            # cumsums on every pass was costing ~8% of a P2 step for numbers
            # that were already known. Only the cheap per-pass g_glob mix is
            # redone below (passes may carry their own g_glob row).
            g_cacheable = attn_cache is not None and not self.dynamic_query
            if g_cacheable and "g_ctx" in attn_cache:
                g_ctx = attn_cache["g_ctx"]
            else:
                g_ctx = self.global_ctx(full_k, full_v, round_idx,
                                        q_add=q_add,
                                        cache=attn_cache if g_cacheable else None)
                if g_cacheable:
                    attn_cache["g_ctx"] = g_ctx
            if self.read_schedule == "split":
                g_row = self.g_glob[round_idx]          # per-pass row, unclamped
            else:
                g_row = self._sel(round_idx, self.g_glob)
            mix = torch.sigmoid(g_row).view(1, n, 1)
            ctx = (1.0 - mix) * ctx + mix * g_ctx.reshape(f, n, self.d_v)

        # exact-identity injection: dot n carries the token at lag n+1 into
        # its value channel (pass-invariant, so it is added after the cache
        # lookup and never written back into the cache).
        if ctx_add is not None:
            ctx = ctx + ctx_add.reshape(f, n, self.d_v)

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
        if self.in_ln is not None:
            x = self.in_ln(x)

        s_f = state.reshape(f, n, self.d_s)
        if self.cell == "lug":
            # Lean Update Gate: ONE input GEMM of width 2*h, diagonal
            # recurrence (u * s), no hidden GEMM.
            if self.share_cell:
                # one GEMM over merged rows: [f*N, in] @ [in, 2h]
                w_lug = self._sel(round_idx, self.w_ih_lug)[0]      # [in, 2h]
                g2 = (x.reshape(f * n, self.in_dim).matmul(w_lug)
                      .reshape(f, n, 2 * self.d_s))
                g2 = g2 + self._sel(round_idx, self.b_lug).unsqueeze(0)
            else:
                g2 = torch.bmm(x.transpose(0, 1),
                               self._sel(round_idx, self.w_ih_lug))  # [N, f, 2h]
                g2 = (g2.transpose(0, 1)
                      + self._sel(round_idx, self.b_lug).unsqueeze(0))
            iz, ih = g2.chunk(2, dim=-1)
            if self.carry_state:
                # SRU-form gate (no state term: it is what makes the scan
                # exact) + cross-position carry.
                z = torch.sigmoid(iz).reshape(b, t, n, self.d_s)
                hh = torch.tanh(ih).reshape(b, t, n, self.d_s)
                s_new = self._carry_scan(
                    z, (1.0 - z) * hh, state.reshape(b, t, n, self.d_s)[:, 0]
                ).reshape(f, n, self.d_s)
            else:
                z = torch.sigmoid(iz + self._sel(round_idx, self.u_lug).unsqueeze(0) * s_f)
                s_new = (1.0 - z) * torch.tanh(ih) + z * s_f                # [f, N, d_s]
        else:
            w_ih_b = self._sel(round_idx, self.w_ih)                    # [N, in, 3h]
            in_gates = torch.bmm(x.transpose(0, 1), w_ih_b)             # [N, f, 3h]
            hid_gates = torch.bmm(s_f.transpose(0, 1),                  # [N, f, d_s]
                                  self._sel(round_idx, self.w_hh))      # [N, d_s, 3h]
            in_gates = (in_gates.transpose(0, 1)                        # [f, N, 3h]
                        + self._sel(round_idx, self.b_ih).unsqueeze(0))
            h3 = hid_gates.transpose(0, 1)                              # [f, N, 3h] (view)
            iz, ir, inet = in_gates.chunk(3, dim=-1)
            hz, hr, hnet = h3.chunk(3, dim=-1)
            z = torch.sigmoid(iz + hz)
            r = torch.sigmoid(ir + hr)
            n_t = torch.tanh(inet + r * hnet)
            s_new = (1.0 - z) * n_t + z * s_f                           # [f, N, d_s]

        # ---- heads --------------------------------------------------------
        # speed opt 6 (Tolu's "don't move data around" hint): with
        # head_layout="shared_gemm" the tied-head GEMM writes the [f, N, V]
        # layout directly — the legacy path wrote [N, f, V] and then paid a
        # FULL permute+reshape copy of the [B, T, N, V] log-probs (~134 MB in
        # bf16 at the bench shape) on every region-pass, forward AND
        # backward. The small [., ., d_s]-scale transpose stays; the big
        # tensor is only ever written once, by the GEMM, in final layout.
        # Same math; only GEMM reassociation differs (last bits).
        gamma = torch.einsum("fnd,nd->fn", s_new, self._sel(round_idx, self.c_w)) \
            + self._sel(round_idx, self.c_b).unsqueeze(0)
        trust = torch.einsum("fnd,nd->fn", s_new, self._sel(round_idx, self.g_w)) \
            + self._sel(round_idx, self.g_b).unsqueeze(0)
        if not want_log_pi and not want_e_state:
            # lean pass: the region only needs gamma/trust (weighting) and
            # the state — skip the p_out projection entirely.
            return DotOutputs(log_pi=None, gamma=gamma.reshape(b, t, n),
                              trust_term=trust.reshape(b, t, n),
                              state=s_new.reshape(b, t, n, self.d_s),
                              e_state=None)
        # [N, f, d_s] @ ([N, d_s, d] or [d_s, d]) -> [N, f, d]
        e_state = torch.matmul(s_new.transpose(0, 1),
                               self._sel(round_idx, self.p_out))
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
