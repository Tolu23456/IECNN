"""A population of dots with separate weights (spec 01).

All N dots of a region are stored with a leading dot dimension N and computed
with batched matrix operations. The GRU cell is written by hand, because
built-in GRUs cannot hold separate weights per dot.

Per pass p, per dot k, per position t (b, f=batch shape B,T):
    q     = q_k                       (dynamic: q_k + U_k s_prev)   [d_q]
    K_t   = windowed memory keys (shared, built by Region)          [w, d_q]
    a     = softmax(K_t q / sqrt(d_q), causal window mask applied)  [w]
    c     = a V_t                                                   [d_v]
    u     = [c ; h_pool ; e ; r_p ; m_in]                          [d_in]
    s'    = GRU_k(u, s)                                             [d_s]
    pi,gamma,trust from s' through the tied embedding E [V, d].
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import torch
import torch.nn as nn
import torch.nn.functional as F

ROUND_DIM = 16          # learned pass-index embedding (spec 01)
ERR_DIM = 3             # error feature e (spec 01, section 5)


def gru_input_dim(d_v: int, d_s: int, message_dim: int = 0) -> int:
    """d_in = value + pooled opinion + error feature + round signal + message."""
    return d_v + d_s + ERR_DIM + ROUND_DIM + message_dim


@dataclass
class DotOutputs:
    log_pi: torch.Tensor        # [B, T, N, V]
    gamma: torch.Tensor         # [B, T, N]
    trust_term: torch.Tensor    # [B, T, N]
    state: torch.Tensor         # [B, T, N, d_s]


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
                 seed: int = 0):
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
        self.in_dim = gru_input_dim(d_v, d_s, message_dim)

        p_layers = 1 if share_pass_params else n_passes   # stacked per-pass depth
        g = torch.Generator().manual_seed(seed)

        self.q = nn.Parameter(torch.stack([_orth_cols(d_q, n_dots, g) for _ in range(p_layers)]))
        self.u_q = nn.Parameter(torch.stack([_xavier(d_s, d_q, g) for _ in range(p_layers)])) \
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

    # ----- helpers ---------------------------------------------------------
    def _sel(self, pass_idx: int, t: torch.Tensor) -> torch.Tensor:
        """[p_layers, N, ...] -> current pass's slice (or layer 0 when shared)."""
        return t[0 if self.shared else pass_idx]

    def initial_state(self, batch_shape) -> torch.Tensor:
        """Broadcast learned s0 to [*batch_shape, N, d_s] for pass 1.

        States reset every position (clarification C1), so pass 1 always
        starts from the learned s0, independent of history.
        """
        s0 = self._sel(0, self.s0)                                  # [N, d_s]
        return s0.expand(*batch_shape, self.N, self.d_s).contiguous()

    # ----- one pass --------------------------------------------------------
    def forward(self, wind_k, wind_v, valid_mask, h_pool, err_feat,
                round_idx: int, msg_in, state, emb_table) -> DotOutputs:
        """One pass for all dots at all positions simultaneously.

        wind_k    [B, T, w, d_q]   windowed, rel-distance-added keys
        wind_v    [B, T, w, d_v]   windowed values
        valid_mask [T, w] bool     True = row exists (t - j < w and j >= 0)
        h_pool    [B, T, N, d_s]   previous pass's pooled opinion (0 at pass 1)
        err_feat  [B, T, N, 3]     error feature (0 at pass 1)
        msg_in    [B, T, N, d_m]   or None
        state     [B, T, N, d_s]   s_prev
        emb_table [V, d]           tied embedding, shared by the whole model
        """
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
        attn = torch.nan_to_num(F.softmax(scores, dim=-1), nan=0.0)    # [B, T, N, w]
        ctx = torch.matmul(attn.unsqueeze(-2), wind_v.unsqueeze(-3)).squeeze(-2)  # [B, T, N, d_v]

        # ---- GRU step (all dots at once) ----------------------------------
        parts = [ctx, h_pool, err_feat,
                 self.round_emb.weight[round_idx].expand(b, t, self.N, ROUND_DIM)]
        if self.d_m > 0:
            parts.append(msg_in if msg_in is not None else
                         torch.zeros(b, t, self.N, self.d_m, device=device, dtype=ctx.dtype))
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
        logits = torch.einsum("bfnm,vm->bfnv", e_state, emb_table)
        log_pi = F.log_softmax(logits, dim=-1)
        gamma = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.c_w)) \
            + self._sel(round_idx, self.c_b)
        trust = torch.einsum("bfnd,nd->bfn", s_new, self._sel(round_idx, self.g_w)) \
            + self._sel(round_idx, self.g_b)
        return DotOutputs(log_pi=log_pi, gamma=gamma, trust_term=trust, state=s_new)
