"""Optional within-region dot communication (spec 05).

Write at pass p, read at pass p+1 (one-pass delay). `use_messages: false`
degenerates exactly to the no-message model, which test_switch_off checks.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def ring_neighbors(n_dots: int, radius: int) -> torch.Tensor:
    """Boolean [N, N]: row k is True at columns of k's ring neighbours.

    Cyclical ring distance min(|i-j|, N-|i-j|); only that, radius or less,
    and never k itself. With N=1 the ring is empty (all-False rows).
    """
    if n_dots <= 1:
        return torch.zeros(n_dots, n_dots, dtype=torch.bool)
    idx = torch.arange(n_dots)
    d = (idx.unsqueeze(1) - idx.unsqueeze(0)).abs()
    dist = torch.minimum(d, n_dots - d)
    return (dist > 0) & (dist <= radius)


class MessageChannel(nn.Module):
    """write: m_k = tanh(W_m s_k); read: attention over ring neighbours."""

    def __init__(self, d_s: int, d_m: int, n_dots: int, radius: int,
                 dropout: float = 0.0, noise_std: float = 0.0):
        super().__init__()
        self.d_m = d_m
        self.n_dots = n_dots
        self.dropout = dropout
        self.noise_std = noise_std
        self.w_write = nn.Linear(d_s, d_m, bias=False)
        self.wq = nn.Linear(d_m, d_m, bias=False)
        self.wk = nn.Linear(d_m, d_m, bias=False)
        self.register_buffer("neigh", ring_neighbors(n_dots, radius), persistent=False)

    def read_mask(self) -> torch.Tensor:
        """[N, N] boolean allowed-neighbour matrix (for tests)."""
        return self.neigh

    def forward(self, states: torch.Tensor, training: bool | None = None) -> torch.Tensor:
        """states [..., N, d_s] -> messages IN for the NEXT pass [..., N, d_m].

        Dot k never reads its own message (ring mask has a False diagonal).
        """
        if training is None:
            training = self.training
        m = torch.tanh(self.w_write(states))                    # [..., N, d_m]
        q = self.wq(m)                                          # reader side
        k = self.wk(m)                                          # writer side, shared keys
        scores = torch.einsum("...nd,...jd->...nj", q, k) / math.sqrt(self.d_m)
        blocked = ~self.neigh                                   # [..., N, N] False = forbidden
        scores = scores.masked_fill(blocked, float("-inf"))
        a = F.softmax(scores, dim=-1)                           # zero weight on self (all -inf row -> nan-safe)
        a = torch.nan_to_num(a, nan=0.0)
        m_in = torch.einsum("...nj,...jd->...nd", a, k)
        if training and self.dropout > 0.0:
            m_in = F.dropout(m_in, p=self.dropout, training=True)
        if training and self.noise_std > 0.0:
            m_in = m_in + self.noise_std * torch.randn_like(m_in)
        return m_in
