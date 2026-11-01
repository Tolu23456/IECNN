"""Optional within-region messages (spec 05)."""
from __future__ import annotations


def ring_neighbors(n_dots: int, radius: int):
    """For each dot, the indices of its neighbors on a ring, never including itself."""
    raise NotImplementedError


class MessageChannel:
    """write: m_k = tanh(W_m s_k). read: attention over neighbors. One-pass delay."""

    def __init__(self, d_s, d_m, n_dots, radius, dropout, noise_std=0.0):
        raise NotImplementedError

    def forward(self, states):
        """Return incoming messages [..., N, d_m] to use at the NEXT pass."""
        raise NotImplementedError
