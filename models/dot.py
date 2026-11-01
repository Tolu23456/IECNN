"""A population of dots with separate weights (spec 01).

All N dots of a region are stored with a leading dot dimension [N, in, out]
and computed with batched matrix operations. The GRU cell is written by hand,
because built-in GRUs cannot hold separate weights per dot.
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass
class DotOutputs:
    log_pi: object      # [..., N, V] log distributions
    gamma: object       # [..., N] confidence logits
    trust_term: object  # [..., N] learned gate contribution
    state: object       # [..., N, d_s] new states


class DotPopulation:
    """N dots in one region. (Becomes a torch.nn.Module when built.)

    Per dot: query q_k, initial state s0_k, GRU weights, output projection P_k,
    confidence head, trust-gate head. Shared in the region: key/value
    projections, relative embedding, round-signal embeddings. Shared in the
    whole model: tied embedding matrix E [V, d].
    """

    def __init__(self, n_dots, d, d_q, d_v, d_s, vocab_size, message_dim=0,
                 n_passes=3, dynamic_query=False):
        raise NotImplementedError

    def initial_state(self, batch_shape):
        """State at pass 1 (learned s0_k), state resets every position (clarification C1)."""
        raise NotImplementedError

    def forward(self, keys, values, mask, h_pool, err_feat, round_idx, msg_in, state) -> DotOutputs:
        """One pass for all dots. GRU input = [context; h_pool; err_feat; round signal; msg_in]."""
        raise NotImplementedError
