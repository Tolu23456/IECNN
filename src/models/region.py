"""One region: N dots, P passes, weights, consensus, summary (spec 03)."""
from __future__ import annotations


class Region:
    """(Becomes a torch.nn.Module when built.)

    Runs P passes with shared dot parameters. Returns the final consensus log
    distribution, the per-position summary h = sum_k w_k s_k, the final weights,
    and per-pass logs used by diagnostics.
    """

    def __init__(self, region_index, config, vocab_size):
        raise NotImplementedError

    def forward(self, memory_rows, mask, tau0, hard_mode=False):
        raise NotImplementedError
