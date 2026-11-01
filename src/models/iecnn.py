"""The full IECNN model: regions stacked, region competition, final output (spec 03)."""
from __future__ import annotations


class IECNN:
    """(Becomes a torch.nn.Module when built.)

    Config switches: n_regions, n_passes, use_weights, use_cross_region,
    use_messages, pooling, share_pass_params, dynamic_query, halting.
    Rung E0 must be exactly a plain uniform ensemble.
    """

    def __init__(self, config: dict, vocab_size: int):
        raise NotImplementedError

    def forward(self, input_ids, tau0=1.0, hard_mode=False):
        """Return a dict: final log distribution, per-region consensus,
        region weights, per-dot log distributions, per-pass logs."""
        raise NotImplementedError
