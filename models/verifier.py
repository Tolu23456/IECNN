"""Verifier head (E6 consensus-v2): oracle-supervised dot picker.

One small MLP per region. For every dot at every position it reads the
label-free per-dot features that already exist inside the consensus loop
(gamma, previous weight, agreement, state norm, error features, pass index,
region entropy) and outputs a trust score. Two uses:

  * at training time the score is supervised by the ORACLE — where the true
    token ranked inside that dot's distribution (rank 1 = target 1.0,
    decaying to 0 beyond the target span). This is the "learned weight that
    picks the right dots" idea (Tolu, 2026-11-14): the answer is free during
    training, so we use it to teach the picker which feature patterns
    predict a dot being right.
  * the (scaled) score is added to the consensus scores u_k before the
    softmax, so the learned trust flows into pooling at inference — the
    oracle itself never runs at inference.

Design notes:
  * when `model.verifier` is false nothing here is constructed and the
    region's forward is bit-identical to E1 (checked by tests).
  * v_scale is initialized small (0.05) so E6 starts near the E1 consensus
    and the verifier earns influence instead of grabbing it.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn

N_FEATURES = 8


class VerifierHead(nn.Module):
    """[B, T, N, F] -> [B, T, N, 1] trust score per dot."""

    def __init__(self, n_features: int = N_FEATURES, hidden: int = 32,
                 seed: int = 0):
        super().__init__()
        self.fc1 = nn.Linear(n_features, hidden)
        self.fc2 = nn.Linear(hidden, 1)
        # deterministic init (same discipline as DotPopulation): a fixed
        # generator per region keeps builds reproducible across machines.
        g = torch.Generator().manual_seed(seed)
        with torch.no_grad():
            bound1 = math.sqrt(6.0 / (n_features + hidden))
            self.fc1.weight.uniform_(-bound1, bound1, generator=g)
            self.fc1.bias.zero_()
            bound2 = math.sqrt(6.0 / (hidden + 1)) * 0.1   # small final layer:
            self.fc2.weight.uniform_(-bound2, bound2, generator=g)
            self.fc2.bias.zero_()

    def forward(self, feats: torch.Tensor) -> torch.Tensor:
        """feats [..., N, F] -> [..., N, 1]."""
        return self.fc2(torch.tanh(self.fc1(feats)))


def verifier_features(gamma: torch.Tensor, w_prev: torch.Tensor,
                      agree: torch.Tensor, state_norm: torch.Tensor,
                      err: torch.Tensor, region_entropy: torch.Tensor,
                      pass_frac: torch.Tensor) -> torch.Tensor:
    """Stack the per-dot feature tensor [..., N, 8].

    gamma [.., N]; w_prev [.., N]; agree [.., N]; state_norm [.., N];
    err [.., N, 3]; region_entropy [..] broadcast over N; pass_frac scalar
    tensor broadcast everywhere. All features are O(1)-ish scale so the MLP
    needs no normalization layer.
    """
    parts = [gamma, w_prev, agree, state_norm,
             err[..., 0], err[..., 1], err[..., 2]]
    n = gamma.shape[-1]
    parts.append(region_entropy.unsqueeze(-1).expand(*region_entropy.shape, n)
                 if region_entropy.dim() >= 1 else
                 region_entropy.expand(gamma.shape))
    parts.append(pass_frac.expand(gamma.shape)
                 if pass_frac.dim() == 0 else pass_frac)
    return torch.stack(parts, dim=-1)
