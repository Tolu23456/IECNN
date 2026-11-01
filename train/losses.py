"""All loss terms (spec 06, section 2). Soft everything; label-free weighting."""
from __future__ import annotations

import torch
import torch.nn.functional as F


def main_loss(final_log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """CE of the final pooled distribution. Inputs [B, T, V] and [B, T]."""
    return F.nll_loss(final_log_probs.reshape(-1, final_log_probs.size(-1)),
                      targets.reshape(-1))


def region_loss(region_log_probs: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
    """Sum of CE of each region's own consensus. [R, B, T, V] or [B, T, V]."""
    if region_log_probs.dim() == 3:
        region_log_probs = region_log_probs[None]
    return sum(
        F.nll_loss(rp.reshape(-1, rp.size(-1)), targets.reshape(-1))
        for rp in region_log_probs
    )


def boosted_dot_loss(dot_log_probs: torch.Tensor, final_log_probs: torch.Tensor,
                     targets: torch.Tensor, gamma_b: float) -> torch.Tensor:
    """Per-dot CE weighted by (1 - consensus prob of truth)^gamma_b, weight detached.

    dot_log_probs [B, T, N, V]; final_log_probs [B, T, V]; targets [B, T].
    This is the training-only way of rewarding dots that are right when the
    consensus is wrong (file 04, section 8).
    """
    with torch.no_grad():
        cons_prob = final_log_probs.exp().gather(-1, targets[..., None]).squeeze(-1)
        boost = (1.0 - cons_prob).clamp(min=0.0) ** gamma_b      # [B, T]
    dot_ce = -dot_log_probs.gather(-1, targets[..., None, None].expand(
        *targets.shape, dot_log_probs.size(-2), 1)).squeeze(-1)   # [B, T, N]
    weighted = dot_ce * boost[..., None]
    if not torch.isfinite(weighted).all():
        raise FloatingPointError("boosted_dot_loss produced non-finite values")
    return weighted.mean()


def decorrelation_penalty(dot_log_probs: torch.Tensor, final_log_probs: torch.Tensor,
                          targets: torch.Tensor) -> torch.Tensor:
    """Weighted pairwise error correlation, positive part, only where crowd wrong.

    Implements spec file 06 section 2.4: e_k = -log p_k(y), weights
    nu_t = 1 - b(y_t) (detached); rho_ij = weighted corr; penalty mean_i<j relu(rho_ij)
    computed within each region's population (dot axis N is the population).
    """
    with torch.no_grad():
        cons_prob = final_log_probs.exp().gather(-1, targets[..., None]).squeeze(-1)
        nu = (1.0 - cons_prob)                                   # [B, T]
    err = -dot_log_probs.gather(-1, targets[..., None, None].expand(
        *targets.shape, dot_log_probs.size(-2), 1)).squeeze(-1)   # [B, T, N]
    dots = dot_log_probs.size(-2)
    penalty = err.new_zeros(())
    denom = nu.sum()
    if denom.item() <= 0:
        return penalty
    mean_k = (err * nu[..., None]).sum(dim=(0, 1)) / denom       # [N]
    centered = err - mean_k
    cov_ij = (centered * (centered * nu[..., None])).sum(dim=(0, 1)) / denom
    var = cov_ij.diagonal().clamp(min=1e-12)
    std = var.sqrt()
    rho = cov_ij / (std[None, :] * std[:, :] + 1e-12)
    mask = torch.triu(torch.ones(dots, dots), diagonal=1).bool()
    penalty = torch.relu(rho[mask]).mean()
    return penalty


def confidence_loss(gamma: torch.Tensor, dot_log_probs: torch.Tensor,
                    targets: torch.Tensor) -> torch.Tensor:
    """BCE between sigmoid(gamma) and whether the dot's argmax was right.

    gamma [B, T, N]; dot_log_probs [B, T, N, V]; targets [B, T].
    """
    correct = (dot_log_probs.argmax(-1) == targets[..., None]).float()
    return F.binary_cross_entropy_with_logits(gamma, correct)


def weight_entropy_bonus(weights: torch.Tensor) -> torch.Tensor:
    """Negative mean entropy of weights (early-training anti-collapse).

    weights [..., N] sums to 1 on the last axis; returns mean over batch of
    H(w) — the caller applies the minus sign per schedule (L_w = -lambda * H).
    """
    ent = -(weights * (weights + 1e-12).log()).sum(dim=-1)
    return ent.mean()
