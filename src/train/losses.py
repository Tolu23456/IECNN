"""All loss terms (spec 06, section 2)."""
from __future__ import annotations


def main_loss(final_log_probs, targets):
    """CE of the final pooled distribution."""
    raise NotImplementedError


def region_loss(region_log_probs, targets):
    """Sum of CE of each region's own consensus."""
    raise NotImplementedError


def boosted_dot_loss(dot_log_probs, final_log_probs, targets, gamma_b):
    """Per-dot CE weighted by (1 - consensus prob of truth)^gamma_b, weight detached."""
    raise NotImplementedError


def decorrelation_penalty(dot_log_probs, final_log_probs, targets):
    """Weighted pairwise error correlation, positive part, only where the crowd was wrong."""
    raise NotImplementedError


def confidence_loss(gamma, dot_log_probs, targets):
    """BCE between sigmoid(gamma) and whether the dot's argmax was right."""
    raise NotImplementedError


def weight_entropy_bonus(weights):
    """Negative mean entropy of weights (early-training anti-collapse)."""
    raise NotImplementedError
