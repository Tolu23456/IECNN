"""Weights, pooling, and pruning (spec 04). All label-free (clarification C2)."""
from __future__ import annotations


def leave_one_out_mean(pi):
    """Mean of the other dots' distributions, per dot: (sum - pi_k) / (N - 1)."""
    raise NotImplementedError


def js_divergence(p, q):
    """Jensen-Shannon divergence, bounded, computed stably with epsilons."""
    raise NotImplementedError


def error_features(pi, loo_mean):
    """Three numbers per dot: JS to crowd, crowd prob of own argmax, own max prob."""
    raise NotImplementedError


def compute_scores(pi, gamma, trust_term, alpha, beta):
    """u_k = alpha * (-JS(pi_k, loo_k)) + beta * gamma_k + trust_term_k."""
    raise NotImplementedError


def update_trust(prev_trust, scores, rho):
    """T^p = rho * T^{p-1} + u^p."""
    raise NotImplementedError


def compute_weights(trust, tau, floor=0.0):
    """softmax(trust / tau) over dots, with optional floor, renormalized."""
    raise NotImplementedError


def pool_log_linear(log_pi, weights):
    """log b = sum_k w_k log pi_k, renormalized with log-softmax."""
    raise NotImplementedError


def pool_mixture(pi, weights):
    """Alternative pooling: sum_k w_k pi_k."""
    raise NotImplementedError


def hard_top_m(weights, m):
    """Keep top-m dots per region, renormalize (inference only)."""
    raise NotImplementedError


def plurality_agreement(pi, consensus):
    """Diagnostic only: fraction of dots whose argmax equals the consensus argmax."""
    raise NotImplementedError


def temperature_schedule(step, total_steps, warmup, tau_max, tau_min):
    """Cosine from tau_max to tau_min after warmup (spec 04, section 6)."""
    raise NotImplementedError
