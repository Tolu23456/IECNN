"""Weights, pooling, and pruning (spec 04). All label-free (clarification C2).

Pure tensor functions: no modules, no parameters. Shapes everywhere use a
leading dot dimension N in second-to-last position: [..., N, V] for
distributions, [..., N] for scalars, [..., N, d_s] for states.
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

_EPS = 1e-9


def leave_one_out_mean(pi: torch.Tensor) -> torch.Tensor:
    """Mean of the other dots' distributions, per dot: (sum - pi_k) / (N - 1).

    Spec 04 section 2. Shape [..., N, V] -> [..., N, V]. The row for dot k
    does not depend on pi_k (leave-one-out), verified by tests.
    """
    n = pi.shape[-2]
    total = pi.sum(dim=-2, keepdim=True)          # [..., 1, V]
    if n <= 1:
        return total.expand_as(pi).clone() * 0.0 + 1.0 / max(n, 1)
    return (total - pi) / (n - 1)


def js_divergence(p: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
    """Jensen-Shannon divergence JS(p, q), bounded [0, ln 2], stable.

    Accepts probability tensors [..., V]; broadcasting supported.
    """
    eps = _EPS
    p_ = p.clamp_min(eps)
    q_ = q.clamp_min(eps)
    m = 0.5 * (p_ + q_)
    logm = m.log()
    kl_pm = (p * (p_.log() - logm)).sum(-1)
    kl_qm = (q * (q_.log() - logm)).sum(-1)
    return 0.5 * kl_pm + 0.5 * kl_qm


def error_features(pi: torch.Tensor, loo_mean: torch.Tensor) -> torch.Tensor:
    """The three label-free numbers per dot (spec 01, section 5):

    e = [ JS(pi_k, m_-k), m_-k[argmax pi_k], max pi_k ]  shape [..., N, 3]
    """
    js = js_divergence(pi, loo_mean)                       # [..., N]
    own_top = pi.argmax(dim=-1)                        # [..., N]
    crowd_on_own_top = loo_mean.gather(-1, own_top.unsqueeze(-1)).squeeze(-1)
    peak = pi.max(dim=-1).values
    return torch.stack([js, crowd_on_own_top, peak], dim=-1)


def compute_scores(pi: torch.Tensor, gamma: torch.Tensor, trust_term: torch.Tensor,
                   alpha: float | torch.Tensor, beta: float | torch.Tensor,
                   return_parts: bool = False):
    """u_k = alpha * (-JS(pi_k, loo_k)) + beta * gamma_k + gate_k (spec 04, section 3).

    Also returns (loo_mean, js) so the caller can reuse them for the error
    feature of the NEXT pass instead of recomputing two leaves over [., N, V].
    """
    loo = leave_one_out_mean(pi)
    js = js_divergence(pi, loo)
    scores = alpha * (-js) + beta * gamma + trust_term
    if return_parts:
        return scores, loo, js
    return scores


def error_features_from_parts(pi: torch.Tensor, loo_mean: torch.Tensor,
                              js: torch.Tensor) -> torch.Tensor:
    """Same as `error_features` but reusing already-computed loo/JS tensors."""
    own_top = pi.argmax(dim=-1)
    crowd_on_own_top = loo_mean.gather(-1, own_top.unsqueeze(-1)).squeeze(-1)
    peak = pi.max(dim=-1).values
    return torch.stack([js, crowd_on_own_top, peak], dim=-1)


def update_trust(prev_trust: torch.Tensor | None, scores: torch.Tensor, rho: float) -> torch.Tensor:
    """T^p = rho * T^(p-1) + u^p ; T^0 = 0 (spec 04, section 3)."""
    if prev_trust is None:
        return scores.clone()           # first pass: trust starts at the score itself
    return rho * prev_trust + scores


def compute_weights(trust: torch.Tensor, tau: float | torch.Tensor, floor: float = 0.0) -> torch.Tensor:
    """softmax(trust / tau) over dots, optional floor then renormalize.

    Scores are clamped to [-20, 20] before softmax (spec 04, section 9).
    """
    t = trust.clamp(-20.0, 20.0)
    w = F.softmax(t / max(float(tau), _EPS), dim=-1)
    if floor > 0.0:
        w = w.clamp_min(floor)
        w = w / w.sum(-1, keepdim=True)
    return w


def pool_log_linear(log_pi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """log b(v) proportional to sum_k w_k log pi_k(v), renormalized (spec 04, section 4).

    log_pi [..., N, V], weights [..., N] -> log_b [..., V].
    """
    agg = (weights.unsqueeze(-1) * log_pi).sum(-2)
    return F.log_softmax(agg, dim=-1)


def pool_mixture(pi: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    """Alternative pooling: b = sum_k w_k pi_k (spec 04, section 4)."""
    b = (weights.unsqueeze(-1) * pi).sum(-2)
    return b / b.sum(-1, keepdim=True).clamp_min(_EPS)


def hard_top_m(weights: torch.Tensor, m: int) -> torch.Tensor:
    """Keep the top-m dots, zero the rest, renormalize (inference only).

    Ties break naturally by top-k order; spec asks confidence-then-index, and
    callers pass final weights which include the gate term.
    """
    n = weights.shape[-1]
    m = max(1, min(int(m), n))
    topk = weights.topk(m, dim=-1).indices            # [..., m]
    hard = torch.zeros_like(weights)
    hard.scatter_(-1, topk, weights.gather(-1, topk))
    return hard / hard.sum(-1, keepdim=True).clamp_min(_EPS)


def plurality_agreement(pi: torch.Tensor, consensus_pi: torch.Tensor) -> torch.Tensor:
    """Diagnostic: fraction of dots whose argmax matches the consensus argmax."""
    dot_top = pi.argmax(-1)                            # [..., N]
    cons_top = consensus_pi.argmax(-1).unsqueeze(-1)   # [..., 1]
    return (dot_top == cons_top).float().mean(-1)


def uniform_weights(n: int, like: torch.Tensor) -> torch.Tensor:
    """Uniform weights over n dots (rung E0): [..., n].

    `like` is any [..., N, V]-style tensor; its leading dims (batch, time)
    are preserved and the two last dims are replaced by the dot axis.
    """
    shape = list(like.shape[:-2]) + [n]
    return torch.full(shape, 1.0 / n, dtype=like.dtype, device=like.device)


def temperature_schedule(step: int, total_steps: int, warmup: int,
                         tau_max: float, tau_min: float) -> float:
    """tau_0 cosine anneal from tau_max to tau_min after warmup (spec 04, section 6)."""
    if step < warmup:
        return tau_max
    frac = (step - warmup) / max(total_steps - warmup, 1)
    frac = min(max(frac, 0.0), 1.0)
    return tau_min + 0.5 * (tau_max - tau_min) * (1.0 + math.cos(math.pi * frac))


def entropy(logits: torch.Tensor) -> torch.Tensor:
    """Shannon entropy H(p) of a log-prob tensor in nats: [..., V] -> [...]."""
    p = logits.exp()
    return -(p * logits).sum(-1)
