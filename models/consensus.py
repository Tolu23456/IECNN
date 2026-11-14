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


def js_divergence(p: torch.Tensor, q: torch.Tensor,
                  log_p: torch.Tensor | None = None) -> torch.Tensor:
    """Jensen-Shannon divergence JS(p, q), bounded [0, ln 2], stable.

    Accepts probability tensors [..., V]; broadcasting supported. `log_p` may
    carry the caller's already-computed log(p) (e.g. a dot's log distribution)
    so the expensive log over [.., N, V] is not recomputed.
    """
    eps = _EPS
    p_ = p.clamp_min(eps)
    q_ = q.clamp_min(eps)
    m = 0.5 * (p_ + q_)
    logm = m.log()
    kl_pm = (p * ((p_.log() if log_p is None else log_p) - logm)).sum(-1)
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
                   return_parts: bool = False,
                   log_pi: torch.Tensor | None = None):
    """u_k = alpha * (-JS(pi_k, loo_k)) + beta * gamma_k + gate_k (spec 04, section 3).

    Also returns (loo_mean, js) so the caller can reuse them for the error
    feature of the NEXT pass instead of recomputing two leaves over [., N, V].
    `log_pi` (optional) saves one log over [., N, V] inside the JS.
    """
    loo = leave_one_out_mean(pi)
    js = js_divergence(pi, loo, log_p=log_pi if log_pi is not None else None)
    scores = alpha * (-js) + beta * gamma + trust_term
    if return_parts:
        return scores, loo, js
    return scores


def js_from_parts_fast(pi: torch.Tensor, log_pi: torch.Tensor,
                       loo: torch.Tensor) -> torch.Tensor:
    """JS(p, q) with q = loo via the entropy identity JS = H(m) - 0.5(H(p)+H(q)).

    Same math as `js_divergence(pi, loo)` for the caller's situation (q IS the
    leave-one-out mean and p = exp(log_pi) is already materialized), but it
    reuses log_pi for H(p) instead of recomputing log(p) over [., N, V], and
    needs no clamped copies of p. Differences vs the reference `js_divergence`
    are last-bit floating point: the eps clamp only affected p values below
    1e-9, whose contribution to JS is ~1e-12 (recorded in the speed-goal note,
    DECISIONS.md 2026-11-14). Pinned by test_js_fast_matches_reference.
    """
    n = pi.shape[-2]
    if n <= 1:
        return torch.zeros_like(pi[..., 0])
    logq = loo.clamp_min(_EPS).log()
    m = 0.5 * (pi + loo)
    logm = m.clamp_min(_EPS).log()
    hm = -(m * logm).sum(-1)
    hp = -(pi * log_pi).sum(-1)
    hq = -(loo * logq).sum(-1)
    return hm - 0.5 * (hp + hq)


def compute_scores_fast(pi: torch.Tensor, log_pi: torch.Tensor,
                        gamma: torch.Tensor, trust_term: torch.Tensor,
                        alpha: float | torch.Tensor, beta: float | torch.Tensor,
                        return_parts: bool = False):
    """Optimized `compute_scores` for the dots path where pi = exp(log_pi).

    Same math (u_k = alpha*(-JS) + beta*gamma + gate_k), two changes, both
    last-bit only: (1) JS via `js_from_parts_fast` (reuses log_pi), (2) the
    leave-one-out mean divides in place on a fresh tensor. Returns
    (scores, loo, js) when return_parts, mirroring compute_scores.
    """
    n = pi.shape[-2]
    total = pi.sum(dim=-2, keepdim=True)                   # [..., 1, V]
    if n <= 1:
        loo = total.expand_as(pi).clone() * 0.0 + 1.0 / max(n, 1)
    else:
        loo = (total - pi) / (n - 1)
    js = js_from_parts_fast(pi, log_pi, loo)
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
    Implemented as a batched matmul ([..., 1, N] @ [..., N, V]) so the
    weighted sum never materializes a [..., N, V] product tensor (the old
    (w.unsqueeze(-1) * log_pi).sum(-2) did — one extra full-size temp per
    pass per region). Same math; GEMM reassociation is last-bit only.
    """
    agg = torch.matmul(weights.unsqueeze(-2), log_pi).squeeze(-2)
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


def consensus_core(log_pi: torch.Tensor, gamma: torch.Tensor,
                   trust_term: torch.Tensor, trust_prev: torch.Tensor,
                   alpha: torch.Tensor, beta: torch.Tensor,
                   v_scale: torch.Tensor, v_out: torch.Tensor,
                   rho: float, tau_t: torch.Tensor, floor_t: torch.Tensor,
                   n_dots: int, pi_in: torch.Tensor | None = None):
    """The whole per-pass [., N, V] consensus chain as ONE function — the
    torch.compile fusion target (speed goal round 2, 2026-11-14).

    Eager PyTorch launches ~10 separate elementwise kernels over [f, N, V]
    here (exp, sum, sub, div, 4x JS entropy chain, gather, max, matmul
    pool); each reads+writes hundreds of MB and the step is bandwidth-bound
    under bf16. Inductor fuses the pointwise chain into 1-2 kernels, cutting
    memory traffic several-fold.

    Same math as the eager inline sequence in Region._pass_consensus
    (compute_scores_fast -> update_trust -> compute_weights -> pooling),
    with two mechanical differences, both caller-controlled:
      * tau and floor arrive as 0-dim TENSORS (per-step annealed floats
        would otherwise be dynamo guards -> recompile every step);
      * the verifier's contribution is folded in as v_scale * v_out (callers
        pass a zero v_out / v_scale when no verifier — bit-exact: x + 0 = x);
      * trust_prev is a real tensor (zeros on pass 1: rho*0 + u == u
        bit-exactly, same as update_trust's first-pass clone);
      * pi_in: callers that already materialized pi = exp(log_pi) (the E6
        verifier features need it) pass it here to skip a second exp; with
        pi_in=None the core computes pi itself (non-verifier rungs). Both
        variants specialize under dynamo.

    Returns (trust, w, log_b, err) where err is error_features_from_parts'
    [.., N, 3] stack. Not used unless training.compile_consensus is on; the
    eager path stays the reference (tests pin equivalence).
    """
    pi = log_pi.exp() if pi_in is None else pi_in
    total = pi.sum(dim=-2, keepdim=True)                       # [..., 1, V]
    loo = (total - pi) / max(n_dots - 1, 1)                    # [..., N, V]
    js = js_from_parts_fast(pi, log_pi, loo)
    scores = alpha * (-js) + beta * gamma + trust_term + v_scale * v_out
    trust = rho * trust_prev + scores
    t = trust.clamp(-20.0, 20.0)
    w = F.softmax(t / tau_t.clamp_min(_EPS), dim=-1)
    w_floor = w.clamp_min(floor_t)
    w = torch.where(floor_t > 0.0, w_floor / w_floor.sum(-1, keepdim=True).clamp_min(1e-9), w)
    own_top = pi.argmax(dim=-1)
    crowd_on_own_top = loo.gather(-1, own_top.unsqueeze(-1)).squeeze(-1)
    peak = pi.max(dim=-1).values
    err = torch.stack([js, crowd_on_own_top, peak], dim=-1)
    log_b = pool_log_linear(log_pi, w)
    return trust, w, log_b, err
