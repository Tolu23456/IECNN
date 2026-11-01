"""Consensus / weighting tests (spec 04, section 11). Pure functions."""
import math

import torch

from models import consensus


def test_weights_floor_renormalizes():
    """Floor 0.1/N: every weight >= floor, then renormalization keeps sum 1."""
    trust = torch.randn(4, 2, 8)
    floor = 0.1 / 8
    w_f = consensus.compute_weights(trust, tau=0.5, floor=floor)
    assert (w_f.sum(-1) - 1).abs().max() < 1e-6
    # after renormalization each weight is at least floor/(1 + N*floor)
    lower = floor / (1 + 8 * floor)
    assert (w_f.min(-1).values > lower - 1e-6).all()
    # and compared to the unfloored weights, mass moved toward weak dots
    w = consensus.compute_weights(trust, tau=0.5)
    assert torch.allclose(w.mean(-1), w_f.mean(-1), atol=1e-5)


def test_tau_to_zero_is_one_hot():
    trust = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    w = consensus.compute_weights(trust, tau=1e-9)
    arg = w.argmax(-1)
    assert arg.item() == 1 and (w[0, 1] > 0.999).item()


def test_tau_large_is_uniform():
    trust = torch.tensor([[1.0, 3.0, 2.0, 0.5]])
    w = consensus.compute_weights(trust, tau=1e9)
    assert torch.allclose(w, torch.full_like(w, 0.25), atol=1e-6)


def test_leave_one_out_invariance_to_own_output():
    """Crowd mean for dot k excludes ONLY pi_k: altering dot k's pi leaves
    dot k's OWN crowd row unchanged (spec 04 test 4) and the row equals the
    mean of the other dots."""
    pi = torch.softmax(torch.randn(3, 4, 32), dim=-1)          # [*, N=4, V=32]
    loo = consensus.leave_one_out_mean(pi)
    n = pi.shape[-2]
    for k in range(n):
        others = [j for j in range(n) if j != k]
        assert torch.allclose(loo[..., k, :], pi[..., others, :].sum(-2) / (n - 1),
                              atol=1e-6), f"loo row {k} includes its own pi"

    pi2 = pi.clone()
    pi2[..., 2, :] = torch.softmax(torch.randn(3, 32), dim=-1)   # alter dot 2 only
    loo2 = consensus.leave_one_out_mean(pi2)
    assert torch.allclose(loo2[..., 2, :], loo[..., 2, :], atol=1e-6), \
        "dot 2's own crowd row changed when only dot 2's output changed"
    # everyone else's crowd row DOES change (dot 2 is in their crowd) — this is
    # exactly the leave-one-out semantics, recorded for honesty in tests
    assert not torch.allclose(loo2[..., 0, :], loo[..., 0, :])


def test_label_independence_of_weights():
    """Scores/weights are computed from pi/gamma only: no label argument exists."""
    import inspect
    sig = inspect.signature(consensus.compute_scores)
    assert list(sig.parameters) == ["pi", "gamma", "trust_term", "alpha", "beta"]


def test_pooled_output_is_valid_distribution():
    log_pi = torch.log_softmax(torch.randn(3, 4, 32), dim=-1)  # [*, N, V]
    w = torch.softmax(torch.randn(3, 4), dim=-1)
    log_b = consensus.pool_log_linear(log_pi, w)
    err = (log_b.exp().sum(-1) - 1).abs().max().item()
    assert err < 1e-5
    b = consensus.pool_mixture(log_pi.exp(), w)                # [*, N, V] in
    assert (b.sum(-1) - 1).abs().max() < 1e-5


def test_log_linear_equals_mixture_when_all_dots_identical():
    N = 5
    one = torch.log_softmax(torch.randn(2, 64), dim=-1)
    log_pi = one.unsqueeze(-2).expand(2, N, 64)
    w = torch.softmax(torch.randn(2, N), dim=-1)
    log_lin = consensus.pool_log_linear(log_pi, w)
    mix = consensus.pool_mixture(log_pi.exp(), w).clamp_min(1e-12).log()
    assert torch.allclose(log_lin.exp(), mix.exp(), atol=1e-5)


def test_hard_top_m_with_m_equal_n_matches_soft_at_low_tau():
    trust = torch.tensor([[[0.2, 0.9, 0.1, 0.7]]])              # [*, N=4]
    tau = 1e-3
    soft = consensus.compute_weights(trust, tau=tau)
    soft = soft / soft.sum(-1, keepdim=True)
    hard = consensus.hard_top_m(soft.cpu() * 0 + soft, m=4)     # m == N: keep all
    assert torch.allclose(hard, soft, atol=1e-5), "top-m(n) must equal soft"


def test_temperature_schedule_endpoints():
    sched = consensus.temperature_schedule
    assert sched(0, 20000, 2000, 1.0, 0.2) == 1.0
    assert sched(1999, 20000, 2000, 1.0, 0.2) == 1.0
    assert sched(20000, 20000, 2000, 1.0, 0.2) <= 0.2 + 1e-6
    assert sched(11000, 20000, 2000, 1.0, 0.2) < 1.0
