"""Loss-term tests (spec 06 section 2). These were skip-stubs until the E6
pilot crashed on the first step inside `decorrelation_penalty` (2026-11-14):
a loss that had never been turned on in any rung had never been executed.
Lesson recorded — every loss the configs can enable needs a real test."""
import math

import torch

from config import load_config


def _synthetic(n: int = 8, v: int = 512, b: int = 2, t: int = 5, seed: int = 0):
    torch.manual_seed(seed)
    dot_logits = torch.randn(b, t, n, v)
    dot_log_probs = dot_logits.log_softmax(-1)
    pooled = torch.log_softmax(
        (torch.full((b, t, n), 1.0 / n).unsqueeze(-1) * dot_log_probs).sum(-2),
        dim=-1)
    targets = torch.randint(0, v, (b, t))
    gamma = torch.randn(b, t, n)
    weights = torch.full((b, t, n), 1.0 / n)
    region_stack = torch.stack([pooled, pooled.roll(1, -1)])
    return dot_log_probs, pooled, targets, gamma, weights, region_stack


def test_all_terms_finite_and_sensible_scale_at_step_zero():
    from train.losses import (main_loss, region_loss, boosted_dot_loss,
                              decorrelation_penalty, confidence_loss,
                              weight_entropy_bonus, oracle_dot_loss,
                              verifier_loss)
    log_pi, pooled, targets, gamma, weights, region_stack = _synthetic()
    v = log_pi.size(-1)
    assert math.isfinite(float(main_loss(pooled, targets)))
    assert math.isfinite(float(region_loss(region_stack, targets)))
    ld = boosted_dot_loss(log_pi, pooled, targets, gamma_b=2.0)
    lc = decorrelation_penalty(log_pi, pooled, targets)
    lk = confidence_loss(gamma, log_pi, targets)
    lw = weight_entropy_bonus(weights)
    lo = oracle_dot_loss(log_pi, pooled, targets, outvote_weight=3.0)
    lv = verifier_loss(gamma, log_pi, pooled, targets, outvote_weight=3.0)
    for name, val, lo_b, hi_b in [("boosted", ld, 0.0, 50.0),
                                  ("decorr", lc, 0.0, 1.0),
                                  ("conf", lk, 0.0, 10.0),
                                  ("w_ent", lw, 0.0, 10.0),
                                  ("oracle", lo, 0.0, 50.0),
                                  ("verifier", lv, 0.0, 10.0)]:
        assert math.isfinite(float(val)), name
        assert lo_b <= float(val) <= hi_b, (name, float(val))


def test_correlation_penalty_zero_when_identical_dots_and_crowd_right():
    """Crowd fully right on every position: nu_t = 0 everywhere -> no signal."""
    from train.losses import decorrelation_penalty
    torch.manual_seed(0)
    b, t, n, v = 1, 4, 8, 64
    log_pi = torch.randn(b, t, n, v).log_softmax(-1)
    targets = torch.randint(0, v, (b, t))
    # crowd one-hot on the truth (probability 1 -> nu = 0)
    pooled = torch.full((b, t, v), -30.0)
    pooled.scatter_(-1, targets[..., None], torch.zeros(b, t, 1))
    assert float(decorrelation_penalty(log_pi, pooled, targets)) == 0.0


def test_correlation_penalty_positive_when_errors_correlated():
    """Dots that fail together (identical errors) must be penalized."""
    from train.losses import decorrelation_penalty
    torch.manual_seed(0)
    b, t, n, v = 1, 6, 4, 64
    targets = torch.randint(0, v, (b, t))
    # two identical error groups: dots (0,1) agree, dots (2,3) agree
    base = torch.randn(b, t, 1, v).log_softmax(-1)
    log_pi = torch.cat([base, base, base.roll(1, -1), base.roll(2, -1)], dim=2)
    pooled = torch.log_softmax(torch.randn(b, t, v), dim=-1)
    penalty = float(decorrelation_penalty(log_pi, pooled, targets))
    assert 0.0 < penalty <= 1.0, penalty


def test_correlation_penalty_gradient_flows():
    """The penalty must be differentiable w.r.t. the dot log-probs (use the
    deterministic correlated-groups construction so relu is positive)."""
    from train.losses import decorrelation_penalty
    torch.manual_seed(0)
    b, t, n, v = 1, 6, 4, 64
    base = torch.randn(b, t, 1, v, requires_grad=True)
    base_pi = base.log_softmax(-1)
    log_pi = torch.cat([base_pi, base_pi, base_pi.roll(1, -1),
                        base_pi.roll(2, -1)], dim=2)
    pooled = torch.log_softmax(torch.randn(b, t, v), dim=-1)
    targets = torch.randint(0, v, (b, t))
    decorrelation_penalty(log_pi, pooled, targets).backward()
    assert base.grad is not None
    assert torch.isfinite(base.grad).all()
    assert float(base.grad.abs().sum()) > 0.0


def test_one_train_step_with_all_e6_losses_on():
    """Full E6 training step on a tiny config — the smoke test that would
    have caught the decorrelation crash before it reached the runner."""
    from train.train import build_model
    from train.losses import (main_loss, boosted_dot_loss,
                              decorrelation_penalty, confidence_loss,
                              weight_entropy_bonus, oracle_dot_loss,
                              verifier_loss)
    torch.manual_seed(0)
    cfg = load_config("configs/e6_consensus_v2_pilot.yaml", "tiny.yaml")
    model = build_model(cfg, 64)
    x = torch.randint(0, 64, (2, 16))
    y = torch.randint(0, 64, (2, 16))
    out = model(x, tau0=1.0, floor=0.1 / 8, hard_mode=False)
    loss = main_loss(out["final_log_probs"], y)
    last = out["per_pass_last_region"][-1]
    loss = loss + 0.1 * boosted_dot_loss(last["log_pi"],
                                         out["final_log_probs"], y, 2.0)
    loss = loss + 0.1 * decorrelation_penalty(last["log_pi"],
                                              out["final_log_probs"], y)
    loss = loss + 0.05 * confidence_loss(last["gamma"], last["log_pi"], y)
    loss = loss + 0.2 * oracle_dot_loss(last["log_pi"],
                                        out["final_log_probs"], y, 3.0)
    loss = loss + 0.5 * verifier_loss(last["verifier"], last["log_pi"],
                                      out["final_log_probs"], y, 3.0)
    loss = loss - 0.05 * weight_entropy_bonus(last["weights"])
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads, "no gradients reached any parameter"
    assert all(torch.isfinite(g).all() for g in grads)
    verifier_grads = [p.grad for n_, p in model.named_parameters()
                      if "verifier" in n_ or "v_scale" in n_]
    assert verifier_grads and any(float(g.abs().sum()) > 0 for g in verifier_grads)
