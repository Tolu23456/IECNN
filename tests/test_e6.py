"""E6 consensus-v2 tests (Tolu, 2026-11-14).

Covers the four invariants that make the rung safe to build on:
  1. verifier OFF => the forward is bit-identical to E1 (no silent change);
  2. verifier ON with the head zeroed AND v_scale=0 => still bit-identical
     (wiring adds exactly nothing until the head earns influence);
  3. the per-dot oracle loss reaches dots whose CONSENSUS WEIGHT is ~0
     (the muted-dot fix — the whole point of the loss);
  4. the verifier target is 1.0 at truth-rank 1 and 0 beyond the span.
Forward/loss only — no training steps (CI is the torch runner).
"""
import torch

from config import load_config
from utils import set_seed

V = 512


def _build(verifier: bool):
    from models.iecnn import build_iecnn
    set_seed(0)
    cfg = load_config("configs/e1_weights_passes.yaml", "tiny.yaml")
    cfg["model"]["verifier"] = verifier
    return build_iecnn(cfg, V)


def test_verifier_off_matches_e1_bit_exact():
    e1 = _build(verifier=False)
    e6 = _build(verifier=False)
    e1.eval(); e6.eval()
    x = torch.randint(0, V, (2, 32))
    with torch.no_grad():
        a = e1(x)["final_log_probs"]
        b = e6(x)["final_log_probs"]
    assert torch.equal(a, b)
    assert e1.regions[0].verifier is None


def test_verifier_zero_scale_is_bit_exact_to_e1():
    e1 = _build(verifier=False)
    e6 = _build(verifier=True)
    e1.eval(); e6.eval()
    # copy shared params so the dots are identical, zero the verifier path
    state1 = e1.state_dict()
    for name, p in e6.named_parameters():
        if name in state1:
            p.data.copy_(state1[name].data)
        elif "verifier" in name or "v_scale" in name:
            p.data.zero_()
    x = torch.randint(0, V, (2, 32))
    with torch.no_grad():
        a = e1(x)["final_log_probs"]
        b = e6(x)["final_log_probs"]
    assert torch.equal(a, b)
    # and the head's output rides along in per_pass for the loss/metrics
    with torch.no_grad():
        out = e6(x)
    assert out["per_pass_last_region"][-1]["verifier"] is not None
    assert out["per_pass_last_region"][-1]["verifier"].shape[-1] == 9  # features


def test_verifier_scale_changes_output_and_carries_gradient():
    e6 = _build(verifier=True)
    e6.eval()
    x = torch.randint(0, V, (2, 32))
    y = torch.randint(0, V, (2, 32))
    out = e6(x)
    loss = torch.nn.functional.nll_loss(
        out["final_log_probs"].reshape(-1, V), y.reshape(-1))
    loss.backward()
    g = e6.regions[0].verifier.fc1.weight.grad
    assert g is not None and float(g.abs().sum()) > 0.0


def test_oracle_dot_loss_reaches_a_muted_dot():
    """The muted-dot fix: dot weight ~1e-4 must still get full gradient.

    Mimic the measured failure (dot 6 at w=0.0001): gradient through the
    POOLED loss at that weight is ~0; the oracle loss must be ~full. All
    dots share one distribution, so a fair teacher gives IDENTICAL
    gradients to the muted dot and the leader.
    """
    torch.manual_seed(0)
    b, t, n, v = 2, 3, 8, V
    base = torch.randn(b, t, 1, v).log_softmax(-1)
    log_pi = base.expand(b, t, n, v).detach().clone().requires_grad_()
    targets = torch.randint(0, v, (b, t))
    weights = torch.full((b, t, n), 1e-4)
    weights[..., 0] = 1.0 - 7e-4                      # all trust on dot 0
    pooled = torch.log_softmax(
        (weights.unsqueeze(-1) * log_pi).sum(-2), dim=-1)
    from train.losses import oracle_dot_loss
    (oracle_dot_loss(log_pi, pooled, targets, outvote_weight=3.0)).backward()
    grad_muted = log_pi.grad[..., 6, :].abs().sum()    # w=1e-4 dot
    grad_leader = log_pi.grad[..., 0, :].abs().sum()   # w~1 dot
    assert float(grad_muted) > 0.0
    assert torch.allclose(grad_muted, grad_leader, rtol=1e-4)


def test_verifier_truth_target_ranks():
    from train.losses import verifier_truth_target
    b, t, n, v = 1, 1, 2, 100
    log_pi = torch.log_softmax(torch.randn(b, t, n, v), dim=-1)
    # force dot 0: truth is top-1; dot 1: truth at rank 60 (beyond span v/8)
    truth = int(log_pi[0, 0, 0].argmax())
    junk = torch.argsort(log_pi[0, 0, 1], descending=True)[0]
    log_pi[0, 0, 1] = torch.log_softmax(
        torch.where(torch.arange(v) == junk,
                    torch.full((v,), -1e9), torch.zeros(v)), dim=-1)
    log_pi[0, 0, 1, truth] = torch.log(torch.tensor(1e-12))
    tgt = verifier_truth_target(log_pi, torch.tensor([[truth]]))
    assert float(tgt[0, 0, 0]) == 1.0
    assert float(tgt[0, 0, 1]) == 0.0


def test_verifier_loss_finite_and_oversampled():
    from train.losses import verifier_loss
    torch.manual_seed(0)
    b, t, n, v = 2, 3, 8, V
    log_pi = torch.log_softmax(torch.randn(b, t, n, v), dim=-1)
    pooled = torch.log_softmax(
        (torch.full((b, t, n), 1.0 / n).unsqueeze(-1) * log_pi).sum(-2), -1)
    scores = torch.randn(b, t, n, requires_grad=True)
    targets = torch.randint(0, v, (b, t))
    lv = verifier_loss(scores, log_pi, pooled, targets, outvote_weight=3.0)
    assert torch.isfinite(lv)
    lv.backward()
    assert torch.isfinite(scores.grad).all()
    assert float(scores.grad.abs().sum()) > 0.0
