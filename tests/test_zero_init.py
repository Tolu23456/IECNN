"""Tolu (2026-11-13): "a test that E1-zero-init output equals E0 at step 0."

The invariant: with the learned-weight pathway at zero init (trust scores
identically zero), softmax gives uniform weights, so E1's consensus pooling
MUST collapse to exactly E0's uniform-ensemble pooling on the same dots.
Forward passes only — no optimizer steps (no training in tests here either).
"""
import math

import torch

from config import load_config
from utils import set_seed


def _build(use_weights: bool):
    """E1 architecture with ONE pass and log-linear pooling; the only
    difference between the two builds is use_weights (E0's switch)."""
    from models.iecnn import build_iecnn
    set_seed(0)
    cfg = load_config("configs/e1_weights_passes.yaml", "tiny.yaml")
    cfg["model"]["n_passes"] = 1
    cfg["model"]["pooling"] = "log_linear"      # same pooling on both sides
    cfg["model"]["use_weights"] = use_weights
    return build_iecnn(cfg, 512)


def test_e1_zero_init_output_equals_e0_at_step0():
    e0 = _build(use_weights=False)
    e1 = _build(use_weights=True)
    e0.eval(); e1.eval()

    # make the dots identical: copy every shared parameter from the E0 build,
    # and ZERO every E1-only parameter (the learned-weight score pathway).
    e0_state = e0.state_dict()
    for name, p in e1.named_parameters():
        if name in e0_state:
            p.data.copy_(e0_state[name].data)
        else:
            p.data.zero_()          # score pathway -> trust == 0 -> uniform w

    x = torch.randint(0, 512, (2, 40))
    with torch.no_grad():
        out0 = e0(x)
        out1 = e1(x, tau0=1.0, floor=0.0)

    # weights are exactly uniform (softmax of zeros) -> pooling collapses
    pp = out1["per_pass_last_region"][0]
    n = e1.regions[0].n_dots
    assert torch.allclose(pp["weights"],
                          torch.full_like(pp["weights"], 1.0 / n), atol=1e-6)
    # and the outputs match E0's plain-ensemble output exactly
    diff = (out1["final_log_probs"] - out0["final_log_probs"]).abs().max().item()
    assert diff < 1e-5, f"E1(zero-init) vs E0 mismatch: {diff}"


def test_grad_norms_by_component_buckets():
    """The per-component gradient logger returns finite positive norms and a
    total consistent with its buckets (one backward, NO optimizer step)."""
    from train.train import grad_norms_by_component
    from models.iecnn import build_iecnn

    set_seed(0)
    cfg = load_config("configs/e1_weights_passes.yaml", "tiny.yaml")
    model = build_iecnn(cfg, 512)
    x = torch.randint(0, 512, (2, 32))
    y = torch.randint(0, 512, (2, 32))
    out = model(x)
    torch.nn.functional.nll_loss(out["final_log_probs"].reshape(-1, 512),
                                 y.reshape(-1)).backward()
    gn = grad_norms_by_component(model)
    assert gn, "no gradient buckets produced"
    for b, v in gn.items():
        assert math.isfinite(v) and v > 0.0, (b, v)
    total_sq = sum(v * v for v in gn.values())
    ref = torch.nn.utils.clip_grad_norm_(model.parameters(), 1e9).item()
    assert abs(total_sq - ref * ref) / (ref * ref) < 1e-4, (total_sq, ref)
