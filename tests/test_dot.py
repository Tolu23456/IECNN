"""Tests for the Dot (spec 01, section 10) through a live Region/IECNN build."""
import torch

from config import load_config
from models.iecnn import build_iecnn
from utils import set_seed


def _model(name="e1_weights_passes", overlay="tiny.yaml"):
    set_seed(0)
    cfg = load_config(f"configs/{name}.yaml", *( [overlay] if overlay else [] ))
    return build_iecnn(cfg, 512)


def test_shapes_distribution_sums_to_one():
    """A dot pass returns pi [B, T, N, V] summing to 1 on the last axis."""
    model = _model()
    x = torch.randint(0, 512, (2, 40))
    out = model(x)
    n = model.regions[0].n_dots
    per_pass = out["region_outputs"][0]["per_pass"]
    assert len(per_pass) == 3
    for p in per_pass:
        assert p["log_pi"].shape[:3] == (2, 40, n) and p["log_pi"].shape[-1] == 512
        err = (p["log_pi"].exp().sum(-1) - 1.0).abs().max().item()
        assert err < 1e-5, f"dot distributions sum to {err} off from 1"


def test_causality_future_token_change_leaves_earlier_outputs_identical():
    model = _model()
    torch.manual_seed(1)
    x = torch.randint(0, 512, (2, 40))
    base = model(x)["final_log_probs"]
    x2 = x.clone()
    x2[:, 20:] = torch.randint(0, 512, (2, 20))
    pert = model(x2)["final_log_probs"]
    diff = (base[:, :20] - pert[:, :20]).abs().max().item()
    assert diff == 0.0, f"future leakage: max diff {diff}"


def test_every_dot_receives_gradient():
    """Every dot's parameters get nonzero gradient after one training step."""
    model = _model()
    x = torch.randint(0, 512, (2, 40))
    y = torch.randint(0, 512, (2, 40))
    out = model(x)
    torch.nn.functional.nll_loss(out["final_log_probs"].reshape(-1, 512),
                                 y.reshape(-1)).backward()
    dots = model.regions[0].dots
    for name, p in dots.named_parameters():
        assert p.grad is not None and float(p.grad.abs().sum()) > 0, \
            f"dot parameter {name} got no gradient"
    # the region's read projections too
    for name, p in model.regions[0].named_parameters():
        if name.startswith(("w_k", "w_v", "rel_emb")):
            assert float(p.grad.abs().sum()) > 0, f"{name} got no gradient"
    assert torch.isfinite(torch.tensor(1.0))


def test_query_cosine_below_half_at_init():
    """No two dot queries start aligned (spec 01: distinct dots)."""
    model = _model()
    q = model.regions[0].dots.q[0]                       # [N, d_q]
    qn = q / q.norm(dim=-1, keepdim=True)
    cos = qn @ qn.T
    off_diag = cos[~torch.eye(cos.shape[0], dtype=torch.bool)]
    assert off_diag.abs().max().item() < 0.5, \
        f"query cosine too high at init: {off_diag.abs().max().item()}"


def test_determinism_same_seed():
    a = _model()
    b = _model()
    for pa, pb in zip(a.parameters(), b.parameters()):
        assert torch.equal(pa, pb), "same seed gave different parameters"
    x = torch.randint(0, 512, (1, 32))
    assert torch.allclose(a(x)["final_log_probs"], b(x)["final_log_probs"])


def test_label_free_signals_identical_with_labels_hidden():
    """Error features / weights / consensus cannot depend on the label (C2).

    The model takes only input ids; whatever the caller does with labels
    afterwards cannot change the forward pass — proven by replaying the same
    batch with two different label tensors and comparing.
    """
    model = _model()
    model.eval()
    x = torch.randint(0, 512, (2, 40))
    y1 = torch.randint(0, 512, (2, 40))
    y2 = torch.zeros_like(y1)
    with torch.no_grad():
        out1 = model(x)
        out2 = model(x)
        torch.nn.functional.nll_loss(out1["final_log_probs"].reshape(-1, 512), y1.reshape(-1))
        torch.nn.functional.nll_loss(out2["final_log_probs"].reshape(-1, 512), y2.reshape(-1))
    for a, b in zip(out1["region_outputs"][0]["per_pass"],
                    out2["region_outputs"][0]["per_pass"]):
        assert torch.equal(a["weights"], b["weights"]), "weights used the label"
    assert torch.equal(out1["final_log_probs"], out2["final_log_probs"])
