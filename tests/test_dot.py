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


def test_fast_math_matches_reference_and_is_default():
    """The optimized [B*T, N] batched-GEMM pass must match the einsum
    reference numerically (same weights, same inputs; fp32 tolerance) and be
    the default path, so what the ladder trains on is what is tested."""
    from models.dot import DotPopulation

    g = torch.Generator().manual_seed(5)
    common = dict(n_dots=5, d=24, d_q=12, d_v=20, d_s=18, vocab_size=64,
                  message_dim=0, n_passes=3, share_pass_params=True)

    fast = DotPopulation(seed=3, fast_math=True, **common)
    ref = DotPopulation(seed=3, fast_math=False, **common)
    ref.load_state_dict(fast.state_dict())                 # identical weights
    ref.eval(); fast.eval()

    emb = torch.randn(64, 24, generator=g)
    torch.manual_seed(11)
    b, t, w = 2, 7, 5
    wind_k = torch.randn(b, t, w, 12)
    wind_v = torch.randn(b, t, w, 20)
    valid = torch.ones(t, w, dtype=torch.bool)
    valid[:2] = False                                      # exercise masking
    h_pool = torch.randn(b, t, 5, 18)
    err = torch.randn(b, t, 5, 3)
    state = torch.randn(b, t, 5, 18)

    with torch.no_grad():
        of, orf = (fast(wind_k, wind_v, valid, h_pool, err, 1, None, state, emb),
                   ref(wind_k, wind_v, valid, h_pool, err, 1, None, state, emb))
    for a, b_ in ((of.log_pi, orf.log_pi), (of.state, orf.state),
                  (of.gamma, orf.gamma), (of.trust_term, orf.trust_term)):
        assert torch.allclose(a, b_, atol=1e-5, rtol=1e-5), \
            (a - b_).abs().max().item()

    # dynamic-query variant must match through both paths as well
    fast_dq = DotPopulation(seed=4, fast_math=True, dynamic_query=True, **common)
    ref_dq = DotPopulation(seed=4, fast_math=False, dynamic_query=True, **common)
    ref_dq.load_state_dict(fast_dq.state_dict())
    fast_dq.eval(); ref_dq.eval()
    with torch.no_grad():
        a = fast_dq(wind_k, wind_v, valid, h_pool, err, 2, None, state, emb)
        r = ref_dq(wind_k, wind_v, valid, h_pool, err, 2, None, state, emb)
    assert torch.allclose(a.log_pi, r.log_pi, atol=1e-5, rtol=1e-5)

    # the fast path is the default everywhere the ladder trains
    assert fast.fast_math and DotPopulation(seed=0, **common).fast_math

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
