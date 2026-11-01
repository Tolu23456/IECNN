"""Region and pass tests (spec 03, section 10)."""
import torch

from config import load_config
from models.iecnn import build_iecnn
from utils import set_seed


def _model(name="e1_weights_passes", overlay="tiny.yaml"):
    set_seed(0)
    cfg = load_config(f"configs/{name}.yaml", *( [overlay] if overlay else [] ))
    return build_iecnn(cfg, 512)


def test_region_weights_sum_to_one():
    model = _model()
    x = torch.randint(0, 512, (2, 32))
    out = model(x)
    for r, ro in enumerate(out["region_outputs"]):
        err = (ro["weights"].sum(-1) - 1).abs().max().item()
        assert err < 1e-5, f"region {r} weights sum off by {err}"


def test_e0_config_reproduces_uniform_ensemble_exactly():
    """E0 (P=1, R=1, uniform weights, mixture pooling) IS the plain average
    of its dots' distributions, exactly."""
    model = _model("e0_plain_ensemble")
    model.eval()
    x = torch.randint(0, 512, (2, 32))
    with torch.no_grad():
        out = model(x)
        log_pi = out["region_outputs"][0]["per_pass"][0]["log_pi"]   # [B,T,N,V]
        plain_mean = log_pi.exp().mean(-2)
        plain_mean = plain_mean / plain_mean.sum(-1, keepdim=True)
    assert torch.allclose(out["final_log_probs"], plain_mean.log(), atol=1e-6), \
        "E0 is not the plain uniform ensemble"


def test_zeroing_round_signal_changes_outputs():
    """The pass-index embedding must matter (otherwise passes are identical)."""
    model = _model("e1_weights_passes")
    model.eval()
    x = torch.randint(0, 512, (1, 24))
    with torch.no_grad():
        base = model(x)["final_log_probs"]
    saved = model.regions[0].dots.round_emb.weight.clone()
    with torch.no_grad():
        model.regions[0].dots.round_emb.weight.zero_()
        pert = model(x)["final_log_probs"]
    model.regions[0].dots.round_emb.weight.data = saved
    assert not torch.allclose(base, pert), "round signal had no effect"


def test_shared_vs_unshared_param_count_differs_by_P():
    """Unsharing the pass parameters multiplies the per-pass dot weights by P."""
    m_shared = _model("e1_weights_passes")
    set_seed(0)
    import copy
    cfg = load_config("configs/e1_weights_passes.yaml", "tiny.yaml")
    cfg = copy.deepcopy(cfg)
    cfg["model"]["share_pass_params"] = False
    m_unshared = build_iecnn(cfg, 512)
    dot_shared = m_shared.regions[0].dots
    dot_unshared = m_unshared.regions[0].dots
    P = dot_shared.P
    # only the per-pass parameters should scale; the round embedding must not
    excl = ()    # all listed stacked params scale by P exactly
    for name in ("w_ih", "w_hh", "p_out", "c_w", "g_w", "s0", "q", "b_ih", "c_b", "g_b"):
        a = getattr(dot_shared, name)
        b = getattr(dot_unshared, name)
        assert b.shape[0] == P and a.shape[0] == 1, (name, a.shape, b.shape)
        assert b.shape[1:] == a.shape[1:], (name, a.shape, b.shape)
    n_shared = sum(p.numel() for p in dot_shared.parameters())
    n_un = sum(p.numel() for p in dot_unshared.parameters())
    round_emb = dot_shared.round_emb.weight.numel()
    assert (n_un - round_emb) == (n_shared - round_emb) * P


def test_final_output_valid_distribution():
    model = _model("e1_weights_passes")
    x = torch.randint(0, 512, (3, 32))
    lp = model(x)["final_log_probs"]
    err = (lp.exp().sum(-1) - 1).abs().max().item()
    assert err < 1e-5 and torch.isfinite(lp).all()
