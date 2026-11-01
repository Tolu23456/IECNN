"""Tests for baselines B0 (GRU) and B1 (transformer), spec 08 section 4."""
import math

import torch

from models.baselines import (
    GRULanguageModel,
    TransformerLanguageModel,
    build_gru_baseline,
    build_transformer_baseline,
)


def _tiny_gru(vocab=64):
    torch.manual_seed(0)
    return GRULanguageModel(vocab, d=16, hidden=32)


def _tiny_transformer(vocab=64):
    torch.manual_seed(0)
    return TransformerLanguageModel(vocab, d=32, n_heads=2, n_layers=2, seq_len=32)


def test_overfit_ten_sequences_gru():
    """B0 can memorize 10 short sequences (spec 06 test: overfit test)."""
    model = _tiny_gru()
    torch.manual_seed(1)
    seqs = torch.randint(0, 64, (10, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=3e-3)
    first = None
    for step in range(250):
        loss = torch.nn.functional.nll_loss(
            model(seqs)["final_log_probs"].reshape(-1, 64), seqs.reshape(-1))
        if step == 0:
            first = float(loss.item())
        opt.zero_grad(); loss.backward(); opt.step()
    print(f"overfit: first {first:.3f} -> final {float(loss.item()):.3f}")
    assert float(loss.item()) < 0.35 * first, "GRU failed to overfit 10 sequences"


def test_causality_transformer_baseline():
    """Perturbing tokens in the future must not change earlier outputs."""
    model = _tiny_transformer()
    torch.manual_seed(2)
    x = torch.randint(0, 64, (2, 20))
    base = model(x)["logits"]
    xp = x.clone()
    xp[:, 10:] = torch.randint(0, 64, (2, 10))
    pert = model(xp)["logits"]
    assert torch.allclose(base[:, :10], pert[:, :10], atol=1e-6), "future leaked into past"


def test_causality_gru_baseline():
    """Same causal check for the GRU (student for free: GRUs are unrolled)."""
    model = _tiny_gru()
    torch.manual_seed(3)
    x = torch.randint(0, 64, (2, 20))
    base = model(x)["logits"]
    xp = x.clone()
    xp[:, 10:] = torch.randint(0, 64, (2, 10))
    pert = model(xp)["logits"]
    assert torch.allclose(base[:, :10], pert[:, :10], atol=1e-8)


def test_parameter_counts_roughly_match_iecnn():
    """Baselines land in the same order of magnitude as the IECNN default (~5M).

    With the default config the transformer should be within x2 of 5M and the
    GRU not below a third of it (equal-FLOP matching happens at rung time).
    """
    import yaml
    cfg = yaml.safe_load(open("configs/base.yaml"))
    V = 512
    n_gru = sum(p.numel() for p in build_gru_baseline(cfg, V).parameters())
    n_tr = sum(p.numel() for p in build_transformer_baseline(cfg, V).parameters())
    print(f"GRU {n_gru:,} params, transformer {n_tr:,} params")
    assert 1.0e6 < n_tr < 10e6
    assert 0.3e6 < n_gru < 10e6
    assert n_gru > 0.25 * n_tr


def test_resume_gives_identical_loss():
    """Losses AFTER the resume point must match a run that never stopped."""
    model = _tiny_gru()
    torch.manual_seed(5)
    seqs = torch.randint(0, 64, (6, 12))
    opt = torch.optim.AdamW(model.parameters(), lr=10e-3)

    def step3(m, o):
        out = []
        for _ in range(3):
            loss = torch.nn.functional.nll_loss(
                m(seqs)["final_log_probs"].reshape(-1, 64), seqs.reshape(-1))
            o.zero_grad(); loss.backward(); o.step()
            out.append(float(loss.item()))
        return out

    step3(model, opt)                          # warm-up segment (before checkpoint)
    import copy as _copy
    pre = {"model": _copy.deepcopy(model.state_dict()), "opt": _copy.deepcopy(opt.state_dict())}

    a = step3(model, opt)                      # segment A: no interruption
    m2 = GRULanguageModel(64, 16, 32)
    o2 = torch.optim.AdamW(m2.parameters(), lr=10e-3)
    m2.load_state_dict(pre["model"]); o2.load_state_dict(pre["opt"])
    b = step3(m2, o2)                          # segment B: fresh loader, same state
    for a_i, b_i in zip(a, b):
        assert math.isclose(a_i, b_i, rel_tol=1e-8), f"resume diverged: {a} vs {b}"


def test_baseline_forward_shapes_and_valid_distribution():
    model = _tiny_gru()
    x = torch.randint(0, 64, (3, 12))
    out = model(x)
    logp = out["final_log_probs"]
    assert logp.shape == (3, 12, 64)
    p = logp.exp()
    assert torch.allclose(p.sum(-1), torch.ones(3, 12), atol=1e-5)
    assert torch.isfinite(logp).all()


def test_init_loss_near_uniform_at_full_config():
    """bpc at init must be ~log(V): catches embedding-scale bugs (bug found on
    Actions where the full transformer started at 173 nats instead of 6.24)."""
    from config import load_config
    from utils import set_seed

    set_seed(0)
    for name, builder in (("b1_transformer", build_transformer_baseline),
                          ("b0_gru", build_gru_baseline)):
        cfg = load_config(f"configs/{name}.yaml")
        model = builder(cfg, 512)
        x = torch.randint(0, 512, (2, 64))
        out = model(x)
        y = torch.randint(0, 512, (2, 64))
        loss = torch.nn.functional.nll_loss(
            out["final_log_probs"].reshape(-1, 512), y.reshape(-1)).item()
        print(f"{name} init loss {loss:.3f}")
        assert 4.0 < loss < 7.5, f"{name} init loss {loss} is not near ln(512)=6.24"
