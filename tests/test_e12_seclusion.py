"""Rung E12 seclusion tests (Tolu's mechanism: every dot reads ONE token)."""
import torch

from config import load_config
from models.region import Region


def _cfg():
    cfg = load_config("configs/e12_seclusion.yaml", "tiny.yaml")
    cfg["model"]["n_regions"] = 1
    cfg["model"]["n_dots"] = 4
    cfg["model"]["n_passes"] = 2
    return cfg


def test_secluded_forward_is_exactly_the_argmax_token():
    """Forward value = the ONE token the dot committed to (no average)."""
    V = 32
    r = Region(0, _cfg(), V)
    assert r.read_mode == "secluded"
    b, t = 2, 8
    wind_k, wind_v, valid = r._window_memory(torch.randn(b, t, r.d),
                                             torch.device("cpu"))
    w = wind_k.shape[2]
    scores = torch.tensordot(wind_k.reshape(b * t, w, r.d_q),
                             r.dots._sel(0, r.dots.q), dims=([2], [1]))
    scores = scores.permute(0, 2, 1) / (r.d_q ** 0.5)          # [f, N, w]
    idx = scores.argmax(-1)
    expected = wind_v.reshape(b * t, w, r.d_v).gather(
        1, idx.unsqueeze(-1).expand(-1, -1, r.d_v))
    ctx = r.dots._secluded_ctx(scores, wind_v.reshape(b * t, w, r.d_v))
    assert torch.allclose(ctx, expected, atol=1e-5)


def test_straight_through_reaches_near_miss_tokens():
    """The soft term routes gradient to tokens the dot did NOT pick — the
    'shouting' learning signal (a token learns its key volume)."""
    V = 32
    r = Region(0, _cfg(), V)
    scores = torch.tensor([[[2.0, -1.0, 0.0]]], requires_grad=False)  # [1,1,3]
    wv = torch.rand(1, 3, 4, requires_grad=True)
    ctx = r.dots._secluded_ctx(scores, wv)
    ctx.sum().backward()
    assert torch.isfinite(ctx).all()
    assert (wv.grad.abs().sum(-1) > 0).all()      # all 3 window tokens got grad
    # forward is hard: equals the argmax token's value exactly
    assert torch.allclose(ctx.detach(), wv.detach()[0, 0, :])


def test_secluded_model_shapes_and_normalized():
    V = 32
    from models.iecnn import build_iecnn
    m = build_iecnn(_cfg(), V)
    m.eval()
    x = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        out = m(x, tau0=1.0, floor=0.0)
    lp = out["final_log_probs"]
    assert lp.shape == (2, 12, V)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 12), atol=1e-4)
    pp = out["per_pass_last_region"]
    assert pp[1]["log_pi"].shape == (2, 12, 4, V)


def test_secluded_model_learns_step():
    V = 32
    from models.iecnn import build_iecnn
    m = build_iecnn(_cfg(), V)
    m.train()
    torch.manual_seed(0)
    x = torch.randint(0, V, (4, 16))
    y = x.clone()
    opt = torch.optim.AdamW(m.parameters(), lr=3e-2)
    lp = m(x, tau0=1.0, floor=0.1 / 4)["final_log_probs"]
    loss0 = -lp[:, :-1].gather(2, y[:, 1:, None]).squeeze(2).mean()
    opt.zero_grad(); loss0.backward(); opt.step()
    lp = m(x, tau0=1.0, floor=0.1 / 4)["final_log_probs"]
    loss1 = -lp[:, :-1].gather(2, y[:, 1:, None]).squeeze(2).mean()
    assert loss1 < loss0
