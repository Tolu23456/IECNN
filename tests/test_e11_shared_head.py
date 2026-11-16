"""Rung E11 shared-head mode (Tolu's prune: one head, dots differ in state)."""
import torch
from config import load_config
from models.iecnn import build_iecnn


def _cfg(tmp_path):
    return load_config("configs/e11_shared_head.yaml", "tiny.yaml")


def test_hidden_mode_shapes_and_normalized_log_b():
    cfg = _cfg(None)
    cfg["model"]["n_regions"] = 1
    cfg["model"]["n_dots"] = 4
    cfg["model"]["n_passes"] = 2
    V = 32
    m = build_iecnn(cfg, V)
    m.eval()
    x = torch.randint(0, V, (2, 12))
    with torch.no_grad():
        out = m(x, tau0=1.0, floor=0.0)
    lp = out["final_log_probs"]
    assert lp.shape == (2, 12, V)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 12), atol=1e-4)
    pp = out["per_pass_last_region"]
    assert len(pp) == 2
    # intermediate passes have no per-dot logits (the pruned GEMMs);
    # the last pass materializes them once for the oracle/metrics
    assert "log_pi" not in pp[0]
    assert pp[1]["log_pi"].shape == (2, 12, 4, V)
    assert torch.allclose(pp[1]["log_pi"].exp().sum(-1),
                          torch.ones(2, 12, 4), atol=1e-4)
    w = pp[1]["weights"]
    assert torch.allclose(w.sum(-1), torch.ones(2, 12), atol=1e-5)


def test_hidden_mode_learns_step():
    """One train step decreases loss on a fixed batch (wiring sanity)."""
    cfg = _cfg(None)
    cfg["model"]["n_regions"] = 1
    cfg["model"]["n_dots"] = 4
    cfg["model"]["n_passes"] = 2
    V = 32
    m = build_iecnn(cfg, V)
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
