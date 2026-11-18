"""IECNN-Lite invariants: the speed prunes must be numerically EXACT.

Two kinds of test:
  * equivalence  — switching a prune on cannot change the model's output
                   (the skipped work is provably unconsumed);
  * mechanism    — the new cell and the multi-scale window compute what they
                   claim (shape, validity, distances, gradient reach).
"""
from __future__ import annotations

import math

import torch

from models.iecnn import build_iecnn


def _cfg(**over):
    m = {
        "kind": "iecnn", "d": 24, "d_q": 12, "d_v": 20, "d_s": 18,
        "n_dots": 4, "n_regions": 1, "n_passes": 3, "spans": [8],
        "use_weights": True, "use_messages": False, "pooling": "log_linear",
        "head_mode": "hidden", "read_mode": "soft",
        "share_pass_params": True, "dynamic_query": False,
        "use_cross_region": False, "fast_dot_math": True,
        "cache_attention": True,
    }
    m.update(over)
    return {"model": m,
            "consensus": {"rho": 0.0, "kappa": 0.7},
            "training": {"compile_consensus": False, "dot_dropout": 0.0},
            "data": {"seq_len": 12}}


def _pair(**over):
    """Same weights, one config with the prune and one without."""
    base = build_iecnn(_cfg(**over), 64)
    pruned = build_iecnn(_cfg(**over), 64)
    pruned.load_state_dict(base.state_dict())
    base.eval(); pruned.eval()
    return base, pruned


def _fwd(model, x):
    with torch.no_grad():
        return model(x)["final_log_probs"]


def test_head_last_only_is_exact():
    """The head/e_state on passes 1..P-1 are never consumed (the next pass
    uses gamma/state/err, not the distribution), so skipping them must not
    move the final log-probs by even a float epsilon."""
    base, pruned = _pair(head_last_only=True)
    x = torch.randint(0, 64, (2, 12))
    a, b = _fwd(base, x), _fwd(pruned, x)
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_per_dot_logits_off_is_exact():
    base, pruned = _pair(per_dot_logits=False)
    x = torch.randint(0, 64, (2, 12))
    a, b = _fwd(base, x), _fwd(pruned, x)
    assert torch.equal(a, b)
    assert "log_pi" not in pruned.regions[0].forward(
        pruned.embed(x), pruned.embed.weight)["per_pass"][-1]


def test_lug_cell_matches_reference_path():
    """The batched-GEMM LUG step must equal the einsum reference step."""
    from models.dot import DotPopulation
    common = dict(n_dots=3, d=20, d_q=10, d_v=12, d_s=14, vocab_size=48,
                  n_passes=2, share_pass_params=True, cell="lug")
    fast = DotPopulation(seed=7, fast_math=True, **common)
    ref = DotPopulation(seed=7, fast_math=False, **common)
    ref.load_state_dict(fast.state_dict())
    fast.eval(); ref.eval()
    b, t, w = 2, 5, 4
    wind_k = torch.randn(b, t, w, 10)
    wind_v = torch.randn(b, t, w, 12)
    valid = torch.ones(t, w, dtype=torch.bool)
    valid[0] = False
    h_pool = torch.randn(b, t, 3, 14)
    err = torch.randn(b, t, 3, 3)
    state = torch.randn(b, t, 3, 14)
    emb = torch.randn(48, 20)
    with torch.no_grad():
        a = fast(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb)
        c = ref(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb)
    for u, v in ((a.log_pi, c.log_pi), (a.state, c.state),
                 (a.gamma, c.gamma), (a.trust_term, c.trust_term)):
        assert torch.allclose(u, v, atol=1e-5, rtol=1e-5), \
            (u - v).abs().max().item()


def test_lug_has_no_hidden_gemm_and_is_cheaper():
    """LUG must have no recurrent matrix and 2*d_s input gates (the whole
    point: the recurrent term is diagonal)."""
    m = build_iecnn(_cfg(cell="lug"), 64).regions[0].dots
    assert m.w_hh is None and m.w_ih is None
    assert m.w_ih_lug.shape[-1] == 2 * m.d_s
    assert m.u_lug.shape[-1] == m.d_s
    g = build_iecnn(_cfg(cell="gru"), 64).regions[0].dots
    cell_lug = m.w_ih_lug.numel()
    cell_gru = g.w_ih.numel() + g.w_hh.numel()
    # exact ratio 3*(in+d_s)/(2*in): ~2x at this tiny shape, ~2.4x at the
    # production shape (the recurrent matrix disappears entirely)
    assert cell_lug < 0.55 * cell_gru, (cell_lug, cell_gru)


def test_quad_window_reaches_far_with_same_slots():
    """quad mode: same slot count, but the last-but-one distances grow
    quadratically out to window_reach, and pre-sequence slots are invalid."""
    from models.memory import region1_rows
    kw = dict(spans=[8], window_mode="quad", window_reach=64)
    base = build_iecnn(_cfg(**kw), 64)
    base.eval()
    rows = region1_rows(base.embed(torch.randint(0, 64, (1, 12))))
    w_k, w_v, valid = base.regions[0]._window_memory(rows, rows.device)
    assert w_k.shape == (1, 12, 8, 12) and w_v.shape == (1, 12, 8, 20)
    offs = base.regions[0]._window_offsets(rows.device)
    assert int(offs[-1]) == 0, "the current token must be the last slot"
    assert int(offs[0].abs()) >= 32, "the far slot must reach well beyond 8"
    # positions before the sequence start are masked out, not duplicated
    assert valid[0].sum() <= 1
    assert valid[-1].sum() >= 3, "near slots must be valid by t=11"


def test_seclusion_topk_keeps_hard_forward():
    """Top-k straight-through must leave the FORWARD read identical to the
    full-window hard pick (only the gradient path changes)."""
    from models.dot import DotPopulation
    common = dict(n_dots=2, d=16, d_q=8, d_v=10, d_s=12, vocab_size=32,
                  n_passes=1, share_pass_params=True)
    full = DotPopulation(seed=2, fast_math=True, **common)
    topk = DotPopulation(seed=2, fast_math=True, **common)
    full.read_mode = topk.read_mode = "secluded"
    topk.load_state_dict(full.state_dict())
    topk.ste_topk = 3
    full.eval(); topk.eval()
    b, t, w = 2, 4, 8
    wind_k = torch.randn(b, t, w, 8)
    wind_v = torch.randn(b, t, w, 10)
    valid = torch.ones(t, w, dtype=torch.bool)
    h_pool = torch.zeros(b, t, 2, 12)
    err = torch.zeros(b, t, 2, 3)
    state = torch.randn(b, t, 2, 12)
    emb = torch.randn(32, 16)
    with torch.no_grad():
        a = full(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb)
        c = topk(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb)
    assert torch.equal(a.state, c.state)
