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


def test_pool_first_identity():
    """The prune's premise: with a SHARED output projection, projecting each
    dot and then pooling is identical to pooling the states and projecting
    once. pool_first relies on exactly this identity."""
    from models.dot import DotPopulation
    dots = DotPopulation(n_dots=5, d=16, d_q=8, d_v=10, d_s=12, vocab_size=32,
                         n_passes=1, share_pass_params=True, share_out=True)
    s = torch.randn(2, 7, 5, 12)                       # [B,T,N,d_s]
    w = torch.softmax(torch.randn(2, 7, 5), dim=-1)    # [B,T,N]
    W = dots.p_out[0]                                  # [d_s, d]
    per_dot = torch.einsum("bfnd,dm->bfnm", s, W)      # project then pool
    h1 = torch.einsum("bfn,bfnd->bfd", w, per_dot)
    s_pool = torch.einsum("bfn,bfnd->bfd", w, s)       # pool then project
    h2 = s_pool.reshape(-1, 12).matmul(W).reshape(2, 7, 16)
    assert torch.allclose(h1, h2, atol=1e-6)


def test_pool_first_runs_and_shrinks_output_projection():
    base = build_iecnn(_cfg(cell="lug", head_last_only=True,
                            per_dot_logits=False), 64)
    pooled = build_iecnn(_cfg(cell="lug", head_last_only=True,
                              per_dot_logits=False, pool_first=True), 64)
    assert pooled.regions[0].dots.p_out.shape == (1, 18, 24)
    assert base.regions[0].dots.p_out.shape == (1, 4, 18, 24)
    x = torch.randint(0, 64, (2, 12))
    lp = _fwd(pooled, x)
    assert torch.isfinite(lp).all()
    n_base = sum(p.numel() for p in base.parameters())
    n_pool = sum(p.numel() for p in pooled.parameters())
    assert n_pool < n_base, (n_base, n_pool)


def test_dist_bias_zero_init_is_exact():
    """dot_dist_bias adds a per-(dot,slot) scalar initialised to ZERO, so
    turning it on cannot change the function at init."""
    base, biased = _pair(dot_dist_bias=True)
    x = torch.randint(0, 64, (2, 12))
    a, b = _fwd(base, x), _fwd(biased, x)
    assert torch.equal(a, b), (a - b).abs().max().item()
    assert biased.regions[0].dots.d_bias.shape == (1, 4, 8)


def test_dist_bias_steers_each_dot_to_its_own_distance():
    """The point of the bias: give dot n a big bonus on slot n and the dots
    must split across DISTINCT distances instead of all picking the last
    slot (the measured 52-70% all-same collapse)."""
    import math as _m
    from models.memory import region1_rows
    m = build_iecnn(_cfg(dot_dist_bias=True, window_mode="hybrid",
                         window_reach=64), 64)
    m.eval()
    d = m.regions[0].dots
    with torch.no_grad():
        d.d_bias.zero_()
        d.d_bias[0, torch.arange(d.N), torch.arange(d.N)] = 20.0
    # T must exceed window_reach, otherwise the far slots are masked out for
    # most positions and a "wrong" pick only means "that slot was invalid".
    x = torch.randint(0, 64, (2, 80))
    rows = region1_rows(m.embed(x))
    wind_k, _wv, valid = m.regions[0]._window_memory(rows, x.device)
    assert bool(valid[-1].all()), "last position must see every slot"
    q = d.q[0]
    scores = torch.einsum("bfwj,nj->bfnw", wind_k, q) / _m.sqrt(d.d_q)
    scores = scores + d.d_bias[0].view(1, d.N, wind_k.shape[2])
    scores = scores.masked_fill(~valid[None, :, None, :], float("-inf"))
    idx = scores.argmax(-1)                      # [B, T, N]
    at_full = idx[:, -1, :]                      # every slot valid here
    assert at_full.tolist() == [list(range(d.N))] * 2, at_full.tolist()
    # the 4 dots therefore sit on 4 DIFFERENT distances (no collapse)
    assert len(set(at_full[0].tolist())) == d.N
    # the bias is what drives the split: move every dot's bonus two slots
    # over and every pick must move with it (deterministic, not luck)
    shifted = torch.roll(d.d_bias[0], shifts=2, dims=-1)
    sc2 = (torch.einsum("bfwj,nj->bfnw", wind_k, q) / _m.sqrt(d.d_q)
           + shifted.view(1, d.N, wind_k.shape[2]))
    sc2 = sc2.masked_fill(~valid[None, :, None, :], float("-inf"))
    assert sc2[:, -1, :].argmax(-1).tolist() == [[2, 3, 4, 5]] * 2


def test_bands_give_each_pass_its_own_window():
    """window_mode "bands": pass 1 = dense local 0..w-1, pass 2 = far band
    tiling (w .. window_reach]. Each pass must get its OWN cache too, or
    pass 2 would silently reuse pass 1's ctx (the fake-depth bug)."""
    from models.memory import region1_rows
    m = build_iecnn(_cfg(spans=[8], window_mode="bands", window_reach=64,
                         n_passes=2, head_last_only=True,
                         per_dot_logits=False), 64)
    m.eval()
    r = m.regions[0]
    x = torch.randint(0, 64, (2, 20))
    rows = region1_rows(m.embed(x))
    near = r._window_offsets(rows.device, 0)
    far = r._window_offsets(rows.device, 1)
    assert int(near[-1]) == 0 and int(near[0]) == -7, near.tolist()
    d_near = (-near).tolist()
    d_far = (-far).tolist()
    # pass 1 = the dense local band, pass 2 = the far band; they must not
    # overlap (pass 2 exists to add information pass 1 cannot see)
    assert d_near[0] <= 8 and d_far[0] >= 8, (d_near[0], d_far[0])
    assert min(d_far) >= 8 and max(d_far) <= 64, d_far
    assert d_far == sorted(d_far, reverse=True), d_far   # slots run far -> near
    # both passes must actually run and produce finite probabilities
    lp = _fwd(m, x)
    assert torch.isfinite(lp).all()
    assert lp.shape == (2, 20, 64)


def test_global_read_is_causal():
    """The prefix-sum read must never leak the future: changing token t+k
    cannot change any log-prob at position <= t+k-1."""
    m = build_iecnn(_cfg(global_read=True, head_last_only=True,
                         per_dot_logits=False, n_passes=2), 64)
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    a = _fwd(m, x)
    x2 = x.clone()
    x2[:, 9] = (x2[:, 9] + 7) % 64                   # perturb the future
    b = _fwd(m, x2)
    assert torch.equal(a[:, :9], b[:, :9]), "global read leaked the future"
    assert not torch.equal(a[:, 9:], b[:, 9:])


def test_global_read_is_exactly_prefix_softmax_attention():
    """The O(T) cumsum trick must equal, to float tolerance, a naive loop
    that runs a real softmax over each position's causal prefix."""
    from models.memory import region1_rows
    m = build_iecnn(_cfg(global_read=True, head_last_only=True,
                         per_dot_logits=False, n_passes=2), 64)
    m.eval()
    r = m.regions[0]
    d = r.dots
    x = torch.randint(0, 64, (2, 10))
    rows = region1_rows(m.embed(x))
    k, v = r._full_kv(rows)                           # [B,T,d_q], [B,T,d_v]
    q = d.q[0]
    sc = torch.einsum("bsj,nj->bns", k, q) / math.sqrt(d.d_q)
    e = torch.exp(sc - sc.amax(-1, keepdim=True))
    ce = torch.cumsum(e, dim=-1)
    cev = torch.cumsum(e.unsqueeze(-1) * v.unsqueeze(1), dim=-2)
    fast = (cev / ce.unsqueeze(-1).clamp_min(1e-9)).permute(0, 2, 1, 3)

    T = sc.shape[-1]
    naive = torch.zeros_like(fast)
    for t in range(T):
        a_t = torch.softmax(sc[..., :t + 1], dim=-1)          # [B, N, t+1]
        naive[:, t] = torch.einsum("bns,bsv->bnv", a_t, v[:, :t + 1])
    assert torch.allclose(fast, naive, atol=1e-5, rtol=1e-5), \
        (fast - naive).abs().max().item()


def test_global_read_mix_is_wired():
    """g_glob must actually gate the two reads: pure-window and pure-global
    settings must give different outputs."""
    m = build_iecnn(_cfg(global_read=True, head_last_only=True,
                         per_dot_logits=False, n_passes=1), 64)
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    dots = m.regions[0].dots
    with torch.no_grad():
        dots.g_glob.fill_(-30.0)                      # sigmoid ~ 0 -> window
        a = _fwd(m, x)
        dots.g_glob.fill_(30.0)                       # sigmoid ~ 1 -> global
        b = _fwd(m, x)
    assert not torch.allclose(a, b)
