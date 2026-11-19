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
    # Not bitwise: `e = exp(sc - sc.amax(-1))` normalises by the max over the
    # whole row, i.e. by a POSITIVE CONSTANT that cancels exactly in the
    # ratio. Mathematically the prefix cannot see the future; in float the
    # cancelled factor leaves last-bit noise, so the real invariant is
    # "no material leak" (a genuine leak is orders of magnitude larger).
    leak = (a[:, :9] - b[:, :9]).abs().max().item()
    assert leak < 1e-6, f"global read leaked the future: {leak}"
    # and the perturbation must actually matter at/after its position
    assert (a[:, 9:] - b[:, 9:]).abs().max().item() > 1e-5


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
    fast = d._global_ctx(k, v, 0)                     # [B, T, N, d_v]  (module)
    sc = torch.einsum("bsj,nj->bns", k, q) / math.sqrt(d.d_q)

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


def test_global_read_cache_is_exact_and_shared_projection_is_exact():
    """Two speed opts must be numerically free:
    (a) caching the pass-invariant global read (g_ctx) across passes, and
    (b) reusing the window read's k/v projections for the global read.
    Same weights, cache off vs on, and pre-projected vs re-projected."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2)
    cached = build_iecnn(_cfg(cache_attention=True, **kw), 64)
    plain = build_iecnn(_cfg(cache_attention=False, **kw), 64)
    plain.load_state_dict(cached.state_dict())
    cached.eval(); plain.eval()
    x = torch.randint(0, 64, (2, 14))
    a, b = _fwd(cached, x), _fwd(plain, x)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), \
        (a - b).abs().max().item()

    # the window read must accept caller-supplied projections unchanged
    r = cached.regions[0]
    rows = torch.randn(2, 14, r.d)
    k, v = r._full_kv(rows)
    w1 = r._window_memory(rows, rows.device, 0)
    w2 = r._window_memory(rows, rows.device, 0, k, v)
    for u, w in zip(w1, w2):
        assert torch.allclose(u, w, atol=0, rtol=0), "projection reuse drifted"


def test_split_read_schedule_is_exact_when_gates_agree():
    """read_schedule='split' must change ONLY the gate rows: with zero gates
    it is bit-identical to the mixed schedule (both reads are the same
    tensors; only the per-pass mix weight differs)."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2)
    mix = build_iecnn(_cfg(**kw), 64)
    split = build_iecnn(_cfg(read_schedule="split", **kw), 64)
    sd = {k: v for k, v in mix.state_dict().items() if "g_glob" not in k}
    split.load_state_dict(sd, strict=False)
    with torch.no_grad():
        split.regions[0].dots.g_glob.zero_()       # matches the 'mix' init
        mix.regions[0].dots.g_glob.zero_()
    mix.eval(); split.eval()
    x = torch.randint(0, 64, (2, 12))
    a, b = _fwd(mix, x), _fwd(split, x)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), \
        (a - b).abs().max().item()


def test_split_read_schedule_gives_each_pass_a_different_read():
    """The point of the split: pass 1 must be window-dominant, pass 2
    global-dominant, so consecutive passes see different input at zero
    added compute."""
    m = build_iecnn(_cfg(global_read=True, read_schedule="split",
                         head_last_only=True, per_dot_logits=False,
                         n_passes=2), 64)
    r = m.regions[0]
    g = r.dots.g_glob
    assert g.shape[0] == 2, g.shape                     # one row per pass
    mix0 = torch.sigmoid(g[0]).mean().item()
    mix1 = torch.sigmoid(g[1]).mean().item()
    assert mix0 < 0.25 < 0.75 < mix1, (mix0, mix1)

    # and each read must still be computed ONCE per region-forward
    calls = {"n": 0}
    real = r.dots._global_ctx

    def counted(*a, **k):
        calls["n"] += 1
        return real(*a, **k)

    r.dots._global_ctx = counted
    x = torch.randint(0, 64, (2, 12))
    _fwd(m, x)
    assert calls["n"] == 1, f"global read ran {calls['n']}x for 2 passes"
    r.cache_attention = False
    _fwd(m, x)
    assert calls["n"] == 3, f"cache off should recompute: {calls['n']}"


def test_share_cell_is_exact_and_enables_more_dots():
    """A shared cell must equal per-dot cells when every dot's cell weights
    are identical (the N batched GEMMs and the one merged GEMM compute the
    same numbers). It must also let N grow without growing the cell."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug")
    base = build_iecnn(_cfg(**kw), 64)
    shared = build_iecnn(_cfg(share_cell=True, **kw), 64)
    sd = {k: v for k, v in base.state_dict().items()
          if "w_ih_lug" not in k and "b_lug" not in k and "u_lug" not in k}
    shared.load_state_dict(sd, strict=False)
    db, ds_ = base.regions[0].dots, shared.regions[0].dots
    with torch.no_grad():
        ds_.w_ih_lug.copy_(db.w_ih_lug[:, :1])
        ds_.b_lug.copy_(db.b_lug[:, :1])
        ds_.u_lug.copy_(db.u_lug[:, :1])
        db.w_ih_lug.copy_(ds_.w_ih_lug.expand_as(db.w_ih_lug))
        db.b_lug.copy_(ds_.b_lug.expand_as(db.b_lug))
        db.u_lug.copy_(ds_.u_lug.expand_as(db.u_lug))
    base.eval(); shared.eval()
    x = torch.randint(0, 64, (2, 12))
    a, b = _fwd(base, x), _fwd(shared, x)
    assert torch.allclose(a, b, atol=1e-6, rtol=1e-6), \
        (a - b).abs().max().item()

    # cell size must not depend on N
    n4 = build_iecnn(_cfg(share_cell=True, **kw), 64)
    n8 = build_iecnn(_cfg(share_cell=True, n_dots=8, **kw), 64)
    def cell_params(m):
        return sum(v.numel() for k, v in m.regions[0].dots.named_parameters()
                   if k in ("w_ih_lug", "b_lug", "u_lug"))

    c4, c8 = cell_params(n4), cell_params(n8)
    assert c4 == c8, (c4, c8)


def test_query_from_input_starts_static_and_stays_causal():
    """query_from_input must (a) be bit-identical to the static-query model
    at init (W=0), (b) actually react to the CURRENT token, and (c) still
    leak nothing from the future — the query depends on position t only."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug")
    base = build_iecnn(_cfg(**kw), 64)
    qin = build_iecnn(_cfg(query_from_input=True, **kw), 64)
    qin.load_state_dict(base.state_dict(), strict=False)   # w_qi stays 0
    base.eval(); qin.eval()
    x = torch.randint(0, 64, (2, 12))
    assert torch.allclose(_fwd(base, x), _fwd(qin, x), atol=1e-7, rtol=1e-7)

    with torch.no_grad():                       # turn the term on
        qin.regions[0].w_qi.normal_(std=0.5)
    a = _fwd(qin, x)
    x2 = x.clone()
    x2[:, 9] = (x2[:, 9] + 7) % 64              # perturb the FUTURE token
    b = _fwd(qin, x2)
    assert (a[:, :9] - b[:, :9]).abs().max().item() < 1e-6
    assert (a[:, 9:] - b[:, 9:]).abs().max().item() > 1e-6

    x3 = x.clone()
    x3[:, 5] = (x3[:, 5] + 11) % 64             # perturb a PAST-position token
    c = _fwd(qin, x3)
    assert (a[:, 5] - c[:, 5]).abs().max().item() > 1e-6, \
        "input-conditioned query did not react to its own token"


def test_id_lags_are_exact_shifted_identities():
    """id_lags: dot k must be injected with rows[t-k-1] (its own lag), zero
    before the sequence start, and never with a future row."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug", id_lags=True)
    model = build_iecnn(_cfg(**kw), 64)
    r = model.regions[0]
    with torch.no_grad():
        r.w_id.normal_(std=0.2)                   # non-trivial projection
    # w_id is [d, N*id]: blocks of `id` are dot-major, so view d x N x id
    w = r.w_id.reshape(r.d, r.n_dots, r.id_dim).permute(1, 0, 2)
    rows = torch.randn(2, 12, r.d) * 0.3
    code = r._id_code(rows)                       # [B,T,N,id_dim]
    T = rows.shape[1]
    for k in range(r.n_dots):
        lag = k + 1
        # position t >= lag reads rows[t-lag] through dot k's projection
        want = rows[:, :T - lag].matmul(w[k])
        got = code[:, lag:, k]
        assert torch.allclose(got, want, atol=1e-6), (k, (got - want).abs().max())
        # positions before the lag read exactly zero (no wrap-around)
        assert code[:, :lag, k].abs().max().item() == 0.0

    model.eval()
    x = torch.randint(0, 64, (2, 12))
    a = _fwd(model, x)
    x2 = x.clone(); x2[:, 9] = (x2[:, 9] + 7) % 64      # future token
    b = _fwd(model, x2)
    assert (a[:, :9] - b[:, :9]).abs().max().item() < 1e-6


def test_ngram_bias_is_a_normalised_order1_prior():
    """ngram_bias adds E1[x_{t-1}] @ E2^T to the head logits: valid log-probs,
    uses exactly the previous token, and changes nothing when ids are absent."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug")
    base = build_iecnn(_cfg(**kw), 64).eval()
    ng = build_iecnn(_cfg(ngram_bias=32, **kw), 64)
    ng.load_state_dict(base.state_dict(), strict=False)
    ng.eval()
    x = torch.randint(0, 64, (2, 12))
    r = ng.regions[0]
    with torch.no_grad():
        a = _fwd(base, x)
        b = _fwd(ng, x)
        # ids=None must skip the term exactly (other callers/tests rely on it)
        no_ids = r(ng.embed(x), ng.embed.weight, ids=None)["log_b"]
        assert torch.allclose(no_ids, a, atol=1e-6), "bias leaked without ids"
        assert torch.allclose(b.exp().sum(-1), torch.ones(2, 12), atol=1e-5)
        # exact wiring: log_b == log_softmax(head + prior), prior at t reads
        # E1[x_{t-1}] (zero at t=0), and nothing else
        raw = torch.randn(2, 12, 64) * 0.5
        orig = r._forward_hidden
        r._forward_hidden = lambda *a, **k: {"log_b": raw}
        try:
            got = r(ng.embed(x), ng.embed.weight, ids=x)["log_b"]
        finally:
            r._forward_hidden = orig
    # position t predicts x_{t+1} and has already seen x_t, so the exact
    # order-1 term is E1[x_t] — always in the causal past, no shift needed
    prior = r.ng_e1[x].matmul(r.ng_e2.t())
    want = torch.log_softmax(raw + prior, dim=-1)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
    assert prior.abs().max().item() > 0.0


def test_head_rank_matches_its_factors_and_is_cheaper():
    """head_rank must compute (h A) B — no hidden extra term — and must beat
    the tied head on MACs (r*(d+V) < d*V)."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug", d=32)
    model = build_iecnn(_cfg(head_rank=12, **kw), 64).eval()
    r = model.regions[0]
    h = torch.randn(7, r.d)
    got = r._head_logits(h, model.embed.weight)
    want = h.matmul(r.w_head_a).matmul(r.w_head_b)
    assert torch.allclose(got, want, atol=1e-6), (got - want).abs().max()
    assert r.head_rank * (r.d + 64) < r.d * 64, "rank head must be cheaper"
    x = torch.randint(0, 64, (2, 12))
    lp = _fwd(model, x)
    assert lp.shape == (2, 12, 64)
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 12), atol=1e-5)


def test_carry_scan_matches_sequential_loop():
    """_carry_scan must implement s_t = a_t*s_{t-1} + b_t EXACTLY, for any T
    (including T not divisible by the chunk size), for any chunk size, and
    with the initial state reaching position 0."""
    dots = build_iecnn(_cfg(cell="lug", state_carry=True), 64).regions[0].dots
    torch.manual_seed(0)
    for t in (1, 7, 8, 12, 16, 64):
        for c in (1, 3, 8, 16):
            dots.carry_chunk = c
            a = torch.sigmoid(torch.randn(2, t, 3, 5))
            bb = torch.randn(2, t, 3, 5) * 0.1
            s0 = torch.randn(2, 3, 5)
            got = dots._carry_scan(a, bb, s0)
            s = s0
            want = []
            for i in range(t):
                s = a[:, i] * s + bb[:, i]
                want.append(s)
            want = torch.stack(want, 1)
            assert got.shape == want.shape, (t, c, got.shape, want.shape)
            err = (got - want).abs().max().item()
            assert err < 1e-6, (t, c, err)


def test_carry_scan_is_differentiable_everywhere():
    """The scan is trained through: every position contributes gradient to
    a, b and to the sequence-start state."""
    dots = build_iecnn(_cfg(cell="lug", state_carry=True), 64).regions[0].dots
    a = torch.sigmoid(torch.randn(2, 12, 3, 5)).requires_grad_(True)
    bb = torch.randn(2, 12, 3, 5, requires_grad=True)
    s0 = torch.randn(2, 3, 5, requires_grad=True)
    out = dots._carry_scan(a, bb, s0)
    out.pow(2).sum().backward()
    for name, g in (("a", a.grad), ("b", bb.grad), ("s0", s0.grad)):
        assert g is not None and torch.isfinite(g).all(), name
    assert a.grad.abs().sum() > 0 and bb.grad.abs().sum() > 0 and s0.grad.abs().sum() > 0


def test_state_carry_reaches_past_the_window_and_stays_causal():
    """With global_read OFF, the only way information can travel further
    back than the read window is the carried state. Perturbing position 2
    (outside position 11's 8-token window) must change position 11 and must
    NOT change positions before 2."""
    kw = dict(global_read=False, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug", state_carry=True, spans=[8])
    m = build_iecnn(_cfg(**kw), 64).eval()
    x = torch.randint(0, 64, (1, 12))
    with torch.no_grad():
        a = _fwd(m, x)
        x2 = x.clone()
        x2[0, 2] = (int(x2[0, 2]) + 9) % 64
        b = _fwd(m, x2)
    assert (a[:, :2] - b[:, :2]).abs().max().item() < 1e-6, "carry leaked the future"
    assert (a[:, 11] - b[:, 11]).abs().max().item() > 1e-6, \
        "carried state did not reach past the window"


def test_state_carry_trains_and_differs_from_reset():
    """state_carry is a real change of computation (not a no-op), it runs a
    full backward, and it keeps every parameter finite."""
    kw = dict(global_read=True, head_last_only=True, per_dot_logits=False,
              n_passes=2, cell="lug")
    base = build_iecnn(_cfg(**kw), 64)
    car = build_iecnn(_cfg(state_carry=True, **kw), 64)
    car.load_state_dict(base.state_dict())
    base.eval(); car.eval()
    x = torch.randint(0, 64, (2, 12))
    assert (_fwd(base, x) - _fwd(car, x)).abs().max().item() > 1e-5

    car.train()
    opt = torch.optim.AdamW(car.parameters(), lr=1e-3)
    y = torch.randint(0, 64, (2, 12))
    for _ in range(2):
        lp = car(x)["final_log_probs"]
        loss = -lp.reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
        opt.zero_grad(); loss.backward(); opt.step()
    assert math.isfinite(loss.item())
    assert all(torch.isfinite(p).all() for p in car.parameters())


def test_gla_matches_a_naive_position_by_position_linear_attention():
    """`_global_ctx_linear` must equal the definition, written the slow way:
    S_t = sum_{s<=t} phi(k_s) v_s^T, z_t = sum_{s<=t} phi(k_s),
    ctx_t = phi(q)^T S_t / phi(q)^T z_t, with phi(z)=elu(z/d^(1/4))+1."""
    import torch.nn.functional as F
    model = build_iecnn(_cfg(global_read=True, global_linear=True, cell="lug"), 64)
    dots = model.regions[0].dots
    b, t, dq, dv, n = 2, 11, dots.d_q, dots.d_v, dots.N
    k = torch.randn(b, t, dq) * 0.5
    v = torch.randn(b, t, dv) * 0.5
    got = dots._global_ctx_linear(k, v, 0)
    assert got.shape == (b, t, n, dv)
    s = dots.d_q ** -0.25
    phik = F.elu(k * s) + 1.0
    phiq = F.elu(dots._sel(0, dots.q) * s) + 1.0                     # [N,dq]
    want = torch.zeros(b, t, n, dv)
    for bb in range(b):
        S = torch.zeros(dq, dv)
        z = torch.zeros(dq)
        for tt in range(t):
            S = S + torch.outer(phik[bb, tt], v[bb, tt])
            z = z + phik[bb, tt]
            for nn in range(n):
                num = phiq[nn] @ S
                den = (phiq[nn] @ z).clamp_min(1e-6)
                want[bb, tt, nn] = num / den
    err = (got - want).abs().max().item()
    assert err < 1e-5, err


def test_gla_is_causal_cached_and_differs_from_the_static_read():
    """Causality (a future token cannot change an earlier read), the S/z
    cache must not change the numbers, and the read must actually differ
    from the static-query prefix sum it replaces."""
    kw = dict(global_read=True, cell="lug", head_last_only=True,
              per_dot_logits=False, n_passes=2)
    static = build_iecnn(_cfg(**kw), 64)
    lin = build_iecnn(_cfg(global_linear=True, **kw), 64)
    lin.load_state_dict(static.state_dict())        # same params, new read
    x = torch.randint(0, 64, (2, 12))
    dots = lin.regions[0].dots
    k = torch.randn(2, 12, dots.d_q)
    v = torch.randn(2, 12, dots.d_v)
    with torch.no_grad():
        a = dots._global_ctx(k, v, 0)
        c1, c2 = {}, {}
        b1 = dots._global_ctx(k, v, 0, cache=c1)     # fills the cache
        b2 = dots._global_ctx(k, v, 0, cache=c1)     # cache hit
        assert torch.allclose(b1, b2, atol=0, rtol=0), "cached read drifted"
        # same k/v, future perturbed: earlier reads must be identical
        k2, v2 = k.clone(), v.clone()
        k2[:, 7:] += 3.0
        b3 = dots._global_ctx(k2, v2, 0, cache=c2)
        assert (a[:, :7] - b3[:, :7]).abs().max().item() < 1e-6
        assert (a[:, 7:] - b3[:, 7:]).abs().max().item() > 1e-6
        # and it is not the static-query read
        sa = static.regions[0].dots._global_ctx(k, v, 0)
        assert (sa - a).abs().max().item() > 1e-3
    # end-to-end: runs, normalised log-probs, and trains a step
    lin.train()
    lp = lin(x)["final_log_probs"]
    assert torch.allclose(lp.exp().sum(-1), torch.ones(2, 12), atol=1e-5)
    y = torch.randint(0, 64, (2, 12))
    loss = -lp.reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in lin.parameters()
               if p.grad is not None)


def test_exact_far_read_equals_the_prefix_sum_and_is_causal():
    """`global_exact` with a STATIC query must reproduce the cumsum read: for a
    query that cannot depend on t, softmax over the causal prefix is exactly
    the normalised prefix sum. That equivalence makes this a free correctness
    check on both implementations, and the content-addressed variant must
    differ (otherwise `query_from_input` is silently a no-op on the far read,
    which it was until now)."""
    kw = dict(global_read=True, cell="lug", head_last_only=True,
              per_dot_logits=False, n_passes=2)
    m = build_iecnn(_cfg(**kw), 64)
    dots = m.regions[0].dots
    b, t, dq, dv = 2, 13, dots.d_q, dots.d_v
    k = torch.randn(b, t, dq) * 0.7
    v = torch.randn(b, t, dv) * 0.7
    with torch.no_grad():
        fast = dots._global_ctx(k, v, 0, cache={})
        exact = dots._global_ctx_exact(k, v, 0)
        err = (fast - exact).abs().max().item()
        assert err < 1e-5, err
        # causality: perturbing the future cannot move an earlier read
        k2, v2 = k.clone(), v.clone()
        k2[:, 7:] += 2.0
        e2 = dots._global_ctx_exact(k2, v2, 0)
        assert (exact[:, :7] - e2[:, :7]).abs().max().item() < 1e-6
        assert (exact[:, 7:] - e2[:, 7:]).abs().max().item() > 1e-6
        # a content-dependent query must actually change the read
        q_add = torch.randn(b, t, dq) * 0.5
        content = dots._global_ctx_exact(k, v, 0, q_add=q_add)
        assert (content - exact).abs().max().item() > 1e-4


def test_head_skip_changes_the_output_and_trains():
    """`head_skip` must add the rows to the pooled state (0 params): the arm
    has to produce different logits from the identical weights, stay
    normalised, and still take a gradient step."""
    base = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                            head_last_only=True, n_passes=2), 64)
    skip = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                            head_last_only=True, n_passes=2, head_skip=True), 64)
    assert (sum(p.numel() for p in base.parameters())
            == sum(p.numel() for p in skip.parameters()))
    skip.load_state_dict(base.state_dict())
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = base(x)["final_log_probs"]
        b = skip(x)["final_log_probs"]
        assert (a - b).abs().max().item() > 1e-3
        assert torch.allclose(b.exp().sum(-1), torch.ones(2, 12), atol=1e-5)
    skip.train()
    y = torch.randint(0, 64, (2, 12))
    lp = skip(x)["final_log_probs"]
    loss = -lp.reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert all(torch.isfinite(p.grad).all() for p in skip.parameters()
               if p.grad is not None)


def test_break_scan_is_numerically_identical():
    """`break_scan` only changes WHERE Inductor may place a graph boundary
    (CUDA workaround for the SplitScan codegen bug), never the arithmetic:
    same weights must give bit-identical logits."""
    base = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                            head_last_only=True, n_passes=2), 64)
    brk = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                           head_last_only=True, n_passes=2, break_scan=True), 64)
    brk.load_state_dict(base.state_dict())
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = base(x)["final_log_probs"]
        b = brk(x)["final_log_probs"]
    assert torch.equal(a, b), (a - b).abs().max().item()


def test_head_post_adds_one_per_position_block_and_trains():
    """`head_post` must be a residual block: identity when off, a real
    per-position nonlinearity when on, same leading shape, normalised
    log-probs, and still differentiable."""
    off = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                           head_last_only=True, n_passes=2), 64)
    on = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                          head_last_only=True, n_passes=2, head_post=32), 64)
    on.load_state_dict(off.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = off(x)["final_log_probs"]
        b = on(x)["final_log_probs"]
    assert a.shape == b.shape
    assert (a - b).abs().max().item() > 1e-4
    assert torch.allclose(b.exp().sum(-1), torch.ones(2, 12), atol=1e-4)
    on.train()
    y = torch.randint(0, 64, (2, 12))
    loss = -on(x)["final_log_probs"].reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    g = on.regions[0].post_blocks[0]["w1"].weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_interleave_rereads_between_passes_and_is_causal():
    """`interleave` must change the function (the second pass reads the rows
    as refined by the first pass's block, not a cached identical ctx), must
    stay causal, must be identity-equivalent when off, and must train."""
    base = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                            head_last_only=True, n_passes=2), 64)
    il = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                          head_last_only=True, n_passes=2, head_post=16,
                          head_post_layers=1, interleave=True), 64)
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = base(x)["final_log_probs"]
        b = il(x)["final_log_probs"]
        # causal: perturbing the last position cannot move earlier log-probs
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 7) % 64
        b2 = il(x2)["final_log_probs"]
        assert (b[:, :-1] - b2[:, :-1]).abs().max().item() < 1e-6
        assert (b[:, -1] - b2[:, -1]).abs().max().item() > 1e-6
        assert torch.allclose(b.exp().sum(-1), torch.ones(2, 12), atol=1e-4)
    # the mechanism must actually change the computation
    il.load_state_dict(base.state_dict(), strict=False)
    with torch.no_grad():
        c = il(x)["final_log_probs"]
    assert (a - c).abs().max().item() > 1e-4
    il.train()
    y = torch.randint(0, 64, (2, 12))
    loss = -il(x)["final_log_probs"].reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    g = il.regions[0].w_k.weight.grad
    assert g is not None and torch.isfinite(g).all()


def test_read_ln_and_head_ln_are_real_and_cheap():
    """Both flags must change the function, add only LayerNorm parameters
    (2*d each), keep log-probs normalised, and train."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2, head_post=16, head_post_layers=1)
    base = build_iecnn(_cfg(**kw), 64)
    ln = build_iecnn(_cfg(**kw, read_ln=True, head_ln=True), 64)
    nb = sum(p.numel() for p in base.parameters())
    nl = sum(p.numel() for p in ln.parameters())
    assert nl - nb == 4 * _cfg(**kw)["model"]["d"], nl - nb
    ln.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = base(x)["final_log_probs"]
        b = ln(x)["final_log_probs"]
    assert (a - b).abs().max().item() > 1e-4
    assert torch.allclose(b.exp().sum(-1), torch.ones(2, 12), atol=1e-4)
    ln.train()
    y = torch.randint(0, 64, (2, 12))
    loss = -ln(x)["final_log_probs"].reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert torch.isfinite(ln.regions[0].head_ln.weight.grad).all()


def test_read_heads_adds_patterns_for_free_and_h1_is_identical():
    """`read_heads=1` must be bit-identical to the single-query read (head 0
    REUSES self.q), and H>1 must only add (H-1)*d_q queries per dot while
    keeping the context width, causality and differentiability intact."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2)
    base = build_iecnn(_cfg(**kw), 64)
    h2 = build_iecnn(_cfg(**kw, read_heads=2), 64)
    dots = base.regions[0].dots
    n_extra = (dots.N * dots.d_q * (2 - 1)
               * (1 if dots.shared else _cfg(**kw)["model"]["n_passes"]))
    assert sum(p.numel() for p in h2.parameters()) - \
        sum(p.numel() for p in base.parameters()) == n_extra
    # H=1 path must be untouched: build the same model twice, compare
    same = build_iecnn(_cfg(**kw, read_heads=1), 64)
    same.load_state_dict(base.state_dict())
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = base(x)["final_log_probs"]
        assert torch.equal(a, same(x)["final_log_probs"])
        b = h2(x)["final_log_probs"]
    assert (a - b).abs().max().item() > 1e-4          # H>1 really changes it
    # causality at H=2
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 5) % 64
    with torch.no_grad():
        b2 = h2(x2)["final_log_probs"]
    assert (b[:, :-1] - b2[:, :-1]).abs().max().item() < 1e-6
    assert (b[:, -1] - b2[:, -1]).abs().max().item() > 1e-6
    h2.train()
    y = torch.randint(0, 64, (2, 12))
    loss = -h2(x)["final_log_probs"].reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert h2.regions[0].dots.q_extra.grad is not None
    assert torch.isfinite(h2.regions[0].dots.q_extra.grad).all()


def test_global_decay_adds_recency_and_vanishes_at_gamma_one():
    """The prefix read must be exactly unchanged when the per-dot decay is ~1,
    and must actually weight recent values more when it is small. N extra
    scalars is the whole parameter cost."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2)
    base = build_iecnn(_cfg(**kw), 64)
    dec = build_iecnn(_cfg(**kw, global_decay=True), 64)
    nb = sum(p.numel() for p in base.parameters())
    nd = sum(p.numel() for p in dec.parameters())
    assert nd - nb == _cfg(**kw)["model"]["n_dots"]
    b, t = 2, 12
    db, dd = base.regions[0].dots, dec.regions[0].dots
    k = torch.zeros(b, t, db.d_q)          # uniform scores: only distance acts
    v = torch.randn(b, t, db.d_v)
    with torch.no_grad():
        plain = db._global_ctx(k, v, 0)
        dd.g_decay.fill_(25.0)             # sigmoid -> 1 (clamped at 0.99999)
        same = dd._global_ctx(k, v, 0)
        # gamma is clamped just below 1, so this is equivalence up to the
        # clamp, not bit-identity: gamma^-12 at 0.99999 is 1.0001.
        d_small = (plain - same).abs().max().item()
        assert d_small < 1e-3, d_small
        dd.g_decay.fill_(-2.0)             # sigmoid -> 0.12 -> clamp 0.5
        rec = dd._global_ctx(k, v, 0)
        # position 0 has exactly one cause, so recency must return v_0 there
        assert (rec[:, 0] - v[:, 0].unsqueeze(1)).abs().max().item() < 1e-5
        # and the last position must be much closer to v_t than the bag is
        d_rec = (rec[:, -1] - v[:, -1].unsqueeze(1)).abs().mean()
        d_bag = (plain[:, -1] - v[:, -1].unsqueeze(1)).abs().mean()
        assert d_rec < d_bag
        # causality
        v2 = v.clone()
        v2[:, 7:] += 3.0
        r2 = dd._global_ctx(k, v2, 0)
        assert (rec[:, :7] - r2[:, :7]).abs().max().item() < 1e-6
    dec.train()
    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))
    loss = -dec(x)["final_log_probs"].reshape(-1, 64).gather(1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert torch.isfinite(dd.g_decay.grad).all()


def test_global_decay_matches_the_direct_weighted_average():
    """Pin the decayed prefix read against the formula it claims to compute,

        ctx_t = sum_{s<=t} e_s gam^(t-s) v_s / sum_{s<=t} e_s gam^(t-s)

    evaluated directly in a loop. The implementation gets this from two
    cumsums with a gamma-folded weight, and the fold has to be a constant
    across s or the ratio is wrong; this is the test that would catch that."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2, global_decay=True)
    d = build_iecnn(_cfg(**kw), 64).regions[0].dots
    b, t = 2, 20
    torch.manual_seed(0)
    k = torch.randn(b, t, d.d_q)
    v = torch.randn(b, t, d.d_v)
    for gv in (-1.0, 1.0, 4.0):                  # gam = .27, .73, .98
        with torch.no_grad():
            d.g_decay.fill_(gv)
            got = d._global_ctx(k, v, 0)                       # [B,T,N,d_v]
            gam = torch.sigmoid(torch.tensor(gv)).clamp(1e-4, 0.99999)
            sc = k @ d.q[0].t() / math.sqrt(d.d_q)             # [B,T,N]
            ref = torch.zeros_like(got)
            for tt in range(t):
                w = torch.exp(sc[:, :tt + 1] - sc[:, tt:tt + 1])  # [B,s,N]
                dist = torch.arange(tt, -1, -1).float().view(1, -1, 1)
                w = w * gam ** dist                               # [B,s,N]
                w = w.permute(0, 2, 1).unsqueeze(-1)              # [B,N,s,1]
                ref[:, tt] = ((w * v[:, None, :tt + 1]).sum(2)
                              / w.sum(2))
            err = (got - ref).abs().max().item()
        assert err < 1e-3, (gv, err)


def test_global_decay_does_not_overflow_at_long_contexts():
    """The gamma^-s fold overflowed float32 for gam < 0.71 at T=256 (the old
    code raised above T=1024 to hide it), and the first fix for that — folding
    by gam^((T-1)-s) — underflowed the whole prefix to zero instead. The fold
    now keeps weights >= 1 and floors gamma at exp(-60/T) so the largest
    weight is e^60. Both ends must be finite at a T far beyond the old guard,
    and history beyond e^-60 must be genuinely unreachable, not merely small."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2, global_decay=True)
    d = build_iecnn(_cfg(**kw), 64).regions[0].dots
    with torch.no_grad():
        d.g_decay.fill_(-20.0)                  # asks for gamma -> 0
        k = torch.randn(1, 4096, d.d_q)
        v = torch.randn(1, 4096, d.d_v)
        ctx = d._global_ctx(k, v, 0)
        assert torch.isfinite(ctx).all()
        # the floored gamma is exp(-60/4095): position 0 is e^-60 of the
        # newest term, so changing it may not move the last read position
        v2 = v.clone()
        v2[:, 0] += 100.0
        ctx2 = d._global_ctx(k, v2, 0)
        assert (ctx[:, -1] - ctx2[:, -1]).abs().max().item() < 1e-4
        # ... and the decay must still be active: with the floored gamma the
        # previous value carries (1-gam) ~ 1.5% of the mass, so moving it by
        # 10 has to move the read, while moving position 0 must not.
        v3 = v.clone()
        v3[:, 1999] += 10.0
        moved = (ctx[:, 2000] - d._global_ctx(k, v3, 0)[:, 2000]).abs().mean()
        assert 0.05 < moved.item() < 1.0, moved


def test_post_share_trades_parameters_for_depth():
    """The block stack is 74% of the model and it is n copies of the same
    shape, so "all" must apply ONE block n times (one block of parameters,
    n blocks of compute) and "up" must share only the up-projection. The
    shared stack must compute exactly what the explicit stack computes."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2, head_post=32, head_post_layers=4)
    base = build_iecnn(_cfg(**kw), 64)
    allsh = build_iecnn(_cfg(**kw, post_share="all"), 64)
    upsh = build_iecnn(_cfg(**kw, post_share="up"), 64)
    n = lambda m: sum(p.numel() for p in m.parameters())
    r = base.regions[0]
    n_blk = sum(p.numel() for p in r.post_blocks[0].parameters())
    assert len(allsh.regions[0].post_blocks) == 1
    assert allsh.regions[0].post_repeat == 4
    assert n(base) - n(allsh) == 3 * n_blk
    # "up" keeps every per-block ln and w2, and only the [d, w] w1 is shared
    n_w1 = sum(p.numel() for p in r.post_blocks[0]["w1"].parameters())
    assert n(base) - n(upsh) == 3 * n_w1
    # and the shared stack is the same function as the explicit one
    h = torch.randn(3, 5, r.post_blocks[0]["ln"].normalized_shape[0])
    with torch.no_grad():
        blk = allsh.regions[0].post_blocks[0]
        want = h
        for _ in range(4):
            want = want + blk["w2"](torch.nn.functional.gelu(
                blk["w1"](blk["ln"](want))))
        got = allsh.regions[0]._post(h)
    assert (got - want).abs().max().item() < 1e-6
    # gradients must reach the single shared block from every application
    allsh.train()
    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))
    loss = -allsh(x)["final_log_probs"].reshape(-1, 64).gather(
        1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert allsh.regions[0].post_blocks[0]["w1"].weight.grad is not None
    assert torch.isfinite(
        allsh.regions[0].post_blocks[0]["w1"].weight.grad).all()


def test_global_decay_mix_k_collapses_to_one_timescale():
    """K geometric reads combined by a softmax: with every gam_k equal and the
    softmax uniform, the mixture must be EXACTLY the single decayed read (the
    same numbers, not merely similar), and the parameter cost must be N*(K-1).
    With distinct gammas it must be a genuine convex mix of the K reads."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2, global_decay=True)
    one = build_iecnn(_cfg(**kw), 64).regions[0].dots
    m_three = build_iecnn(_cfg(**kw, global_decay_k=3), 64)
    three = m_three.regions[0].dots
    n = lambda m: sum(p.numel() for p in m.parameters())
    assert n(three) - n(one) == one.N * 2
    b, t = 2, 16
    torch.manual_seed(0)
    k = torch.randn(b, t, one.d_q)
    v = torch.randn(b, t, one.d_v)
    with torch.no_grad():
        one.g_decay.fill_(2.0)
        three.g_decay.fill_(2.0)               # tied gammas, uniform mix
        a = one._global_ctx(k, v, 0)
        c = three._global_ctx(k, v, 0)
        assert (a - c).abs().max().item() < 1e-6
        # distinct timescales: the mix must sit between the fastest and the
        # slowest single read (convexity), and differ from both
        three.g_decay.copy_(
            torch.tensor([-3.0, 2.0, 6.0]).view(1, 1, 3).expand_as(
                three.g_decay).contiguous())        # three distinct gammas
        m = three._global_ctx(k, v, 0)
        reads = []
        for j in range(3):
            t_one = build_iecnn(_cfg(**kw), 64).regions[0].dots
            t_one.g_decay.copy_(three.g_decay[:, :, j])
            reads.append(t_one._global_ctx(k, v, 0))
        lo = torch.stack(reads).min(0).values
        hi = torch.stack(reads).max(0).values
        assert (m >= lo - 1e-5).all() and (m <= hi + 1e-5).all()
        assert (m - reads[1]).abs().max().item() > 1e-4
    m_three.train()
    x = torch.randint(0, 64, (2, 12))
    y = torch.randint(0, 64, (2, 12))
    loss = -m_three(x)["final_log_probs"].reshape(-1, 64).gather(
        1, y.reshape(-1, 1)).mean()
    loss.backward()
    assert torch.isfinite(three.g_decay.grad).all()


def test_ablate_dot_is_a_diagnostic_only_path():
    """`ablate_dot` exists so the audit can ask what one dot adds by dropping
    it from the pool. It must not touch the default path AT ALL (bit-identical
    log-probs at -1) and dropping a dot must give a finite, different answer."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2)
    m = build_iecnn(_cfg(**kw), 64)
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = m(x)["final_log_probs"]
        b = m(x, ablate_dot=-1)["final_log_probs"]
        assert torch.equal(a, b)
        c = m(x, ablate_dot=0)["final_log_probs"]
        d = m(x, ablate_dot=1)["final_log_probs"]
    assert torch.isfinite(c).all() and torch.isfinite(d).all()
    assert (a - c).abs().max().item() > 1e-5
    assert (c - d).abs().max().item() > 1e-5


def test_pool_only_dot_is_the_trained_path_one_hot():
    """`pool_only_dot=k` is how the audit judges a dot ALONE (per-dot logits
    are an untrained side output, so they cannot answer "is this dot any
    good"). It must be inert at -1 and exact: the one-hot pool has to
    reproduce the pooled answer when there is only ONE dot to pick."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=2)
    m = build_iecnn(_cfg(**kw), 64)
    m.eval()
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        a = m(x)["final_log_probs"]
        assert torch.equal(a, m(x, pool_only_dot=-1)["final_log_probs"])
        outs = [m(x, pool_only_dot=k)["final_log_probs"] for k in range(4)]
    for o in outs:
        assert torch.isfinite(o).all()
        assert o.shape == a.shape
    assert (outs[0] - outs[1]).abs().max().item() > 1e-5


def test_gate_agree_is_inert_at_init_and_trains():
    """gate_agree gives the gate the one thing it cannot see: whether a dot
    agrees with the committee. Zero init means the arm must start
    bit-identical to the arm without it, and the scalars must actually
    receive gradient (otherwise the mechanism silently does nothing)."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=1, n_dots=3)
    base = build_iecnn(_cfg(**kw), 64).eval()
    m = build_iecnn(_cfg(gate_agree=True, **kw), 64).eval()
    # copy BASE's weights INTO m (m has one extra parameter, c_agree, which
    # must stay at its zero init) — loading the other way round would leave
    # the two models with different random weights and the test would fail
    # for the wrong reason.
    m.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        assert torch.allclose(base(x)["final_log_probs"],
                              m(x)["final_log_probs"], atol=1e-6)
        # a non-zero scale must move the output
        m.regions[0].c_agree.fill_(2.0)
    assert not torch.allclose(base(x)["final_log_probs"],
                              m(x)["final_log_probs"], atol=1e-6)
    m.train()
    with torch.no_grad():
        m.regions[0].c_agree.zero_()   # leaf param: in-place needs no_grad
    out = m(x)
    out["final_log_probs"].exp().gather(
        -1, torch.zeros(2, 12, 1, dtype=torch.long)).sum().backward()
    g = m.regions[0].c_agree.grad
    assert g is not None and torch.isfinite(g).all()


def test_reread_is_inert_at_init_causal_and_trains():
    """reread adds a causal span-8 read of the head block's own output between
    blocks. Three things must hold or the mechanism is a lie: the zero-init
    out projection makes step 0 bit-identical to the arm without it, the read
    cannot see the future (causality is the whole reason it is a window and
    not a full read), and the new parameters receive gradient."""
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=1, n_dots=3, head_post=32, head_post_layers=3)
    base = build_iecnn(_cfg(**kw), 64).eval()
    m = build_iecnn(_cfg(reread=2, **kw), 64).eval()
    m.load_state_dict(base.state_dict(), strict=False)
    x = torch.randint(0, 64, (2, 12))
    with torch.no_grad():
        assert torch.allclose(base(x)["final_log_probs"],
                              m(x)["final_log_probs"], atol=1e-6)

    # causality: perturb ONE late position and check that no earlier position
    # moves (the read is causal and the rest of the model is per position).
    reg = m.regions[0]
    with torch.no_grad():
        torch.nn.init.normal_(reg.rr_o.weight, std=0.5)   # switch the read on
        h = torch.randn(2, 12, reg.d)
        h2 = h.clone()
        # NOT a uniform +1: the read starts with a LayerNorm, which is
        # invariant to a constant shift along d (it subtracts the mean), so a
        # uniform perturbation is the one change the read cannot see.
        h2[:, -1] += torch.randn(reg.d)
        a, b = reg._reread(h), reg._reread(h2)
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-6)
        assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-6)

    m.train()
    out = m(x)
    out["final_log_probs"].exp().gather(
        -1, torch.zeros(2, 12, 1, dtype=torch.long)).sum().backward()
    for p in (reg.rr_q.weight, reg.rr_k.weight, reg.rr_v.weight,
              reg.rr_o.weight):
        assert p.grad is not None and torch.isfinite(p.grad).all()
    assert reg.rr_o.weight.grad.abs().sum() > 0


def test_generate_bytes_is_deterministic_and_greedy_is_argmax():
    """The long runs report what the model says, so the decoder has to agree
    with the logits the harness scores, reproduce for a fixed seed, and leave
    the model untouched (eval mode, no gradients, prompt not modified).

    It is also the only decode path shared by the harness and the probes, so
    this test is what stops a probe from sampling differently than a run.
    """
    from models.generate import generate_bytes
    m = build_iecnn(_cfg(global_read=True, per_dot_logits=False,
                         n_passes=1, n_dots=3, head_post=32,
                         head_post_layers=2), 64).eval()
    prompt = torch.randint(0, 64, (1, 6))
    lp = m(prompt)["final_log_probs"][:, -1]
    g = generate_bytes(m, prompt, 10, mode="greedy", ctx=12)
    assert len(g) == 10 and all(0 <= b < 64 for b in g)
    assert g[0] == int(lp.argmax(-1))          # greedy == the scored argmax

    p0 = prompt.clone()
    a = generate_bytes(m, prompt, 16, mode="sample", seed=11, ctx=12)
    b = generate_bytes(m, prompt, 16, mode="sample", seed=11, ctx=12)
    assert a == b and torch.equal(prompt, p0)
    assert all(q.grad is None for q in m.parameters())
    assert m.training is False

    # a prompt longer than the context still decodes (truncated, not an error)
    assert len(generate_bytes(m, torch.randint(0, 64, (1, 20)), 4, ctx=8)) == 4
    # and the shape contract is enforced rather than guessed
    try:
        generate_bytes(m, torch.zeros(4, dtype=torch.long), 1)
    except ValueError:
        pass
    else:
        raise AssertionError("expected ValueError for a 1-D prompt")


def test_reread_span_ladder_defaults_identical_and_is_causal():
    """reread_spans gives each reread site its own exact horizon (2/8/32).

    The default is uniform `reread_span` at every site, so the flag must leave
    an arm bit-identical when it is not set -- this is a re-allocation of the
    existing read, not a new mechanism. It also must not change the parameter
    count (rr_* stay shared across sites) and it must stay causal at the
    longest rung, since span 32 is a real window and not a masked full read.
    """
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=1, n_dots=3, head_post=32, head_post_layers=3)
    x = torch.randint(0, 64, (2, 12))

    # 1) default and explicit-uniform are the same arm
    a = build_iecnn(_cfg(reread=2, **kw), 64).eval()
    b = build_iecnn(_cfg(reread=2, reread_spans=(8, 8), **kw), 64).eval()
    b.load_state_dict(a.state_dict())
    with torch.no_grad():
        assert torch.allclose(a(x)["final_log_probs"],
                              b(x)["final_log_probs"], atol=1e-6)
    assert a.regions[0].reread_spans == (8, 8) == b.regions[0].reread_spans

    # 2) a short ladder repeats its last rung, and the ladder adds no params
    lad = build_iecnn(_cfg(reread=3, reread_spans=(2, 8, 32), **kw), 64)
    assert lad.regions[0].reread_spans == (2, 8, 32)
    assert lad.regions[0]._reread_span_at(0) == 2
    assert lad.regions[0]._reread_span_at(2) == 32
    short = build_iecnn(_cfg(reread=3, reread_spans=(4,), **kw), 64)
    assert short.regions[0].reread_spans == (4, 4, 4)
    ref = build_iecnn(_cfg(reread=3, **kw), 64)
    assert (sum(p.numel() for p in lad.parameters())
            == sum(p.numel() for p in ref.parameters()))

    # 3) the ladder actually reads different horizons (switch the read on,
    #    because rr_o is zero-initialised and would hide every difference)
    lad.load_state_dict(ref.state_dict())
    with torch.no_grad():
        for reg in (lad.regions[0], ref.regions[0]):
            torch.nn.init.normal_(reg.rr_o.weight, std=0.5)
        assert not torch.allclose(lad(x)["final_log_probs"],
                                  ref(x)["final_log_probs"], atol=1e-4)

    # 4) causality at the longest rung, and end to end through the ladder
    reg = lad.regions[0]
    h = torch.randn(2, 12, reg.d)
    h2 = h.clone()
    h2[:, -1] += torch.randn(reg.d)     # not a uniform shift: LN ignores those
    with torch.no_grad():
        p, q = reg._reread(h, 32), reg._reread(h2, 32)
        assert torch.allclose(p[:, :-1], q[:, :-1], atol=1e-6)
        assert not torch.allclose(p[:, -1], q[:, -1], atol=1e-6)
        x2 = x.clone()
        x2[:, -1] = (x2[:, -1] + 1) % 64
        lad.eval()
        l1, l2 = lad(x)["final_log_probs"], lad(x2)["final_log_probs"]
        assert torch.allclose(l1[:, :-1], l2[:, :-1], atol=1e-5)

    # 5) the shared read still trains through a ladder
    lad.train()
    lad(x)["final_log_probs"].exp().gather(
        -1, torch.zeros(2, 12, 1, dtype=torch.long)).sum().backward()
    for p_ in (reg.rr_q.weight, reg.rr_k.weight, reg.rr_v.weight,
               reg.rr_o.weight):
        assert p_.grad is not None and torch.isfinite(p_.grad).all()
    assert reg.rr_o.weight.grad.abs().sum() > 0


def test_gather_windows_plain_flag_matches_the_tensor_check():
    """The compile-safe `plain` argument must be a pure optimisation: for the
    default sliding window it has to return bit-identical windows to the eager
    tensor comparison, and for a non-default window it must NOT take the fast
    path. (The tensor comparison itself is what made torch.compile fail with a
    data-dependent guard, so it can never come back into a trained forward.)"""
    from models.memory import gather_windows
    x = torch.randn(2, 9, 4)
    for span in (3, 8):
        offs = torch.arange(span) - span + 1
        a, va = gather_windows(x, span, offs, None)   # eager tensor check
        b, vb = gather_windows(x, span, offs, True)   # compile-safe path
        assert torch.equal(a, b) and torch.equal(va, vb)
        odd = offs.clone()
        odd[0] = offs[0] - 1        # one slot further back: a real difference
        c, _ = gather_windows(x, span, odd, False)
        assert not torch.equal(a, c)


def test_hillis_scan_matches_cumsum_causally_and_trains():
    """scan_mode="hillis" must be the SAME inclusive prefix sum as cumsum (up
    to float association) and stay causal.

    It exists because Inductor's CUDA SplitScan lowering crashes on the
    cumsum in the global read, which forces compiled dot arms to carry an
    eager boundary (`break_scan`) and costs them half the compile win. The
    Hillis-Steele form is pad+add only, so it can be compiled inline; this
    test is the guard that "inline" does not change the maths.
    """
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=1, n_dots=3, d_q=12, d_v=20, head_post=32,
              head_post_layers=2, global_decay=True)
    m0 = build_iecnn(_cfg(scan_mode="cumsum", **kw), 64).eval()
    m1 = build_iecnn(_cfg(scan_mode="hillis", **kw), 64).eval()
    m1.load_state_dict(m0.state_dict())
    d0, d1 = m0.regions[0].dots, m1.regions[0].dots

    # 1) the scan helper itself, every small T and both rank shapes
    for t in (1, 2, 3, 7, 8, 12):
        x = torch.randn(2, t, 5)
        assert torch.allclose(d0._prefix_sum(x), d1._prefix_sum(x), atol=1e-5)
        y = torch.randn(2, t, 3, 4)
        assert torch.allclose(d0._prefix_sum(y), d1._prefix_sum(y), atol=1e-5)

    # 2) causality: a change at position 4 cannot reach positions < 4
    x = torch.randn(1, 9, 3)
    a = d1._prefix_sum(x)
    x2 = x.clone()
    x2[:, 4] += 100.0
    b = d1._prefix_sum(x2)
    assert torch.allclose(a[:, :4], b[:, :4], atol=1e-5)
    assert not torch.allclose(a[:, 4:], b[:, 4:], atol=1e-5)

    # 3) the whole global read agrees and the scan carries gradients
    fk = torch.randn(2, 12, d1.d_q)
    fv = torch.randn(2, 12, d1.d_v)
    with torch.no_grad():
        c0 = d0._global_ctx(fk, fv, 0)
        c1 = d1._global_ctx(fk, fv, 0)
    assert torch.allclose(c0, c1, atol=1e-4)
    fk.requires_grad_(True)
    fv.requires_grad_(True)
    d1._global_ctx(fk, fv, 0).sum().backward()
    assert fk.grad is not None and torch.isfinite(fk.grad).all()
    assert fv.grad is not None and torch.isfinite(fv.grad).all()
    assert fv.grad.abs().sum() > 0


def test_k_lag_shifts_only_the_key_stream_and_stays_causal():
    """k_lag=1 is the induction circuit: key[s] = x[s-1], value[s] = x[s].

    Two things have to hold for that sentence to be true rather than a
    comment: only the KEY stream moves (values must stay at their own
    position, otherwise the read returns the matched token instead of its
    successor), and the shift must be causal (perturbing the last token may
    not change an earlier position's output).
    """
    kw = dict(global_read=True, per_dot_logits=False, head_last_only=True,
              n_passes=1, n_dots=3, d_q=12, d_v=20, head_post=32,
              head_post_layers=2, query_from_input=True, global_exact=True)
    m0 = build_iecnn(_cfg(k_lag=0, **kw), 64).eval()
    m1 = build_iecnn(_cfg(k_lag=1, **kw), 64).eval()
    m1.load_state_dict(m0.state_dict())
    reg = m1.regions[0]

    rows = torch.randn(2, 12, reg.d)
    with torch.no_grad():
        k1, v1 = reg._full_kv(rows)
        k0, v0 = m0.regions[0]._full_kv(rows)
        want_k = reg.w_k(torch.cat([torch.zeros(2, 1, reg.d), rows[:, :-1]],
                                   dim=1))
        assert torch.allclose(k1, want_k, atol=1e-6)
        # the value stream is NOT shifted: v1[t] must be the row at t
        assert torch.allclose(v1, reg.w_v(rows), atol=1e-6)
        assert torch.allclose(v0, v1, atol=1e-6)
        # and the key stream really did move
        assert not torch.allclose(k1[1:], k0[1:])   # position 0 is zeroed

    # causality: k_lag only ever looks BACK, so a change at the last position
    # cannot reach an earlier position's log-probs.
    with torch.no_grad():
        reg.w_qi.normal_(std=0.5)                  # switch the query on
    x = torch.randint(0, 64, (2, 12))
    x2 = x.clone()
    x2[:, -1] = (x2[:, -1] + 1) % 64
    with torch.no_grad():
        a = m1(x)["final_log_probs"]
        b = m1(x2)["final_log_probs"]
        assert torch.allclose(a[:, :-1], b[:, :-1], atol=1e-5)
        assert not torch.allclose(a[:, -1], b[:, -1], atol=1e-5)

    # the query projection (the part that turns k_lag into induction) trains
    m1.train()
    out = m1(x)["final_log_probs"]
    out.exp().gather(-1, torch.zeros(2, 12, 1, dtype=torch.long)).sum().backward()
    assert reg.w_qi.grad is not None and torch.isfinite(reg.w_qi.grad).all()
    assert reg.w_qi.grad.abs().sum() > 0
