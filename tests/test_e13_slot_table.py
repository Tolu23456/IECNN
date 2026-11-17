"""Rung E13 slot-table tests (Tolu's mechanism: per-dot persistent slots)."""
import torch

from config import load_config
from models.dot import DotPopulation, gru_input_dim
from models.region import Region


def _cfg():
    cfg = load_config("configs/e13_slot_table.yaml", "tiny.yaml")
    cfg["model"]["n_regions"] = 1
    cfg["model"]["n_dots"] = 4
    cfg["model"]["n_passes"] = 2
    return cfg


def _pop(**over):
    return DotPopulation(4, 32, 8, 16, 32, 33, n_passes=2, seed=3, **over)


def test_slots_change_gru_input_dim_and_off_is_unchanged():
    base = gru_input_dim(16, 32, 0)
    p_on = _pop(slot_slots=4, slot_dim=32)
    p_off = _pop()
    assert p_on.S == 4 and p_on.d_w == 32
    assert p_on.in_dim == base + 32          # slot read joins the GRU input
    assert p_off.in_dim == base and p_off.S == 0
    assert p_off.sv is None and p_off.sr is None


def test_table_starts_empty_and_write_fills_it():
    """Sequence boundary: fresh table is zero; a gated write makes it not."""
    p = _pop(slot_slots=4, slot_dim=16)
    b, t = 2, 6
    m0 = p.empty_slots(b, torch.device("cpu"), torch.float32)
    assert m0.shape == (b, p.N, p.S, p.d_w) and (m0 == 0).all()
    state = torch.randn(b, t, p.N, p.d_s)
    with torch.no_grad():                    # push every write gate to ~1
        p.sgb.fill_(10.0)
    m1 = p.update_slots(state, m0, 0)
    assert (m1.abs() > 0).all()              # every slot received a write
    # decay: writing zeros on top of m1 shrinks it toward 0, never exactly 0
    m2 = p.update_slots(torch.zeros_like(state), m1, 0)
    assert (m2.abs() < m1.abs()).all() and (m2.abs() > 0).any()


def test_writes_are_detached_read_is_not():
    """The fast-weights invariant: grads reach the read path only."""
    p = _pop(slot_slots=4, slot_dim=16)
    b, t = 2, 5
    m0 = p.empty_slots(b, torch.device("cpu"), torch.float32)
    state = torch.randn(b, t, p.N, p.d_s, requires_grad=True)
    m1 = p.update_slots(state, m0, 0)
    r = p.slot_context(m1, 0)
    r.sum().backward()
    assert p.sr.grad is not None and p.sr.grad.abs().sum() > 0
    assert p.sg.grad is None and p.sv.grad is None   # writes: no grad ever
    assert state.grad is None                        # nor through the state


def test_region_forward_with_slots_both_head_modes():
    """Full region pass loop: slots ride along in hidden AND prob modes."""
    for mode in ("hidden", "prob"):
        cfg = _cfg()
        cfg["model"]["head_mode"] = mode
        r = Region(0, cfg, 33)
        assert r.dots.S == 4
        out = r(torch.randn(2, 8, r.d), torch.randn(33, r.d))
        assert torch.isfinite(out["log_b"]).all()
        # 2 passes -> the table was updated between them (nonzero by pass 2)
        assert (r.dots.sr is not None)


def test_fast_and_reference_paths_agree_with_slots():
    """The slot read must be identical in the optimized and reference paths."""
    torch.manual_seed(0)
    kw = dict(slot_slots=4, slot_dim=16)
    pf = _pop(fast_math=True, **kw)
    pr = _pop(fast_math=False, **kw)
    for (a, b) in zip(pf.parameters(), pr.parameters()):
        b.data.copy_(a.data)
    bsz, t, w = 2, 4, 3
    wind_k = torch.randn(bsz, t, w, 8)
    wind_v = torch.randn(bsz, t, w, 16)
    valid = torch.ones(t, w, dtype=torch.bool)
    state = torch.randn(bsz, t, 4, 32)
    h_pool = torch.randn(bsz, t, 4, 32)
    err = torch.randn(bsz, t, 4, 3)
    tbl = torch.randn(bsz, 4, 4, 16)
    emb = torch.randn(33, 32)
    of = pf(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb,
            want_log_pi=True, slot_table=tbl)
    orf = pr(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb,
             want_log_pi=True, slot_table=tbl)
    assert torch.allclose(of.log_pi, orf.log_pi, atol=1e-5)
    assert torch.allclose(of.state, orf.state, atol=1e-5)


def test_slots_off_matches_plain_path_exactly():
    """slot_slots=0 must be bit-identical to the pre-e13 behavior."""
    torch.manual_seed(1)
    pf = _pop(fast_math=True)
    pr = _pop(fast_math=False)
    for (a, b) in zip(pf.parameters(), pr.parameters()):
        b.data.copy_(a.data)
    bsz, t, w = 2, 4, 3
    wind_k = torch.randn(bsz, t, w, 8)
    wind_v = torch.randn(bsz, t, w, 16)
    valid = torch.ones(t, w, dtype=torch.bool)
    state = torch.randn(bsz, t, 4, 32)
    h_pool = torch.randn(bsz, t, 4, 32)
    err = torch.randn(bsz, t, 4, 3)
    emb = torch.randn(33, 32)
    of = pf(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb,
            want_log_pi=True, slot_table=None)
    orf = pr(wind_k, wind_v, valid, h_pool, err, 0, None, state, emb,
             want_log_pi=True, slot_table=None)
    assert torch.allclose(of.log_pi, orf.log_pi, atol=1e-5)


def test_config_loads_e13():
    cfg = load_config("configs/e13_slot_table.yaml", "tiny.yaml")
    m = cfg["model"]
    assert m["slot_slots"] == 4 and m["slot_dim"] == 32
    assert m["head_mode"] == "hidden"        # built on e11b (shared head)
    assert cfg["consensus"]["rho"] == 0.0    # ...and no-trust
    assert cfg["name"] == "e13_slot_table"
