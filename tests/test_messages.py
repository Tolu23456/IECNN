"""Optional within-region messages (spec 05): ring topology + one-pass delay."""
import torch

from models.messages import MessageChannel, ring_neighbors


def test_switch_off_identical_to_no_message_model():
    """The switch-off D9 identity, structurally:

    (1) a zeroed channel produces exactly-zero messages (announced channel
        with no signal = the thin channel at zero),
    (2) the Region passes msg None at pass 1 (no matter how the channel is
        configured) and zeros when a message would be un-set,
    (3) non-zero channel weights must change pass>=2 outputs (messages are
        actually read at p+1), while pass-1 outputs are untouched.
    """
    import copy
    from config import load_config
    from models.iecnn import build_iecnn
    from utils import set_seed

    set_seed(7)
    cfg = copy.deepcopy(load_config("configs/e1_weights_passes.yaml", "tiny.yaml"))
    cfg["model"]["use_messages"] = True
    model = build_iecnn(cfg, 512)
    region = model.regions[0]
    ch = region.messages
    assert ch is not None

    # (1) zeroed channel -> exactly zero messages in
    for lin in (ch.w_write, ch.wq, ch.wk):
        torch.nn.init.zeros_(lin.weight)
    states = torch.randn(3, 4, region.n_dots, region.dots.d_s)
    m_in = ch(states, training=False)
    assert (m_in == 0).all(), "zeroed channel still emits messages"

    # (2) the no-message configuration has no channel at all
    set_seed(7)
    cfg_off = copy.deepcopy(load_config("configs/e1_weights_passes.yaml", "tiny.yaml"))
    model_off = build_iecnn(cfg_off, 512)
    assert model_off.regions[0].messages is None

    # (3) a live channel changes pass 2+ but cannot touch pass 1 (one-pass delay)
    set_seed(7)
    cfg_live = copy.deepcopy(load_config("configs/e1_weights_passes.yaml", "tiny.yaml"))
    cfg_live["model"]["use_messages"] = True
    model_live = build_iecnn(cfg_live, 512)
    model_live.eval()
    model_zero = copy.deepcopy(model_live)
    torch.nn.init.zeros_(model_zero.regions[0].messages.w_write.weight)
    ch.zero_grad()
    x = torch.randint(0, 512, (1, 24))
    live = model_live(x)["region_outputs"][0]["per_pass"]
    zero = model_zero(x)["region_outputs"][0]["per_pass"]
    assert torch.equal(live[0]["log_pi"], zero[0]["log_pi"]), \
        "pass 1 must not see messages (they write at p, read at p+1)"
    assert not torch.equal(live[1]["log_pi"], zero[1]["log_pi"]), \
        "pass 2 must see the pass-1 messages"


def test_dot_never_reads_own_message():
    """No dot's incoming message CONTAINS its own write: values flow only from
    ring neighbours (mask has no self-loops). With queries zeroed, attention is
    uniform and dot 0's read is exactly the mean of its NEIGHBOURS' keys.
    """
    ch = MessageChannel(d_s=16, d_m=8, n_dots=6, radius=1)
    torch.nn.init.zeros_(ch.wq.weight)
    states = torch.randn(2, 5, 6, 16)
    m_in = ch(states, training=False)
    assert m_in.shape == (2, 5, 6, 8)
    assert (~ch.read_mask().diagonal()).all(), "self-loops exist in ring mask"

    neighbors = ch.read_mask()                       # [N, N] ring, radius 1
    keys = ch.wk(torch.tanh(ch.w_write(states)))      # [..., N, d_m] writer side
    for k in range(6):
        idx = torch.nonzero(neighbors[k]).flatten()
        expected = keys[:, :, idx, :].mean(dim=2)
        assert torch.allclose(m_in[..., k, :], expected, atol=1e-6), k

    # dot 0 writes (neighbors read it), but its own write never enters its read:
    states2 = states.clone()
    states2[..., 0, :] = torch.randn(2, 5, 16)
    m2 = ch(states2, training=False)
    assert not torch.allclose(m_in[..., 1, :], m2[..., 1, :]), "neighbor must receive the write"


def test_messages_cannot_affect_same_pass():
    """Write at pass p, read at p+1: calling the channel ONCE per pass and
    feeding the result only at the next pass guarantees the delay. The unit:
    messages read at pass p come from states of pass p-1 (structural)."""
    ch = MessageChannel(d_s=16, d_m=8, n_dots=6, radius=1)
    # the channel returns 'to use at the NEXT pass', i.e. from CURRENT states
    s = torch.randn(1, 1, 6, 16)
    m_next = ch(s, training=False)
    assert m_next.shape[-1] == 8
    # if states change, and we feed OLD messages, this pass cannot see it:
    old = ch(torch.zeros_like(s), training=False)
    s2 = s * 2 + 1
    s2_out = ch(s2, training=False)
    assert not torch.allclose(old, s2_out)


def test_ring_wraps_correctly():
    n = ring_neighbors(6, radius=2)
    expected = {
        0: {4, 5, 1, 2},
        1: {5, 0, 2, 3},
        3: {1, 2, 4, 5},
    }
    for k, cols in expected.items():
        got = set(torch.nonzero(n[k]).flatten().tolist())
        assert got == cols, (k, got)
    n1 = ring_neighbors(1, radius=2)
    assert not n1.any(), "single dot has no neighbours"
