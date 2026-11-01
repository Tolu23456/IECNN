"""Shared memory tests (spec 02, section 9)."""
import torch

from config import load_config
from models.iecnn import build_iecnn
from models.memory import banded_causal_mask, gather_windows, next_region_rows
from utils import set_seed


def _model(name="e2_regions", overlay="tiny.yaml"):
    set_seed(0)
    cfg = load_config(f"configs/{name}.yaml", *( [overlay] if overlay else [] ))
    return build_iecnn(cfg, 512)


def test_future_token_change_changes_no_visible_row():
    """Region 1 rows are embeddings of x_j: changing x_{t+1} cannot touch any
    row with index <= t."""
    x = torch.randint(0, 512, (2, 40))
    x2 = x.clone()
    x2[:, 30:] = torch.randint(0, 512, (2, 10))
    model = _model()
    with torch.no_grad():
        rows1 = model.embed(x)
        rows2 = model.embed(x2)
    assert (rows1[:, :30] == rows2[:, :30]).all(), "embedding rows leaked the future"


def test_window_edge_masked_rows_have_zero_attention():
    """At small t, out-of-window rows get exactly zero attention weight."""
    model = _model()
    region = model.regions[0]
    x = torch.randint(0, 512, (1, 16))
    with torch.no_grad():
        rows = model.embed(x)
        wind_k, wind_v, valid = region._window_memory(rows, rows.device)
        q = region.dots._sel(0, region.dots.q)                     # [N, d_q]
        scores = torch.einsum("bfwj,nj->bfnw", wind_k, q) / q.shape[-1] ** 0.5
        masked = scores.masked_fill(~valid[None, :, None, :], float("-inf"))
        attn = torch.softmax(masked, dim=-1)
    # positions t < span-1 must put zero mass on (any) invalid slots
    small_t = attn[0, 0]              # t=0: only slot span-1 (itself) visible
    assert (small_t[..., :-1] < 1e-12).all(), "attention leaked to masked rows"
    assert (small_t[..., -1] > 0).any()
    # every row of the (masked-softmax) attention sums to 1 over the window
    row_sums = attn.sum(-1)
    assert (row_sums - 1).abs().max().item() < 1e-5


def test_parallel_training_equals_token_by_token_inference():
    """Running all positions in parallel must equal one-position-at-a-time
    reads on the same text (within fp32 tolerance)."""
    model = _model(name="e1_weights_passes")
    model.eval()
    torch.manual_seed(3)
    x = torch.randint(0, 512, (1, 24))
    with torch.no_grad():
        full = model(x)["final_log_probs"]
        # token-by-token: feed prefixes of length 2..T and read the last logit
        step_ids = []
        for t in range(2, x.shape[1] + 1):
            out = model(x[:, :t])
            step_ids.append(out["final_log_probs"][0, -1:])
        serial = torch.cat(step_ids, dim=0)                # [T-1, V]
    diff = (full[0, 1:] - serial).abs().max().item()
    assert diff < 1e-3, f"parallel vs serial mismatch: {diff}"


def test_region_chaining_projection_ablation_changes_loss():
    """Zeroing the Proj_r path of the chained memory must change the output."""
    model = _model()
    model.eval()
    x = torch.randint(0, 512, (1, 24))
    with torch.no_grad():
        base = model(x)["final_log_probs"]
        for proj in model.chain_proj:
            proj.weight.zero_()
        ablated = model(x)["final_log_probs"]
    assert not torch.allclose(base, ablated), "Proj_r ablation changed nothing"
