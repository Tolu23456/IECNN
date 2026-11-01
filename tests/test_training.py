"""Tests for the training loop and helpers (spec 06, section 11)."""
import math

import numpy as np
import torch

from train.evaluate import evaluate_bpc, bpc_from_nll
from train.schedules import lr_at, correlation_lambda, use_hard_batch
from data.dataset import pack_sequences


def test_single_step_reduces_loss_on_tiny_fixed_batch():
    """One optimizer step lowers the loss on a fixed tiny batch (overfit sanity)."""
    from models.baselines import GRULanguageModel

    torch.manual_seed(0)
    model = GRULanguageModel(64, d=16, hidden=32)
    x = torch.randint(0, 64, (4, 16))
    y = torch.randint(0, 64, (4, 16))
    opt = torch.optim.AdamW(model.parameters(), lr=1e-2)

    def current_loss():
        with torch.no_grad():
            return float(torch.nn.functional.nll_loss(
                model(x)["final_log_probs"].reshape(-1, 64), y.reshape(-1)).item())

    l0 = current_loss()
    loss = torch.nn.functional.nll_loss(
        model(x)["final_log_probs"].reshape(-1, 64), y.reshape(-1))
    opt.zero_grad(); loss.backward(); opt.step()
    assert current_loss() < l0, f"loss did not decrease: {l0} -> {current_loss()}"


class _UniformModel:
    """A model with a fixed uniform distribution (hand-check the bpc pipeline)."""

    context_length = 8

    def __init__(self, vocab: int):
        self.v = vocab
        self.param = torch.nn.Parameter(torch.zeros(1))

    def parameters(self):
        return iter([self.param])

    def forward(self, x, hard_mode=False):
        logp = torch.full((x.shape[0], x.shape[1], self.v), -(math.log(self.v)))
        return {"final_log_probs": logp}

    def __call__(self, x, hard_mode=False):
        return self.forward(x, hard_mode)


def _tiny_ubm(tmp_path):
    """A UBM whose 64 'characters' are single bytes in the printable range."""
    from UBM import UBM, BaseRecord
    ubm = UBM()
    recs = []
    start = len(ubm.bases)
    for i, ch in enumerate("abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789+-"):
        recs.append(BaseRecord(id=start + i, bytes=ch.encode()))
    ubm.add_bases(recs)
    return ubm

def test_bpc_hand_computed_example(tmp_path):
    """evaluate_bpc on a uniform model equals n*log(V) / (ln2 * chars), verifiable by hand."""
    text = "abc def ghi jkl"
    ubm = _tiny_ubm(tmp_path)
    model = _UniformModel(len(ubm.bases))
    ids = ubm.encode(text)
    n_tokens = len(ids) - 1                      # predicted events
    # every token here expands to exactly 1 char, so chars = n_tokens (round trip)
    chars = sum(len(ubm.expand(int(i))) for i in ids[1:])
    bpc = evaluate_bpc(model, ubm, text, seq_len=8)
    expected = bpc_from_nll(n_tokens * math.log(len(ubm.bases)), chars)
    # tolerance covers float32 summation-order rounding, not more
    assert math.isclose(bpc, expected, rel_tol=1e-6), f"bpc {bpc} vs expected {expected}"
    # and determinism: two identical calls give identical bpc
    assert evaluate_bpc(model, ubm, text, seq_len=8) == bpc


def test_pack_sequences_shift_and_disjoint():
    ids = np.arange(0, 30, dtype=np.int32)
    x, y = pack_sequences(ids, seq_len=8)
    assert x.shape == y.shape and x.shape[0] == 3
    assert (x[:, 1:] == y[:, :-1]).all()         # inputs and targets align inside a window
    assert x[0, 0] == 0 and x[1, 0] == 8         # windows are disjoint


def test_lr_warmup_cosine():
    peak, warmup, total = 1e-3, 10, 100
    assert abs(lr_at(0, total, warmup, peak) - peak / warmup) < 1e-6
    assert abs(lr_at(warmup, total, warmup, peak) - peak) < 1e-9     # ramp ends AT peak
    assert lr_at(warmup - 2, total, warmup, peak) < peak            # ramp is linear up
    mid = lr_at((warmup + total) // 2, total, warmup, peak)
    assert mid < peak
    assert lr_at(total, total, warmup, peak) < 1e-4                 # decays to ~0


def test_correlation_lambda_ramp():
    assert correlation_lambda(0, 1000, 0.01) == 0.0
    assert abs(correlation_lambda(500, 1000, 0.01) - 0.005) < 1e-9
    assert abs(correlation_lambda(1000, 1000, 0.01) - 0.01) < 1e-12


def test_hard_batch_fraction():
    import random as _r
    rng = _r.Random(0)          # ONE generator driven through all the calls
    early = sum(use_hard_batch(s, 100, 0.7, 0.5, rng) for s in range(50))
    late = sum(use_hard_batch(s, 100, 0.7, 0.5, rng) for s in range(70, 100))
    assert early == 0 and 0 < late < 30


def test_ladder_rung_configs_all_load():
    from config import load_config
    expected_keys = ["model", "training"]
    for fname in ["base", "tiny", "b0_gru", "b1_transformer", "e0_plain_ensemble",
                  "e1_weights_passes", "e2_regions", "e3_messages",
                  "e4_decorrelation", "e5_ubm_promotion"]:
        cfg = load_config(f"configs/{fname}.yaml", "tiny.yaml")
        for key in expected_keys:
            assert key in cfg, f"{fname} missing {key}"
    e0 = load_config("configs/e0_plain_ensemble.yaml", "tiny.yaml")
    assert e0["model"]["n_passes"] == 1 and not e0["model"]["use_weights"]
