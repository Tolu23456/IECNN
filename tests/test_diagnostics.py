"""Spec-08 §6 diagnostic suite (built 2026-11-14): shared collector alignment,
ablation damage + Gini, dispersion AUROC, mode gap, checkpoint loader guard."""
import json
import math

import pytest
import torch

from config import load_config
from utils import set_seed


def _tiny_iecnn(vocab_size, seed=0):
    from models.iecnn import build_iecnn
    set_seed(seed)
    cfg = load_config("configs/e0_plain_ensemble.yaml", "tiny.yaml")
    m = build_iecnn(cfg, vocab_size)
    m.eval()
    return m


def _tiny_ubm():
    from UBM import UBM
    return UBM()          # 256 bytes + 8 specials: every ASCII char is a base


def test_gini_and_auroc_math():
    from diagnostics.common import auroc, gini
    assert gini([1.0, 1.0, 1.0]) == 0.0                    # identical = no spread
    assert gini([0.0, 0.0, 0.0]) == 0.0
    assert gini([0.0, 0.0, 3.0]) > 0.5                     # one dot matters
    assert auroc([1, 2, 3, 4], [0, 0, 1, 1]) == 1.0        # perfect separator
    assert auroc([1, 2, 3, 4], [1, 1, 0, 0]) == 0.0
    assert auroc([1, 1], [1, 0]) == 0.5
    assert auroc([1.0], []) is None                        # empty class = None


def test_token_types():
    from diagnostics.common import token_types
    ubm = _tiny_ubm()
    ids = ubm.encode("Alice, 42 wonderland")
    types = token_types(ubm, torch.tensor(ids))
    assert types[0] == "capitalized"                       # 'Alice'
    assert "punct" in types                                # ','
    assert "digit" in types                                # '42'
    assert "whitespace" in types


def test_collect_matches_evaluate_bpc_exactly():
    """collect()'s flattened NLL/char accounting must reproduce the headline
    evaluate_bpc number on the same positions — the diagnostics sit on top of
    it, so any drift here silently corrupts every spec-08 metric."""
    from diagnostics.common import collect
    from train.evaluate import evaluate_bpc
    ubm = _tiny_ubm()
    m = _tiny_iecnn(len(ubm.bases))
    text = "the quick brown fox jumps over the lazy dog again and again"
    bpc = evaluate_bpc(m, ubm, text, seq_len=8)
    c = collect(m, ubm, text, seq_len=8, need=())
    mine = float(c["nll"].sum()) / (math.log(2) * float(c["expand_len"].sum()))
    assert abs(mine - bpc) < 1e-6, (mine, bpc)
    assert c["n_positions"] == len(ubm.encode(text)) - 1


def test_ablation_gini_zero_for_identical_dots():
    """Dots with identical distributions and uniform weights: masking any one
    costs the same -> damage vector flat -> Gini 0 (the collapse warning)."""
    from diagnostics.ablation import compute
    m_pos, n, v = 6, 4, 11
    g = torch.Generator().manual_seed(3)
    one = torch.randn(m_pos, v, generator=g)               # shared distribution
    coll = {"pass_index": 0,
            "per_pass": {0: {"log_pi": one[:, None, :].expand(m_pos, n, v).clone(),
                             "weights": torch.full((m_pos, n), 1.0 / n)}},
            "ys": torch.randint(0, v, (m_pos,), generator=g),
            "n_positions": m_pos, "n_chars": float(m_pos)}
    out = compute(coll)
    assert out["gini"] == 0.0
    assert max(out["damage_nats_per_token"]) - min(out["damage_nats_per_token"]) < 1e-6


def test_ablation_flags_the_only_useful_dot():
    """One dot near-perfect, the rest garbage: masking the good dot must be
    by far the most damaging (wide spread, high Gini)."""
    from diagnostics.ablation import compute
    m_pos, n, v = 8, 4, 11
    g = torch.Generator().manual_seed(5)
    y = torch.randint(0, v, (m_pos,), generator=g)
    junk = torch.full((m_pos, v), -math.log(v))            # uniform junk
    good = torch.full((m_pos, v), -30.0)
    good[torch.arange(m_pos), y] = -0.05                   # near one-hot truth
    log_pi = junk[:, None, :].repeat(1, n, 1).clone()
    log_pi[:, 0, :] = good                                 # dot 0 is the star
    coll = {"pass_index": 0,
            "per_pass": {0: {"log_pi": log_pi,
                             "weights": torch.full((m_pos, n), 1.0 / n)}},
            "ys": y, "n_positions": m_pos, "n_chars": float(m_pos)}
    out = compute(coll)
    assert out["damage_nats_per_token"][0] == max(out["damage_nats_per_token"])
    assert out["most_important_dot"] == 0
    assert out["gini"] > 0.2


def test_dispersion_runs_and_reports_auroc_range():
    from diagnostics.dispersion import compute
    from diagnostics.common import collect
    ubm = _tiny_ubm()
    set_seed(1)
    from models.iecnn import build_iecnn
    m = build_iecnn(load_config("configs/e1_weights_passes.yaml", "tiny.yaml"),
                    len(ubm.bases))          # 3 passes: exercises the halting sim
    m.eval()
    text = "to be or not to be that is the question whether nobler"
    c_first = collect(m, ubm, text, region="first", passes="first",
                      need=("log_pi", "weights"), seq_len=8)
    c_all = collect(m, ubm, text, region="first", passes="all",
                    need=("log_b",), seq_len=8)
    out = compute(c_first, c_all)
    aroc = out["auroc_dispersion_vs_final_wrong"]
    assert aroc is None or 0.0 <= aroc <= 1.0
    assert out["n_positions"] == c_first["n_positions"] > 0
    assert isinstance(out["halting_simulation"], list)


def test_mode_gap_is_small_on_a_soft_trained_model():
    """At init, weights are near-uniform so top-4-of-4... renormalized hard
    weights ≈ soft weights -> |mode gap| must be small, and either sign is
    reportable (spec 08 §6.5: report both bpc and the gap)."""
    from diagnostics.mode_gap import run
    from train.evaluate import mode_gap
    ubm = _tiny_ubm()
    m = _tiny_iecnn(len(ubm.bases), seed=2)
    text = "alpha beta gamma delta"
    mg = mode_gap(m, ubm, text)
    assert abs(mg["mode_gap"]) <= 0.5


def test_mode_gap_module_on_saved_checkpoint(tmp_path):
    from diagnostics.mode_gap import run
    from diagnostics.common import load_run
    from UBM.tokenizer import fingerprint
    ubm = _tiny_ubm()
    m = _tiny_iecnn(len(ubm.bases), seed=3)
    d = tmp_path / "ck"
    d.mkdir()
    vocab_file = tmp_path / "v.json"
    ubm.save(vocab_file)
    torch.save({"model": m.state_dict(), "best_step": 0,
                "vocab_sha256": fingerprint(vocab_file)}, d / "checkpoint.pt")
    (d / "config_used.json").write_text(json.dumps(
        load_config("configs/e0_plain_ensemble.yaml", "tiny.yaml")))
    out = run(str(d), None, tmp_path, vocab=str(vocab_file),
              text="alpha beta gamma delta")
    assert "mode_gap" in out and not out.get("skipped")


def test_load_run_refuses_vocab_mismatch(tmp_path):
    from diagnostics.common import load_run
    from UBM.tokenizer import fingerprint
    ubm = _tiny_ubm()
    m = _tiny_iecnn(len(ubm.bases), seed=4)
    d = tmp_path / "ck"
    d.mkdir()
    torch.save({"model": m.state_dict(), "best_step": 0,
                "vocab_sha256": "deadbeef" * 8}, d / "checkpoint.pt")
    (d / "config_used.json").write_text(json.dumps(
        load_config("configs/e0_plain_ensemble.yaml", "tiny.yaml")))
    with pytest.raises(RuntimeError, match="VOCAB MISMATCH"):
        load_run(str(d), None, vocab="UBM/vocab_v0.json")
