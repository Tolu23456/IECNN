"""UBM v1 promotion (spec 07 §6.2, built 2026-11-14): CandidateLog ranking,
mean-of-parts embedding init, settle hook, rollback tombstone + restore,
round-trip integrity after promotion."""
import math

import torch


def _tiny_ubm():
    from UBM import UBM
    return UBM()          # 256 byte bases + 8 specials; letters are single ids


class _EmbedOnly(torch.nn.Module):
    """Minimal stand-in: promotion only needs `model.embed` (and the optional
    vocab_size attribute)."""

    def __init__(self, vocab_size, d=8):
        super().__init__()
        self.embed = torch.nn.Embedding(vocab_size, d)
        self.vocab_size = vocab_size


def _candidates(ubm, pairs):
    return [{"base_ids": list(p), "count": 5,
             "mean_plurality_agreement": 0.9, "mean_confidence": 0.8,
             "score": 0.9 * 0.8 * math.log(6)} for p in pairs]


def test_candidate_log_ranking_and_length_filter():
    from UBM.promotion import CandidateLog
    log = CandidateLog()
    log.update([1, 2], 0.5, 0.5)                 # once, mediocre
    log.update([3, 4], 0.9, 0.9)                 # once, great
    log.update([3, 4], 0.9, 0.9)                 # twice, great -> wins
    log.update([5], 1.0, 1.0)                    # too short -> ignored
    log.update([6, 7, 8, 9, 10, 11, 12], 1.0, 1.0)  # too long -> ignored
    top = log.top_candidates(2)
    assert [tuple(c["base_ids"]) for c in top] == [(3, 4), (1, 2)]
    assert top[0]["count"] == 2
    assert abs(top[0]["mean_plurality_agreement"] - 0.9) < 1e-9
    assert log.top_candidates(0) == []


def test_promote_creates_base_and_grows_embedding_mean_of_parts():
    from UBM.promotion import promote
    ubm = _tiny_ubm()
    v0 = len(ubm.bases)
    model = _EmbedOnly(v0)
    torch.manual_seed(0)
    a = ubm.encode("q")[0]
    b = ubm.encode("z")[0]
    report = promote(ubm, model, _candidates(ubm, [(a, b)]),
                     settle_steps=0, rollback_epsilon=1e9,   # never roll back
                     val_text="nothing to score", eval_bpc=lambda t: 2.0)
    assert not report["rolled_back"]
    assert len(report["promoted_ids"]) == 1
    new_id = report["promoted_ids"][0]
    rec = ubm.bases[new_id]
    assert rec.origin == "promotion"
    assert rec.parts == (a, b)
    assert ubm.expand(new_id) == b"qz"
    assert model.embed.weight.shape[0] == v0 + 1
    assert model.vocab_size == v0 + 1
    # embedding init = MEAN of the two constituent rows (spec 07 6.2 step 4)
    expected = (model.embed.weight[a].detach() + model.embed.weight[b].detach()) / 2
    assert torch.allclose(model.embed.weight[new_id].detach(), expected)
    # encoding now uses the promoted base; decode round-trip holds
    assert ubm.encode("qz") == [new_id]
    assert ubm.decode([new_id]) == "qz"
    assert report["bpc_before"] == report["bpc_after"] == 2.0


def test_promote_rollback_tombstones_and_restores():
    from UBM.promotion import promote
    ubm = _tiny_ubm()
    v0 = len(ubm.bases)
    model = _EmbedOnly(v0)
    torch.manual_seed(0)
    a, b = ubm.encode("q")[0], ubm.encode("z")[0]
    orig_weight = model.embed.weight.detach().clone()
    # epsilon = -1 -> "after > before + eps" is always true -> forced rollback
    report = promote(ubm, model, _candidates(ubm, [(a, b)]),
                     settle_steps=0, rollback_epsilon=-1.0,
                     val_text="nothing", eval_bpc=lambda t: 2.0)
    assert report["rolled_back"]
    assert report["promoted_ids"] == [] and report["tombstoned_ids"]
    tid = report["tombstoned_ids"][0]
    assert not ubm.bases[tid].active
    # id is reserved (never removed), but encoding ignores the tombstone
    assert len(ubm.bases) == v0 + 1
    assert ubm.encode("qz") == [a, b]
    assert ubm.decode([a, b]) == "qz"
    # the original embedding table is back in place
    assert model.embed.weight.shape[0] == v0
    assert torch.equal(model.embed.weight.detach(), orig_weight)
    assert model.vocab_size == v0


def test_promote_skips_candidates_that_do_not_encode_to_two_bases():
    from UBM.promotion import promote
    ubm = _tiny_ubm()
    model = _EmbedOnly(len(ubm.bases))
    three = ubm.encode("qzb")            # 3 single-byte bases
    two = ubm.encode("qz")
    report = promote(ubm, model, _candidates(ubm, [tuple(three), tuple(two)]),
                     settle_steps=0, rollback_epsilon=1e9,
                     val_text="nothing", eval_bpc=lambda t: 1.0)
    assert len(report["promoted_ids"]) == 1
    assert len(report["skipped"]) == 1
    assert "need exactly 2" in report["skipped"][0]["reason"]


def test_promote_calls_settle_fn_with_new_ids():
    from UBM.promotion import promote
    ubm = _tiny_ubm()
    model = _EmbedOnly(len(ubm.bases))
    a, b = ubm.encode("q")[0], ubm.encode("z")[0]
    calls = []

    def settle_fn(m, u, new_ids, steps):
        calls.append((list(new_ids), steps))

    promote(ubm, model, _candidates(ubm, [(a, b)]), settle_steps=7,
            rollback_epsilon=1e9, val_text="nothing", eval_bpc=lambda t: 1.0,
            settle_fn=settle_fn)
    assert calls == [([len(ubm.bases) - 1], 7)]


def test_promote_with_real_model_and_bpc_end_to_end():
    """Full loop on a tiny IECNN: candidates -> promotion -> real held-out
    bpc rollback test -> encoding uses the new base when kept."""
    from config import load_config
    from utils import set_seed
    from models.iecnn import build_iecnn
    from train.evaluate import evaluate_bpc
    from UBM.promotion import CandidateLog, promote

    ubm = _tiny_ubm()
    v0 = len(ubm.bases)
    set_seed(0)
    cfg = load_config("configs/e0_plain_ensemble.yaml", "tiny.yaml")
    model = build_iecnn(cfg, v0)
    model.eval()

    # harvest-style candidates: from the model's own predicted stream
    log = CandidateLog()
    log.update(tuple(ubm.encode("th")), 0.9, 0.9)
    log.update(tuple(ubm.encode("th")), 0.8, 0.9)
    cands = log.top_candidates(4)

    val_text = "the quick brown fox"
    before = evaluate_bpc(model, ubm, val_text, seq_len=8)
    report = promote(ubm, model, cands, settle_steps=0,
                     rollback_epsilon=1e9, val_text=val_text,
                     eval_bpc=lambda t: evaluate_bpc(model, ubm, t, seq_len=8))
    assert abs(report["bpc_before"] - before) < 1e-5   # report rounds to 6 dp
    new_id = report["promoted_ids"][0]
    assert ubm.encode("th") == [new_id]
    after = evaluate_bpc(model, ubm, val_text, seq_len=8)
    assert math.isfinite(after)
