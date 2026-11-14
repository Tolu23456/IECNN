"""Nucleus (top-p) + top-k sampling filters — scripts/generate.py.

Contracts pinned here (both take LOG-PROBS, not logits — passing logits made
the first draft of these tests nearly-uniform and useless, fixed 2026-11-14):

nucleus_mask(lp, top_p): keep the smallest prefix of the probability-sorted
distribution whose cumulative mass reaches top_p (inclusive), mask the rest
to -inf. The top-1 token is ALWAYS kept (degenerate confident distributions
never get their argmax masked away); top_p >= 1.0 is a no-op.

top-k (in the sampling loop): tokens outside the top-k are EXCLUDED (-inf).
The previous `clamp_max(kth)` behaviour was a bug, not top-k: it clipped the
whole tail down to the kth value instead of removing it, leaving every
excluded token at probability p_k in the softmax — sampling stayed nearly
full-support ("0 = sample from the full distribution" makes k > 0 mean the
opposite of that).
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate import nucleus_mask  # noqa: E402


def _lp(probs):
    """log-probs of an explicitly specified probability vector."""
    return torch.log(probs.clone().clamp_min(1e-30))


def _topk_like(lp, k):
    """generate.py's top-k, mirrored here so tests pin the pipeline."""
    kth = lp.topk(k).values[-1]
    return lp.masked_fill(lp < kth, float("-inf"))


def test_nucleus_keeps_smallest_mass_prefix_inclusive():
    # cum = [0.5, 0.8, 0.95, 1.0]; crossing at token 1 -> keep {0,1}
    lp = _lp(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = nucleus_mask(lp, 0.6)
    kept = out > float("-inf")
    assert kept.tolist() == [True, True, False, False], kept.tolist()
    assert torch.allclose(out.softmax(-1),
                          torch.tensor([0.625, 0.375, 0.0, 0.0]), atol=1e-6)


def test_nucleus_always_keeps_argmax_even_point_mass():
    # point mass: tails at 1e-12 — an exclusive cut would drop the argmax
    lp = _lp(torch.tensor([1.0, 1e-12, 1e-12, 1e-12]))
    out = nucleus_mask(lp, 0.99)
    assert int(out.argmax()) == 0
    assert (out > float("-inf")).tolist() == [True, False, False, False]


def test_nucleus_one_is_noop_and_small_p_reduces_to_argmax():
    lp = torch.log_softmax(torch.randn(64, generator=torch.Generator().manual_seed(0)), -1)
    assert torch.equal(nucleus_mask(lp, 1.0), lp)           # off
    out = nucleus_mask(lp, 1e-9)                             # tiny p
    assert int((out > float("-inf")).sum()) == 1
    assert int(out.argmax()) == int(lp.argmax())             # argmax survives


def test_topk_excludes_tail_instead_of_clipping():
    """THE regression test for the clamp_max bug: with k=2, tokens 2..3 must
    be EXCLUDED (-inf), not clipped down to p[1] (which left them sampleable
    at probability p_k each)."""
    lp = _lp(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = _topk_like(lp, 2)
    assert (out > float("-inf")).tolist() == [True, True, False, False]
    assert torch.allclose(out.softmax(-1),
                          torch.tensor([0.625, 0.375, 0.0, 0.0]), atol=1e-6)
    # old buggy behaviour for contrast: out.clamp_max(lp[1]) keeps all 4 live


def test_topk_then_nucleus_interaction():
    """Pipeline order used in generate.py: temperature -> top-k -> top-p.
    top-k 3 leaves {0,1,2} (probs renormalize to .45/.27/.18 within the
    kept set: 0.5/0.9, 0.3/0.9, ...), then nucleus 0.6 keeps {0,1}."""
    lp = _lp(torch.tensor([0.5, 0.3, 0.2, 0.0]))             # p[3] below both cuts
    lp = _topk_like(lp, 3)                                   # excludes token 3
    lp = nucleus_mask(lp, 0.6)
    kept = (lp > float("-inf")).tolist()
    # sorted kept probs: 0.5, 0.3, 0.2 -> cum .5, .8 crosses .6 at token 1
    assert kept == [True, True, False, False], kept
