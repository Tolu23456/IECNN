"""Nucleus (top-p) sampling filter — scripts/generate.py.nucleus_mask.

The semantics that matter for coherent text: the kept set is the smallest
prefix of the probability-sorted distribution whose cumulative mass reaches
top_p (inclusive), the top-1 token is ALWAYS kept (degenerate confident
distributions never get their argmax masked away), and top_p >= 1.0 is a
no-op. Found-in-the-wild failure it guards: an exclusive cut (crossing token
dropped) makes the kept mass always < top_p and can mask the argmax when the
distribution is a point mass.
"""
import sys
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.generate import nucleus_mask  # noqa: E402


def _lp(logits):
    return torch.log_softmax(logits, dim=-1)


def test_nucleus_keeps_smallest_mass_prefix_inclusive():
    # probs = [0.5, 0.3, 0.15, 0.05]: top_p 0.6 -> cum 0.5, 0.8 ...
    # crossing at token 1 -> keep {0,1}, kept mass 0.8 (>= 0.6)
    lp = _lp(torch.tensor([0.5, 0.3, 0.15, 0.05]))
    out = nucleus_mask(lp, 0.6)
    kept = out > float("-inf")
    assert kept.tolist() == [True, True, False, False]
    # renormalized kept mass sums to 1 over survivors
    assert torch.allclose(out.softmax(-1), torch.tensor([0.625, 0.375, 0.0, 0.0]),
                          atol=1e-6)


def test_nucleus_always_keeps_argmax_even_point_mass():
    # point-mass distribution: exclusive-cut variants would mask the argmax
    lp = _lp(torch.tensor([1.0, 0.0, 0.0, 0.0]))
    out = nucleus_mask(lp, 0.99)
    assert int(out.argmax()) == 0
    kept = out > float("-inf")
    assert kept.tolist() == [True, False, False, False]


def test_nucleus_one_is_noop_and_small_p_reduces_to_argmax():
    lp = _lp(torch.randn(64, generator=torch.Generator().manual_seed(0)))
    assert torch.equal(nucleus_mask(lp, 1.0), lp)          # off
    out = nucleus_mask(lp, 1e-9)                            # tiny p
    kept = out > float("-inf")
    assert int(kept.sum()) == 1
    assert int(out.argmax()) == int(lp.argmax())            # argmax survives


def test_nucleus_after_top_k_interaction():
    """Pipeline order used in generate.py: temperature -> top-k -> top-p.
    top-k clamps first; top-p then prunes within the kept tail."""
    lp = _lp(torch.tensor([0.5, 0.25, 0.15, 0.06, 0.04]))
    k = 3
    kth = lp.topk(k).values[-1]
    lp = lp.clamp_max(kth)                                  # keeps {0,1,2}
    lp = nucleus_mask(lp, 0.55)                             # cum .5, .75 -> keep {0,1}
    kept = lp > float("-inf")
    assert kept.tolist() == [True, True, False, False, False]
