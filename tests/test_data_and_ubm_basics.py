import pytest

from src.data.prepare import normalize_nfc, split_text
from src.ubm.tokenizer import FIRST_LEARNED_ID, normalize_text


def test_split_is_contiguous_and_complete():
    text = "abcdefghij" * 100
    train, val, test = split_text(text, 0.05, 0.05)
    assert train + val + test == text
    assert len(val) == 50 and len(test) == 50


def test_nfc_composes_combining_marks():
    assert normalize_nfc("e\u0301") == "\u00e9"
    assert normalize_text("e\u0301") == "\u00e9"


def test_vocab_layout_constants():
    assert FIRST_LEARNED_ID == 264


def test_banded_mask_shape_and_rule():
    torch = pytest.importorskip("torch")
    from src.models.memory import banded_causal_mask
    m = banded_causal_mask(6, 3)
    assert m.shape == (6, 6)
    assert bool(m[4, 4]) and bool(m[4, 2]) and not bool(m[4, 1]) and not bool(m[2, 4])
