"""Packing tokenized text into fixed-length sequences (spec 06, section 1).

All splits are cached as `.npy` int32 arrays in `data/` keyed by vocabulary
version, so repeated runs tokenize once. Sequences are packed without overlap;
block boundaries lose full context on a few positions, standard practice, and
the same for every model (noted in the phase report).
"""
from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np


def tokenize_corpus(text: str, vocab_size: int = 512, cache_dir: str = "data") -> np.ndarray:
    """Encode text with the fixed v0 vocabulary, caching ids as npy.

    Uses `UBM/vocab_v0.json` (the fixed 512-base vocabulary, DECISIONS.md).
    Returns an int32 array of token ids (no specials inserted).
    """
    from UBM import UBM

    cache = Path(cache_dir) / f"tokens_v{vocab_size}_{_text_hash(text)[:10]}.npy"
    if cache.exists():
        return np.load(cache)
    ubm = UBM.load("UBM/vocab_v0.json")
    ids = ubm.encode(text)
    arr = np.asarray(ids, dtype=np.int32)
    cache.parent.mkdir(parents=True, exist_ok=True)
    np.save(cache, arr)
    return arr


def _text_hash(text: str) -> str:
    """sha256 of utf-8 bytes of the text (cache key)."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def pack_sequences(ids: np.ndarray, seq_len: int):
    """Return inputs and targets (targets shifted by one) of shape [n, seq_len].

    The token stream is chopped into disjoint windows of seq_len (a small
    remainder at the very end is dropped). inputs[i] = ids[k..k+seq_len),
    targets[i] = ids[k+1..k+seq_len+1) where the last target is the first token
    of the next window; the final piece is clamped so nothing is out of range.
    """
    n_full = (len(ids) - 1) // seq_len
    if n_full == 0:
        raise ValueError(f"need more than seq_len+1 tokens, got {len(ids)}")
    steps = np.arange(seq_len)
    starts = np.arange(n_full)[:, None] * seq_len
    idx_in = starts + steps
    idx_out = starts + steps + 1
    idx_out[:, -1] = np.minimum(idx_out[:, -1], len(ids) - 1)
    inputs = ids[idx_in]
    targets = ids[idx_out]
    return inputs.astype(np.int64), targets.astype(np.int64)


class Batcher:
    """Deterministic batch iterator over packed sequences.

    With shuffle=True the sequence order is permuted once per epoch with a
    seed-derived generator, so runs are reproducible.
    """

    def __init__(self, ids: np.ndarray, seq_len: int, batch_size: int, seed: int = 0, shuffle: bool = True):
        self.inputs, self.targets = pack_sequences(ids, seq_len)
        self.batch_size = batch_size
        self.shuffle = shuffle
        self.rng = np.random.default_rng(seed)
        self._order = self.rng.permutation(len(self.inputs))
        self._pos = 0
        self.epoch = 0

    def next_batch(self) -> tuple[np.ndarray, np.ndarray]:
        """Return (inputs, targets) of the next batch; reshuffles at epoch end."""
        if self.shuffle and self._pos + self.batch_size > len(self.inputs):
            self._order = self.rng.permutation(len(self.inputs))
            self._pos = 0
            self.epoch += 1
        sl = slice(self._pos, self._pos + self.batch_size)
        idx = self._order[sl]
        self._pos += self.batch_size
        return self.inputs[idx], self.targets[idx]

    def __len__(self) -> int:
        """Number of full batches per epoch."""
        return len(self.inputs) // self.batch_size
