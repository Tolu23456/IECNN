"""Packing tokenized text into fixed-length sequences."""
from __future__ import annotations


def pack_sequences(ids: list[int], seq_len: int):
    """Return inputs and targets (targets shifted by one) of shape [n, seq_len]."""
    raise NotImplementedError


def make_dataloader(ids: list[int], seq_len: int, batch_size: int, seed: int, shuffle: bool):
    """Deterministic dataloader for one split."""
    raise NotImplementedError
