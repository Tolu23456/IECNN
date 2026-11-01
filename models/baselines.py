"""Baselines B0 (small GRU) and B1 (small causal transformer)."""
from __future__ import annotations


def build_gru_baseline(config: dict, vocab_size: int):
    """Small GRU language model with roughly the same parameters as IECNN."""
    raise NotImplementedError


def build_transformer_baseline(config: dict, vocab_size: int):
    """Small causal transformer, matched parameters."""
    raise NotImplementedError
