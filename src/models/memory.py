"""Shared fixed-window memory (spec 02)."""
from __future__ import annotations


def banded_causal_mask(seq_len: int, span: int):
    """Boolean mask [T, T]: entry [t, j] is True iff j <= t and t - j < span.

    True means "position t may read memory row j".
    """
    import torch
    t = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    return (j <= t) & ((t - j) < span)


def region1_rows(token_embeddings):
    """Region 1 memory rows are the token embeddings (no position added to rows)."""
    raise NotImplementedError


def next_region_rows(prev_region_summary, prev_rows):
    """m^r_j = LayerNorm(Proj(h^{r-1}_j) + m^{r-1}_j)  (spec 02, section 2)."""
    raise NotImplementedError


def relative_key_embedding(max_span: int, d_q: int):
    """Learned embedding over distances 0 .. max_span-1 added to keys."""
    raise NotImplementedError
