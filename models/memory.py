"""Shared fixed-window memory (spec 02).

Functional helpers; the learnable pieces (W_K, W_V, rel embedding, Proj_r +
LayerNorm for chaining) live in `Region` / `IECNN` and are passed in.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


def banded_causal_mask(seq_len: int, span: int):
    """Boolean mask [T, T]: entry [t, j] is True iff j <= t and t - j < span.

    True means "position t may read memory row j".
    """
    t = torch.arange(seq_len).unsqueeze(1)
    j = torch.arange(seq_len).unsqueeze(0)
    return (j <= t) & ((t - j) < span)


def region1_rows(token_embeddings: torch.Tensor) -> torch.Tensor:
    """Region 1 memory rows are the token embeddings, position NOT added to rows.

    Position enters only through the relative-distance key embedding, so the
    same row means different things at different distances (spec 02, section 2).
    """
    return token_embeddings


def next_region_rows(prev_summary: torch.Tensor, prev_rows: torch.Tensor,
                     proj: nn.Module | None, ln: nn.Module) -> torch.Tensor:
    """m^r_j = LayerNorm(Proj(h^{r-1}_j) + m^{r-1}_j)   (spec 02, section 2).

    proj may be None to ablate the chaining path (memory test 4); then only
    the residual path survives, inside the LayerNorm.
    """
    if proj is None:
        return ln(prev_rows)
    return ln(proj(prev_summary) + prev_rows)


def relative_key_embedding(max_span: int, d_q: int, table: nn.Embedding | None = None):
    """Learned embedding over distances 0 .. max_span-1, added to keys.

    If `table` is given it is used directly (the model's owned embedding);
    otherwise a fresh uninitialized one is returned for tests/inspection.
    Distance semantics: distance 0 = the position t itself, increasing
    backwards in time.
    """
    if table is None:
        table = nn.Embedding(max_span, d_q)
    return table


def gather_windows(rows: torch.Tensor, span: int, window_offsets: torch.Tensor | None = None):
    """Sliding window gather [B, T, d] -> key/value rows per position.

    Returns (windowed_rows [B, T, span, d], valid_mask [*, span]) where
    windowed_rows[..., t, i] is the row of position t + i - span + 1
    (i.e. i = span-1 is position t itself, i = 0 is the furthest past).
    Rows before the sequence start are duplicated from row 0 but flagged
    invalid in valid_mask (masked, not zero-filled — spec 02 section 1).
    """
    b, t, d = rows.shape
    if window_offsets is None:
        window_offsets = torch.arange(span) - span + 1            # [-(span-1) .. 0]
    window_offsets = window_offsets.to(rows.device)
    idx = torch.arange(t, device=rows.device).unsqueeze(1) + window_offsets.unsqueeze(0)
    valid = idx >= 0
    idx = idx.clamp_min(0)
    windowed = rows[:, idx]          # fancy indexing: [B, T, span, d]
    return windowed, valid
