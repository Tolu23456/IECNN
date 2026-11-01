"""UBM: Universal Basemapping (spec 07)."""
from __future__ import annotations

from UBM.tokenizer import (
    FIRST_LEARNED_ID,
    FIRST_SPECIAL_ID,
    NORMALIZATION,
    NUM_BYTES,
    PRETOKENIZER_PATTERN,
    SPECIAL_TOKENS,
    BaseRecord,
    UBM,
    chunk_text,
    normalize_text,
)

__all__ = [
    "FIRST_LEARNED_ID",
    "FIRST_SPECIAL_ID",
    "NORMALIZATION",
    "NUM_BYTES",
    "PRETOKENIZER_PATTERN",
    "SPECIAL_TOKENS",
    "BaseRecord",
    "UBM",
    "chunk_text",
    "normalize_text",
]
