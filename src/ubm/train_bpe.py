"""UBM v0: frequency-based merge training (spec 07, section 6.1)."""
from __future__ import annotations


def train_ubm_v0(text: str, target_vocab_size: int, max_base_len: int = 32):
    """Train merges on text. Merges never cross pre-token chunk boundaries."""
    raise NotImplementedError
