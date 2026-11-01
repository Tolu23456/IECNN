"""Evaluation: bits per character, soft and hard mode."""
from __future__ import annotations


def evaluate_bpc(model, ubm, text: str, hard_mode: bool = False) -> float:
    """Total NLL over text / (ln 2 * characters of the ORIGINAL text)."""
    raise NotImplementedError


def mode_gap(model, ubm, text: str) -> dict:
    """Soft bpc, hard bpc, and the gap."""
    raise NotImplementedError
