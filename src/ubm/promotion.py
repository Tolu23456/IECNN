"""UBM v1: model-driven checkpoint promotion (spec 07, section 6.2)."""
from __future__ import annotations


class CandidateLog:
    """Logs short base sequences that region 1 predicted with high agreement."""

    def update(self, base_ids, plurality_agreement, confidence) -> None:
        raise NotImplementedError

    def top_candidates(self, k: int) -> list:
        """Rank by frequency, agreement, and confidence together."""
        raise NotImplementedError


def promote(ubm, model, candidates, settle_steps: int, rollback_epsilon: float, val_text: str) -> dict:
    """Create bases, init embeddings as mean of parts, retokenize, settle, roll back if worse."""
    raise NotImplementedError
