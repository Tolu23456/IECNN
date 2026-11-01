"""UBM v0: frequency-based merge training (spec 07, sections 4/6.1).

Deterministic byte-level BPE over pre-tokenizer chunks: repeatedly merge the
most frequent adjacent pair; ties are broken by the smaller (left, right) pair.
Merges never cross chunk boundaries by construction.
"""
from __future__ import annotations

from collections import Counter

from UBM.tokenizer import (
    FIRST_LEARNED_ID,
    BaseRecord,
    UBM,
    chunk_text,
)


def _chunk_to_ids(chunk: str) -> list[int]:
    """The UTF-8 bytes of one chunk (byte value == byte id, spec 07, section 4)."""
    return list(chunk.encode("utf-8"))


def train_ubm_v0(text: str, target_vocab_size: int, max_base_len: int = 32) -> tuple[UBM, list[BaseRecord]]:
    """Train merges on text until `target_vocab_size` total bases exist.

    Returns (ubm, merge_records). Merges never cross pre-token chunk
    boundaries. Deterministic: most frequent pair first, ties by smaller pair.
    """
    ubm = UBM()
    chunks: list[list[int]] = [_chunk_to_ids(c) for c in chunk_text(text) if c]

    # Global counts, indexed per pair. `banned` holds pairs whose merged form
    # exceeds max_base_len; they are never applied and never re-registered.
    banned: set[tuple[int, int]] = set()
    pair_counts: Counter = Counter()
    pair_chunks: dict[tuple[int, int], set[int]] = {}

    def _bump(idx: int, ch: list[int], sign: float) -> None:
        """Add (sign>0) or remove (sign<0) chunk idx's adjacent-pair counts."""
        for i in range(len(ch) - 1):
            pair = (ch[i], ch[i + 1])
            if sign > 0:
                if pair in banned:
                    continue
                pair_counts[pair] += 1
                pair_chunks.setdefault(pair, set()).add(idx)
            else:
                pair_counts[pair] -= 1
                if pair_counts[pair] <= 0:
                    del pair_counts[pair]
                    s = pair_chunks.get(pair)
                    if s is not None:
                        s.discard(idx)
                        if not s:
                            del pair_chunks[pair]

    for idx, ch in enumerate(chunks):
        _bump(idx, ch, +1)

    while len(ubm.bases) < target_vocab_size and pair_counts:
        # Best pair: most frequent, ties by smaller (left, right).
        best: tuple[int, int] | None = None
        best_count = 0
        for pair, count_ in pair_counts.items():
            if count_ < 2:                       # a merge seen once never pays off
                continue
            if best is None or count_ > best_count or (count_ == best_count and pair < best):
                best, best_count = pair, count_
        if best is None:
            break                                # only singleton-pair merges remain
        left, right = best
        merged_bytes = ubm.expand(left) + ubm.expand(right)
        if len(merged_bytes) > max_base_len:     # very long bases are not learned
            banned.add(best)
            pair_counts.pop(best, None)
            pair_chunks.pop(best, None)
            continue

        merged_id = len(ubm.bases)
        rec = BaseRecord(
            id=merged_id, bytes=merged_bytes, parts=(left, right),
            birth_step=0, origin="merge", count=best_count,
        )
        ubm.add_bases([rec])

        # Rewrite every chunk that contains this pair, updating counts by delta.
        for idx in list(pair_chunks.get(best, ())):
            ch = chunks[idx]
            _bump(idx, ch, -1)
            ubm._replace_pair(ch, left, right, merged_id)
            _bump(idx, ch, +1)
        pair_counts.pop(best, None)
        pair_chunks.pop(best, None)

    ubm.vocab_version = 1
    return ubm, [r for r in ubm.bases if r.origin == "merge"]


def ubm_from_merges(merge_records: list[BaseRecord], up_to: int) -> UBM:
    """Build a fresh UBM with only the first `up_to` merge records applied.

    Used to report compression at intermediate vocabulary sizes (264/512/1024)
    without re-training.
    """
    ubm = UBM()
    for rec in merge_records[:up_to]:
        ubm.add_bases([rec])
    return ubm
