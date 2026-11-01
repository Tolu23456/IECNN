"""UBM: Universal Basemapping tokenizer (spec 07).

Every piece of text is built from numbered *bases*: raw bytes at the bottom,
specials next, merged pieces above, and (later, phase 9) promoted words at the
top. Every base records the two parts it came from, so any base expands to
bytes: a new word is never "unknown".
"""
from __future__ import annotations

import base64
import json
import unicodedata
from dataclasses import dataclass

import regex

NUM_BYTES = 256
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<mask>", "<r1>", "<r2>", "<r3>"]
FIRST_SPECIAL_ID = NUM_BYTES                          # 256
FIRST_LEARNED_ID = NUM_BYTES + len(SPECIAL_TOKENS)    # 264

# Pre-tokenizer (spec 07, sections 4/10): words with an attached leading space
# marker, digit groups of up to 3, punctuation runs, whitespace runs.
# Merges never cross chunk boundaries, so chunking is a hard rule.
PRETOKENIZER_PATTERN = (
    r" ?\p{L}+"                                    # words (" cat" becomes a chunk)
    r"| ?\p{N}{1,3}"                               # digit groups of up to 3
    r"| ?[^\s\p{L}\p{N}]+"                         # punctuation runs
    r"|(?:\s+(?!\S))|\s+"                          # whitespace runs
)

NORMALIZATION = "NFC"


def normalize_text(text: str) -> str:
    """NFC normalization applied before any encoding."""
    return unicodedata.normalize(NORMALIZATION, text)


def chunk_text(text: str) -> list[str]:
    """Split text into pre-tokenizer chunks; merges never cross them."""
    return regex.findall(PRETOKENIZER_PATTERN, text)


@dataclass
class BaseRecord:
    """One vocabulary entry. IDs are never reused or removed (tombstone instead)."""
    id: int
    bytes: bytes
    parts: tuple[int, int] | None = None
    birth_step: int = 0
    origin: str = "byte"        # byte | merge | promotion
    count: int = 0
    active: bool = True


class UBM:
    """Byte-level base vocabulary with learned merges and promotions."""

    def __init__(self) -> None:
        self.vocab_version = 0
        self.bases: list[BaseRecord] = []
        self._init_static_vocab()
        self._pair_to_merged: dict[tuple[int, int], int] = {}

    # ------------------------------------------------------------- static --
    def _init_static_vocab(self) -> None:
        """IDs 0..255 are byte values; 256..263 are the specials (spec 07, sec 2)."""
        for i in range(NUM_BYTES):
            self.bases.append(BaseRecord(id=i, bytes=bytes([i])))
        for j, _ in enumerate(SPECIAL_TOKENS):
            # Specials are control tokens; encode() never produces them. Their
            # byte payload is empty so decode() simply skips them.
            self.bases.append(BaseRecord(id=FIRST_SPECIAL_ID + j, bytes=b"", origin="special"))

    # ------------------------------------------------------------ encoding --
    def encode(self, text: str) -> list[int]:
        """Normalize, pre-tokenize, apply merges by creation order. Never fails."""
        out: list[int] = []
        for chunk in chunk_text(normalize_text(text)):
            seq = list(chunk.encode("utf-8"))        # byte value == its id (0..255)
            self._apply_merges_in_place(seq)
            out.extend(seq)
        return out

    def _apply_merges_in_place(self, seq: list[int]) -> None:
        """Greedy: always apply the lowest-created applicable merge first."""
        while True:
            best_id: int | None = None
            for i in range(len(seq) - 1):
                pair = (seq[i], seq[i + 1])
                merged = self._pair_to_merged.get(pair)
                if merged is not None and self.bases[merged].active:
                    if all(self.bases[p].active for p in pair):
                        if best_id is None or merged < best_id:
                            best_id = merged
                            if best_id == FIRST_LEARNED_ID:  # cannot be beaten
                                break
            if best_id is None:
                return
            left, right = self.bases[best_id].parts  # type: ignore[misc]
            self._replace_pair(seq, left, right, best_id)

    @staticmethod
    def _replace_pair(seq: list[int], left: int, right: int, merged: int) -> None:
        """Replace every non-overlapping occurrence of (left, right) with merged."""
        out: list[int] = []
        i = 0
        n = len(seq)
        while i < n:
            if i + 1 < n and seq[i] == left and seq[i + 1] == right:
                out.append(merged)
                i += 2
            else:
                out.append(seq[i])
                i += 1
        seq[:] = out

    # ------------------------------------------------------------ decoding --
    def decode(self, ids: list[int]) -> str:
        """Concatenate bytes, decode UTF-8 with replacement for invalid sequences."""
        raw = b"".join(self.expand(int(i)) for i in ids)
        return raw.decode("utf-8", errors="replace")

    def expand(self, base_id: int) -> bytes:
        """Bytes of one base (recursive over parts)."""
        rec = self.bases[base_id]
        if rec.parts is None:
            return rec.bytes
        return self.expand(rec.parts[0]) + self.expand(rec.parts[1])

    def parts(self, base_id: int):
        """(left_id, right_id) for merged bases, else None."""
        return self.bases[base_id].parts

    # ------------------------------------------------------ vocabulary ops --
    def add_bases(self, records: list[BaseRecord]) -> list[int]:
        """Add new bases at the end of the vocabulary, return their ids.

        Each must be the next free id (ids are never reused). Raises on conflict.
        """
        ids: list[int] = []
        for rec in records:
            if rec.id != len(self.bases):
                raise ValueError(f"new base id {rec.id} must equal next free id {len(self.bases)}")
            if rec.parts is not None and (rec.parts[0] < 0 or rec.parts[1] < 0
                                          or rec.parts[0] >= len(self.bases)
                                          or rec.parts[1] >= len(self.bases)):
                raise ValueError(f"parts {rec.parts} must already exist")
            self.bases.append(rec)
            if rec.parts is not None:
                self._pair_to_merged[rec.parts] = rec.id
            ids.append(rec.id)
        return ids

    def tombstone(self, base_id: int) -> None:
        """Deactivate a base without removing its id (rollback, spec 07 sec 6.2)."""
        self.bases[base_id].active = False

    def revive(self, base_id: int) -> None:
        """Undo a tombstone (used by rollback testing)."""
        self.bases[base_id].active = True

    # ------------------------------------------------------- serialization --
    def save(self, path) -> None:
        """Save the vocabulary as JSON (header + base records)."""
        payload = {
            "vocab_version": self.vocab_version,
            "normalization": NORMALIZATION,
            "pretokenizer": PRETOKENIZER_PATTERN,
            "first_learned_id": FIRST_LEARNED_ID,
            "bases": [
                {
                    "id": b.id,
                    "bytes": base64.b64encode(b.bytes).decode("ascii"),
                    "parts": list(b.parts) if b.parts else None,
                    "birth_step": b.birth_step,
                    "origin": b.origin,
                    "count": b.count,
                    "active": b.active,
                }
                for b in self.bases
            ],
        }
        with open(path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=1)

    @classmethod
    def load(cls, path) -> "UBM":
        """Load a vocabulary saved by `save`."""
        with open(path, encoding="utf-8") as f:
            payload = json.load(f)
        ubm = cls.__new__(cls)          # skip the default byte-base init
        ubm.vocab_version = payload["vocab_version"]
        ubm.bases = [
            BaseRecord(
                id=b["id"],
                bytes=base64.b64decode(b["bytes"]),
                parts=tuple(b["parts"]) if b["parts"] is not None else None,
                birth_step=b["birth_step"],
                origin=b["origin"],
                count=b["count"],
                active=b["active"],
            )
            for b in payload["bases"]
        ]
        ubm._pair_to_merged = {}
        for b in ubm.bases:
            if b.parts is not None:
                ubm._pair_to_merged[b.parts] = b.id
        return ubm
