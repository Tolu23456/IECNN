"""UBM: Universal Basemapping tokenizer (spec 07)."""
from __future__ import annotations

import unicodedata
from dataclasses import dataclass, field

NUM_BYTES = 256
SPECIAL_TOKENS = ["<pad>", "<bos>", "<eos>", "<sep>", "<mask>", "<r1>", "<r2>", "<r3>"]
FIRST_SPECIAL_ID = NUM_BYTES                  # 256
FIRST_LEARNED_ID = NUM_BYTES + len(SPECIAL_TOKENS)  # 264


def normalize_text(text: str) -> str:
    """NFC normalization applied before any encoding."""
    return unicodedata.normalize("NFC", text)


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
        self.bases: list[BaseRecord] = []   # TODO(agent): initialize 256 byte bases and specials

    def encode(self, text: str) -> list[int]:
        """Normalize, pre-tokenize, apply merges by creation order. Never fails."""
        raise NotImplementedError

    def decode(self, ids: list[int]) -> str:
        """Concatenate bytes, decode UTF-8 with replacement for invalid sequences."""
        raise NotImplementedError

    def expand(self, base_id: int) -> bytes:
        """Bytes of one base."""
        raise NotImplementedError

    def parts(self, base_id: int):
        """(left_id, right_id) for merged bases, else None."""
        raise NotImplementedError

    def add_bases(self, records: list[BaseRecord]) -> list[int]:
        """Add new bases at the end of the vocabulary, return their ids."""
        raise NotImplementedError

    def tombstone(self, base_id: int) -> None:
        """Deactivate a base without removing its id."""
        raise NotImplementedError

    def save(self, path) -> None:
        raise NotImplementedError

    @classmethod
    def load(cls, path) -> "UBM":
        raise NotImplementedError
