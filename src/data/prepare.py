"""Corpus download, normalization, and splitting (spec 08, section 2)."""
from __future__ import annotations

import unicodedata
from pathlib import Path


def normalize_nfc(text: str) -> str:
    """Normalize text to Unicode NFC."""
    return unicodedata.normalize("NFC", text)


def split_text(text: str, val_fraction: float, test_fraction: float):
    """Split into (train, val, test) by contiguous blocks. Never shuffled."""
    n = len(text)
    n_test = int(n * test_fraction)
    n_val = int(n * val_fraction)
    n_train = n - n_val - n_test
    return text[:n_train], text[n_train:n_train + n_val], text[n_train + n_val:]


def download_corpus(dest_dir: Path) -> Path:
    """Download a small public-domain corpus into dest_dir and return its path.

    TODO(agent): try a public-domain source; if offline, ask the user for text
    files or build a synthetic corpus and LABEL results as synthetic.
    """
    raise NotImplementedError


def prepare_all(config: dict) -> dict:
    """Download, normalize, split, save train/val/test files, record checksums."""
    raise NotImplementedError
