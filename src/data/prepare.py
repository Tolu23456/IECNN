"""Corpus download, normalization, and splitting (spec 08, section 2)."""
from __future__ import annotations

import hashlib
import json
import unicodedata
from collections import Counter
from pathlib import Path

# Public-domain corpus: The Huntington Shakespeare "tiny Shakespeare" text
# (karpathy/char-rnn), ~1.1M characters. Internet access is required; if the
# download fails the caller should fall back to any user-supplied text file.
CORPUS_URL = (
    "https://raw.githubusercontent.com/karpathy/char-rnn/master/"
    "data/tinyshakespeare/input.txt"
)


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
    """Download the public-domain corpus into dest_dir and return its path.

    Uses the standard library (urllib). Raises if there is no internet.
    """
    import urllib.request

    dest_dir.mkdir(parents=True, exist_ok=True)
    dest = dest_dir / "tinyshakespeare.txt"
    if not dest.exists() or dest.stat().st_size == 0:
        urllib.request.urlretrieve(CORPUS_URL, dest)
    return dest


def audit_text(text: str) -> dict:
    """Data-cleaning audit: counts of everything that might need cleaning.

    Reports characters changed by NFC normalization, Unicode control
    characters, and mojibake/fraction-style artifacts.
    """
    nfc = unicodedata.normalize("NFC", text)
    n_changed_by_nfc = sum(1 for a, b in zip(text, nfc) if a != b)
    n_changed_by_nfc += abs(len(text) - len(nfc))

    control = Counter()
    other = Counter()
    for ch in nfc:
        cat = unicodedata.category(ch)
        if not cat.startswith("C"):
            continue
        if ch in ("\n", "\t"):           # expected whitespace controls
            control["\\n or \\t"] += 1
        elif cat == "Cc":                # unexpected actual control characters
            control[unicodedata.name(ch, "UNNAMED U+%04X" % ord(ch))] += 1
        else:                            # Cf: format chars, Cs: surrogates, Co: private use
            other[f"C:{cat}"] += 1
    printable = not (control or other)
    return {
        "num_chars": len(text),
        "num_nfc_chars": len(nfc),
        "chars_changed_by_nfc": n_changed_by_nfc,
        "num_lines": text.count("\n"),
        "control_characters": dict(control),
        "format_or_unassigned": dict(other),
        "fully_printable": printable,
        "num_unique_chars_nfc": len(set(nfc)),
    }


def sha256_of(text: str) -> str:
    """sha256 hex digest of the UTF-8 bytes of the text."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def prepare_all(config: dict) -> dict:
    """Download, normalize, split, save train/val/test files, record checksums.

    Returns a summary dict, also written to data/split_info.json.
    """
    data_cfg = config.get("data", {})
    dest_dir = Path(config.get("data_dir", data_cfg.get("dir", "data")))
    val_fraction = float(data_cfg.get("val_fraction", 0.05))
    test_fraction = float(data_cfg.get("test_fraction", 0.05))

    raw_path = download_corpus(dest_dir)
    raw_text = raw_path.read_text(encoding="utf-8")

    audit = audit_text(raw_text)

    # Cleaning rules (deliberate, minimal, recorded in DECISIONS.md):
    # 1. NFC normalization (spec 07, section 4).
    # 2. UnicodeCRLF -> LF line endings ("\\r\\n" -> "\\n') for consistency.
    # 3. Drop ASCII 26 (EOF marker) if present; keep all other controls.
    cleaned = unicodedata.normalize("NFC", raw_text)
    cleaned = cleaned.replace("\r\n", "\n")
    cleaned = cleaned.replace("\x1a", "")

    train, val, test = split_text(cleaned, val_fraction, test_fraction)
    out = {}
    for name, part in [("train", train), ("val", val), ("test", test)]:
        p = dest_dir / f"{name}.txt"
        p.write_text(part, encoding="utf-8")
        out[name] = {
            "chars": len(part),
            "sha256": sha256_of(part),
            "file": str(p),
        }

    info = {
        "source_url": CORPUS_URL,
        "raw_file": str(raw_path),
        "raw_chars": audit["num_chars"],
        "cleaned_chars": len(cleaned),
        "nfc_changed_chars": audit["chars_changed_by_nfc"],
        "unique_chars_nfc": audit["num_unique_chars_nfc"],
        "control_characters_found": audit["control_characters"],
        "format_or_unassigned_found": audit["format_or_unassigned"],
        "split": {"val_fraction": val_fraction, "test_fraction": test_fraction},
        "parts": out,
    }
    (dest_dir / "split_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return info
