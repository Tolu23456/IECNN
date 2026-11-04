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

    `data.corpus: big` selects corpus v1 (whole-book held-out split).
    Returns a summary dict, also written to data/split_info.json.
    """
    if str(config.get("data", {}).get("corpus", "v0")) == "big":
        return prepare_big_corpus(config)
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


# ---------------------------------------------------------------------------
# Corpus v1 ("big"): 38 public-domain Project Gutenberg books, split by WHOLE
# HELD-OUT BOOKS (Tolu's preference) instead of the contiguous tail split used
# for v0. Gutenberg boilerplate is stripped between the license markers and
# each book's body sha is recorded; duplicate ids AND duplicate bodies are
# rejected so a book appears exactly once, in exactly one split part.
# ---------------------------------------------------------------------------

BOOKS = [
    (10,   "King James Bible"), (100, "Complete Works of William Shakespeare"),
    (2701, "Moby Dick; Or, The Whale"), (2600, "War and Peace"),
    (1400, "Great Expectations"), (766, "David Copperfield"),
    (768,  "Wuthering Heights"), (1260, "Jane Eyre"),
    (1342, "Pride and Prejudice"), (158, "Emma"),
    (161,  "Sense and Sensibility"), (46, "A Christmas Carol"),
    (76,   "Adventures of Huckleberry Finn"), (74, "The Adventures of Tom Sawyer"),
    (1184, "The Count of Monte Cristo"), (345, "Dracula"),
    (6130, "The Iliad"), (1727, "The Odyssey"),
    (205,  "Walden, and On The Duty Of Civil Disobedience"), (1232, "The Prince"),
    (3207, "Leviathan"), (219, "Heart of Darkness"),
    (1661, "The Adventures of Sherlock Holmes"), (84, "Frankenstein"),
    (2148, "The Works of Edgar Allan Poe, Volume 1"), (1080, "A Modest Proposal"),
    (55,   "The Wonderful Wizard of Oz"), (174, "The Picture of Dorian Gray"),
    (113,  "The Secret Garden"), (45, "Anne of Green Gables"),
    (35,   "The Time Machine"), (36, "The War of the Worlds"),
    (41,   "The Legend of Sleepy Hollow"), (1497, "The Decameron"),
    (996,  "Don Quixote"), (4300, "Ulysses"),
    (2591, "Grimms' Fairy Tales"), (1952, "The Yellow Wallpaper"),
]

# Held-out books (whole book goes to exactly one split). Val is one book
# (The Wonderful Wizard of Oz, ~213k chars) so a full-val pass costs ~2-5 min
# at E1 speed; test spans three small books of mixed genres.
VAL_BOOK_IDS = [55]
TEST_BOOK_IDS = [219, 1080, 1952]    # Heart of Darkness, Modest Proposal, Yellow Wallpaper


def corpus_v1_url(pg_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt"


def strip_gutenberg_boilerplate(text: str):
    """Return (body, markers_found): only the text between the Gutenberg
    license START/END markers."""
    low = text.lower()
    start = low.find("*** start of the project gutenberg ebook")
    end = low.find("*** end of the project gutenberg")
    if start == -1 or end == -1 or end <= start:
        return text, False
    body = text[start:end].split("\n", 1)[1]      # drop the marker line itself
    return body, True


def download_big_books(raw_dir: Path) -> list:
    """Download each manifest book once into raw_dir; returns (id, title, body,
    markers_ok) records with body sha16. Skips already-downloaded files."""
    import time
    import urllib.request

    raw_dir.mkdir(parents=True, exist_ok=True)
    ids_seen, shas_seen, records = set(), set(), []
    for pg_id, title in BOOKS:
        if pg_id in ids_seen:
            raise ValueError(f"duplicate book id in manifest: {pg_id}")
        ids_seen.add(pg_id)
        dest = raw_dir / f"pg{pg_id}.txt"
        if not dest.exists() or dest.stat().st_size == 0:
            last = None
            for attempt in range(3):
                try:
                    req = urllib.request.Request(
                        corpus_v1_url(pg_id),
                        headers={"User-Agent": "IECNN-corpus-builder/2"})
                    with urllib.request.urlopen(req, timeout=180) as r:
                        data = r.read()
                    dest.write_bytes(data)
                    break
                except Exception as exc:                    # noqa: BLE001
                    last = exc
                    time.sleep(2 * (attempt + 1))
            else:
                raise RuntimeError(f"download failed pg{pg_id}: {last}")
        text = dest.read_text(encoding="utf-8", errors="replace")
        body, ok = strip_gutenberg_boilerplate(text)
        cleaned = __import__("unicodedata").normalize("NFC", body)
        cleaned = cleaned.replace("\r\n", "\n").replace("\x1a", "")
        sha16 = sha256_of(cleaned)[:16]
        if sha16 in shas_seen:
            raise ValueError(f"duplicate book body detected (same text): {title}")
        shas_seen.add(sha16)
        records.append({"pg_id": pg_id, "title": title, "raw_bytes": dest.stat().st_size,
                        "chars": len(cleaned), "body_sha16": sha16,
                        "markers_found": ok})
        print(f"pg{pg_id:>6} {len(cleaned):>10,} chars {'~ok' if ok else '**MARKERS NOT FOUND**'} {title}", flush=True)
    return records


def prepare_big_corpus(config: dict) -> dict:
    """Whole-book-held-out split of corpus v1 (Tolu's requested design)."""
    data_cfg = config.get("data", {})
    dest_dir = Path(config.get("data_dir", data_cfg.get("dir", "data")))
    raw_dir = dest_dir / "raw_big"

    records = download_big_books(raw_dir)
    by_id = {r["pg_id"]: r for r in records}
    for b in VAL_BOOK_IDS + TEST_BOOK_IDS:
        if b not in by_id:
            raise ValueError(f"held-out book {b} not in manifest")

    parts = {"train": [], "val": [], "test": []}
    heldout = set(VAL_BOOK_IDS) | set(TEST_BOOK_IDS)
    for r in sorted(records, key=lambda r: r["pg_id"]):
        key = "val" if r["pg_id"] in VAL_BOOK_IDS else \
            "test" if r["pg_id"] in TEST_BOOK_IDS else "train"
        body = (raw_dir / f"pg{r['pg_id']}.txt").read_text(
            encoding="utf-8", errors="replace")
        body, ok = strip_gutenberg_boilerplate(body)
        body = __import__("unicodedata").normalize("NFC", body).replace("\r\n", "\n").replace("\x1a", "")
        parts[key].append((r["pg_id"], body))

    summary_books = {"train": len(records) - len(heldout), "val": len(VAL_BOOK_IDS),
                     "test": len(TEST_BOOK_IDS)}
    out = {}
    for name, parts_list in parts.items():
        text = "\n".join(b for _, b in parts_list)
        p = dest_dir / f"{name}.txt"
        p.write_text(text, encoding="utf-8")
        out[name] = {"chars": len(text), "sha256": sha256_of(text),
                     "books": [pid for pid, _ in parts_list], "file": str(p)}

    info = {
        "source": "public-domain-english-v1 (Project Gutenberg, whole-book held-out split)",
        "split_design": "whole held-out books (no book appears in two parts); "
                        "val[%s] test[%s]" % (VAL_BOOK_IDS, TEST_BOOK_IDS),
        "n_books_total": len(records),
        "books": summary_books,
        "split": {"held_out_by": "book", "val_book_ids": VAL_BOOK_IDS,
                  "test_book_ids": TEST_BOOK_IDS},
        "boilerplate": "all bodies stripped between Gutenberg license markers",
        "duplicate_check": "ids + body sha16 both unique",
        "audit_total": {k: out[k]["chars"] for k in out},
        "parts": out,
    }
    (dest_dir / "split_info.json").write_text(
        json.dumps(info, indent=2, ensure_ascii=False), encoding="utf-8")
    print("corpus v1 ready:", {k: v["chars"] for k, v in out.items()})
    return info
