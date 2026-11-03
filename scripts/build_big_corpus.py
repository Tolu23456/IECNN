"""Build a large public-domain corpus (tens of MB) for Phases 4+ (Tolu's request).

Downloads a curated list of Project Gutenberg books (English prose/verse),
strips the Gutenberg license headers/footers, NFC-normalizes and cleans with
the SAME rules as data/prepare.py (NFC, CRLF->LF, drop \\x1a), concatenates,
splits (contiguous, never shuffled), audits, and writes checksums to a
manifest. Deterministic: the book order, split points, and checksums are
recorded so the runner rebuilds a byte-identical corpus.

Usage:  python3 scripts/build_big_corpus.py [--dest data_big] [--val 0.005]
        (fractions are small on purpose: a 5% val of 40 MB would be 2 MB and
        a full-val bpc pass would cost an hour at E1 speed; 0.5% ~= 200k
        characters evaluates in minutes)
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from data.prepare import audit_text, split_text  # noqa: E402

BOOKS = [  # (project gutenberg id, title)  -- all public domain
    (10,   "King James Bible"),
    (100,  "Complete Works of William Shakespeare"),
    (2701, "Moby Dick; Or, The Whale"),
    (2600, "War and Peace"),
    (1400, "Great Expectations"),
    (766,  "David Copperfield"),
    (768,  "Wuthering Heights"),
    (1260, "Jane Eyre"),
    (1342, "Pride and Prejudice"),
    (158,  "Emma"),
    (161,  "Sense and Sensibility"),
    (46,   "A Christmas Carol"),
    (76,   "Adventures of Huckleberry Finn"),
    (74,   "The Adventures of Tom Sawyer"),
    (1184, "The Count of Monte Cristo"),
    (345,  "Dracula"),
    (6130, "The Iliad"),
    (1727, "The Odyssey"),
    (205,  "Walden, and On The Duty Of Civil Disobedience"),
    (1232, "The Prince"),
    (3207, "Leviathan"),
    (219,  "Heart of Darkness"),
    (1661, "The Adventures of Sherlock Holmes"),
    (84,   "Frankenstein"),
    (2148, "The Works of Edgar Allan Poe, Volume 1"),
    (1080, "A Modest Proposal"),
    (55,   "The Wonderful Wizard of Oz"),
    (174,  "The Picture of Dorian Gray"),
    (113,  "The Secret Garden"),
    (45,   "Anne of Green Gables"),
    (35,   "The Time Machine"),
    (36,   "The War of the Worlds"),
    (41,   "The Legend of Sleepy Hollow"),
    (1497, "The Decameron"),
    (996,  "Don Quixote"),
    (4300, "Ulysses"),
    (2591, "Grimms' Fairy Tales"),
    (1952, "The Yellow Wallpaper"),
]


def gutenberg_url(pg_id: int) -> str:
    return f"https://www.gutenberg.org/cache/epub/{pg_id}/pg{pg_id}.txt"


def strip_gutenberg_boilerplate(text: str) -> tuple[str, bool]:
    """Keep only the text between the START/END markers (case-insensitive
    plain ASCII finds; Gutenberg's markers are the canonical license edges)."""
    low = text.lower()
    start = low.find("*** start of the project gutenberg ebook")
    end = low.find("*** end of the project gutenberg")
    if start == -1 or end == -1 or end <= start:
        return text, False
    body = text[start:end]
    body = body.split("\n", 1)[1]      # drop the marker line itself
    return body, True


def fetch(url: str, dest: Path, retries: int = 3) -> int:
    if dest.exists() and dest.stat().st_size > 0:
        return dest.stat().st_size
    import urllib.error
    for attempt in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "IECNN-corpus-builder/1"})
            with urllib.request.urlopen(req, timeout=180) as resp:
                data = resp.read()
            dest.write_bytes(data)
            return len(data)
        except Exception as exc:                      # noqa: BLE001 - retry anything
            if attempt == retries - 1:
                raise
            print(f"  retry {attempt + 1} after error: {exc}", flush=True)
            time.sleep(3 * (attempt + 1))
    return 0


def main() -> None:
    args = sys.argv[1:]
    dest_dir = Path(args[args.index("--dest") + 1]) if "--dest" in args else Path("data_big")
    val_f = float(args[args.index("--val") + 1]) if "--val" in args else 0.005
    test_f = float(args[args.index("--test") + 1]) if "--test" in args else 0.005

    raw_dir = dest_dir / "raw"
    raw_dir.mkdir(parents=True, exist_ok=True)

    manifest, bodies = [], []
    total_stripped = 0
    for pg_id, title in BOOKS:
        dest = raw_dir / f"pg{pg_id}.txt"
        size = fetch(gutenberg_url(pg_id), dest)
        text = dest.read_text(encoding="utf-8", errors="replace")
        body, ok = strip_gutenberg_boilerplate(text)
        cleaned = body.replace("\r\n", "\n").replace("\x1a", "")
        cleaned = __import__("unicodedata").normalize("NFC", cleaned)
        stripped_bytes = len(cleaned.encode("utf-8"))
        total_stripped += stripped_bytes
        sha = hashlib.sha256(cleaned.encode("utf-8")).hexdigest()[:16]
        manifest.append({"pg_id": pg_id, "title": title, "raw_bytes": size,
                         "stripped_bytes": stripped_bytes, "body_sha16": sha,
                         "markers_found": ok})
        bodies.append(cleaned)
        print(f"pg{pg_id:>6}  {size:>9,} raw -> {stripped_bytes:>9,} chars "
              f"{'OK' if ok else 'MARKERS NOT FOUND'}  {title}", flush=True)
        time.sleep(2.0)

    corpus = "\n".join(bodies)
    audit = audit_text(corpus)

    train, val, test = split_text(corpus, val_f, test_f)
    n = len(corpus)

    paths, shas = {}, {}
    for name, part in (("train", train), ("val", val), ("test", test)):
        p = dest_dir / f"{name}.txt"
        p.write_text(part, encoding="utf-8")
        shas[name] = hashlib.sha256(part.encode("utf-8")).hexdigest()
        paths[name] = str(p)

    info = {
        "corpus": "public-domain-english-v1",
        "books": manifest,
        "total_chars": n,
        "total_raw_bytes": sum(b["raw_bytes"] for b in manifest),
        "total_stripped_bytes": total_stripped,
        "audit": audit,
        "split_fraction": {"val": val_f, "test": test_f},
        "files": paths,
        "sha256": shas,
    }
    (dest_dir / "corpus_info.json").write_text(json.dumps(info, indent=1))
    print("\n=== SUMMARY ===")
    print(f"books: {len(manifest)}  | markers missing: "
          f"{sum(1 for b in manifest if not b['markers_found'])}")
    print(f"total chars: {n:,}  (train {len(train):,} / val {len(val):,} / "
          f"test {len(test):,})")
    print(f"audit unique chars: {audit['num_unique_chars_nfc']}  "
          f"printable: {audit['fully_printable']}")
    print(f"sha256: {json.dumps(shas, indent=1)}")


if __name__ == "__main__":
    main()
