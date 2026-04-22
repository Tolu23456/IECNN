#!/usr/bin/env python3
"""
Download a public English text corpus and write the first N non-empty lines
to a plain-text file (one sentence-ish unit per line).

Defaults to WikiText-2 raw (CC-BY-SA), 10,000 lines, output: corpus_10k.txt.
"""
import argparse
import os
import re
import sys
import urllib.request

WIKITEXT2_URL = (
    "https://huggingface.co/datasets/Salesforce/wikitext/resolve/main/"
    "wikitext-2-raw-v1/test-00000-of-00001.parquet"
)
# Easier: use a pre-extracted plain-text mirror.
WIKITEXT2_PLAIN_URL = (
    "https://raw.githubusercontent.com/pytorch/examples/main/word_language_model/"
    "data/wikitext-2/train.txt"
)


def download(url: str, dest: str) -> None:
    print(f"[corpus] Downloading {url}")
    req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0 IECNN-corpus"})
    with urllib.request.urlopen(req, timeout=60) as resp, open(dest, "wb") as fh:
        fh.write(resp.read())
    sz = os.path.getsize(dest)
    print(f"[corpus] Wrote {dest} ({sz/1024:.1f} KB)")


_HEADER_RE = re.compile(r"^=.*=\s*$")


def extract_lines(src: str, dst: str, limit: int) -> int:
    sent_split = re.compile(r"(?<=[.!?])\s+")
    kept = 0
    with open(src, "r", encoding="utf-8", errors="replace") as fh, \
         open(dst, "w", encoding="utf-8") as out:
        for raw in fh:
            line = raw.strip()
            if not line or _HEADER_RE.match(line):
                continue
            # Split paragraphs into sentence-ish chunks
            for chunk in sent_split.split(line):
                chunk = chunk.strip()
                # Keep moderate-length lines
                if 20 <= len(chunk) <= 400:
                    out.write(chunk + "\n")
                    kept += 1
                    if kept >= limit:
                        return kept
    return kept


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", default="corpus_10k.txt", help="Output plain-text file")
    ap.add_argument("--limit", type=int, default=10000, help="Max non-empty lines")
    ap.add_argument("--raw", default="wikitext2_raw.txt",
                    help="Where to cache the raw download")
    args = ap.parse_args()

    if not os.path.exists(args.raw):
        try:
            download(WIKITEXT2_PLAIN_URL, args.raw)
        except Exception as e:
            print(f"[corpus] Download failed: {e}", file=sys.stderr)
            sys.exit(1)
    else:
        print(f"[corpus] Using cached raw file: {args.raw}")

    n = extract_lines(args.raw, args.out, args.limit)
    print(f"[corpus] Wrote {n} lines to {args.out}")


if __name__ == "__main__":
    main()
