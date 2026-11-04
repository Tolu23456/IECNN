"""Train a UBM vocabulary (deterministic byte-level BPE merge set) on a corpus.

Usage: python3 scripts/train_vocab.py --train data/train.txt \
        --val data/val.txt --size 512 --out UBM/vocab_v1.json

Saves the vocabulary and prints/writes a compression report (average
characters per token on train/val, like the v0 report).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from UBM.train_bpe import train_ubm_v0  # noqa: E402
from UBM.tokenizer import UBM  # noqa: E402


def compression_stats(ubm: UBM, text: str, name: str) -> dict:
    """chars / token over the whole text (the v0 report's statistic)."""
    t0 = time.time()
    ids = ubm.encode(text)
    chars_per_token = len(text) / max(1, len(ids))
    out = {"split": name, "chars": len(text), "tokens": len(ids),
           "chars_per_token": round(chars_per_token, 4),
           "bits_per_char_if_uniform": round(
               __import__("math").log2(len(ubm.bases)) / chars_per_token, 4),
           "encode_seconds": round(time.time() - t0, 1)}
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--train", required=True)
    ap.add_argument("--val", required=True)
    ap.add_argument("--size", type=int, default=512)
    ap.add_argument("--out", default="UBM/vocab_v1.json")
    args = ap.parse_args()

    text = Path(args.train).read_text(encoding="utf-8")
    print(f"corpus: {len(text):,} chars -> training {args.size}-base vocabulary",
          flush=True)
    t0 = time.time()
    ubm, merges = train_ubm_v0(text, target_vocab_size=args.size)
    ubm.save(args.out)
    print(f"vocabulary saved: {args.out} ({len(ubm.bases)} bases, "
          f"{len(merges)} merges) in {time.time()-t0:.0f}s", flush=True)

    val_text = Path(args.val).read_text(encoding="utf-8")
    report = {
        "vocab_file": args.out,
        "bases": len(ubm.bases),
        "merges": len(merges),
        "first_20_merges": [m.bytes.decode("utf-8", "replace") for m in merges[:20]],
        "stats": [compression_stats(ubm, text, "train"),
                  compression_stats(ubm, val_text, "val")],
    }
    out_report = args.out.replace("UBM/", "results/").replace(".json", "")
    report_path = out_report + "_compression.json"
    Path(report_path).parent.mkdir(parents=True, exist_ok=True)
    Path(report_path).write_text(json.dumps(report, indent=1))
    print(json.dumps(report["stats"], indent=1))
    print("report:", report_path)


if __name__ == "__main__":
    main()
