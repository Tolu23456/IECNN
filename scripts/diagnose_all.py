"""Run the whole spec-08 §6 diagnostic suite on one checkpoint (runner job).

Usage:
  python scripts/diagnose_all.py --ckpt-dir checkpoints/<run>_seed0 \
      [--vocab UBM/vocab_v1.json] [--config-fallback configs/<rung>.yaml] \
      [--out results/diagnostics/<run>] [--max-chars 8000] [--seq-len N]

Writes one JSON per diagnostic + summary.json, and prints a one-line verdict
per diagnostic. Any diagnostic that errors is recorded, not fatal — one bad
module must not hide the other seven.
"""
from __future__ import annotations

import argparse
import importlib
import json
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

MODULES = ["ablation", "calibration", "dispersion", "error_correlation",
           "message_influence", "mode_gap", "region_weights", "win_map"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt-dir", required=True)
    ap.add_argument("--vocab", default=None)
    ap.add_argument("--config-fallback", default=None)
    ap.add_argument("--out", required=True)
    ap.add_argument("--max-chars", type=int, default=8000)
    ap.add_argument("--seq-len", type=int, default=None)
    ap.add_argument("--text-file", default="data/val.txt")
    args = ap.parse_args()

    text = Path(args.text_file).read_text(encoding="utf-8")
    text = text[: args.max_chars * 4]          # raw char headroom for expand

    results = {}
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)   # summary.json needs the dir
                                        # even when every module errors
    print(f"diagnose-all on {args.ckpt_dir} "
          f"(<= {args.max_chars} chars per diagnostic)", flush=True)
    for name in MODULES:
        mod = importlib.import_module(f"diagnostics.{name}")
        t0 = time.perf_counter()
        try:
            summary = mod.run(args.ckpt_dir, args.config_fallback, args.out,
                              vocab=args.vocab, text=text,
                              max_chars=args.max_chars,
                              seq_len=args.seq_len)
            status = "skipped" if summary.get("skipped") else "ok"
        except Exception as exc:                        # noqa: BLE001
            summary = {"error": f"{type(exc).__name__}: {exc}"}
            status = "ERROR"
        results[name] = summary
        dt = time.perf_counter() - t0
        print(f"  [{status:>7}] {name:<18} {dt:6.1f}s", flush=True)
        if status == "ERROR":
            print(f"           {summary['error']}", flush=True)

    out = Path(args.out)
    (out / "summary.json").write_text(json.dumps(
        {"ckpt_dir": args.ckpt_dir, "max_chars": args.max_chars,
         "results": results}, indent=1, default=str))
    ok = [n for n, s in results.items() if not s.get("error")
          and not s.get("skipped")]
    print(f"done: {len(ok)}/{len(MODULES)} diagnostics succeeded -> {out}",
          flush=True)


if __name__ == "__main__":
    main()
