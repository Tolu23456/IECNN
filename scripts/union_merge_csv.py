#!/usr/bin/env python3
"""Union-merge a git-conflicted results CSV (train.yml commit step, 2026-11-14).

Two chained chunks can add the SAME results/*.csv path: chunk A's checkout
predates chunk B's results commit, so `git pull --rebase` aborts with
"untracked working tree files would be overwritten" and, once committed
first, hits a real add/add conflict. This script resolves it by union:
dedupe rows by the `step` column (first field), keep the LAST occurrence of
a step (the fresher chunk's row), sort rows by step ascending. Conflict
markers are dropped. Anything that is not a step-keyed CSV fails loudly.

Usage: python3 union_merge_csv.py <file.csv> [<file2.csv> ...]
"""
import sys
from pathlib import Path


def union_merge(path: Path) -> int:
    lines = path.read_text().splitlines()
    header, rows, seen, order = None, {}, set(), []
    for ln in lines:
        s = ln.strip()
        if not s or s.startswith(("<<<<<<<", "=======", ">>>>>>>")):
            continue
        if header is None and s.startswith("step"):
            header = s
            continue
        if s.startswith("step"):
            continue
        step = s.split(",", 1)[0]
        if step in seen:
            order.remove(step)      # keep LAST occurrence (freshest chunk)
        seen.add(step)
        order.append(step)
        rows[step] = s
    if header is None:
        print(f"UNION_MERGE_FAIL: {path} has no 'step' header — refusing",
              file=sys.stderr)
        return 1
    try:
        order.sort(key=int)
    except ValueError:
        pass
    path.write_text("\n".join([header] + [rows[k] for k in order]) + "\n")
    print(f"union-merged {path}: {len(order)} rows")
    return 0


def main() -> int:
    if len(sys.argv) < 2:
        print("usage: union_merge_csv.py <file.csv> [...]", file=sys.stderr)
        return 2
    rc = 0
    for arg in sys.argv[1:]:
        rc |= union_merge(Path(arg))
    return rc


if __name__ == "__main__":
    sys.exit(main())
