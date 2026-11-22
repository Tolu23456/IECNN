"""Matched-compute tables from an ab_small run log.

The objective is bpc per unit compute, but a 900 s batch is equal WALL CLOCK:
arms with a cheaper step simply train on more tokens. This reads the JSON row a
kernel prints at the end and reports bpc at matched TOTAL MACs (tokens x
MACs/token), which is the comparison the objective actually asks for.

Two protocol rules are encoded rather than left to the caller:

  * Markers only. Every arm's last logged marker is a raw eval; `bpc_final` is
    SWA-averaged and is 0.045-0.061 better than that marker on every arm in
    every run so far. Mixing the SWA endpoint into an interpolation changes the
    eval protocol mid-curve, so it is excluded unless --with-final is passed.
  * No extrapolation. A cell is printed as "--" when the budget lies outside an
    arm's measured token range, so a number never appears beyond the data.

The MAC map is the one `scripts/flops_audit.py` computes; it lives here as a
single dict so token->compute conversions cannot drift between analyses. (A
first version of this table was off by 1000x because 147.6M tokens x 430,080
MACs is 63.5 TERA-MACs, not 63.5 giga.)

Usage: python3 scripts/mac_table.py log.txt [--with-final] [--budgets 25,50,...]
       (log.txt is the raw `kaggle kernels logs` output; JSON-per-line)
"""
from __future__ import annotations

import argparse
import json
import sys

# MACs per token at T=256, from scripts/flops_audit.py.
TF_MACS = 430_080
X5_MACS = 829_760
X4_MACS = 698_688
X3_MACS = 567_616


def macs_per_token(name: str) -> int:
    if name.startswith("tf"):
        return TF_MACS
    if "-x4-" in name:
        return X4_MACS
    if "-x3-" in name:
        return X3_MACS
    return X5_MACS


def short(name: str) -> str:
    return (name.replace("iecnn-n3-", "").replace("iecnn-", "")
            .replace("-p1-ln-decay-d128-w512-rr2-c", ""))


def _concat_data(raw: str) -> str:
    """Kernel logs are either one JSON array or one JSON object per line."""
    try:
        obj = json.loads(raw)
        if isinstance(obj, list):
            return "".join(e.get("data", "") for e in obj if isinstance(e, dict))
        if isinstance(obj, dict):
            return obj.get("data", "")
    except Exception:
        pass
    out = []
    for line in raw.splitlines():
        line = line.strip().lstrip(",")
        if not line:
            continue
        try:
            out.append(json.loads(line).get("data", ""))
        except Exception:
            pass
    return "".join(out)


def parse_rows(text: str):
    """The kernel prints one `JSON [...]` line inside the log stream."""
    rows = None
    for line in _concat_data(text).splitlines():
        if line.startswith("JSON "):
            rows = json.loads(line[5:])
    if not rows:
        raise SystemExit("no JSON row found in the log")
    return rows


def points(row, with_final: bool):
    xs = [c[0] for c in row["curve"]]
    ys = [c[1] for c in row["curve"]]
    if with_final:
        xs.append(row["tokens"])
        ys.append(row["bpc_final"])
    return xs, ys


def bpc_at(row, tok: float, with_final: bool):
    xs, ys = points(row, with_final)
    if tok < xs[0] or tok > xs[-1]:
        return None                      # no extrapolation, by rule
    for i in range(len(xs) - 1):
        if xs[i] <= tok <= xs[i + 1]:
            f = 0.0 if xs[i + 1] == xs[i] else (tok - xs[i]) / (xs[i + 1] - xs[i])
            return ys[i] + f * (ys[i + 1] - ys[i])
    return None


def macs_to(row, target: float, with_final: bool):
    """Total MACs (T) at which the arm first reaches `target` raw bpc."""
    xs, ys = points(row, with_final)
    m = macs_per_token(row["arm"])
    for i in range(len(xs) - 1):
        if ys[i] >= target >= ys[i + 1]:
            f = (ys[i] - target) / (ys[i] - ys[i + 1]) if ys[i] != ys[i + 1] else 0
            return (xs[i] + f * (xs[i + 1] - xs[i])) * m / 1e12
    return None


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("log")
    ap.add_argument("--with-final", action="store_true",
                    help="include the SWA endpoint (changes eval protocol)")
    ap.add_argument("--budgets", default="25,50,75,100,125,150,175,200")
    ap.add_argument("--targets", default="2.10,2.05,2.00,1.95,1.90")
    ap.add_argument("--base", default="tf-swa-c")
    args = ap.parse_args()
    rows = parse_rows(open(args.log, errors="replace").read())
    budgets = [float(b) for b in args.budgets.split(",")]
    targets = [float(t) for t in args.targets.split(",")]

    print(f"{'arm':<24}{'seed':>4}{'params':>8}{'Mtok':>9}{'s/step':>8}"
          f"{'bpc':>8}{'T-MAC':>8}")
    for r in rows:
        print(f"{short(r['arm'])[:24]:<24}{r['seed']:>4}{r['params']:>8}"
              f"{r['tokens'] / 1e6:>9.1f}{r['s_per_step']:>8.4f}"
              f"{r['bpc_final']:>8.4f}"
              f"{r['tokens'] * macs_per_token(r['arm']) / 1e12:>8.1f}")

    print(f"\nbpc at matched TOTAL MACs"
          f"{' (incl. SWA endpoint)' if args.with_final else ' (raw markers)'}")
    print(f"{'arm':<24}{'seed':>4}" + "".join(f"{b:>7.0f}T" for b in budgets))
    for r in rows:
        line = f"{short(r['arm'])[:24]:<24}{r['seed']:>4}"
        for b in budgets:
            v = bpc_at(r, b * 1e12 / macs_per_token(r["arm"]), args.with_final)
            line += f"{v:>8.4f}" if v is not None else f"{'--':>8}"
        print(line)

    base = next((r for r in rows if r["arm"] == args.base), None)
    if base is not None:
        bm = macs_per_token(base["arm"])
        ref = {b: bpc_at(base, b * 1e12 / bm, args.with_final) for b in budgets}
        print(f"\ndelta vs {args.base} (negative = ahead)")
        print(f"{'arm':<24}{'seed':>4}" + "".join(f"{b:>7.0f}T" for b in budgets))
        for r in rows:
            if r is base:
                continue
            line = f"{short(r['arm'])[:24]:<24}{r['seed']:>4}"
            for b in budgets:
                v = bpc_at(r, b * 1e12 / macs_per_token(r["arm"]), args.with_final)
                line += (f"{v - ref[b]:>+8.4f}"
                         if (v is not None and ref[b] is not None) else f"{'--':>8}")
            print(line)

    print("\nMACs (T) to reach a target raw bpc")
    for t in targets:
        print(f"  {t:.2f}:", end="")
        for r in rows:
            got = macs_to(r, t, args.with_final)
            if got is None:
                print(f"  {short(r['arm'])[:16]}=never", end="")
            else:
                print(f"  {short(r['arm'])[:16]}={got:.1f}", end="")
        print()


if __name__ == "__main__":
    main()
