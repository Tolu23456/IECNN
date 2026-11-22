"""Does a LONGER exact context pay? The k of the k-gram, which was never swept.

Every copy arm in the registry is `suffix_copy=3`, and the k was never chosen by
measurement — it was chosen as "three bytes is enough to be specific". This
sweeps k = 2..6 on trained checkpoints with every weight frozen, and prices each
k the same way twice:

  * best CONSTANT gate (a post-hoc ceiling, so the k curve is not confounded by
    a gate that was trained for k=3);
  * the checkpoint's own learned (a, b), which is the deployable read.

Two facts make the answer predictable in advance, and the probe is designed to
distinguish them rather than confirm them:

  * a longer k is a stricter requirement, so coverage must fall and precision
    must rise. Which effect wins is an empirical question about this corpus.
  * extension-based SELECTION (already measured at -0.0062..-0.0076) is the
    soft version of the same idea: keep the occurrence whose match extends
    furthest rather than demanding a longer exact match from every position.
    So the interesting comparison is longer-k against extvote, not against the
    single pointer.

The `k=3, ext>=m` rows are the bridge: they show what a hard threshold at
"match at least k-3 more bytes" would have kept, i.e. the coverage a k+m-gram
demands, measured on the same pointer set.

Causality and the hash are the model's own (`_suffix_copy`), mirrored here only
because that method is bound to `self.suffix_copy`; a self-check asserts the
k=3 case is bit-identical to the model's, so this cannot drift from the mechanism.

Usage: PYTHONPATH=. python3 scripts/probe_copy_k.py ckpt.pt [...]
Reads ARM_T / ARM_VAL / ARM_NVAL / ARM_VALW / ARM_VALSTRIDE / COPY_EXTRA_CAP.
"""
from __future__ import annotations

import argparse
import math
import os

import torch

from models.iecnn import build_iecnn
from models.baselines import build_transformer_baseline

V = 256
T = int(os.environ.get("ARM_T", "256"))
CAP = int(os.environ.get("COPY_EXTRA_CAP", "8"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", "5_000_000"))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "122"))
KS = (2, 3, 4, 5, 6)
GATES = [i / 20 for i in range(0, 20)] + [0.99]


def suffix_k(x: torch.Tensor, k: int, cap: int):
    """Pointer, validity and extension for an exact k-gram match.

    Mirrors `Region._suffix_copy` exactly (same rolling base-256 hash, same
    `s < t` / `s + 1 <= t` mask, same most-recent-wins, same contiguous
    extension), with k as a parameter.
    """
    b, t = x.shape
    zero = torch.zeros(b, 1, dtype=x.dtype, device=x.device)
    h = torch.zeros_like(x)
    for j in range(k):
        h = h + torch.cat((zero.expand(b, j), x), dim=1)[:, :t] * (256 ** j)
    idx = torch.arange(t, device=x.device)
    full = idx >= (k - 1)
    cand = (h.unsqueeze(2) == h.unsqueeze(1))
    cand = cand & full.view(1, 1, t) & full.view(1, t, 1)
    cand = cand & (idx.view(1, 1, t) < idx.view(1, t, 1))
    pos = torch.where(cand, idx.view(1, 1, t).expand(b, t, t),
                      torch.full((), -1, dtype=torch.long, device=x.device))
    s, _ = pos.max(dim=-1)
    valid = s >= 0
    src = (s + 1).clamp(min=0, max=t - 1)
    copy = torch.where(valid, x.gather(1, src), torch.zeros_like(x))
    extra = torch.zeros_like(x)
    for m in range(cap):
        ai = idx.view(1, t) - k - m
        si = s - k - m
        av = x.gather(1, ai.clamp_min(0).expand(b, t))
        sv = x.gather(1, si.clamp_min(0))
        ok = (ai >= 0) & (si >= 0) & (av == sv) & (extra == m)
        extra = extra + ok.to(extra.dtype)
    return copy, valid, extra


def bpc(p_true: torch.Tensor, mass: torch.Tensor, g: torch.Tensor) -> float:
    mix = (1 - g) * p_true + g * mass
    return float(-torch.log2(mix.clamp_min(1e-9)).mean())


def gate(extra, a, b, valid, cap):
    logit = (torch.full_like(extra, a, dtype=torch.float32) if b is None
             else a + b * extra.clamp(max=cap).to(torch.float32))
    return torch.sigmoid(logit) * valid.to(torch.float32)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpts", nargs="+")
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(bytearray(raw), dtype=torch.uint8).to(torch.int64)
    starts = [w * T * VALSTRIDE for w in range(VALW)
              if w * T * VALSTRIDE + T + 1 <= val.numel()]
    y = torch.cat([val[i + 1:i + 1 + T] for i in starts]).to(device)
    print(f"val {len(starts)}x{T} = {len(starts) * T} positions | stride "
          f"{VALSTRIDE} | cap {CAP} | {len(raw)} bytes", flush=True)

    for path in args.ckpts:
        ck = torch.load(path, map_location="cpu", weights_only=False)
        build = (build_transformer_baseline if ck["kind"] == "transformer"
                 else build_iecnn)
        model = build(ck["cfg"], V).to(device)
        model.load_state_dict(ck["state_dict"])
        model.eval()
        name = os.path.basename(path).replace(".pt", "")
        r = model.regions[0] if getattr(model, "regions", None) else None
        print(f"\n===== {name}", flush=True)
        if r is None or getattr(r, "sc_gate", None) is None:
            print("  no copy channel (base only)", flush=True)
            continue
        a = float(r.sc_gate.detach())
        b = float(r.sc_slope.detach()) if r.sc_slope is not None else None
        print(f"  learned gate a={a:.4f} b="
              f"{'none' if b is None else f'{b:.4f}'}", flush=True)

        # base p from the same weights with the gate shut
        saved_a = r.sc_gate.detach().clone()
        saved_b = r.sc_slope.detach().clone() if r.sc_slope is not None else None
        if r.sc_slope is not None:
            r.sc_slope.data.zero_()
        r.sc_gate.data.fill_(-50.0)
        with torch.no_grad():
            p = torch.cat([model(val[i:i + T].unsqueeze(0).to(device))
                           ["final_log_probs"][0].exp() for i in starts])
        r.sc_gate.data.copy_(saved_a)
        if saved_b is not None:
            r.sc_slope.data.copy_(saved_b)
        p_true = p.gather(1, y.unsqueeze(1)).squeeze(1)
        shut = bpc(p_true, torch.zeros_like(y, dtype=torch.float32),
                   torch.zeros_like(y, dtype=torch.float32))
        print(f"  base (gate shut) {shut:.4f}", flush=True)

        rows = {}
        for k in KS:
            cp, va, ex = [], [], []
            with torch.no_grad():
                for i in starts:
                    x = val[i:i + T].unsqueeze(0).to(device)
                    c, v, e = suffix_k(x, k, CAP)
                    cp.append(c[0]); va.append(v[0]); ex.append(e[0])
            rows[k] = (torch.cat(cp), torch.cat(va), torch.cat(ex))

        # self-check: k=3 must be the model's own channel, bit for bit
        cp3, va3, ex3 = [], [], []
        with torch.no_grad():
            for i in starts:
                x = val[i:i + T].unsqueeze(0).to(device)
                c, v, e = r._suffix_copy(x, with_extra=True)
                cp3.append(c[0]); va3.append(v[0]); ex3.append(e[0])
        c3, v3, e3 = rows[3]
        assert torch.equal(c3, torch.cat(cp3)), "copy differs from the model"
        assert torch.equal(v3, torch.cat(va3)), "valid differs from the model"
        assert torch.equal(e3, torch.cat(ex3)), "extension differs from the model"
        print("  self-check ok: k=3 reproduces _suffix_copy exactly", flush=True)

        ref = bpc(p_true, (c3 == y).to(torch.float32), gate(e3, a, b, v3, CAP))
        print(f"\n  {'k':>3}{'cover':>8}{'acc':>7}{'best g':>8}{'ceiling':>10}"
              f"{'gain':>8}{'at learned':>12}{'vs k3':>9}", flush=True)
        for k in KS:
            c, v, e = rows[k]
            mass = (c == y).to(torch.float32)
            best = min(((bpc(p_true, mass, g * v.to(torch.float32)), g)
                        for g in GATES))
            got = bpc(p_true, mass, gate(e, a, b, v, CAP))
            print(f"  {k:>3}{100 * float(v.float().mean()):>7.1f}%"
                  f"{100 * float((c == y)[v].float().mean()):>6.1f}%"
                  f"{best[1]:>8.2f}{best[0]:>10.4f}"
                  f"{shut - best[0]:>+8.4f}{got:>12.4f}{got - ref:>+9.4f}",
                  flush=True)

        # the bridge: a hard threshold on the k=3 pointer's extension
        c, v, e = rows[3]
        print(f"\n  {'k3 & ext>=':>11}{'cover':>8}{'acc':>7}{'best g':>8}"
              f"{'ceiling':>10}{'gain':>8}", flush=True)
        for m in (0, 1, 2, 3, 4):
            msk = v & (e >= m)
            mass = ((c == y) & msk).to(torch.float32)
            best = min(((bpc(p_true, mass, g * msk.to(torch.float32)), g)
                        for g in GATES))
            print(f"  {m:>11}{100 * float(msk.float().mean()):>7.1f}%"
                  f"{100 * float((c == y)[msk].float().mean()):>6.1f}%"
                  f"{best[1]:>8.2f}{best[0]:>10.4f}{shut - best[0]:>+8.4f}",
                  flush=True)


if __name__ == "__main__":
    main()
