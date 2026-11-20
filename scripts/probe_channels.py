"""Which channel is the bottleneck? Ablate a trained checkpoint, no retraining.

The 3 h run says the dot model underfits (train 1.8073 vs tf 1.7119), so the
question is which part of the pipe cannot carry more. The thinnest links are
the SHARED projections: w_k (128 -> 16) and w_v (128 -> 32), i.e. every dot
reads the same 16 key numbers and the same 32 value numbers per slot.

This measures how much of each channel the trained model actually uses, by
EXACT subspace truncation at the weights -- no forward patch, no hook, no
change to models/:

    v = rows @ W_v^T                                    [d_v]
    keep only the subspace spanned by orthonormal P      [d_v, k]:
    v' = P P^T v    <=>    W_v <- P P^T W_v

and the same projection on the consumer side (the d_v rows of the cell input
weight, the d_v columns of the reread output, the d_v rows of w_v again --
P P^T is idempotent, so applying it on both sides is still exactly one
projection). With P = I every tensor is bit-identical, so every delta is
measured against a base pass that must reproduce the checkpoint's own val bpc.

Bases compared per k -- variance is not value, so a low-variance direction is
not assumed to be useless:
  pca   top-k principal directions of the model's own value distribution
  rand  a random k-subspace (control: is this just "damage"?)
  tail  the last k canonical dims (the naive "chop the end" ablation)

Usage:
  PYTHONPATH=. python3 scripts/probe_channels.py ckpt.pt [--windows 160]
Reads the same corpus/val knobs as scripts/ab_small.py.
"""
from __future__ import annotations

import argparse
import json
import math
import os

import torch
import torch.nn.functional as F

from models.baselines import build_transformer_baseline
from models.iecnn import build_iecnn
from models.memory import region1_rows

V = 256
T = int(os.environ.get("ARM_T", "256"))
VAL_PATH = os.environ.get("ARM_VAL", "data/toy_val.txt")
N_VAL = int(os.environ.get("ARM_NVAL", 5_000_000))
VALW = int(os.environ.get("ARM_VALW", "160"))
VALSTRIDE = int(os.environ.get("ARM_VALSTRIDE", "1"))


# --------------------------------------------------------------------------
# measurement primitives
# --------------------------------------------------------------------------
def spectrum(x: torch.Tensor) -> dict:
    """Eigenvalues of the covariance of rows of x, plus summaries.

    participation ratio = (sum l)^2 / sum l^2, the standard "how many
    directions is this really using" number: r for r equal eigenvalues, 1 for
    rank 1. Reported next to the cumulative variance curve because the two
    disagree exactly when a few directions dominate.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    cov = x.t() @ x / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)   # descending
    tot = ev.sum().clamp_min(1e-12)
    frac = (ev / tot).tolist()
    cum = torch.tensor(frac).cumsum(0).tolist()
    pr = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12))
    return {"dim": int(ev.numel()), "participation_ratio": round(pr, 3),
            "top1": round(frac[0], 4), "top4": round(cum[min(3, len(cum) - 1)], 4),
            "top8": round(cum[min(7, len(cum) - 1)], 4),
            "top16": round(cum[min(15, len(cum) - 1)], 4),
            "frac": [round(f, 5) for f in frac]}


def basis_pca(x: torch.Tensor, k: int) -> torch.Tensor:
    """Top-k principal directions of x's rows: [d, k], orthonormal."""
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    cov = x.t() @ x / max(x.shape[0] - 1, 1)
    _, evec = torch.linalg.eigh(cov.cpu())
    return evec.flip(1)[:, :k]


def basis_random(d: int, k: int, seed: int = 0) -> torch.Tensor:
    g = torch.Generator().manual_seed(seed)
    q, _ = torch.linalg.qr(torch.randn(d, k, generator=g))
    return q


def basis_tail(d: int, k: int) -> torch.Tensor:
    """The last k canonical dims -- what "chop the end" actually does."""
    return torch.eye(d)[:, d - k:]


def project_out(w: torch.Tensor, P: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Zero the component of `w` outside span(P) along `dim`.

    dim=0: w is [d, ...] (a projection OUTPUT, e.g. w_v), so P P^T on the left.
    dim=-1: w is [..., d] (a projection INPUT, e.g. rr_o), so P P^T on the right.
    """
    M = P @ P.t()
    return M @ w if dim == 0 else w @ M


# --------------------------------------------------------------------------
# which tensors to touch
# --------------------------------------------------------------------------
def _find(sd: dict, suffix: str, n: int = 1) -> list:
    hits = [k for k in sd if k.endswith(suffix)]
    if len(hits) != n:
        raise SystemExit(f"state_dict has {len(hits)} keys ending in {suffix!r}, "
                         f"expected {n}: {hits}")
    return hits


def truncate_values(sd: dict, P: torch.Tensor) -> dict:
    """Restrict the VALUE channel v (d_v) to span(P), exactly.

    Four places consume the value vector, all of them linear in v:
      * w_v            -> v                    (the projection itself)
      * dots.w_ih(_lug) rows [:d_v]             (the cell input, ctx block)
      * rr_v / rr_o    -> the reread's own value channel (same d_v)
    Projecting the projection and the consumer is one projection, not two:
    (P P^T)(P P^T) = P P^T.
    """
    sd = {k: v.clone() for k, v in sd.items()}
    for key in _find(sd, ".w_v.weight"):
        pre = key[:-len("w_v.weight")]
        dv = sd[key].shape[0]
        sd[key] = project_out(sd[key], P, 0)
        for cell_key in (pre + "dots.w_ih_lug", pre + "dots.w_ih"):
            if cell_key in sd:
                w = sd[cell_key]
                sd[cell_key] = torch.cat(
                    [project_out(w[..., :dv, :], P, -2), w[..., dv:, :]], dim=-2)
    for key in [k for k in sd if k.endswith(".rr_v.weight")]:
        pre = key[:-len("rr_v.weight")]
        dv = sd[key].shape[0]
        sd[key] = project_out(sd[key], P, 0)
        sd[pre + "rr_o.weight"] = project_out(
            sd[pre + "rr_o.weight"], P, -1)
        assert sd[pre + "rr_o.weight"].shape[-1] == dv, (pre, dv)
    return sd


def truncate_keys(sd: dict, P: torch.Tensor) -> dict:
    """Restrict the KEY channel k (d_q) to span(P).

    Only w_k (and the reread's rr_k) project into key space; the consumer is a
    dot product against a query, so there is no weight matrix on the other
    side. Truncating the key subspace is therefore strictly a loss of geometry
    -- it says "how much do the directions outside P matter", not "how wide
    should the pipe be".
    """
    sd = {k: v.clone() for k, v in sd.items()}
    for key in _find(sd, ".w_k.weight") + _find(sd, ".rr_k.weight"):
        sd[key] = project_out(sd[key], P, 0)
    return sd


# --------------------------------------------------------------------------
# evaluation
# --------------------------------------------------------------------------
def load_model(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, kind = ck["cfg"], ck["kind"]
    torch.manual_seed(0)
    build = build_transformer_baseline if kind == "transformer" else build_iecnn
    model = build(cfg, V).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def val_bpc(model, val: torch.Tensor, windows: int, device) -> float:
    tot, n = 0.0, 0
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            y = val[i + 1:i + 1 + T].unsqueeze(0).to(device)
            logp = model(x)["final_log_probs"].reshape(-1, V)
            tot += float(F.nll_loss(logp, y.reshape(-1), reduction="sum"))
            n += y.numel()
    return tot / n / math.log(2)


def collect_vectors(model, val: torch.Tensor, windows: int, device) -> dict:
    """Value / key / ctx / state rows over the val windows (one pass)."""
    r = model.regions[0]
    outs = {"v": [], "k": [], "ctx": [], "s": []}
    with torch.no_grad():
        for w in range(windows):
            i = w * T * VALSTRIDE
            x = val[i:i + T].unsqueeze(0).to(device)
            rows = region1_rows(model.embed(x))
            k_all = r.w_k(rows)
            v_all = r.w_v(rows)
            wind_k, wind_v, valid = r._window_memory(
                rows, device, 0, k_all=k_all, v_all=v_all)
            # Same read as DotPopulation.forward, reused here so the ctx
            # spectrum is the vector the cell actually receives (window read
            # only: the global read is a separate, already O(T) path).
            q = r.dots._sel(0, r.dots.q)
            sc = torch.einsum("bfwj,nj->bfnw", wind_k, q) / math.sqrt(r.d_q)
            sc = sc.masked_fill(~valid[None, None, None, :], float("-inf"))
            a = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
            ctx = torch.matmul(a.unsqueeze(-2), wind_v.unsqueeze(-3)).squeeze(-2)
            out = model(x)
            outs["v"].append(v_all)
            outs["k"].append(k_all)
            outs["ctx"].append(ctx)
            outs["s"].append(out["region_outputs"][0]["region_summary"])
    return {k: torch.cat(v, dim=0).reshape(-1, v[-1].shape[-1]).float()
            for k, v in outs.items()}


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--windows", type=int, default=VALW)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    model, ck = load_model(args.ckpt, device)
    row = ck.get("row", {})
    base_sd = ck["state_dict"]

    base = val_bpc(model, val, args.windows, device)
    print(f"=== {os.path.basename(args.ckpt)}  {ck['kind']}  "
          f"windows {args.windows}x{T}", flush=True)
    print(f"base val {base:.4f} bpc (checkpoint row said "
          f"{row.get('bpc_final', 'n/a')})", flush=True)

    vec = collect_vectors(model, val, min(args.windows, 40), device)
    print("\n-- channel spectra (covariance participation ratio) --", flush=True)
    for name in ("v", "k", "ctx", "s"):
        s = spectrum(vec[name])
        print(f"{name:>4} dim {s['dim']:>3}  PR {s['participation_ratio']:>6.2f}"
              f"  cumvar top4 {s['top4']:.3f} top8 {s['top8']:.3f}"
              f" top16 {s['top16']:.3f}", flush=True)

    def run_basis(kind: str, d: int, ks: list) -> None:
        print(f"\n-- {kind} channel truncation (delta vs base, + = worse) --",
              flush=True)
        for k in ks:
            for basis_name in ("pca", "rand", "tail"):
                if basis_name == "pca":
                    P = basis_pca(vec["v" if kind == "value" else "k"], k)
                elif basis_name == "rand":
                    P = basis_random(d, k, seed=1000 + k)
                else:
                    P = basis_tail(d, k)
                if k == d:                      # identity check, must be 0
                    P = torch.eye(d)
                sd = (truncate_values(base_sd, P) if kind == "value"
                      else truncate_keys(base_sd, P))
                model.load_state_dict(sd)
                b = val_bpc(model, val, args.windows, device)
                print(f"  {kind:>5} k {k:>2}/{d}  {basis_name:<4}"
                      f"  {b:.4f}  {b - base:+.4f}", flush=True)
                model.load_state_dict(base_sd)

    dv = base_sd[_find(base_sd, ".w_v.weight")[0]].shape[0]
    dq = base_sd[_find(base_sd, ".w_k.weight")[0]].shape[0]
    run_basis("value", dv, [1, 2, 4, 6, 8, 12, 16, 24, 32, dv] if dv == 32
              else [1, 2, 4, 8, dq])
    run_basis("key", dq, [1, 2, 4, 6, 8, 12, 16] if dq == 16 else [1, 2, 4, 8, dq])

    model.load_state_dict(base_sd)
    final = val_bpc(model, val, args.windows, device)
    print(f"\nrestored val {final:.4f} bpc (drift {final - base:+.4f})", flush=True)
    print("JSON " + json.dumps({"ckpt": os.path.basename(args.ckpt),
                                "base": round(base, 4),
                                "restored": round(final, 4),
                                "spectra": {n: spectrum(vec[n]) for n in
                                            ("v", "k", "ctx", "s")}}), flush=True)


if __name__ == "__main__":
    main()
