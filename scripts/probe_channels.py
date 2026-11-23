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

and the same projection on the consumer side (the d_v rows of the cell input,
the d_v columns of the reread output). P P^T is idempotent, so applying it on
both sides is still exactly ONE projection. With P = I every tensor is
bit-identical, so the k = d row of every table is the zero point.

WHAT IS AND IS NOT ABLATED
  value  the dot channel (w_v -> cell input) AND, when the arm has one, the
         reread's own value channel (rr_v -> rr_o). Both are needed: if only
         the dot channel is cut, the reread still reads the raw rows and a
         null result could not tell "32 dims are enough" from "the reread
         supplies the rest".
  key    the dot channel (w_k) only. rr_k is a separate projection with its
         own geometry; cutting half of one key space and not the other would
         report a mixture of two different questions.

Bases compared per k -- variance is not value, so a low-variance direction is
not assumed to be useless:
  pca     top-k principal directions of the model's own activation
  rand    a random k-subspace (control: is this just "damage"?)
  prefix  the first k canonical dims, i.e. what "chop the channel" does

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
    rank 1. Reported next to the cumulative curve because the two disagree
    exactly when a few directions dominate.
    """
    x = x.reshape(-1, x.shape[-1]).float()
    x = x - x.mean(0, keepdim=True)
    cov = x.t() @ x / max(x.shape[0] - 1, 1)
    ev = torch.linalg.eigvalsh(cov).flip(0).clamp_min(0)   # descending
    frac = (ev / ev.sum().clamp_min(1e-12)).tolist()
    cum = torch.tensor(frac).cumsum(0).tolist()
    pr = float(ev.sum() ** 2 / (ev ** 2).sum().clamp_min(1e-12))
    return {"dim": int(ev.numel()), "participation_ratio": round(pr, 3),
            "cum": [round(c, 4) for c in cum],
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


def basis_prefix(d: int, k: int) -> torch.Tensor:
    """The first k canonical dims -- what "chop the channel" does."""
    return torch.eye(d)[:, :k]


def project_out(w: torch.Tensor, P: torch.Tensor, dim: int = 0) -> torch.Tensor:
    """Zero the component of `w` outside span(P) along `dim`.

    dim=0/-2: `w` indexes the space on a ROW axis -- the result is P P^T w,
    which torch broadcasts over any leading dims (the cell input is
    [p, n, in_dim, 2*d_s] and only its first d_v rows are the ctx block).
    dim=-1: the space is the LAST axis (rr_o is [.., d_out, d_v]), so the
    projection goes on the right: w P P^T.
    """
    M = P @ P.t()
    return M @ w if dim in (0, -2) else w @ M


# --------------------------------------------------------------------------
# which tensors to touch
# --------------------------------------------------------------------------
def _find(sd: dict, suffix: str) -> list:
    """Every key ending in `suffix`; at least one is required."""
    hits = [k for k in sd if k.endswith(suffix)]
    if not hits:
        raise SystemExit(f"state_dict has no key ending in {suffix!r}")
    return hits


def truncate_values(sd: dict, P: torch.Tensor, P_rr: torch.Tensor | None = None) -> dict:
    """Restrict the value channel(s) to span(P) / span(P_rr), exactly.

    Two independent value channels, each linear in its own v:
      * dot w_v -> cell input rows [:d_v]         (P,     the shared pipe)
      * reread rr_v -> rr_o input columns [:dv]   (P_rr,  optional, own width)
    The reread is cut too when a basis is given: leaving it intact would let
    the model recover the value information through a second path and make a
    null result uninterpretable.
    """
    sd = {k: v.clone() for k, v in sd.items()}
    for key in _find(sd, ".w_v.weight"):
        pre = key[:-len("w_v.weight")]
        dv = sd[key].shape[0]
        if P.shape[0] != dv:
            raise SystemExit(f"basis {tuple(P.shape)} does not match {key} "
                             f"dim {dv}")
        sd[key] = project_out(sd[key], P, 0)
        for cell_key in (pre + "dots.w_ih_lug", pre + "dots.w_ih"):
            if cell_key in sd:
                w = sd[cell_key]
                sd[cell_key] = torch.cat(
                    [project_out(w[..., :dv, :], P, -2), w[..., dv:, :]], dim=-2)
    for key in [k for k in sd if k.endswith(".rr_v.weight")]:
        if P_rr is None:
            continue
        pre = key[:-len("rr_v.weight")]
        dv = sd[key].shape[0]
        if P_rr.shape[0] != dv:
            raise SystemExit(f"reread basis {tuple(P_rr.shape)} does not match "
                             f"{key} dim {dv}")
        sd[key] = project_out(sd[key], P_rr, 0)
        sd[pre + "rr_o.weight"] = project_out(
            sd[pre + "rr_o.weight"], P_rr, -1)
    return sd


def truncate_keys(sd: dict, P: torch.Tensor) -> dict:
    """Restrict the DOT key channel (w_k) to span(P).

    The consumer is a dot product against a query, so there is no weight
    matrix on the other side: truncating the key subspace is strictly a loss
    of geometry. rr_k is deliberately left alone (see the module docstring).
    """
    sd = {k: v.clone() for k, v in sd.items()}
    for key in _find(sd, ".w_k.weight"):
        if P.shape[0] != sd[key].shape[0]:
            raise SystemExit(f"basis {tuple(P.shape)} does not match {key}")
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
    """Value / key / ctx / state / reread-value rows over the val windows."""
    r = model.regions[0]
    outs = {"v": [], "k": [], "ctx": [], "s": [], "rv": []}
    got_rv = []
    hook = (r.rr_v.register_forward_hook(
        lambda _m, _i, o: got_rv.append(o.detach())) if r.rr_v is not None
        else None)
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
            # valid is [T, span] (see gather_windows), so the mask is
            # [1, T, 1, span] against scores [B, T, N, span].
            sc = sc.masked_fill(~valid[None, :, None, :], float("-inf"))
            a = torch.nan_to_num(F.softmax(sc, dim=-1), nan=0.0)
            ctx = torch.matmul(a.unsqueeze(-2), wind_v.unsqueeze(-3)).squeeze(-2)
            out = model(x)
            outs["v"].append(v_all)
            outs["k"].append(k_all)
            outs["ctx"].append(ctx)
            outs["s"].append(out["region_outputs"][0]["region_summary"])
    if hook is not None:
        hook.remove()
        outs["rv"] = got_rv
    return {k: torch.cat(v, dim=0).reshape(-1, v[-1].shape[-1]).float()
            for k, v in outs.items() if v}


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
    base_sd = ck["state_dict"]
    row = ck.get("row", {})

    base = val_bpc(model, val, args.windows, device)
    print(f"=== {os.path.basename(args.ckpt)}  {ck['kind']}  "
          f"windows {args.windows}x{T}", flush=True)
    print(f"base val {base:.4f} bpc (checkpoint row said "
          f"{row.get('bpc_final', 'n/a')})", flush=True)
    if base_sd.get("regions.0.w_v.weight") is None:
        raise SystemExit("no dots value channel in this checkpoint (not IECNN)")

    vec = collect_vectors(model, val, min(args.windows, 24), device)
    print("\n-- channel spectra (covariance participation ratio) --", flush=True)
    for name in ("v", "k", "ctx", "s", "rv"):
        if name not in vec:
            continue
        s = spectrum(vec[name])
        c = s["cum"]
        print(f"{name:>4} dim {s['dim']:>3}  PR {s['participation_ratio']:>6.2f}"
              f"  cumvar@4 {c[min(3, len(c) - 1)]:.3f}"
              f"  @8 {c[min(7, len(c) - 1)]:.3f}"
              f"  @16 {c[min(15, len(c) - 1)]:.3f}", flush=True)

    dv = base_sd[_find(base_sd, ".w_v.weight")[0]].shape[0]
    dq = base_sd[_find(base_sd, ".w_k.weight")[0]].shape[0]
    rv_keys = [k for k in base_sd if k.endswith(".rr_v.weight")]
    drr = base_sd[rv_keys[0]].shape[0] if rv_keys else None
    # the hook only fires if the arm actually runs its reread sites (an arm
    # that builds rr_* but never calls them has no recordable channel)
    if "rv" not in vec:
        drr = None

    def basis(kind: str, act: str, d: int, k: int) -> torch.Tensor:
        if k >= d:
            return torch.eye(d)
        if kind == "pca":
            return basis_pca(vec[act], k)
        if kind == "rand":
            return basis_random(d, k, seed=1000 + k)
        return basis_prefix(d, k)

    def run_basis(which: str, dims: list, ks: list, act: str,
                  act_rr: str | None = None, d_rr: int | None = None) -> None:
        print(f"\n-- {which} channel truncation (delta vs base, + = worse) --",
              flush=True)
        for k in ks:
            for kind in ("pca", "rand", "prefix"):
                P = basis(kind, act, dims[0], k)
                P_rr = (basis(kind, act_rr, d_rr, min(k, d_rr))
                        if d_rr is not None else None)
                sd = (truncate_values(base_sd, P, P_rr) if which == "value"
                      else truncate_keys(base_sd, P))
                model.load_state_dict(sd)
                b = val_bpc(model, val, args.windows, device)
                tag = f"k {k:>2}/{dims[0]}"
                if d_rr is not None:
                    tag += f" (rr {min(k, d_rr)}/{d_rr})"
                print(f"  {which:>5} {tag:<16} {kind:<6} {b:.4f}"
                      f"  {b - base:+.4f}", flush=True)
                model.load_state_dict(base_sd)

    run_basis("value", [dv], [1, 2, 4, 6, 8, 12, 16, 24, 32, dv], "v",
              act_rr="rv", d_rr=drr)
    run_basis("key", [dq], [1, 2, 4, 6, 8, 12, dq], "k")

    model.load_state_dict(base_sd)
    final = val_bpc(model, val, args.windows, device)
    print(f"\nrestored val {final:.4f} bpc (drift {final - base:+.4f})", flush=True)
    print("JSON " + json.dumps({
        "ckpt": os.path.basename(args.ckpt), "base": round(base, 4),
        "restored": round(final, 4),
        "spectra": {n: spectrum(vec[n]) for n in vec}}), flush=True)


if __name__ == "__main__":
    main()
