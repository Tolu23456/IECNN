"""What does each dot actually add? Ablate a trained checkpoint, no retraining.

Motivation: three measurements now point the same way --

  * every session has one dot parked on distance 0 (`frac_end` 0.97-1.00),
  * `input_res` removes that dot and all three become the SAME distance-1
    reader (`distinct_per_pos` 2.97 -> 1.0),
  * head capacity (x5) pays 0.046 while adding dots never has.

so the question is no longer "are the dots specialised" but the harder one:
**does the committee use more than one dot at all?** A diversity score cannot
answer that. This does, by changing only the pool weights on the trained path:

  drop-one   remove dot k from the consensus and renormalise (`ablate_dot=k`).
             The cost is what that dot contributes to the committee.
  alone      force a one-hot pool onto dot k (`pool_only_dot=k`) and run the
             SAME trained pool->head stack. This is the honest per-dot quality
             number: per-dot logits are an untrained side output in arms that
             train with `per_dot_logits=False`.
  oracle     per position, take the LOWEST per-dot nll. An upper bound on what
             any routing could extract from these dots; if it sits at the
             committee, routing is not the missing piece and the dots carry no
             complementary information.

Plus the gate's own allocation (mean weight per dot, effective N = 1/sum w^2)
and the read geometry, so "which dot is the redundant one" is answerable.

Usage:
  PYTHONPATH=. python3 scripts/probe_population.py ckpt.pt [--windows 96]
Reads the same corpus/val knobs as scripts/ab_small.py, so the val slice and
stride match the run being audited.
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


def load_model(path: str, device):
    ck = torch.load(path, map_location="cpu", weights_only=False)
    cfg, kind = ck["cfg"], ck["kind"]
    torch.manual_seed(0)
    build = build_transformer_baseline if kind == "transformer" else build_iecnn
    model = build(cfg, V).to(device)
    model.load_state_dict(ck["state_dict"])
    model.eval()
    return model, ck


def windows(val: torch.Tensor, n: int, device):
    for w in range(n):
        i = w * T * VALSTRIDE
        if i + T + 1 > val.numel():
            return
        yield (val[i:i + T].unsqueeze(0).to(device),
               val[i + 1:i + 1 + T].unsqueeze(0).to(device))


def nats_per_token(model, val, n, device, **kw) -> torch.Tensor:
    """Per-position nats (not bits), so callers can combine position-wise."""
    out = []
    with torch.no_grad():
        for x, y in windows(val, n, device):
            lp = model(x, **kw)["final_log_probs"].reshape(-1, V)
            out.append(F.nll_loss(lp, y.reshape(-1), reduction="none").reshape(-1))
    return torch.cat(out) if out else torch.zeros(0)


def js(p, q, eps=1e-12):
    m = 0.5 * (p + q)
    kl = lambda a, b: (a.clamp_min(eps) * ((a + eps) / (b + eps)).log()).sum(-1)
    return (0.5 * kl(p, m) + 0.5 * kl(q, m)).clamp_min(0.0).mean().item()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("ckpt")
    ap.add_argument("--windows", type=int, default=96)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    device = torch.device(args.device)

    raw = open(VAL_PATH, "rb").read(N_VAL)
    val = torch.frombuffer(raw, dtype=torch.uint8).long()
    model, ck = load_model(args.ckpt, device)
    if "regions.0.dots.q" not in ck["state_dict"]:
        raise SystemExit("checkpoint has no dot population (not an IECNN arm)")
    row = ck.get("row", {})
    reg, dots = model.regions[0], model.regions[0].dots
    n = dots.N

    base_nats = nats_per_token(model, val, args.windows, device)
    ntok = base_nats.numel()
    base = float(base_nats.mean()) / math.log(2)

    print(f"=== {os.path.basename(args.ckpt)}  {n} dots  n_passes "
          f"{getattr(reg, 'n_passes', '?')}  windows {args.windows}x{T}",
          flush=True)
    print(f"base val {base:.4f} bpc  "
          f"(checkpoint row said {row.get('bpc_final', 'n/a')})", flush=True)

    drop, alone = {}, {}
    alone_nats = {}
    for k in range(n):
        d = nats_per_token(model, val, args.windows, device, ablate_dot=k)
        a = nats_per_token(model, val, args.windows, device, pool_only_dot=k)
        drop[k] = float((d.mean() - base_nats.mean())) / math.log(2)
        alone[k] = float(a.mean()) / math.log(2)
        alone_nats[k] = a
        print(f"drop dot {k}: {drop[k]:+.4f} bpc   alone: {alone[k]:.4f} bpc",
          flush=True)

    # oracle per position over the SAME per-dot one-hot streams
    stack = torch.stack([alone_nats[k] for k in range(n)], dim=-1)
    oracle = float(stack.min(-1).values.mean()) / math.log(2)
    best_mean = min(alone.values())
    print(f"\ncommittee {base:.4f} | best single dot {best_mean:.4f} | "
          f"per-position oracle {oracle:.4f}", flush=True)
    print(f"routing headroom (committee - oracle): {base - oracle:+.4f} bpc",
          flush=True)

    # gate allocation + read geometry on the trained path
    wsum = torch.zeros(n)
    wsq = torch.zeros(n)
    r_ent = torch.zeros(n)
    r_dist = torch.zeros(n)
    r_end = torch.zeros(n)
    read_p = [[] for _ in range(n)]
    with torch.no_grad():
        for x, _y in windows(val, args.windows, device):
            out = model(x)
            w = out["per_pass_last_region"][-1]["weights"].reshape(-1, n)
            wsum += w.sum(0).cpu()
            wsq += (w ** 2).sum(0).cpu()
            rows = region1_rows(model.embed(x))
            wk, _wv, valid = reg._window_memory(rows, x.device)
            q = dots._sel(0, dots.q)
            sc = torch.einsum("bfwj,nj->bfnw", wk, q) / math.sqrt(dots.d_q)
            sc = sc.masked_fill(~valid[None, :, None, :], float("-inf"))
            if getattr(dots, "no_lag0", False) and wk.shape[2] > 1:
                sc = sc.masked_fill(
                    torch.arange(wk.shape[2], device=sc.device)
                    == wk.shape[2] - 1, float("-inf"))
            p = torch.nan_to_num(sc.softmax(-1), nan=0.0)
            p = p.reshape(-1, n, p.shape[-1])
            r_ent += (-(p.clamp_min(1e-12).log() * p).sum(-1)).sum(0).cpu()
            offs = reg._window_offsets(x.device, 0)
            idx = p.argmax(-1)
            r_dist += (-offs)[idx].clamp_min(0).sum(0).float().cpu()
            r_end += (idx == p.shape[-1] - 1).sum(0).float().cpu()
            for k in range(n):
                read_p[k].append(p[:, k])

    wbar = (wsum / ntok)
    eff = float(1.0 / (wbar ** 2).sum())
    print(f"\ngate: mean weight {[round(float(v), 3) for v in wbar]}  "
          f"effective N {eff:.2f} of {n}", flush=True)
    print("read: dot  dist_back  frac_end  entropy", flush=True)
    for k in range(n):
        print(f"      {k}    {float(r_dist[k] / ntok):.2f}"
              f"       {float(r_end[k] / ntok):.3f}"
              f"     {float(r_ent[k] / ntok):.3f}", flush=True)
    for a in range(n):
        for b in range(a + 1, n):
            v = js(torch.cat(read_p[a]), torch.cat(read_p[b]))
            print(f"      read JS dot {a} vs {b}: {v:.4f}", flush=True)

    # ------------------------------------------------------------------
    # IS THE ABLATION MEASURING THE DOT, OR THE HEAD'S OFF-MANIFOLD INPUT?
    #
    # The 3 h rr2-c checkpoint gives drop-one +4.60/+1.62/+0.34 bpc and any
    # dot ALONE 4.0-7.9 bpc against a 1.83 committee. A convex blend of three
    # near-uninformative states cannot be 3 nats better than each of them, so
    # one of these is true:
    #   (a) the pool WEIGHTS carry context (they are a 2-dof function of the
    #       position), and removing/normalising them deletes information;
    #   (b) the head is brittle: it only ever saw near-uniform weights, so any
    #       reweighting is off-manifold and the result is garbage.
    # A time-shuffle of w separates them: same values, same marginal, but the
    # per-position pairing with the context is destroyed. If shuffled w costs
    # like drop-one, the weights are information (a). If it stays at the
    # committee, the weights are only a mixture and the ablation is (b).
    orig_hidden = reg._pass_hidden
    perm = torch.randperm(n)          # fixed, so the control is reproducible

    def patched(mode):
        def fn(gam, trust_term, trust, mask, tau, floor):
            tr, w = orig_hidden(gam, trust_term, trust, mask, tau, floor)
            if mode == "uniform":
                return tr, torch.full_like(w, 1.0 / n)
            if mode == "shuffle_t":
                return tr, w.flip(1)
            if mode == "shuffle_n":
                return tr, w[..., perm.to(w.device)]
            if mode.startswith("floor"):
                k = int(mode[-1])
                keep = torch.ones_like(w)
                keep[..., k] = 0.1 / n
                w = w * keep
                return tr, w / w.sum(-1, keepdim=True).clamp_min(1e-9)
            raise ValueError(mode)
        return fn

    wmodes = ["uniform", "shuffle_t", "shuffle_n"] + [f"floor{k}" for k in range(n)]
    wrep = {}
    print("\n-- pool-weight controls (bpc; delta vs committee) --", flush=True)
    for mode in wmodes:
        reg._pass_hidden = patched(mode)
        try:
            b = float(nats_per_token(model, val, min(args.windows, 64),
                                     device).mean()) / math.log(2)
        finally:
            reg._pass_hidden = orig_hidden
        wrep[mode] = {"bpc": b, "delta": b - base}
        print(f"{mode:<12} {b:.4f}  {b - base:+.4f}", flush=True)

    rep = {
        "ckpt": os.path.basename(args.ckpt), "dots": n,
        "windows": args.windows, "tokens": ntok,
        "bpc": base, "best_single_dot_bpc": best_mean, "oracle_bpc": oracle,
        "drop_one_delta_bpc": {str(k): drop[k] for k in range(n)},
        "alone_bpc": {str(k): alone[k] for k in range(n)},
        "mean_weight": [float(v) for v in wbar],
        "effective_dots": eff,
        "dot_dist_back": [float(r_dist[k] / ntok) for k in range(n)],
        "dot_frac_end": [float(r_end[k] / ntok) for k in range(n)],
        "dot_read_entropy": [float(r_ent[k] / ntok) for k in range(n)],
        "pool_weight_controls": wrep,
    }
    out = os.environ.get("POP_OUT")
    if out:
        with open(out, "w") as f:
            json.dump(rep, f, indent=1)
        print("WROTE", out, flush=True)


if __name__ == "__main__":
    main()
