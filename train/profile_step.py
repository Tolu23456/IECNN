"""Component timing profile for IECNN runs (diagnostic only; no training).

Dispatched on GitHub Actions instead of the trainer, e.g.

    python -m train.profile_step --config configs/e1_weights_passes.yaml --seed 0

Times each part of the training step at full size: the attention read, the
GRU, the output heads, the consensus machinery, the aux losses, forward as a
whole, and backward — so a measured 10x slowdown can be attributed to a real
component instead of guessed at. Results go to stdout and results/profile_*.txt.
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from config import load_config
from data.dataset import Batcher
from train.losses import main_loss
from utils import set_seed, set_torch_threads


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="checkpoints/profile")
    ap.add_argument("--steps", type=int, default=4)
    ap.add_argument("--resume", action="store_true")     # accepted, ignored
    ap.add_argument("--max-steps", default="")           # accepted, ignored
    args = ap.parse_args()

    set_torch_threads(2)
    set_seed(args.seed)
    config = load_config(args.config, *args.overlay)
    model_cfg = dict(config["model"])
    model_cfg["kind"] = "iecnn"

    from models.iecnn import build_iecnn

    model = build_iecnn({"model": model_cfg, "consensus": config.get("consensus", {}),
                         "data": config.get("data", {})}, 512)
    model.train()
    opt = torch.optim.AdamW(model.parameters(), lr=1e-4)

    # synthetic batch of the real shape: [64, 256] within the 512 vocab
    rng = np.random.default_rng(args.seed)
    batch = (rng.integers(0, 512, size=(64, 257))).astype(np.int64)
    x = batch[:, :-1]
    y = batch[:, 1:]

    timers: dict[str, list[float]] = {}

    def timed(name, fn, *a, **k):
        t0 = time.perf_counter()
        out = fn(*a, **k)
        dt = time.perf_counter() - t0
        timers.setdefault(name, []).append(dt)
        return out

        # warmup (allocator, einsum paths)
    print("profile: warmup forward/backward", flush=True)
    out = model(torch.from_numpy(x))
    loss = main_loss(out["final_log_probs"], torch.from_numpy(y))
    loss.backward()
    opt.zero_grad(set_to_none=True)

    for i in range(args.steps):
        if i % 1 == 0:
            print(f"profile step {i} starting", flush=True)
        xb, yb = torch.from_numpy(x), torch.from_numpy(y)

        # ---- forward, split into the pieces Region.forward does ----------
        print("profile: split forward", flush=True)
        def fwd():
            emb_table = model.embed.weight
            rows = model.embed(xb)
            rops = []
            for r, region in enumerate(model.regions):
                wind_k, wind_v, valid = timed(f"r{r}.window_proj",
                                              region._window_memory, rows, xb.device)
                b, t = rows.shape[0], rows.shape[1]
                state = region.dots.initial_state((b, t))
                trust = None
                for p in range(1, region.n_passes + 1):
                    tau_p = 1.0 * (region.kappa ** (p - 1))
                    if p == 1 or not region.use_weights:
                        h_pool = torch.zeros(b, t, region.n_dots, region.d_s)
                        err = torch.zeros(b, t, region.n_dots, 3)
                    else:
                        from models import consensus
                        err = consensus.error_features(
                            pi_prev, consensus.leave_one_out_mean(pi_prev))
                    o = timed(f"r{r}.p{p}.dots_forward",
                              region.dots, wind_k, wind_v, valid, h_pool, err,
                              round_idx=p - 1, msg_in=None, state=state,
                              emb_table=emb_table)
                    state_prev, pi_prev = o.state, o.log_pi.exp()
                    if region.use_weights:
                        from models import consensus
                        scores = timed(f"r{r}.p{p}.consensus",
                                       consensus.compute_scores, pi_prev, o.gamma,
                                       o.trust_term, region.alpha, region.beta)
                        trust = consensus.update_trust(trust, scores, region.rho)
                        consensus.compute_weights(trust, tau_p)
                    else:
                        from models import consensus
                        consensus.uniform_weights(region.n_dots, pi_prev)
                    pooled = torch.einsum("bfn,bfnd->bfd",
                                          torch.full(o.state.shape[:-1], 1.0 / region.n_dots),
                                          state_prev)
                    if p == region.n_passes:
                        rops.append((region, o, state_prev, pooled))
                if r + 1 < len(model.regions):
                    rows = region.chain_ln[min(r, len(region.chain_ln) - 1)](
                        region.chain_proj[min(r, len(region.chain_proj) - 1)](state_prev * 0) + rows)
            return rops

        rops = timed("forward_full", lambda: timed("forwardRegions", fwd))

        # full training-style forward to get real grads (the split above is
        # approximate; the full one is what the trainer actually runs)
        def real_fwd():
            return model(xb, tau0=1.0, floor=0.1 / _n_dots(model))

        out = timed("forward_model_real", real_fwd)
        loss = timed("main_loss", main_loss, out["final_log_probs"], yb)

        lambdas = {k: float(config.get("training", {}).get(k, 0.0))
                   for k in ("lambda_d", "lambda_k", "lambda_w")}
        if any(v > 0 for v in lambdas.values()):
            timed("aux_losses", _aux, out, yb, lambdas)

        opt.zero_grad(set_to_none=True)
        timed("backward", loss.backward)
        timed("clip+step", lambda: (torch.nn.utils.clip_grad_norm_(
            model.parameters(), 0.5), opt.step()))
        opt.zero_grad(set_to_none=True)

    # ---- report ---------------------------------------------------------
    lines = [f"params {sum(p.numel() for p in model.parameters()):,}"]
    for name, ts in sorted(timers.items()):
        med = sorted(ts)[len(ts) // 2]
        lines.append(f"{name:40s} median {med*1000:9.1f} ms   (n={len(ts)})")
    total_fwd = np.median(timers["forward_model_real"])
    total_bwd = np.median(timers["backward"])
    lines.append(f"{'TOTAL fwd+bwd':40s} median {(total_fwd + total_bwd)*1000:9.1f} ms/step")
    text = "\n".join(lines)
    print(text)
    Path("results").mkdir(exist_ok=True)
    name = Path(args.config).stem
    Path(f"results/profile_{name}_seed{args.seed}.txt").write_text(text + "\n")


def _n_dots(model) -> int:
    return int(model.regions[0].dots.N)


def _aux(out, yb, lambdas):
    from train.losses import (boosted_dot_loss, confidence_loss,
                             weight_entropy_bonus)
    last = out["per_pass_last_region"][-1]
    if lambdas["lambda_d"] > 0:
        boosted_dot_loss(last["log_pi"], out["final_log_probs"], yb, 2.0)
    if lambdas["lambda_k"] > 0:
        confidence_loss(last["gamma"], last["log_pi"], yb)
    if lambdas["lambda_w"] > 0:
        weight_entropy_bonus(last["weights"])


if __name__ == "__main__":
    main()
