"""Speed benchmark for one config on the runner (speed goal C1/C2, 2026-11-14).

Times real training steps of `train.train`'s step body (same building blocks:
set_seed, threads, UBM vocab, batcher, model, tau/floor/hard schedules, aux
losses, grad-norm buckets, clip, AdamW step) so the numbers reflect what a
real run costs, then reports s/step, tok/s, projected wall-to-600-steps and
the cost of one eval pair.

Usage:
  python scripts/bench.py --config configs/e1_weights_passes_vocab2.yaml \
      [--overlay tiny.yaml] --tag baseline_vocab2 [--steps 40] [--warmup 5]

Writes results/bench_<tag>.json. This mirrors train_one's step body rather
than calling it so mid-run evals/checkpoints do not pollute the timing; if
train.py's step body changes materially, update both.
"""
from __future__ import annotations

import argparse
import json
import statistics
import subprocess
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from config import load_config
from data.dataset import Batcher, tokenize_corpus
from train.losses import main_loss
from train.evaluate import evaluate_bpc, evaluate_bpc_per_pass, prepare_text
from train.schedules import (temperature_at, weight_floor_at, use_hard_batch,
                             lr_at, lr_at_constant)
from utils import set_seed, set_torch_threads
from train.train import (build_model, _iecnn_n_dots, grad_norms_by_component)


def build_setup(config: dict, seed: int) -> dict:
    """Build everything one timed step needs (model, batcher, schedules)."""
    set_seed(seed)
    set_torch_threads(int(config.get("training", {}).get("torch_threads", 2)))

    vocab_file = config.get("ubm", {}).get("vocab_file", "UBM/vocab_v0.json")
    from UBM import UBM
    ubm = UBM.load(vocab_file)
    vocab_size = len(ubm.bases)

    train_text = Path(config.get("data", {}).get(
        "train_file", "data/train.txt")).read_text(encoding="utf-8")
    val_text = Path(config.get("data", {}).get(
        "val_file", "data/val.txt")).read_text(encoding="utf-8")
    train_ids = tokenize_corpus(train_text, vocab_size,
                                cache_dir=config.get("data", {}).get("dir", "data"),
                                vocab_file=vocab_file)

    tcfg = config.get("training", {})
    seq_len = int(config.get("data", {}).get("seq_len", 256))
    peak_lr = float(tcfg.get("lr", 1e-3))
    warmup_steps = int(tcfg.get("warmup_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    lr_sched = str(tcfg.get("lr_schedule", "cosine"))
    lr_fn = lr_at_constant if lr_sched == "constant" else lr_at
    precision = str(tcfg.get("precision", "fp32")).lower()
    use_bf16 = precision == "bf16"
    batch_size = int(tcfg.get("batch_size", 16))

    batcher = Batcher(train_ids, seq_len, batch_size, seed=seed, shuffle=True)
    model = build_model(config, vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr,
                            betas=tuple(tcfg.get("betas", (0.9, 0.95))),
                            weight_decay=float(tcfg.get("weight_decay", 0.01)))
    model.train()

    is_iecnn = config["model"]["kind"] == "iecnn"
    cons = config.get("consensus", {})
    tau_max = float(cons.get("tau_max", 1.0))
    tau_min = float(cons.get("tau_min", 0.2))
    tau_warmup = int(cons.get("tau_warmup", 2000))
    lam_ramp = int(cons.get("lambda_c_ramp_steps", 5000))
    floor_frac = float(cons.get("weight_floor_fraction", 0.1))
    hard_start = float(tcfg.get("hard_batch_start", 0.7))
    hard_frac = float(tcfg.get("hard_batch_fraction", 0.2))
    hard_rng = __import__("random").Random(seed * 7919 + 3)
    total_steps = int(tcfg.get("steps", 1000))
    lambdas = {k: float(tcfg.get(k, 0.0)) for k in
               ("lambda_r", "lambda_d", "lambda_c", "lambda_k", "lambda_w",
                "lambda_o", "lambda_v")}
    outvote_weight = float(tcfg.get("outvote_weight", 3.0))
    gamma_b = float(tcfg.get("gamma_b", 2.0))
    from train.schedules import correlation_lambda, weight_entropy_lambda
    from train.losses import (region_loss, boosted_dot_loss, decorrelation_penalty,
                              confidence_loss, weight_entropy_bonus,
                              oracle_dot_loss, verifier_loss)

    def run_step(step: int) -> None:
        # paired A/B: whichever setup was built last would otherwise own the
        # thread count for both; pin it per step (cheap no-op when equal)
        want = int(config.get("training", {}).get("torch_threads", 2))
        if want > 0 and torch.get_num_threads() != want:
            torch.set_num_threads(want)
        lr = lr_fn(step, total_steps, warmup_steps, peak_lr)
        for group in opt.param_groups:
            group["lr"] = lr
        x, y = batcher.next_batch()
        xb = torch.from_numpy(np.ascontiguousarray(x))
        yb = torch.from_numpy(np.ascontiguousarray(y))
        kwargs = {}
        if is_iecnn:
            tau0 = temperature_at(step, total_steps, tau_warmup, tau_max, tau_min)
            floor = weight_floor_at(step, total_steps,
                                    floor_frac / _iecnn_n_dots(model))
            hard = use_hard_batch(step, total_steps, hard_start, hard_frac, hard_rng)
            kwargs = {"tau0": tau0, "hard_mode": bool(hard), "floor": floor}
        with torch.autocast("cpu", dtype=torch.bfloat16, enabled=use_bf16):
            out = model(xb, **kwargs)
            loss = main_loss(out["final_log_probs"], yb)
            if is_iecnn and any(v != 0 for v in lambdas.values()):
                last_pass = out["per_pass_last_region"][-1]
                if lambdas["lambda_r"] > 0 and len(out["region_outputs"]) > 1:
                    stack = torch.stack([ro["log_b"] for ro in out["region_outputs"]])
                    loss = loss + lambdas["lambda_r"] * region_loss(stack, yb)
                if lambdas["lambda_d"] > 0:
                    loss = loss + lambdas["lambda_d"] * boosted_dot_loss(
                        last_pass["log_pi"], out["final_log_probs"], yb, gamma_b)
                if lambdas["lambda_c"] > 0:
                    lc_val = correlation_lambda(step, lam_ramp, lambdas["lambda_c"])
                    loss = loss + lc_val * decorrelation_penalty(
                        last_pass["log_pi"], out["final_log_probs"], yb)
                if lambdas["lambda_k"] > 0:
                    loss = loss + lambdas["lambda_k"] * confidence_loss(
                        last_pass["gamma"], last_pass["log_pi"], yb)
                if lambdas["lambda_w"] > 0:
                    lw_val = weight_entropy_lambda(step, total_steps,
                                                   lambdas["lambda_w"])
                    loss = loss - lw_val * weight_entropy_bonus(last_pass["weights"])
                if lambdas["lambda_o"] > 0:
                    loss = loss + lambdas["lambda_o"] * oracle_dot_loss(
                        last_pass["log_pi"], out["final_log_probs"], yb,
                        outvote_weight)
                if lambdas["lambda_v"] > 0 and last_pass.get("verifier") is not None:
                    loss = loss + lambdas["lambda_v"] * verifier_loss(
                        last_pass["verifier"], last_pass["log_pi"],
                        out["final_log_probs"], yb, outvote_weight)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        grad_norms_by_component(model)
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()

    return {
        "config": config, "model": model, "opt": opt, "batcher": batcher,
        "ubm": ubm, "vocab_size": vocab_size, "train_text": train_text,
        "val_text": val_text, "seq_len": seq_len,
        "batch_size": batcher.batch_size, "precision": precision,
        "run_step": run_step, "is_iecnn": is_iecnn,
        "tau_max": tau_max, "tau_min": tau_min, "tau_warmup": tau_warmup,
        "total_steps": total_steps,
        "eval_chars": int(tcfg.get("eval_chars", 12000)),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", default="")
    ap.add_argument("--tag", required=True)
    ap.add_argument("--steps", type=int, default=40)
    ap.add_argument("--warmup", type=int, default=5)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="")
    ap.add_argument("--profile", type=int, default=0,
                    help="after timing, profile N steps with torch.profiler "
                         "and attach the top ops by self CPU time to the JSON")
    ap.add_argument("--ab-config", default="",
                    help="paired A/B: a SECOND config timed in the SAME "
                         "process/runner, steps interleaved (kills the ~40%% "
                         "cross-runner fleet variance)")
    ap.add_argument("--ab-overlay", default="",
                    help="overlay for the --ab-config model")
    args = ap.parse_args()

    overlays = [args.overlay] if args.overlay else []
    config = load_config(args.config, *overlays)
    su_a = build_setup(config, args.seed)
    su_b = None
    if args.ab_config:
        ab_overlays = [args.ab_overlay] if args.ab_overlay else []
        cfg_b = load_config(args.ab_config, *ab_overlays)
        su_b = build_setup(cfg_b, args.seed)

    # ---- warmup (JIT-ish allocations, cache warm) ------------------------
    for step in range(args.warmup):
        su_a["run_step"](step)
    if su_b is not None:
        for step in range(args.warmup):
            su_b["run_step"](step)

    def time_steps(su, steps):
        times = []
        for _ in range(steps):
            t0 = time.perf_counter()
            su["run_step"](0)
            times.append(time.perf_counter() - t0)
        return times

    if su_b is None:
        step_times = time_steps(su_a, args.steps)
        ab_times = None
    else:
        # interleaved A/B on the same runner: A,B,A,B... per step index
        step_times, ab_times = [], []
        for _ in range(args.steps):
            t0 = time.perf_counter()
            su_a["run_step"](0)
            step_times.append(time.perf_counter() - t0)
            t0 = time.perf_counter()
            su_b["run_step"](0)
            ab_times.append(time.perf_counter() - t0)
    train_wall = sum(step_times)
    tokens_timed = args.steps * su_a["batch_size"] * su_a["seq_len"]

    # ---- one eval pair, timed (val subsample + train slice) --------------
    ubm, val_text, train_text = su_a["ubm"], su_a["val_text"], su_a["train_text"]
    eval_chars = su_a["eval_chars"]
    model = su_a["model"]
    prepared_val = prepare_text(ubm, val_text)
    eval_slice = train_text[:max(eval_chars * 6, 512_000)]
    prepared_train = prepare_text(ubm, eval_slice)
    model.eval()
    tau0 = temperature_at(args.steps, su_a["total_steps"], su_a["tau_warmup"],
                          su_a["tau_max"], su_a["tau_min"]) \
        if su_a["is_iecnn"] else 1.0
    seq_len = su_a["seq_len"]
    context = min(seq_len, getattr(model, "context_length", seq_len))
    t0 = time.perf_counter()
    evaluate_bpc_per_pass(model, ubm, val_text, seq_len=context,
                          max_chars=eval_chars, tau0=tau0, prepared=prepared_val)
    evaluate_bpc(model, ubm, eval_slice, seq_len=context, tau0=tau0,
                 max_chars=eval_chars, prepared=prepared_train)
    eval_pair_s = time.perf_counter() - t0

    mean = statistics.mean(step_times)
    median = statistics.median(step_times)
    n_evals_600 = 6
    proj600 = 600 * mean + n_evals_600 * eval_pair_s

    try:
        sha = subprocess.run(["git", "rev-parse", "HEAD"], capture_output=True,
                             text=True, check=True).stdout.strip()
    except Exception:
        sha = "unknown"

    # ---- optional torch.profiler pass (where does the step time go?) -----
    profile_ops = None
    if args.profile > 0:
        from torch.profiler import profile, ProfilerActivity
        model.train()
        with profile(activities=[ProfilerActivity.CPU],
                     record_shapes=False) as prof:
            for step in range(args.profile):
                su_a["run_step"](args.warmup + args.steps + step)
        table = prof.key_averages().table(sort_by="self_cpu_time_total",
                                          row_limit=25)
        print(table)
        profile_ops = [
            {"op": e.key, "self_cpu_s": round(e.self_cpu_time_total / 1e6, 4),
             "calls": e.count}
            for e in sorted(prof.key_averages(),
                            key=lambda e: -e.self_cpu_time_total)[:25]
        ]

    result = {
        "tag": args.tag,
        "config": args.config,
        "overlay": args.overlay,
        "precision": su_a["precision"],
        "git_sha": sha,
        "torch_version": torch.__version__,
        "torch_threads": torch.get_num_threads(),
        "params": int(sum(p.numel() for p in model.parameters())),
        "batch_size": su_a["batch_size"],
        "seq_len": su_a["seq_len"],
        "vocab_size": su_a["vocab_size"],
        "steps_timed": args.steps,
        "warmup_steps": args.warmup,
        "s_per_step_mean": round(mean, 4),
        "s_per_step_median": round(median, 4),
        "s_per_step_min": round(min(step_times), 4),
        "s_per_step_p90": round(sorted(step_times)[int(0.9 * len(step_times)) - 1], 4),
        "tokens_per_s": round(tokens_timed / train_wall, 1),
        "eval_pair_s": round(eval_pair_s, 2),
        "projected_wall_600_steps_s": round(proj600, 1),
        "projected_wall_600_steps_min": round(proj600 / 60, 1),
        "step_times_s": [round(t, 4) for t in step_times],
        "date_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
    }
    if ab_times is not None:
        ab_mean = statistics.mean(ab_times)
        ab_median = statistics.median(ab_times)
        result["ab"] = {
            "config": args.ab_config, "overlay": args.ab_overlay,
            "s_per_step_mean": round(ab_mean, 4),
            "s_per_step_median": round(ab_median, 4),
            "s_per_step_min": round(min(ab_times), 4),
            "tokens_per_s": round(tokens_timed / sum(ab_times), 1),
            "a_median_over_b_median": round(median / ab_median, 3),
            "step_times_s": [round(t, 4) for t in ab_times],
        }
    if profile_ops is not None:
        result["profile_ops"] = profile_ops
    out = Path(args.out) if args.out else Path(f"results/bench_{args.tag}.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=1))
    print(json.dumps({k: v for k, v in result.items()
                      if k not in ("step_times_s", "ab")}, indent=1))


if __name__ == "__main__":
    main()
