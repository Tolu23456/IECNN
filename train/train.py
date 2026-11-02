"""Training loop (spec 06, sections 1/3/4/10): configs, seeds, CSV logging,
checkpoints, resume, gradient clipping, AdamW with warmup + cosine decay.

Usage:
  python -m train.train --config configs/b0_gru.yaml [--overlay tiny.yaml] \
      --seed 0 [--out-dir checkpoints/x] [--max-steps N] [--fresh] [--resume]
"""
from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from config import load_config
from data.dataset import Batcher, tokenize_corpus
from train.losses import main_loss
from train.evaluate import evaluate_bpc
from models.baselines import build_gru_baseline, build_transformer_baseline
from train.schedules import lr_at
from utils import set_seed, set_torch_threads, count_parameters


def _iecnn_n_dots(model) -> int:
    """Dots per region of an IECNN (for the weight floor eps = 0.1 / N)."""
    try:
        return int(model.regions[0].dots.N)
    except AttributeError:
        return 1


def build_model(config: dict, vocab_size: int) -> torch.nn.Module:
    """Model factory for the 'kind' switch (gru | transformer | iecnn)."""
    kind = config["model"]["kind"]
    if kind == "gru":
        return build_gru_baseline(config, vocab_size)
    if kind == "transformer":
        return build_transformer_baseline(config, vocab_size)
    if kind == "iecnn":
        from models.iecnn import build_iecnn
        return build_iecnn(config, vocab_size)
    raise ValueError(f"unknown model kind '{kind}'")


def save_checkpoint(path: Path, state: dict) -> None:
    """Atomic checkpoint write (write temp, then rename)."""
    tmp = path.with_suffix(".tmp")
    torch.save(state, tmp)
    tmp.replace(path)


def _ema(loss: float, prev: float | None, beta: float = 0.9) -> float:
    """Exponential moving average for log display."""
    return loss if prev is None else beta * prev + (1 - beta) * loss


class _CSVLog:
    """Small CSV writer that appends; writes the header on first row."""

    def __init__(self, path: Path, columns: list[str], append: bool):
        self.path = path
        self.columns = columns
        self._open = open(path, "a" if append else "w", newline="")
        self._wrote_header = append and path.exists()
        if not self._wrote_header:
            csv.writer(self._open).writerow(columns)
            self._open.flush()

    def row(self, values: list) -> None:
        """Append one row and flush (so resume/UI can read partial logs)."""
        csv.writer(self._open).writerow(values)
        self._open.flush()

    def close(self) -> None:
        self._open.close()


def train_one(config: dict, seed: int, out_dir: Path, max_steps: int | None = None,
              resume: bool = False) -> dict:
    """Train one seed; returns a summary dict; logs CSV; checkpoints; can resume."""
    name = config.get("name", "run")
    out_dir.mkdir(parents=True, exist_ok=True)

    set_seed(seed)
    set_torch_threads(int(config.get("training", {}).get("torch_threads", 2)))

    vocab_file = config.get("ubm", {}).get("vocab_file", "UBM/vocab_v0.json")
    from UBM import UBM
    ubm = UBM.load(vocab_file)
    vocab_size = len(ubm.bases)

    train_text = Path("data/train.txt").read_text(encoding="utf-8")
    val_text = Path("data/val.txt").read_text(encoding="utf-8")
    train_ids = tokenize_corpus(train_text, vocab_size)

    tcfg = config.get("training", {})
    seq_len = int(config.get("data", {}).get("seq_len", 256))
    batch_size = int(tcfg.get("batch_size", 16))
    peak_lr = float(tcfg.get("lr", 1e-3))
    warmup = int(tcfg.get("warmup_steps", 200))
    grad_clip = float(tcfg.get("grad_clip", 1.0))
    eval_every = int(tcfg.get("eval_every", 100))
    eval_chars = int(tcfg.get("eval_chars", 12000))
    checkpoint_every = int(tcfg.get("checkpoint_every", 1000))
    betas = tuple(tcfg.get("betas", (0.9, 0.95)))
    weight_decay = float(tcfg.get("weight_decay", 0.01))
    steps = max_steps if max_steps is not None else int(tcfg.get("steps", 1000))

    batcher = Batcher(train_ids, seq_len, batch_size, seed=seed, shuffle=True)
    used_batch = batcher.batch_size

    model = build_model(config, vocab_size)
    opt = torch.optim.AdamW(model.parameters(), lr=peak_lr, betas=betas,
                            weight_decay=weight_decay)

    ckpt_path = out_dir / "checkpoint.pt"
    start_step = 0
    best_val = float("inf")
    best_step = 0
    best_state = None
    if resume and ckpt_path.exists():
        state = torch.load(ckpt_path, weights_only=False)
        model.load_state_dict(state["model"])
        opt.load_state_dict(state["opt"])
        batcher._pos = int(state["batcher_pos"])
        batcher._order = np.asarray(state["batcher_order"])
        batcher.epoch = int(state["batcher_epoch"])
        random.setstate(state["py_rng"])
        batcher.rng.bit_generator.state = state["batcher_rng"]
        start_step = int(state["step"]) + 1
        best_val = float(state.get("best_val", best_val))
        best_step = int(state.get("best_step", 0))
        best_state = state.get("best_state")          # best weights so far
        print(f"resumed from {ckpt_path} at step {start_step}")

    csv_path = Path("results") / f"train_{name}_seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log = _CSVLog(csv_path, ["step", "lr", "loss", "bpc_val", "elapsed_s"],
                  append=start_step > 0)

    # ---- IECNN-specific schedules and aux-loss weights (spec 04/06) ------
    is_iecnn = config["model"]["kind"] == "iecnn"
    tcfg_all = config.get("training", {})
    lambdas = {
        "lambda_r": float(tcfg_all.get("lambda_r", 0.0)),   # region a.u.x
        "lambda_d": float(tcfg_all.get("lambda_d", 0.0)),   # boosted per-dot
        "lambda_c": float(tcfg_all.get("lambda_c", 0.0)),   # decorrelation (E4)
        "lambda_k": float(tcfg_all.get("lambda_k", 0.0)),   # confidence calib.
        "lambda_w": float(tcfg_all.get("lambda_w", 0.0)),   # weight entropy bonus
        "gamma_b": float(tcfg_all.get("gamma_b", 2.0)),
    }
    cons = config.get("consensus", {})
    tau_max = float(cons.get("tau_max", 1.0)); tau_min = float(cons.get("tau_min", 0.2))
    tau_warmup = int(cons.get("tau_warmup", 2000))
    lam_ramp = int(cons.get("lambda_c_ramp_steps", 5000))
    weight_floor_fraction = float(cons.get("weight_floor_fraction", 0.1))
    hard_start = float(tcfg_all.get("hard_batch_start", 0.7))
    hard_frac = float(tcfg_all.get("hard_batch_fraction", 0.2))
    hard_rng = random.Random(seed * 7919 + 3)

    from train.schedules import (temperature_at, weight_floor_at,
                                 correlation_lambda, weight_entropy_lambda,
                                 use_hard_batch)

    context = min(seq_len, getattr(model, "context_length", seq_len))
    smooth: float | None = None
    t0 = time.perf_counter()
    tokens_seen = 0

    for step in range(start_step, steps):
        lr = lr_at(step, steps, warmup, peak_lr)
        for group in opt.param_groups:
            group["lr"] = lr

        x, y = batcher.next_batch()
        xb = torch.from_numpy(np.ascontiguousarray(x))
        yb = torch.from_numpy(np.ascontiguousarray(y))
        kwargs = {}
        if is_iecnn:
            tau0 = temperature_at(step, steps, tau_warmup, tau_max, tau_min)
            floor = weight_floor_at(step, steps, weight_floor_fraction / _iecnn_n_dots(model))
            hard = use_hard_batch(step, steps, hard_start, hard_frac, hard_rng)
            kwargs = {"tau0": tau0, "hard_mode": bool(hard), "floor": floor}
        out = model(xb, **kwargs)
        loss = main_loss(out["final_log_probs"], yb)
        aux_parts = {"main": float(loss.item())}

        if is_iecnn and any(v != 0 for k, v in lambdas.items() if k != "gamma_b"):
            from train.losses import (region_loss, boosted_dot_loss,
                                      decorrelation_penalty, confidence_loss,
                                      weight_entropy_bonus)
            last_pass = out["per_pass_last_region"][-1]
            if lambdas["lambda_r"] > 0 and len(out["region_outputs"]) > 1:
                stack = torch.stack([ro["log_b"] for ro in out["region_outputs"]])
                lr_term = region_loss(stack, yb)
                loss = loss + lambdas["lambda_r"] * lr_term
                aux_parts["L_region"] = float(lr_term.detach())
            if lambdas["lambda_d"] > 0:
                ld = boosted_dot_loss(last_pass["log_pi"], out["final_log_probs"],
                                      yb, lambdas["gamma_b"])
                loss = loss + lambdas["lambda_d"] * ld
                aux_parts["L_dot"] = float(ld.detach())
            if lambdas["lambda_c"] > 0:
                lc_val = correlation_lambda(step, lam_ramp, lambdas["lambda_c"])
                lc = decorrelation_penalty(last_pass["log_pi"],
                                           out["final_log_probs"], yb)
                loss = loss + lc_val * lc
                aux_parts["L_corr"] = float(lc.detach()) * lc_val
            if lambdas["lambda_k"] > 0:
                lk = confidence_loss(last_pass["gamma"], last_pass["log_pi"], yb)
                loss = loss + lambdas["lambda_k"] * lk
                aux_parts["L_conf"] = float(lk.detach())
            if lambdas["lambda_w"] > 0:
                lw_val = weight_entropy_lambda(step, steps, lambdas["lambda_w"])
                lw = weight_entropy_bonus(last_pass["weights"])
                loss = loss - lw_val * lw
                aux_parts["L_w"] = -float(lw.detach()) * lw_val

        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tokens_seen += int(x.shape[0]) * int(x.shape[1])
        smooth = _ema(float(loss.item()), smooth)

        last = step == steps - 1
        if (step > 0 and step % eval_every == 0) or last:
            cut = None if last else eval_chars
            bpc = evaluate_bpc(model, ubm, val_text, seq_len=context, max_chars=cut,
                               tau0=kwargs.get("tau0", 1.0) if is_iecnn else 1.0)
            elapsed = time.perf_counter() - t0
            log.row([step, lr, round(smooth, 5), round(bpc, 5), round(elapsed, 1)])
            if bpc < best_val:
                best_val = bpc
                best_step = step
                # keep the weights of the best point (they are what full-val is
                # finally scored on): measured B0 overfits well before 50k steps
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            if step % (eval_every * 10) == 0 or last:
                aux_str = " ".join(f"{k} {v:.3f}" for k, v in aux_parts.items())
                print(f"[{name} seed{seed}] step {step} {aux_str} "
                      f"val bpc {bpc:.4f} lr {lr:.2e} ({elapsed:.0f}s)", flush=True)

        if (step > 0 and step % checkpoint_every == 0) or last:
            save_checkpoint(ckpt_path, {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": best_val,
                "best_step": best_step,
                "best_state": best_state,
                "batcher_pos": batcher._pos,
                "batcher_order": batcher._order,
                "batcher_epoch": batcher.epoch,
                "batcher_rng": batcher.rng.bit_generator.state,
                "py_rng": random.getstate(),
                "tokens_seen": tokens_seen,
            })

    elapsed_total = time.perf_counter() - t0
    final_eval_kwargs = {}
    if is_iecnn:
        final_eval_kwargs = {"tau0": temperature_at(steps - 1, steps, tau_warmup,
                                                    tau_max, tau_min)}
    final_bpc = evaluate_bpc(model, ubm, val_text, seq_len=context,
                             **final_eval_kwargs)
    mode_gap_val = None
    if is_iecnn:            # spec 08: report the soft-to-hard gap at every eval
        hard_bpc = evaluate_bpc(model, ubm, val_text, seq_len=context,
                                hard_mode=True, **final_eval_kwargs)
        mode_gap_val = round(hard_bpc - final_bpc, 5)

    # ---- best checkpoint, scored on FULL validation ----------------------
    # In-loop evals score a subsample; the honest "best" headline is the best
    # tracked point re-scored on the full validation text (B0 measured on the
    # runner: val bpc degrades monotonically long before the 50k target, so
    # final-only numbers would report the most-overfit point of every model).
    best_full_val_bpc = None
    if best_state is not None and best_step != steps - 1:
        last_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
        model.load_state_dict(best_state)
        best_full_val_bpc = evaluate_bpc(model, ubm, val_text, seq_len=context,
                                         **final_eval_kwargs)
        mode_gap_best = None
        if is_iecnn:
            mb = evaluate_bpc(model, ubm, val_text, seq_len=context,
                              hard_mode=True, **final_eval_kwargs)
            mode_gap_best = round(mb - best_full_val_bpc, 5)
        model.load_state_dict(last_state)
        print(f"[{name} seed{seed}] BEST at step {best_step}: full-val bpc "
              f"{best_full_val_bpc:.4f}", flush=True)
    log.row([steps - 1, lr_at(steps - 1, steps, warmup, peak_lr), smooth, round(final_bpc, 5), round(elapsed_total, 1)])
    log.close()

    summary = {
        "name": name,
        "seed": seed,
        "steps_trained_this_run": steps - start_step,
        "steps_final": steps,
        "parameters": count_parameters(model),
        "final_val_bpc": round(final_bpc, 5),
        "best_val_bpc": round(min(best_val, final_bpc), 5),
        "vocab_size": vocab_size,
        "batch_size": used_batch,
        "seq_len": seq_len,
        "wall_seconds": round(elapsed_total, 1),
        "tokens_per_s": round(tokens_seen / max(elapsed_total, 1e-9), 1),
        "estimated_train_flops": int(6 * count_parameters(model) * tokens_seen),
    }
    if mode_gap_val is not None:
        summary["final_hard_bpc"] = round(final_bpc + mode_gap_val, 5)
        summary["mode_gap"] = mode_gap_val
    if best_full_val_bpc is not None:
        summary["best_full_val_bpc"] = round(best_full_val_bpc, 5)
        summary["best_step"] = best_step
        if mode_gap_best is not None:
            summary["best_mode_gap"] = mode_gap_best
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1))
    (out_dir / "config_used.json").write_text(json.dumps(config, indent=1, default=str))
    print(f"DONE {name} seed{seed}: val bpc {final_bpc:.4f} "
          f"({count_parameters(model):,} params, {elapsed_total:.0f}s)")
    return summary


def main() -> None:
    """CLI entry point."""
    ap = argparse.ArgumentParser()
    ap.add_argument("--config", required=True)
    ap.add_argument("--overlay", default="", help="e.g. tiny.yaml")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out-dir", default="")
    ap.add_argument("--max-steps", type=int, default=None)
    ap.add_argument("--resume", action="store_true")
    args = ap.parse_args()
    overlays = [args.overlay] if args.overlay else []
    cfg = load_config(args.config, *overlays)
    out_dir = Path(args.out_dir) if args.out_dir else Path(f"checkpoints/{cfg.get('name', 'run')}_seed{args.seed}")
    train_one(cfg, args.seed, out_dir, max_steps=args.max_steps, resume=args.resume)


if __name__ == "__main__":
    main()
