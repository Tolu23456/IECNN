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
from train.evaluate import evaluate_bpc, evaluate_bpc_per_pass, prepare_text
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

    train_text = Path(config.get("data", {}).get(
        "train_file", "data/train.txt")).read_text(encoding="utf-8")
    val_text = Path(config.get("data", {}).get(
        "val_file", "data/val.txt")).read_text(encoding="utf-8")
    train_ids = tokenize_corpus(train_text, vocab_size,
                                cache_dir=config.get("data", {}).get("dir", "data"),
                                vocab_file=vocab_file)

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
    # early stopping (config training.early_stop): stop after N evals without
    # improvement, plus a total wall-clock cap. Both survive chunked resume:
    # best_val/best_step/best_state come from the checkpoint below, and the
    # accumulated training seconds are carried in it too.
    es_cfg = tcfg.get("early_stop", {})
    es_patience = int(es_cfg.get("patience_evals", 0))
    es_min_improve = float(es_cfg.get("min_improvement", 0.001))
    es_wall_hours = float(es_cfg.get("wall_hours", 0.0))
    train_seconds = 0.0
    since_best_evals = 0
    stopped_reason = None
    stop_kind = None            # "patience" | "wall_cap" | None (target reached)
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
        train_seconds = float(state.get("train_seconds", 0.0))
        since_best_evals = int(state.get("since_best_evals", 0))
        print(f"resumed from {ckpt_path} at step {start_step}")

    csv_path = Path(tcfg.get("csv_dir", "results")) / f"train_{name}_seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    # per-pass columns (p1..p3) are blank for baselines; tau0 is the schedule
    # temperature at the eval step; L_w is the weight-entropy bonus value.
    log = _CSVLog(csv_path, ["step", "lr", "loss", "bpc_val",
                             "bpc_p1", "bpc_p2", "bpc_p3", "bpc_train",
                             "tau0", "weight_entropy", "L_w", "elapsed_s"],
                  append=start_step > 0)
    # encode the eval texts ONCE (full val + a fixed training slice); evals
    # reuse the prepared (ids, expand_len) pairs instead of re-encoding.
    prepared_val = prepare_text(ubm, val_text)
    eval_slice = train_text[:max(eval_chars * 6, 512_000)]
    prepared_train = prepare_text(ubm, eval_slice)

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
        t_step = time.perf_counter()
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
        train_seconds += time.perf_counter() - t_step

        last = step == steps - 1
        if (step > 0 and step % eval_every == 0) or last:
            cut = None if last else eval_chars
            ev = evaluate_bpc_per_pass(model, ubm, val_text, seq_len=context,
                                       max_chars=cut, tau0=kwargs.get("tau0", 1.0)
                                       if is_iecnn else 1.0,
                                       prepared=prepared_val)
            bpc = ev["final"]
            bpc_tr = evaluate_bpc(model, ubm, eval_slice, seq_len=context,
                                  tau0=kwargs.get("tau0", 1.0) if is_iecnn else 1.0,
                                  max_chars=eval_chars, prepared=prepared_train)
            elapsed = time.perf_counter() - t0
            p3 = (ev["passes"] + [""] * 3)[:3]
            log.row([step, lr, round(smooth, 5), round(bpc, 5),
                     p3[0], p3[1], p3[2], round(bpc_tr, 5),
                     round(kwargs.get("tau0", 1.0), 5) if is_iecnn else "",
                     ev["weight_entropy"] if is_iecnn else "",
                     round(aux_parts.get("L_w", 0.0), 5) if "L_w" in aux_parts else "",
                     round(elapsed, 1)])
            prev_best = best_val
            if bpc < best_val:
                best_val = bpc
                best_step = step
                # keep the weights of the best point (they are what full-val is
                # finally scored on): measured B0 overfits well before 50k steps
                best_state = {k: v.detach().clone() for k, v in model.state_dict().items()}
            # early stopping (Tolu's rule): N evals without a real improvement,
            # or a total wall-clock cap. Improvement must beat min_improvement
            # so eval noise cannot reset the patience counter.
            if bpc <= prev_best - es_min_improve:
                since_best_evals = 0
            else:
                since_best_evals += 1
            if es_patience > 0 and since_best_evals >= es_patience and not last:
                stop_kind = "patience"
                stopped_reason = (f"patience: {since_best_evals} evals without "
                                  f"improvement > {es_min_improve}")
            elif 0 < es_wall_hours < train_seconds / 3600:
                stop_kind = "wall_cap"
                stopped_reason = (f"wall cap {es_wall_hours}h reached at "
                                  f"{train_seconds / 3600:.2f}h")
            if step % (eval_every * 10) == 0 or last:
                aux_str = " ".join(f"{k} {v:.3f}" for k, v in aux_parts.items())
                pp_str = " ".join(f"p{i+1} {b:.3f}" for i, b in enumerate(ev["passes"]))
                print(f"[{name} seed{seed}] step {step} {aux_str} val bpc {bpc:.4f} "
                      f"({pp_str}) train bpc {bpc_tr:.4f} "
                      f"wtH {ev['weight_entropy']:.3f} lr {lr:.2e} ({elapsed:.0f}s)",
                      flush=True)
            if stopped_reason is not None:
                print(f"EARLY STOP ({stop_kind}) at step {step}: {stopped_reason}",
                      flush=True)
                save_checkpoint(ckpt_path, {          # stop-state is resumable
                    "step": step, "model": model.state_dict(),
                    "opt": opt.state_dict(), "best_val": best_val,
                    "best_step": best_step, "best_state": best_state,
                    "batcher_pos": batcher._pos, "batcher_order": batcher._order,
                    "batcher_epoch": batcher.epoch,
                    "batcher_rng": batcher.rng.bit_generator.state,
                    "py_rng": random.getstate(), "tokens_seen": tokens_seen,
                    "train_seconds": train_seconds,
                    "since_best_evals": since_best_evals,
                })
                break

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
                "train_seconds": train_seconds,
                "since_best_evals": since_best_evals,
            })

    elapsed_total = time.perf_counter() - t0
    final_eval_kwargs = {}
    if is_iecnn:
        final_eval_kwargs = {"tau0": temperature_at(steps - 1, steps, tau_warmup,
                                                    tau_max, tau_min)}
    final_ev = evaluate_bpc_per_pass(model, ubm, val_text, seq_len=context,
                                     **final_eval_kwargs, prepared=prepared_val)
    final_bpc = final_ev["final"]
    mode_gap_val = None
    if is_iecnn:            # spec 08: report the soft-to-hard gap at every eval
        hard_bpc = evaluate_bpc(model, ubm, val_text, seq_len=context,
                                hard_mode=True, prepared=prepared_val,
                                **final_eval_kwargs)
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
        "final_bpc_per_pass": final_ev["passes"],
        "final_bpc_train": None,          # filled below (fixed training slice)
        "final_weight_entropy": final_ev["weight_entropy"] if is_iecnn else None,
        "final_tau0": final_eval_kwargs.get("tau0") if is_iecnn else None,
        "best_val_bpc": round(min(best_val, final_bpc), 5),
        "vocab_size": vocab_size,
        "batch_size": used_batch,
        "stop_kind": stop_kind,           # "patience" | "wall_cap" | None
        "stopped_early": stopped_reason is not None,
        "hit_wall_cap": stop_kind == "wall_cap",
        "reached_step_target": stopped_reason is None,
        "stopped_reason": stopped_reason,
        "train_seconds_total": round(train_seconds, 1),
        "seq_len": seq_len,
        "wall_seconds": round(elapsed_total, 1),
        "tokens_per_s": round(tokens_seen / max(elapsed_total, 1e-9), 1),
        "estimated_train_flops": int(6 * count_parameters(model) * tokens_seen),
    }
    # train bpc on the FIXED slice (same text every run: fit, not generalization)
    summary["final_bpc_train"] = round(evaluate_bpc(
        model, ubm, eval_slice, seq_len=context, max_chars=eval_chars,
        tau0=final_eval_kwargs.get("tau0", 1.0) if is_iecnn else 1.0,
        prepared=prepared_train), 5)
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
