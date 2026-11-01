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
from models.baselines import build_gru_baseline, build_transformer_baseline
from train.evaluate import evaluate_bpc
from train.schedules import lr_at
from utils import set_seed, set_torch_threads, count_parameters


def build_model(config: dict, vocab_size: int) -> torch.nn.Module:
    """Model factory for the 'kind' switch (gru | transformer; iecnn: Phase 3)."""
    kind = config["model"]["kind"]
    if kind == "gru":
        return build_gru_baseline(config, vocab_size)
    if kind == "transformer":
        return build_transformer_baseline(config, vocab_size)
    raise NotImplementedError(f"model kind '{kind}' is built in a later phase")


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
        print(f"resumed from {ckpt_path} at step {start_step}")

    csv_path = Path("results") / f"train_{name}_seed{seed}.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    log = _CSVLog(csv_path, ["step", "lr", "loss", "bpc_val", "elapsed_s"],
                  append=start_step > 0)

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
        out = model(xb)
        loss = F.nll_loss(out["final_log_probs"].reshape(-1, out["final_log_probs"].size(-1)),
                          yb.reshape(-1))
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        opt.step()
        tokens_seen += int(x.shape[0]) * int(x.shape[1])
        smooth = _ema(float(loss.item()), smooth)

        last = step == steps - 1
        if (step > 0 and step % eval_every == 0) or last:
            cut = None if last else eval_chars
            bpc = evaluate_bpc(model, ubm, val_text, seq_len=context, max_chars=cut)
            elapsed = time.perf_counter() - t0
            log.row([step, lr, round(smooth, 5), round(bpc, 5), round(elapsed, 1)])
            best_val = min(best_val, bpc)
            if step % (eval_every * 10) == 0 or last:
                print(f"[{name} seed{seed}] step {step} loss {smooth:.4f} "
                      f"val bpc {bpc:.4f} lr {lr:.2e} ({elapsed:.0f}s)")

        if (step > 0 and step % checkpoint_every == 0) or last:
            save_checkpoint(ckpt_path, {
                "step": step,
                "model": model.state_dict(),
                "opt": opt.state_dict(),
                "best_val": best_val,
                "batcher_pos": batcher._pos,
                "batcher_order": batcher._order,
                "batcher_epoch": batcher.epoch,
                "batcher_rng": batcher.rng.bit_generator.state,
                "py_rng": random.getstate(),
                "tokens_seen": tokens_seen,
            })

    elapsed_total = time.perf_counter() - t0
    final_bpc = evaluate_bpc(model, ubm, val_text, seq_len=context)
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
