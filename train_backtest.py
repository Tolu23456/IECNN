"""
train_backtest.py — IECNN 300k Corpus Training with Checkpoint Backtests

Trains IECNN on a 300k-line corpus and runs a backtest evaluation
every 40k sentences. Reports MaxEff, MeanEff, and dot learning signals
at each checkpoint, then prints a full final report.

Usage:
    python train_backtest.py
    python train_backtest.py --corpus /tmp/corpus_300k.txt
    python train_backtest.py --corpus /tmp/corpus_300k.txt --checkpoint-every 40000
"""

import argparse
import os
import sys
import time
import json
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

BRAIN_PATH   = "global_brain.pkl"
REPORT_PATH  = "training_report.json"
CORPUS_PATH  = "/tmp/corpus_300k.txt"
CHECKPOINT_N = 40_000

BAR = "─" * 72

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _build_c_if_needed():
    import subprocess
    so_files = [
        "formulas/formulas_c.so", "basemapping/basemapping_c.so",
        "aim/aim_c.so", "convergence/convergence_c.so",
        "pruning/pruning_c.so", "neural_dot/neural_dot_c.so",
        "decoding/decoder_c.so", "fast_count_c.so",
        "fast_scan_c.so", "pipeline/pipeline_c.so",
    ]
    missing = [p for p in so_files if not os.path.exists(p)]
    if missing:
        print(f"[build] Missing {len(missing)} C extensions — compiling...")
        r = subprocess.run(["bash", "build.sh"], capture_output=True, text=True)
        if r.returncode != 0:
            print(f"[build] FAILED:\n{r.stderr.strip()}")
        else:
            print("[build] All C extensions ready.")


def _make_model():
    from pipeline.pipeline import IECNN
    model = IECNN(
        feature_dim=256, num_dots=128, n_heads=4,
        max_iterations=12, evolve=True, seed=42,
        persistence_path=BRAIN_PATH,
    )
    if not model.base_mapper.is_fitted:
        seed = [
            "the quick brown fox jumps over the lazy dog",
            "neural networks learn patterns from data",
            "a sentence is a sequence of words",
        ]
        model.fit(seed)
        model.save_brain()
    return model


def _snapshot(model, sentences_done: int, t_start: float,
              total_words: int, prev_max_eff: float, prev_mean_eff: float):
    """Collect a checkpoint snapshot dict."""
    dots     = model._dots or []
    dot_ids  = [d.dot_id for d in dots]
    effs_arr = model.dot_memory.all_effectivenesses(dot_ids)
    effs     = np.array(effs_arr, dtype=np.float32) if len(effs_arr) > 0 else np.array([0.0])

    max_eff  = float(np.max(effs))
    mean_eff = float(np.mean(effs))
    std_eff  = float(np.std(effs))
    above_50 = int(np.sum(effs > 0.50))
    above_60 = int(np.sum(effs > 0.60))
    above_70 = int(np.sum(effs > 0.70))
    above_80 = int(np.sum(effs > 0.80))

    elapsed   = time.time() - t_start
    words_ps  = total_words / max(elapsed, 1e-6)

    # Learning signal: positive delta from prior checkpoint
    max_delta  = max_eff  - prev_max_eff
    mean_delta = mean_eff - prev_mean_eff
    learning   = "YES" if (max_delta > 0.005 or mean_delta > 0.005) else \
                 ("STABLE" if abs(max_delta) <= 0.005 else "REGRESSING")

    # Win-rate proxy from dot_memory
    total_wins = sum(model.dot_memory._success_counts.get(d, 0.0) for d in dot_ids)
    total_pred = sum(model.dot_memory._total_counts.get(d, 0.0)   for d in dot_ids)
    win_rate   = total_wins / max(total_pred, 1.0)

    # Dot diversity (mean pairwise W-cosine)
    try:
        W_stack = np.stack([d.W for d in dots[:32]])  # sample 32 dots for speed
        W_flat  = W_stack.reshape(len(W_stack), -1)
        norms   = np.linalg.norm(W_flat, axis=1, keepdims=True).clip(1e-10)
        W_norm  = W_flat / norms
        cosines = W_norm @ W_norm.T
        mask    = ~np.eye(len(cosines), dtype=bool)
        diversity = 1.0 - float(cosines[mask].mean())
    except Exception:
        diversity = 0.0

    snap = {
        "sentences_done":  sentences_done,
        "elapsed_s":       round(elapsed, 1),
        "words_per_s":     round(words_ps, 0),
        "max_eff":         round(max_eff,  6),
        "mean_eff":        round(mean_eff, 6),
        "std_eff":         round(std_eff,  6),
        "max_eff_delta":   round(max_delta, 6),
        "mean_eff_delta":  round(mean_delta, 6),
        "dots_above_50pct": above_50,
        "dots_above_60pct": above_60,
        "dots_above_70pct": above_70,
        "dots_above_80pct": above_80,
        "win_rate":        round(win_rate, 6),
        "dot_diversity":   round(diversity, 6),
        "learning_signal": learning,
        "pool_size":       len(dots),
    }
    return snap


def _print_checkpoint(snap: dict, checkpoint_num: int):
    """Pretty-print one checkpoint."""
    sents  = snap["sentences_done"]
    print(f"\n{BAR}")
    print(f"  CHECKPOINT {checkpoint_num}  —  {sents:,} sentences trained")
    print(BAR)
    print(f"  MaxEff        : {snap['max_eff']:.4f}  (Δ {snap['max_eff_delta']:+.4f})")
    print(f"  MeanEff       : {snap['mean_eff']:.4f}  (Δ {snap['mean_eff_delta']:+.4f})")
    print(f"  StdEff        : {snap['std_eff']:.4f}")
    print(f"  Win Rate      : {snap['win_rate']:.2%}")
    print(f"  Dots learning : {snap['learning_signal']}")
    print(f"  Dots >50%     : {snap['dots_above_50pct']} / {snap['pool_size']}")
    print(f"  Dots >70%     : {snap['dots_above_70pct']} / {snap['pool_size']}")
    print(f"  Dots >80%     : {snap['dots_above_80pct']} / {snap['pool_size']}")
    print(f"  Dot diversity : {snap['dot_diversity']:.4f}  (1.0=max diverse)")
    print(f"  Speed         : {snap['words_per_s']:,.0f} w/s")
    print(f"  Elapsed       : {snap['elapsed_s']:.0f}s")
    print(BAR)


def _print_final_report(checkpoints: list, total_sentences: int,
                        elapsed: float, total_words: int):
    """Print the final full training report."""
    print(f"\n\n{'═'*72}")
    print(f"  IECNN 300k TRAINING — FULL REPORT")
    print(f"{'═'*72}")
    print(f"  Total sentences trained : {total_sentences:,}")
    print(f"  Total words processed   : {total_words:,}")
    print(f"  Total time              : {elapsed:.0f}s  ({elapsed/60:.1f} min)")
    print(f"  Average speed           : {total_words/max(elapsed,1):.0f} w/s")
    print(f"  Checkpoints recorded    : {len(checkpoints)}")

    if checkpoints:
        final = checkpoints[-1]
        first = checkpoints[0]
        print(f"\n  ── Learning Progression ─────────────────────────────────────────")
        print(f"  {'Checkpoint':<12} {'Sents':>9} {'MaxEff':>8} {'MeanEff':>8} "
              f"{'Win%':>7} {'Lrn':>10} {'w/s':>9}")
        print(f"  {'─'*65}")
        for i, s in enumerate(checkpoints):
            print(f"  {i+1:<12} {s['sentences_done']:>9,} {s['max_eff']:>8.4f} "
                  f"{s['mean_eff']:>8.4f} {s['win_rate']:>7.2%} "
                  f"{s['learning_signal']:>10} {s['words_per_s']:>9,.0f}")

        # Trend analysis
        max_effs  = [s['max_eff']  for s in checkpoints]
        mean_effs = [s['mean_eff'] for s in checkpoints]
        total_max_gain  = max_effs[-1]  - max_effs[0]
        total_mean_gain = mean_effs[-1] - mean_effs[0]
        trend_max  = "↑ IMPROVING" if total_max_gain  >  0.02 else \
                     ("→ STABLE"   if abs(total_max_gain)  <= 0.02 else "↓ DECLINING")
        trend_mean = "↑ IMPROVING" if total_mean_gain >  0.02 else \
                     ("→ STABLE"   if abs(total_mean_gain) <= 0.02 else "↓ DECLINING")

        print(f"\n  ── Aggregate Statistics ─────────────────────────────────────────")
        print(f"  MaxEff  start→end : {max_effs[0]:.4f} → {max_effs[-1]:.4f}  "
              f"(total gain: {total_max_gain:+.4f})  {trend_max}")
        print(f"  MeanEff start→end : {mean_effs[0]:.4f} → {mean_effs[-1]:.4f}  "
              f"(total gain: {total_mean_gain:+.4f})  {trend_mean}")
        print(f"  Peak MaxEff       : {max(max_effs):.4f} at checkpoint "
              f"{max_effs.index(max(max_effs))+1}")
        print(f"  Final Dots >50%   : {final['dots_above_50pct']} / {final['pool_size']}")
        print(f"  Final Dots >70%   : {final['dots_above_70pct']} / {final['pool_size']}")
        print(f"  Final Dots >80%   : {final['dots_above_80pct']} / {final['pool_size']}")
        print(f"  Final Win Rate    : {final['win_rate']:.2%}")
        print(f"  Final Dot Diversity: {final['dot_diversity']:.4f}")

        # Verdict
        print(f"\n  ── Verdict ──────────────────────────────────────────────────────")
        if final['max_eff'] > 0.70:
            verdict = "STRONG LEARNING — dots have significantly specialised"
        elif final['max_eff'] > 0.55:
            verdict = "MODERATE LEARNING — top dots are above chance baseline"
        elif final['max_eff'] > 0.50:
            verdict = "WEAK LEARNING — marginal improvement above 0.50 prior"
        else:
            verdict = "NO LEARNING DETECTED — dots at or below 0.50 random prior"

        if final['dot_diversity'] < 0.30:
            diversity_note = "⚠  LOW DIVERSITY — dot collapse detected (W-matrices too similar)"
        elif final['dot_diversity'] < 0.50:
            diversity_note = "~  MODERATE DIVERSITY — some collapse, consider repulsion term"
        else:
            diversity_note = "✓  HEALTHY DIVERSITY — dots have specialised to different directions"

        print(f"  {verdict}")
        print(f"  {diversity_note}")

    print(f"\n  Brain saved to: {BRAIN_PATH}")
    print(f"  Full JSON report: {REPORT_PATH}")
    print(f"{'═'*72}\n")


# ─────────────────────────────────────────────────────────────────────────────
# Main training loop
# ─────────────────────────────────────────────────────────────────────────────

def run_training(corpus_path: str, checkpoint_every: int):
    _build_c_if_needed()

    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus not found: {corpus_path}")
        sys.exit(1)

    # Count lines
    with open(corpus_path, encoding='utf-8', errors='replace') as f:
        all_lines = [l.strip() for l in f if l.strip()]
    total_sentences = len(all_lines)
    total_words_est = sum(len(l.split()) for l in all_lines[:1000]) // 1000 * total_sentences

    print(f"\n{BAR}")
    print(f"  IECNN 300k Training + Backtest")
    print(BAR)
    print(f"  Corpus           : {corpus_path}")
    print(f"  Total sentences  : {total_sentences:,}")
    print(f"  Est. total words : {total_words_est:,}")
    print(f"  Checkpoint every : {checkpoint_every:,} sentences")
    print(f"  Checkpoints      : {total_sentences // checkpoint_every}")
    print(f"  Brain path       : {BRAIN_PATH}")
    print(BAR)

    model = _make_model()

    # Phase 1: Vocab seeding (skip if already fitted)
    if not model.base_mapper.is_fitted or len(model.base_mapper._base_vocab) < 1000:
        print(f"\n[Phase 1] Vocab seeding on first 10k sentences...")
        t_vocab0 = time.time()
        with open(corpus_path, encoding="utf-8", errors="replace") as fh:
            seed_lines = [l.strip() for l, _ in zip(fh, range(10_000)) if l.strip()]
        model.fit(seed_lines)
        model.save_brain()
        t_vocab = time.time() - t_vocab0
        vocab_size = len(model.base_mapper._base_vocab)
        print(f"[Phase 1] Done in {t_vocab:.1f}s — vocab: {vocab_size:,} tokens")
    else:
        vocab_size = len(model.base_mapper._base_vocab)
        print(f"\n[Phase 1] Vocab already fitted — {vocab_size:,} tokens (skipping seed)")

    # Phase 2: Causal training with checkpoints
    print(f"\n[Phase 2] Causal training with checkpoint backtests every {checkpoint_every:,} sents...")
    print(f"  (Architecture target: 16k w/s — using causal_batch=400, max_pos=6)\n")

    t_start      = time.time()
    checkpoints  = []
    words_trained = 0
    prev_max_eff  = 0.50
    prev_mean_eff = 0.50
    checkpoint_n  = 0

    # Process in chunks of checkpoint_every
    for chunk_start in range(0, total_sentences, checkpoint_every):
        chunk_end   = min(chunk_start + checkpoint_every, total_sentences)
        chunk       = all_lines[chunk_start:chunk_end]
        chunk_words = sum(len(s.split()) for s in chunk)

        print(f"\n  Training sentences {chunk_start+1:,}–{chunk_end:,}  "
              f"({len(chunk):,} sents, ~{chunk_words:,} words)...")

        chunk_t0 = time.time()
        model.causal_train_pass(
            sentences=chunk,
            max_pos=6,
            causal_batch=400,
            save_every=0,        # We save manually at each checkpoint
            verbose=True,
        )
        chunk_elapsed = time.time() - chunk_t0
        words_trained += chunk_words
        chunk_wps = chunk_words / max(chunk_elapsed, 1e-6)

        print(f"\n  Chunk done: {chunk_elapsed:.1f}s  ({chunk_wps:,.0f} w/s)")

        # Save brain
        model.save_brain()

        # Snapshot
        checkpoint_n += 1
        snap = _snapshot(
            model         = model,
            sentences_done = chunk_end,
            t_start        = t_start,
            total_words    = words_trained,
            prev_max_eff   = prev_max_eff,
            prev_mean_eff  = prev_mean_eff,
        )
        checkpoints.append(snap)
        _print_checkpoint(snap, checkpoint_n)

        prev_max_eff  = snap["max_eff"]
        prev_mean_eff = snap["mean_eff"]

    total_elapsed = time.time() - t_start

    # Final report
    _print_final_report(
        checkpoints       = checkpoints,
        total_sentences   = total_sentences,
        elapsed           = total_elapsed,
        total_words       = words_trained,
    )

    # Save JSON report
    report = {
        "corpus_path":       corpus_path,
        "total_sentences":   total_sentences,
        "total_words":       words_trained,
        "elapsed_s":         round(total_elapsed, 2),
        "avg_words_per_s":   round(words_trained / max(total_elapsed, 1), 1),
        "checkpoint_every":  checkpoint_every,
        "checkpoints":       checkpoints,
    }
    with open(REPORT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print(f"[done] JSON report written to {REPORT_PATH}")


# ─────────────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--corpus",           default=CORPUS_PATH)
    parser.add_argument("--checkpoint-every", type=int, default=CHECKPOINT_N)
    args = parser.parse_args()

    run_training(
        corpus_path      = args.corpus,
        checkpoint_every = args.checkpoint_every,
    )
