"""
evaluate_iecnn.py — IECNN performance evaluation suite (v2).

Tests:
  T1: Next-word top-1 / top-5 accuracy using predict_word() NN inference
  T2: Causal win-rate vs random 50% baseline
  T3: Dot specialization metrics (effectiveness, win-rate distribution)
  T4: Dot diversity (inter-dot W cosine — the collapse metric)
  T5: Qualitative next-word samples

Usage:
  python evaluate_iecnn.py [--brain global_brain.pkl] [--corpus /tmp/corpus_300k.txt]
"""

import sys, os, time, argparse, random
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from pipeline.pipeline import IECNN

# ── CLI ───────────────────────────────────────────────────────────────────────
parser = argparse.ArgumentParser(description="Evaluate IECNN performance")
parser.add_argument("--brain",  default="global_brain.pkl",     help="Brain file path")
parser.add_argument("--corpus", default="/tmp/corpus_300k.txt", help="Corpus for held-out test")
parser.add_argument("--n_test", type=int, default=500,          help="Held-out sentences")
parser.add_argument("--seed",   type=int, default=99,           help="RNG seed")
args = parser.parse_args()

random.seed(args.seed)
np.random.seed(args.seed)

print("=" * 68)
print("  IECNN EVALUATION SUITE  v2")
print("=" * 68)

# ── Load brain ────────────────────────────────────────────────────────────────
print(f"\nLoading brain from {args.brain!r} ...")
if not os.path.exists(args.brain):
    print("ERROR: brain file not found. Run training first.")
    sys.exit(1)

nn   = IECNN(persistence_path=args.brain, num_dots=128, feature_dim=256, max_iterations=12)
dots = nn._ensure_dots()
dim  = nn.feature_dim

print(f"  dots     : {len(dots)}")
print(f"  dim      : {dim}")
print(f"  vocab    : {len(nn.base_mapper._base_vocab):,} tokens")

# ── Load corpus ───────────────────────────────────────────────────────────────
print(f"\nLoading corpus from {args.corpus!r} ...")
with open(args.corpus, encoding="utf-8", errors="replace") as fh:
    all_lines = [l.strip() for l in fh if len(l.split()) >= 5]

random.shuffle(all_lines)
n_test   = min(args.n_test, len(all_lines) // 10)
held_out = all_lines[:n_test]
print(f"  total lines : {len(all_lines):,}")
print(f"  held-out    : {n_test:,}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 1: Next-word accuracy using predict_word() (NN inference)
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("TEST 1: Next-word prediction accuracy (NN inference via predict_word)")
print("─" * 68)
print("  Method: effectiveness-weighted mean prediction → cosine NN over top-3000 vocab\n")

top1_hits = 0
top5_hits = 0
skipped   = 0
t0        = time.time()

for sent in held_out:
    toks = nn.base_mapper._tokenize(sent)
    if len(toks) < 3:
        skipped += 1
        continue
    pos    = random.randint(1, min(len(toks) - 1, 6))
    prefix = " ".join(toks[:pos])
    truth  = toks[pos]

    if truth not in nn.base_mapper._base_vocab:
        skipped += 1
        continue

    preds = nn.predict_word(prefix, top_k=5, n_cands=3000)

    if preds and preds[0] == truth:
        top1_hits += 1
    if truth in preds:
        top5_hits += 1

n_eval  = n_test - skipped
elapsed = time.time() - t0
print(f"  Evaluated : {n_eval} sentences  ({elapsed:.1f}s, {n_eval/max(elapsed,1e-6):.0f} sent/s)")
print(f"  Skipped   : {skipped} (too short or OOV target)")
print()
print(f"  ┌────────────────────────────────────────┐")
print(f"  │  Top-1 accuracy : {top1_hits/max(n_eval,1)*100:5.2f}%              │")
print(f"  │  Top-5 accuracy : {top5_hits/max(n_eval,1)*100:5.2f}%              │")
print(f"  │  Random baseline: ~0.033% top-1 (1/3k) │")
print(f"  └────────────────────────────────────────┘")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 2: Causal win-rate vs random 50% baseline
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("TEST 2: Causal win-rate vs random baseline")
print("─" * 68)
print("  Method: for each (prefix, next-token) pair, check what fraction of")
print("  dots predict in the correct direction.  Random = 50%.\n")

sample_sents = held_out[:100]
_dots_c  = nn._ensure_dots()
n_dots_c = len(_dots_c)
W_eval   = np.ascontiguousarray(np.stack([d.W for d in _dots_c]), dtype=np.float32)
W_flat_e = W_eval.reshape(n_dots_c * dim, dim)

total_pairs = 0
total_wins  = 0

for sent in sample_sents:
    toks = nn.base_mapper._tokenize(sent)
    if len(toks) < 2:
        continue
    for pos in range(1, min(len(toks), 7)):
        pfx     = " ".join(toks[:pos])
        ctx     = nn._get_ctx_cached(pfx)               # (dim,)
        tgt_tok = toks[pos]
        v       = nn._tok_emb_cache.get(tgt_tok)
        if v is None:
            te = nn.base_mapper._token_embedding(
                tgt_tok, nn.base_mapper._base_types.get(tgt_tok, "word")
            )
            nc = min(len(te), dim)
            v  = np.zeros(dim, dtype=np.float32)
            v[:nc] = np.real(te[:nc])
            n = float(np.linalg.norm(v))
            if n > 1e-10:
                v /= n
        # One BLAS call for all dots
        preds   = (ctx @ W_flat_e.T).reshape(n_dots_c, dim)  # (n_dots, dim)
        sims    = preds @ v                                    # (n_dots,)
        wins    = (sims > 0).sum()
        total_wins  += int(wins)
        total_pairs += n_dots_c

win_rate = total_wins / max(total_pairs, 1) * 100
print(f"  Total (pos, dot) pairs : {total_pairs:,}")
print(f"  Total wins             : {total_wins:,}")
print()
print(f"  ┌──────────────────────────────────────────┐")
print(f"  │  Causal win-rate : {win_rate:5.1f}%              │")
print(f"  │  Random baseline : 50.0%              │")
print(f"  │  Delta           : {win_rate - 50.0:+.1f}%              │")
print(f"  └──────────────────────────────────────────┘")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 3: Dot specialization
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("TEST 3: Dot specialization (effectiveness & win-rate distribution)")
print("─" * 68)

all_dot_ids = [d.dot_id for d in dots]
effs        = nn.dot_memory.all_effectivenesses(all_dot_ids)

win_rates = []
for did in all_dot_ids:
    tc = nn.dot_memory._total_counts.get(did, 0)
    sc = nn.dot_memory._success_counts.get(did, 0)
    if tc > 0:
        win_rates.append(sc / tc)
wr_arr = np.array(win_rates) if win_rates else np.array([])

if len(effs) > 0:
    max_eff  = float(np.max(effs))
    mean_eff = float(np.mean(effs))
    above50  = int((effs > 0.5).sum())
    above60  = int((effs > 0.6).sum())
    above70  = int((effs > 0.7).sum())

    print(f"  Dots active           : {len(effs)}")
    print(f"  MaxEff                : {max_eff:.4f}")
    print(f"  MeanEff               : {mean_eff:.4f}")
    print(f"  Dots with eff > 0.5   : {above50} / {len(effs)}")
    print(f"  Dots with eff > 0.6   : {above60} / {len(effs)}")
    print(f"  Dots with eff > 0.7   : {above70} / {len(effs)}")
    print()

    print(f"  Effectiveness histogram:")
    bins   = [0.0, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.01]
    labels = ["0.0–0.3","0.3–0.4","0.4–0.5","0.5–0.6","0.6–0.7","0.7–0.8","0.8–0.9","0.9–1.0"]
    hist, _ = np.histogram(effs, bins=bins)
    for label, count in zip(labels, hist):
        bar = "█" * int(count * 32 / max(hist.max(), 1))
        print(f"    {label}: {bar} {count}")

    if len(wr_arr) > 0:
        print(f"\n  Win-rate per dot (WTA → each dot wins ~1/n_dots positions):")
        print(f"    Mean : {wr_arr.mean()*100:.3f}%   (WTA expected: {100/n_dots_c:.3f}%)")
        print(f"    Min  : {wr_arr.min()*100:.3f}%")
        print(f"    Max  : {wr_arr.max()*100:.3f}%")
        print(f"    Std  : {wr_arr.std()*100:.4f}%")
else:
    print("  No effectiveness data (run training first).")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 4: Dot diversity — the collapse metric
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("TEST 4: Dot diversity (inter-dot W cosine — collapse metric)")
print("─" * 68)
print("  Goal: < 0.30 inter-dot cosine (was 0.657 pre-fix, random = ~0.00)\n")

W_stack = np.stack([d.W for d in dots])               # (n_dots, dim, dim)
Wf      = W_stack.reshape(len(dots), -1)              # (n_dots, dim*dim)
Wn      = Wf / (np.linalg.norm(Wf, axis=1, keepdims=True) + 1e-10)
pw      = Wn @ Wn.T                                   # (n_dots, n_dots)
np.fill_diagonal(pw, 0.0)
n_pairs = len(dots) * (len(dots) - 1)

mean_cos = pw.sum() / n_pairs
max_cos  = pw.max()
min_cos  = pw.min()
# Fraction of pairs with cosine > 0.5 (still-similar pairs)
frac_sim = (pw > 0.5).sum() / n_pairs * 100

print(f"  Inter-dot W cosine  (mean) : {mean_cos:.4f}  {'✓ GOOD' if mean_cos < 0.30 else '✗ still collapsed'}")
print(f"  Inter-dot W cosine  (max)  : {max_cos:.4f}")
print(f"  Inter-dot W cosine  (min)  : {min_cos:.4f}")
print(f"  Pairs with cosine > 0.5    : {(pw > 0.5).sum()//2} ({frac_sim:.1f}%)")

# Context-sensitivity: variance of predictions across different contexts
print(f"\n  Context-sensitivity test (10 diverse prefixes → pred variance):")
test_pfxs = [
    "the old", "she looked", "the government", "children love",
    "the city", "science and", "in the beginning", "he could not",
    "the weather", "a large number",
]
W_loc = np.ascontiguousarray(W_stack, dtype=np.float32)
W_flat_loc = W_loc.reshape(len(dots) * dim, dim)
ctx_vecs = []
for pfx in test_pfxs:
    ctx_vecs.append(nn._get_ctx_cached(pfx))
ctx_mat = np.stack(ctx_vecs)                          # (10, dim)
mean_preds = (ctx_mat @ W_flat_loc.T).reshape(10, len(dots), dim).mean(axis=1)
                                                      # (10, dim) mean pred per prefix
mp_n  = mean_preds / (np.linalg.norm(mean_preds, axis=1, keepdims=True) + 1e-10)
# Pairwise cosine of mean predictions across prefixes
pp = mp_n @ mp_n.T
np.fill_diagonal(pp, 0)
ctx_sens = 1.0 - pp.sum() / (10 * 9)
print(f"    Mean pred pairwise cosine : {pp.sum()/(10*9):.4f}")
print(f"    Context-sensitivity score : {ctx_sens:.4f}  {'✓ GOOD' if ctx_sens > 0.30 else '✗ low — model ignores context'}")

# ─────────────────────────────────────────────────────────────────────────────
# TEST 5: Qualitative samples
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "─" * 68)
print("TEST 5: Qualitative next-word samples (predict_word)")
print("─" * 68)

sample_prefixes = [
    "the old man",
    "she looked at",
    "the government has",
    "in the beginning",
    "children love to",
    "the city was",
    "he could not",
    "the book was",
    "science and technology",
    "the weather today",
    "a large number of",
    "they walked into the",
]

print(f"  {'Prefix':<38} {'Top-1':<14} {'Top-2..5'}")
print(f"  {'─'*38} {'─'*14} {'─'*28}")
for pfx_str in sample_prefixes:
    preds = nn.predict_word(pfx_str, top_k=5, n_cands=3000)
    if preds:
        top1 = preds[0]
        rest = ", ".join(preds[1:])
        print(f"  {pfx_str:<38} {top1:<14} [{rest}]")

# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
print("\n" + "=" * 68)
print("  SUMMARY")
print("=" * 68)
_above50 = int((effs > 0.5).sum()) if len(effs) > 0 else 0
_max_eff = float(np.max(effs)) if len(effs) > 0 else 0.0
_mean_eff = float(np.mean(effs)) if len(effs) > 0 else 0.0
print(f"""
  Metric                 | Value        | Target / Baseline
  ─────────────────────────────────────────────────────────
  Top-1 accuracy         | {top1_hits/max(n_eval,1)*100:5.2f}%       | >> 0.03% (random)
  Top-5 accuracy         | {top5_hits/max(n_eval,1)*100:5.2f}%       | >> 0.15% (random)
  Causal win-rate        | {win_rate:5.1f}%       | > 50% (random = 50%)
  MaxEff                 | {_max_eff:.4f}       | > 0.70 (specialized)
  MeanEff                | {_mean_eff:.4f}       | > 0.50
  Dots > 0.5 eff         | {_above50:3d} / {len(effs)}    | > 64/128
  Inter-dot cosine       | {mean_cos:.4f}       | < 0.30 (was 0.657)
  Context-sensitivity    | {ctx_sens:.4f}       | > 0.30 (was 0.19)
""")
