"""
train_300k.py — Full IECNN training run on the 271k-line corpus.

Phase 1: Fast parallel vocab training (BaseMapper.fit_fast) — scans corpus
         to build token/ngram frequency tables; takes ~30s for 271k lines.

Phase 2: Causal streaming training (causal_train_file) — iterates over the
         corpus in 5k-line chunks, updating dot W matrices on-the-fly.
         Streams file so peak RAM stays bounded.

Usage:
  python train_300k.py [--corpus /tmp/corpus_300k.txt] [--brain global_brain.pkl]
  python train_300k.py --passes 2   # train 2 full passes over the corpus
"""

import sys, os, time, argparse
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

parser = argparse.ArgumentParser()
parser.add_argument("--corpus", default="/tmp/corpus_300k.txt")
parser.add_argument("--brain",  default="global_brain.pkl")
parser.add_argument("--passes", type=int, default=1)
parser.add_argument("--skip_vocab", action="store_true",
                    help="Skip phase 1 (vocab already built in brain)")
parser.add_argument("--max_pos", type=int, default=6)
parser.add_argument("--chunk",   type=int, default=5000)
args = parser.parse_args()

CORPUS = args.corpus
BRAIN  = args.brain

if not os.path.exists(CORPUS):
    print(f"ERROR: corpus not found at {CORPUS}")
    sys.exit(1)

n_lines = sum(1 for _ in open(CORPUS, errors="replace"))
print("╔══════════════════════════════════════════════════════════════╗")
print("║            IECNN 300k CORPUS TRAINING RUN                   ║")
print("╚══════════════════════════════════════════════════════════════╝")
print(f"  corpus : {CORPUS}  ({n_lines:,} lines)")
print(f"  brain  : {BRAIN}")
print(f"  passes : {args.passes}")
print(f"  max_pos: {args.max_pos}")
print()

t_total = time.time()

# ── Phase 1: Vocabulary training ──────────────────────────────────────────────
if not args.skip_vocab:
    print("━━━  PHASE 1: Vocabulary (BaseMapper.fit_fast)  ━━━")
    from fast_train import fast_vocab_train_omp
    fast_vocab_train_omp(
        corpus_path  = CORPUS,
        brain_path   = BRAIN,
        n_workers    = 6,
        batch_size   = 50_000,
        skip_subwords= True,
        verbose      = True,
    )
    print(f"  Phase 1 done in {time.time()-t_total:.1f}s\n")
else:
    print("  [skip_vocab] Skipping Phase 1 — using existing brain.\n")

# ── Phase 2: Causal dot training ──────────────────────────────────────────────
print("━━━  PHASE 2: Causal dot training (causal_train_file)  ━━━")
from pipeline.pipeline import IECNN

nn = IECNN(
    persistence_path = BRAIN,
    num_dots         = 128,
    feature_dim      = 256,
    max_iterations   = 12,
    seed             = 42,
)
dots = nn._ensure_dots()
print(f"  Model loaded: {len(dots)} dots, vocab={len(nn.base_mapper._base_vocab):,}")
print()

t_phase2 = time.time()
for pass_no in range(1, args.passes + 1):
    print(f"  ── Pass {pass_no}/{args.passes} ─────────────────────────────────")
    nn.causal_train_file(
        path         = CORPUS,
        chunk_size   = args.chunk,
        max_pos      = args.max_pos,
        causal_batch = 200,
        save_every   = 10_000,
        verbose      = True,
    )
    print()

elapsed_p2 = time.time() - t_phase2
elapsed_tot = time.time() - t_total

print("═" * 62)
print(f"  Training complete!")
print(f"  Phase 2 time : {elapsed_p2:.1f}s")
print(f"  Total time   : {elapsed_tot:.1f}s  ({elapsed_tot/60:.1f} min)")
print(f"  Brain saved  : {BRAIN}")
print("═" * 62)
