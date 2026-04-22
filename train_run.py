#!/usr/bin/env python3
"""
Long-running training script: trains the IECNN brain on corpus_10k.txt with
fast settings, checkpointing every 250 sentences. Resumable.
"""
import sys, time, os
from pipeline.pipeline import IECNN

CORPUS = "corpus_10k.txt"
BRAIN  = "global_brain.pkl"

def main():
    n_target = int(sys.argv[1]) if len(sys.argv) > 1 else 10000
    print(f"[train] target lines: {n_target}", flush=True)

    m = IECNN(
        feature_dim=256,
        num_dots=64,            # smaller pool for tractable speed
        n_heads=2,
        max_iterations=2,
        evolve=True,
        seed=42,
        persistence_path=BRAIN,
    )

    # Build vocab if empty (cap vocab corpus at 500 lines to keep base count manageable)
    if not m.base_mapper.is_fitted:
        print("[train] Phase 1: fitting vocab on first 500 lines...", flush=True)
        with open(CORPUS) as fh:
            vocab_lines = [ln.strip() for ln in fh if ln.strip()][:500]
        m.fit(vocab_lines)
        m.save_brain()
        print(f"[train] vocab fitted: {len(m.base_mapper._base_vocab)} bases", flush=True)

    # Read sentences
    with open(CORPUS) as fh:
        sentences = [ln.strip() for ln in fh if ln.strip()][:n_target]
    print(f"[train] Phase 2: learning pass on {len(sentences)} sentences", flush=True)

    # Reduce per-dot work to keep training tractable
    dots = m._ensure_dots()
    for d in dots: d.n_heads = 1
    m.aim.max_variants = 0

    # Training loop with checkpointing
    t0 = time.time()
    save_every = 100
    import traceback, signal
    def _save_on_term(sig, frame):
        print(f"[train] caught signal {sig}, saving...", flush=True)
        try: m.save_brain()
        except Exception: pass
        sys.exit(0)
    signal.signal(signal.SIGTERM, _save_on_term)
    signal.signal(signal.SIGINT,  _save_on_term)

    for i, sent in enumerate(sentences, 1):
        try:
            m.run(sent, verbose=False)
        except Exception as e:
            print(f"[train] err sent {i}: {e!r}", flush=True)
            traceback.print_exc()
            continue
        if i % 10 == 0 or i == len(sentences):
            elapsed = time.time() - t0
            rate = i / max(elapsed, 1e-6)
            eta_s = (len(sentences) - i) / max(rate, 1e-6)
            dm = m.dot_memory.summary()
            print(f"[train] {i:>5}/{len(sentences)}  "
                  f"{rate:5.2f} ex/s  ETA {eta_s/60:5.1f}m  "
                  f"gen={m.evolution.generation:>4}  "
                  f"mean_eff={dm['mean_eff']:.4f}  max={dm['max_eff']:.4f}  "
                  f"active={dm['active_dots']}",
                  flush=True)
        if i % save_every == 0:
            m.save_brain()

    m.save_brain()
    print(f"[train] DONE — {len(sentences)} sentences in {(time.time()-t0)/60:.1f} min",
          flush=True)
    dm = m.dot_memory.summary()
    print(f"[train] final: gen={m.evolution.generation}  active_dots={dm['active_dots']}  "
          f"mean_eff={dm['mean_eff']:.4f}  max_eff={dm['max_eff']:.4f}", flush=True)

if __name__ == "__main__":
    main()
