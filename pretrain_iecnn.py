"""
IECNN Pretraining Script — Masked BaseMap Modeling (MBM).
"""

import os
import sys
import time
import numpy as np
from pipeline.pipeline import IECNN

def main():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║                   IECNN UNSUPERVISED PRETRAINING                 ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    corpus_path = "corpus_10k.txt"
    brain_path = "pretrained_iecnn"

    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus not found at {corpus_path}")
        return

    # 1. Initialize IECNN
    model = IECNN(
        num_dots=64,
        max_iterations=3,
        phase_coding=True,
        persistence_path=brain_path
    )

    # 2. Load and Fit Vocabulary (Phase 1)
    print(f"\n[PHASE 1] Fitting BaseMapper vocabulary from {corpus_path}...")
    with open(corpus_path, 'r', encoding='utf-8') as f:
        sentences = [line.strip() for line in f if line.strip()][:100]

    model.fit(sentences)
    print(f"  ‣ Vocab Size: {len(model.base_mapper._base_vocab)} bases")

    # 3. MBM Pretraining (Phase 2)
    print(f"\n[PHASE 2] Starting Masked BaseMap Modeling (MBM) Pretraining...")
    print(f"  ‣ Mask Ratio: 0.15 (15% of tokens hidden)")
    print(f"  ‣ Population size: 64 dots")

    start_time = time.time()
    model.train_pass(
        sentences,
        max_iterations=2,
        mask_ratio=0.15,
        verbose=True,
        save_every=50,
        prune_every=100
    )

    duration = time.time() - start_time
    print(f"\n\n[SUCCESS] Pretraining complete in {duration:.1f}s")

    # 4. Save final brain
    model.save_brain()
    print(f"\n[INFO] Pretrained brain saved to '{brain_path}.*'")

if __name__ == "__main__":
    main()
