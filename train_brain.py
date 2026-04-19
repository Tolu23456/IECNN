import sys
import os
import argparse
from typing import List

# Ensure we can import from the root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import IECNN

def train_global_brain(corpus_path: str, persistence_path: str):
    print(f"╔══════════════════════════════════════════════════════════════════╗")
    print(f"║                IECNN GLOBAL BRAIN KNOWLEDGE INGESTION            ║")
    print(f"╚══════════════════════════════════════════════════════════════════╝")

    if not os.path.exists(corpus_path):
        print(f"[ERROR] Corpus file not found: {corpus_path}")
        return

    print(f"\n[TRAIN] Initializing IECNN Model with persistence: {persistence_path}")
    model = IECNN(persistence_path=persistence_path)

    print(f"[TRAIN] Loading corpus from {corpus_path}...")
    with open(corpus_path, "r", encoding="utf-8") as f:
        # Read lines and filter empty ones
        lines = [line.strip() for line in f if line.strip()]

    print(f"[TRAIN] Ingesting {len(lines)} sentences into BaseMapping...")

    # Process in batches to manage memory and provide updates
    batch_size = 100
    total_batches = (len(lines) + batch_size - 1) // batch_size

    for i in range(0, len(lines), batch_size):
        batch = lines[i:i+batch_size]
        batch_num = i // batch_size + 1

        print(f"  → Batch {batch_num}/{total_batches} ({len(batch)} sentences)...", end="\r")

        # Fit the BaseMapper on this batch
        model.base_mapper.fit(batch)

    print(f"\n[TRAIN] Knowledge ingestion complete.")

    # Final smoothing and save
    print(f"[TRAIN] Applying final cooccurrence smoothing...")
    model.base_mapper._apply_cooc_smoothing()

    print(f"[TRAIN] Saving global brain to {persistence_path}...")
    model.base_mapper.save(persistence_path)

    bm = model.base_mapper
    n_words  = sum(1 for t in bm._base_types.values() if t == "word")
    n_phrase = sum(1 for t in bm._base_types.values() if t == "phrase")
    print(f"\n[STAT] Current Global Brain State:")
    print(f"  ‣ Word bases  : {n_words}")
    print(f"  ‣ Phrase bases: {n_phrase}")
    print(f"  ‣ Primitives  : {len(bm._primitive_embeddings)}")
    print(f"  ‣ Persistence : {persistence_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Ingest knowledge into IECNN Global Brain")
    parser.add_argument("--corpus", type=str, default="corpus.txt", help="Path to text corpus")
    parser.add_argument("--brain", type=str, default="global_brain.pkl", help="Path to persistent brain file")

    args = parser.parse_args()

    # Create a dummy corpus if none exists for demo
    if not os.path.exists(args.corpus):
        print(f"[INFO] Creating demo corpus file: {args.corpus}")
        with open(args.corpus, "w") as f:
            f.write("Artificial intelligence is reaching new heights.\n")
            f.write("Neural dots are the future of gradient-free learning.\n")
            f.write("Transformers rely on backpropagation and massive compute.\n")
            f.write("IECNN focuses on emergent agreement and iteration.\n")
            f.write("Convergence is the key to stable representations.\n")
            f.write("The global brain stores collective knowledge.\n")

    train_global_brain(args.corpus, args.brain)
