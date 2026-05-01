import time
import os
import sys
from pipeline.pipeline import IECNN

def main():
    corpus_path = "corpus_300k.txt"
    persistence_path = "global_brain.pkl"
    batch_size = 10
    max_time = 330

    start_time = time.time()
    model = IECNN(persistence_path=persistence_path)

    start_line = 0
    print(f"Starting training session...")

    lines_trained = 0
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as f:
        while time.time() - start_time < max_time:
            batch = []
            for _ in range(batch_size):
                line = next(f, None)
                if line:
                    line = line.strip()
                    if line: batch.append(line)
                else: break
            if not batch: break
            model.train_pass(batch, max_iterations=2, verbose=False)
            lines_trained += len(batch)
            elapsed = time.time() - start_time
            print(f"\rTrained {lines_trained} lines | Time: {elapsed:.1f}s", end="", flush=True)

    model.save_brain()
    print("\nTraining session complete.")

if __name__ == "__main__":
    main()
