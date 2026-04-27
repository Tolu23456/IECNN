"""
Comparison script to verify pretraining impact.
"""
import numpy as np
from pipeline.pipeline import IECNN

def verify():
    # Load pretrained model
    model = IECNN(persistence_path="pretrained_iecnn")

    # Text with missing info
    text = "The [MASK] girl went to the park."

    # 1. Similarity check: 'The little girl' vs 'The [MASK] girl'
    v_full = model.encode("The little girl went to the park.")
    v_mask = model.encode("The girl went to the park.")

    sim = model.similarity("The little girl went to the park.", "The girl went to the park.")
    print(f"Similarity (Full vs Masked): {sim:.4f}")

    # 2. Convergence check
    res = model.run("The little girl went to the park.")
    print(f"Full text convergence score: {res.top_cluster.score:.4f}")

    res_mask = model.run("The girl went to the park.")
    print(f"Masked text convergence score: {res_mask.top_cluster.score:.4f}")

if __name__ == "__main__":
    verify()
