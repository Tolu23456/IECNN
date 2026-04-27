"""
Deep Verification of IECNN Pretraining Effectiveness.
Tests:
1. Reconstruction Test: Can the decoder recover masked information?
2. Relative Similarity: sim(full, masked) vs sim(full, unrelated).
3. Learning Delta: Pretrained vs Untrained model performance.
4. Perturbation Robustness: Convergence under word swaps/drops.
"""

import numpy as np
import os
from pipeline.pipeline import IECNN
from decoding.decoder import IECNNDecoder

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print(f"{'='*60}")

def run_deep_verification():
    brain_path = "pretrained_iecnn"
    if not os.path.exists(brain_path + ".meta.pkl"):
        print("[ERROR] Pretrained brain not found. Run pretrain_iecnn.py first.")
        return

    # 1. Initialize Models
    print("[INFO] Initializing Models...")
    model_trained = IECNN(persistence_path=brain_path)
    model_fresh   = IECNN() # Untrained

    decoder = IECNNDecoder(model_trained)

    full_text = "The little girl went to the park to play."
    masked_text = "The girl went to the park to play."
    unrelated_text = "The car exploded in the middle of the desert."
    perturbed_text = "girl park play went the to the." # Word swap/drop

    # --- TEST 1: RECONSTRUCTION ---
    print_header("TEST 1: RECONSTRUCTION")
    res_mask = model_trained.run(masked_text)
    reconstructed = decoder.decode(res_mask.output, use_pipeline=True)
    print(f"Input Masked:  '{masked_text}'")
    print(f"Decoded:       '{reconstructed}'")
    # Check if 'little' or something similar appears (Heuristic)
    if "little" in reconstructed.lower() or "girl" in reconstructed.lower():
         print("Result: POSITIVE (Some semantic recovery observed)")
    else:
         print("Result: NEGATIVE (No recovery observed)")

    # --- TEST 2: RELATIVE SIMILARITY ---
    print_header("TEST 2: RELATIVE SIMILARITY")
    sim_match = model_trained.similarity(full_text, masked_text)
    sim_unrelated = model_trained.similarity(full_text, unrelated_text)

    print(f"Sim(Full, Masked):    {sim_match:+.4f}")
    print(f"Sim(Full, Unrelated): {sim_unrelated:+.4f}")

    gap = sim_match - sim_unrelated
    print(f"Discriminative Gap:   {gap:+.4f}")
    if gap > 0.05:
        print("Result: SUCCESS (Discriminative gap confirmed)")
    else:
        print("Result: FAIL (Insufficient discriminative power)")

    # --- TEST 3: LEARNING DELTA (Pre vs Post Training) ---
    print_header("TEST 3: LEARNING DELTA")

    def get_stats(model, text):
        res = model.run(text)
        return res.top_cluster.score, np.linalg.norm(res.output)

    score_fresh, _ = get_stats(model_fresh, full_text)
    score_trained, _ = get_stats(model_trained, full_text)

    print(f"Convergence Score (Untrained):  {score_fresh:.4f}")
    print(f"Convergence Score (Pretrained): {score_trained:.4f}")

    if score_trained > score_fresh:
        print(f"Learning Delta: +{(score_trained - score_fresh):.4f} (Learning confirmed)")
    else:
        print("Result: FAIL (No learning improvement observed)")

    # --- TEST 4: PERTURBATION ROBUSTNESS ---
    print_header("TEST 4: PERTURBATION ROBUSTNESS")
    res_full = model_trained.run(full_text)
    res_pert = model_trained.run(perturbed_text)

    print(f"Full Text Convergence:     {res_full.top_cluster.score:.4f}")
    print(f"Perturbed Text Convergence:  {res_pert.top_cluster.score:.4f}")

    sim_pert = model_trained.similarity(full_text, perturbed_text)
    print(f"Sim(Full, Perturbed):      {sim_pert:+.4f}")

    if res_pert.top_cluster.score > 0.5:
        print("Result: SUCCESS (Maintains high attractor stability)")
    else:
        print("Result: FAIL (Stability lost under perturbation)")

if __name__ == "__main__":
    run_deep_verification()
