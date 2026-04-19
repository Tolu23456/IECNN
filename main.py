"""
IECNN — Iterative Emergent Convergent Neural Network
Demo script: runs the full pipeline on text examples.
"""

import numpy as np
from iecnn import IECNN


CORPUS = [
    "neural networks learn from data using gradient descent",
    "deep learning models use many layers to extract features",
    "transformers use self-attention to model relationships",
    "natural language processing understands human language",
    "computer vision enables machines to interpret images",
    "reinforcement learning trains agents through rewards",
    "convolutional networks process spatial data efficiently",
    "recurrent networks handle sequential information over time",
    "generative models can create new data from learned distributions",
    "embeddings map words into continuous vector spaces",
]

EXAMPLES = [
    "IECNN uses neural dots that independently generate predictions",
    "convergence emerges from agreement among many independent units",
    "BaseMapping converts text into structured maps of bases and modifiers",
]


def print_section(title: str):
    print(f"\n{'─'*60}")
    print(f"  {title}")
    print(f"{'─'*60}")


def run_demo():
    print("=" * 60)
    print("  IECNN — Iterative Emergent Convergent Neural Network")
    print("  Version 0.1.0  |  Text Processing Demo")
    print("=" * 60)

    print_section("1. Initialising IECNN")
    model = IECNN(
        feature_dim=128,
        num_dots=64,
        max_iterations=8,
        similarity_threshold=0.45,
        dominance_threshold=0.70,
        novelty_threshold=0.05,
        seed=42,
    )
    print(f"  Dots       : {model.num_dots}")
    print(f"  Feature dim: {model.feature_dim}")
    print(f"  Max iters  : {model.iteration_ctrl.max_iterations}")

    print_section("2. Fitting BaseMapper on corpus")
    model.fit(CORPUS)
    print(f"  Corpus size   : {len(CORPUS)} texts")
    print(f"  Word vocab    : {len(model.base_mapper.word_vocab)} entries")
    print(f"  N-gram vocab  : {len(model.base_mapper.ngram_vocab)} entries")

    print_section("3. BaseMapping a sample sentence")
    sample = EXAMPLES[0]
    bmap = model.base_mapper.transform(sample)
    print(f"  Input  : '{sample}'")
    print(f"  {bmap}")
    print(f"  Bases  : {bmap.bases}")
    print(f"  Matrix : shape {bmap.matrix.shape}, dtype {bmap.matrix.dtype}")

    print_section("4. Running the full IECNN pipeline (verbose)")
    result = model.run(EXAMPLES[0], verbose=True)

    print_section("5. Iteration summary")
    for r in result.all_rounds:
        print(
            f"  Round {r['round']}: "
            f"{r['num_candidates']} candidates → "
            f"{r['num_clusters']} clusters | "
            f"dom={r['dominance']:.3f}"
        )
    print(f"\n  Stop reason : {result.stop_reason}")
    print(f"  Rounds      : {result.iteration_summary['rounds_completed']}")
    if result.top_cluster:
        print(f"  Top cluster : size={result.top_cluster.size}, score={result.top_cluster.score:.4f}")
        print(f"  Sources     : {result.top_cluster.sources()}")

    print_section("6. Encoding all examples")
    encodings = []
    for text in EXAMPLES:
        enc = model.encode(text)
        encodings.append(enc)
        norm = float(np.linalg.norm(enc))
        print(f"  '{text[:50]}...' → norm={norm:.3f}")

    print_section("7. Semantic similarity between examples")
    from iecnn.formulas import similarity_score
    for i in range(len(EXAMPLES)):
        for j in range(i + 1, len(EXAMPLES)):
            sim = similarity_score(encodings[i], encodings[j])
            print(f"  [{i}] vs [{j}]: similarity = {sim:.4f}")

    print_section("8. Bias vector after learning")
    bv = model.base_bias
    print(f"  Attention bias    : {bv.attention_bias:.4f}")
    print(f"  Granularity bias  : {bv.granularity_bias:.4f}")
    print(f"  Abstraction bias  : {bv.abstraction_bias:.4f}")
    print(f"  Inversion bias    : {bv.inversion_bias:.4f}")
    print(f"  Sampling temp     : {bv.sampling_temperature:.4f}")

    print(f"\n{'='*60}")
    print("  IECNN demo complete.")
    print(f"{'='*60}\n")


if __name__ == "__main__":
    run_demo()
