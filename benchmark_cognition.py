import numpy as np
from pipeline.pipeline import IECNN
import time

def benchmark_cognition():
    print("=== IECNN Cognition Layer (AGI Control) Benchmark ===")

    model = IECNN(num_dots=64, max_iterations=10, seed=42)

    # Text inputs representing different cognitive loads
    test_inputs = [
        "A simple statement.",
        "A complex logical paradox that requires deep reasoning and abstraction to resolve properly.",
        "Planning a multi-step procedure for aligning high-dimensional latent vectors in a gradient-free regime.",
        "Exploring unknown semantic territories where entropy is high and stability is low."
    ]

    print(f"\nProcessing {len(test_inputs)} inputs through cognitive control...")

    for i, text in enumerate(test_inputs):
        print(f"\nInput {i+1}: {text}")
        res = model.run(text, verbose=True)

        # Verbose mode in model.run already prints the cognition footer

    print("\nVerifying Self-Model Persistence...")
    print(f"Self-Model state: {model.cognition.self_model}")
    assert not np.all(model.cognition.self_model == 0), "Self-model should have been updated."

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    benchmark_cognition()
