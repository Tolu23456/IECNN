import numpy as np
from pipeline.pipeline import IECNN
import time

def benchmark_energy():
    print("=== IECNN Energy-Based Optimization Benchmark ===")

    # Initialize IECNN with default settings (now includes energy objective)
    model = IECNN(num_dots=64, max_iterations=10, seed=42)

    # Test texts
    test_texts = [
        "Neural convergence emerges from independent agents.",
        "Optimization of energy functions leads to stable states.",
        "Gradient-free architectures utilize evolutionary selection.",
        "Multi-modal alignment requires robust cross-modal binding."
    ]

    print(f"\nProcessing {len(test_texts)} test inputs...")

    for i, text in enumerate(test_texts):
        print(f"\nInput {i+1}: {text}")
        start_time = time.time()
        result = model.run(text, verbose=True)
        end_time = time.time()

        print(f"\nResult Summary:")
        print(f"  Stop Reason: {result.stop_reason}")
        print(f"  Rounds:      {len(result.rounds)}")
        if result.rounds:
            last_round = result.rounds[-1]
            print(f"  Final Objective: {last_round.get('objective', 'N/A'):.4f}")
            print(f"  Final Energy:    {last_round.get('energy', 'N/A'):.4f}")
            print(f"  Final Stability: {last_round.get('stability', 'N/A'):.4f}")
            print(f"  Final EUG:       {last_round.get('eug', 'N/A'):.4f}")
        print(f"  Time taken:      {end_time - start_time:.3f}s")

        # Verify that energy metrics are present in the history
        if result.rounds:
            assert "energy" in result.rounds[0]
            assert "objective" in result.rounds[0]
            assert "stability" in result.rounds[0]

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    try:
        benchmark_energy()
    except Exception as e:
        print(f"\nBenchmark failed with error: {e}")
        import traceback
        traceback.print_exc()
