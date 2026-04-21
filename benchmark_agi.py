import numpy as np
from pipeline.pipeline import IECNN
import time

def benchmark_agi_stack():
    print("=== IECNN AGI Stack (World, Memory, Planning) Benchmark ===")

    model = IECNN(num_dots=16, max_iterations=5, seed=42)

    # Scenario: A sequential interaction where the model needs to build a world state
    scenario = [
        "A heavy object is placed on the table.",
        "The table begins to creak under the weight.",
        "A support beam is added to the table.",
        "The creaking stops and the table is now stable."
    ]

    print(f"\nProcessing AGI Scenario ({len(scenario)} steps)...")

    for i, step in enumerate(scenario):
        print(f"\n--- STEP {i+1}: {step} ---")
        res = model.run(step, verbose=True)

        # Verify AGI stack outputs
        agi_layer = model.cognition
        print(f"World Vector Norm: {np.linalg.norm(agi_layer.world.global_vector):.3f}")
        print(f"Memory Count: {agi_layer.memory.count}")

    print("\nVerifying Causal Score in World Model...")
    # Relationship between (dummy) entities
    agi_layer.world.add_entity(1, np.random.randn(256), {"name": "object"})
    agi_layer.world.add_entity(2, np.random.randn(256), {"name": "table"})
    agi_layer.world.add_relationship(1, 2, "on_top_of")

    # Simulate a surprise event that improves objective J
    agi_layer.world.construct_causal_graph(surprise=0.8, delta_j=0.5)
    causal_score = agi_layer.world.relationships[(1, 2)]["causal_score"]
    print(f"Causal Score for Relationship (1,2): {causal_score:.4f}")
    assert causal_score > 0, "Causal score should be reinforced."

    print("\nBenchmark completed successfully!")

if __name__ == "__main__":
    benchmark_agi_stack()
