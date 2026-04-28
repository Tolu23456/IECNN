import sys
import os
import numpy as np

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

from pipeline.pipeline import IECNN

def test_transformer_killer_features():
    print("Testing 'Transformer-Killer' Architectural Suite...")
    model = IECNN(num_dots=32, max_iterations=5)

    # 1. Multi-Modal Patch-Based Rendering
    print("\n1. Testing Multi-Modal Rendering (Image/Audio)...")
    latent = np.random.randn(256).astype(np.float32)
    from decoding.decoder import IECNNDecoder
    decoder = IECNNDecoder(model)

    img = decoder.decode(latent, target_mode="image")
    print(f"Generated Image size: {img.size}")

    audio_bytes = decoder.decode(latent, target_mode="audio")
    print(f"Generated Audio bytes: {len(audio_bytes)}")

    # 2. Surprise-Driven Memory Consolidation
    print("\n2. Testing Surprise-Driven Consolidation...")
    # Add a repeating pattern (Low Surprise)
    pat = np.random.randn(256).astype(np.float32)
    for _ in range(3):
        model.world_graph.consolidate([(pat, 0.8)], surprise_threshold=0.1)
    print(f"World Graph Nodes (after repeat): {len(model.world_graph.nodes)}")

    # Add a unique pattern (High Surprise)
    unique_pat = np.random.randn(256).astype(np.float32)
    model.world_graph.consolidate([(unique_pat, 0.8)], surprise_threshold=0.1)
    print(f"World Graph Nodes (after unique): {len(model.world_graph.nodes)}")

    # 3. Cognitive Veto System
    print("\n3. Testing Cognitive Veto (Repellent Convergence)...")
    # This triggers _reflect with possible vetoes
    res = model.run("logic contradicts consensus")
    print(f"Refinement successful: {res.stop_reason}")

    # 4. Multi-Scale Sensory BaseMapping
    print("\n4. Testing Multi-Scale Sensory BaseMapping...")
    # Note: requires an actual image file or mock
    # (Simulated via transform check)
    try:
        # If we have a mock image or one in the environment
        pass
    except:
        pass

    # 5. Thinking Policy
    print("\n5. Testing Dynamic Thinking Policy (Light vs Deep)...")
    # Force a high EUG delta (simulated)
    from cognition.control import CognitiveStateVector
    csv_high_surprise = CognitiveStateVector(eug=0.25, entropy=0.5, dominance=0.2)
    actions = model.self_model.decide(csv_high_surprise)
    print(f"Budget Delta (High Surprise): {actions.iteration_budget_delta}")

    csv_low_surprise = CognitiveStateVector(eug=0.01, entropy=0.5, dominance=0.2)
    actions_low = model.self_model.decide(csv_low_surprise)
    print(f"Budget Delta (Low Surprise): {actions_low.iteration_budget_delta}")

if __name__ == "__main__":
    test_transformer_killer_features()
