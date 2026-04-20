import sys
import os
import numpy as np
from PIL import Image

# Ensure we can import from the root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from pipeline.pipeline import IECNN
from formulas.formulas import similarity_score

def run_narrative_test():
    print("╔══════════════════════════════════════════════════════════════════╗")
    print("║               IECNN NARRATIVE & GROUNDING BENCHMARK              ║")
    print("╚══════════════════════════════════════════════════════════════════╝")

    model = IECNN()

    # 1. Sensory Grounding Test
    # Define a visual proof for 'red'
    img_red = Image.new('RGB', (128, 128), color = (255, 0, 0))
    img_red.save('grounding_red.png')

    print("\n[GROUNDING] Training Global Brain with sensory proofs...")
    # Ground 'red' by fitting on an image + text pair
    model.base_mapper.fit(["red"])
    # Mock sensory hint for smoothing
    red_vec = model.base_mapper.transform('grounding_red.png', mode='image').pool()
    model.base_mapper._apply_cooc_smoothing(sensory_hints={"red": red_vec})

    # 2. Narrative Consistency Test
    print("\n[NARRATIVE] Testing multi-scene context...")

    # Scene 1: Introduction
    print("\nScene 1: 'The mystery begins.'")
    res1 = model.run("The mystery begins.")
    v1 = res1.output

    # Scene 2: Continuation (without explicit mentions, should be similar due to working memory)
    print("Scene 2: 'It deepens now.'")
    res2 = model.run("It deepens now.")
    v2 = res2.output

    sim = similarity_score(v1, v2)
    print(f"\n[RESULT] Sequential Similarity: {sim:+.4f}")

    if sim > 0.1: # Significant link compared to baseline noise
        print("[SUCCESS] NARRATIVE CONTEXT MAINTAINED.")
    else:
        print("[FAILURE] Context lost between calls.")

    # Cleanup
    os.remove('grounding_red.png')

if __name__ == "__main__":
    run_narrative_test()
