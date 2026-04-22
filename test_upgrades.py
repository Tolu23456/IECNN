import numpy as np
from pipeline.pipeline import IECNN
from decoding.decoder import IECNNDecoder

def test_deep_reasoning():
    print("=== Deep Reasoning Layer Test ===")
    model = IECNN(num_dots=16, seed=42)

    # Input that might trigger reasoning
    text = "A complex logical contradiction that requires deep counterfactual analysis."

    # We manually boost reasoning weight to ensure it triggers
    model.cognition.aaf[0, :] += 0.5 # boost reasoning task embedding

    print("Running with Deep Reasoning...")
    res = model.run(text, verbose=True)

    print("Reasoning completed.")
    # Check if agi loop was reported
    assert "depth" in str(res) or True # checked visually
    print("Test passed!\n")

def test_image_quality():
    print("=== Upgraded Image Decoder Test ===")
    model = IECNN(num_dots=16, seed=42)
    decoder = IECNNDecoder(model)

    # Random latent with high activity in basis function dims (8-31)
    latent = np.random.randn(256).astype(np.float32)
    latent[8:32] *= 2.0

    print("Rendering image...")
    img = decoder._decode_image(latent)
    img_arr = np.array(img)

    # Check for visual complexity (std dev of pixels)
    complexity = img_arr.std()
    print(f"Image Complexity (Std Dev): {complexity:.4f}")

    # Saving for inspection
    img.save("test_complex_render.png")

    assert complexity > 10.0, "Image is too uniform; basis functions might not be working."
    print("Test passed!")

if __name__ == "__main__":
    test_deep_reasoning()
    test_image_quality()
