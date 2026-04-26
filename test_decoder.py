import numpy as np
from pipeline.pipeline import IECNN
from decoding.decoder import IECNNDecoder
import os

def test_decoding():
    # 1. Setup Model and fit on a tiny corpus
    model = IECNN(num_dots=32, max_iterations=2)
    model.base_mapper.min_base_freq = 1
    corpus = [
        "The red apple is sweet.",
        "A blue bird flies in the sky.",
        "Green grass grows fast."
    ]
    model.fit(corpus)

    decoder = IECNNDecoder(model)

    # 2. Encode a concept
    text = "red apple"
    latent = model.encode(text)
    print(f"Encoded '{text}', Latent norm: {np.linalg.norm(latent):.4f}")

    # 3. Decode back to text
    decoded_text = decoder.decode(latent, target_mode="text", max_tokens=3)
    print(f"Decoded text: {decoded_text}")

    # 3b. Test Pipeline-in-the-Loop decoding
    print("Testing Pipeline-in-the-Loop decoding...")
    refined_text = decoder.decode(latent, target_mode="text", max_tokens=3, use_pipeline=True)
    print(f"Refined text: {refined_text}")

    # 4. Decode to image
    decoded_img = decoder.decode(latent, target_mode="image")
    decoded_img.save("decoded_apple.png")
    print("Decoded image saved to decoded_apple.png")

    # 5. Decode to audio
    decoded_aud = decoder.decode(latent, target_mode="audio")
    decoder.save_output(decoded_aud, "audio", "decoded_apple.wav")
    print("Decoded audio saved to decoded_apple.wav")

if __name__ == "__main__":
    test_decoding()
