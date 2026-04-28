# IECNN Multi-Modal: Sensory BaseMapping & Rendering

IECNN treats all data—Text, Image, Audio, and Video—as "sentences of tokens" in a unified 256-dimensional representation space.

---

## 1. Sensory BaseMapping (Encoding)

### Text
Words and multi-word phrases are mapped to single vectors. Version 5 uses **Morpheme Decomposition** to capture sub-word features for unknown tokens.

### Image (Multi-Scale Sensory Patches)
Images are decomposed into hierarchical patches:
-   **4x4**: Captures fine texture and edges.
-   **8x8**: Captures local structure.
-   **16x16**: Captures global context.

### Audio (FFT Spectrum)
Audio is transformed into a frequency spectrum using multiple window sizes, allowing the model to process both temporal detail and harmonic precision.

### Video (Motion-Vector Encoding)
Video is treated as a sequence of image tokens. Version 5 introduces **8-dim Motion Signatures** that encode the pixel-level differences between consecutive frames, allowing "Neural Dots" to attend specifically to movement.

---

## 2. Neural Rendering (Decoding)

Decoding in IECNN is an iterative process of **Generative Convergence**.

### Tournament Decoding (Text)
Instead of a simple probability check, the decoder picks the top-10 vocabulary candidates and runs a "Mini-Convergence" pass. The word that best aligns with the latent vector's internal features wins.

### Patch-Based Rendering (Image)
The decoder reconstructs images by iteratively proposing 8x8 patches and accepting those that increase the global similarity between the canvas's BaseMap and the target latent vector.

### Additive Spectral Synthesis (Audio)
Complex audio is reconstructed by treating latent dimensions as harmonic weights and frequency modulators, creating rich, multi-tonal sound from a single vector.

---

## 3. The Unified Representation

All modalities share the same **FEATURE_DIM = 256** layout:
-   `[0:224]`: Base Embedding
-   `[224:232]`: Position Encoding
-   `[232:236]`: Frequency Features
-   `[236:252]`: Modifier & Modality Flags
-   `[252:256]`: Context Summary
