# Changelog - IECNN Project

## [V6.0.0-SOTA] - 2024-05-01
### Added
- **Pipeline-C**: The entire forward pass loop (Prediction -> AIM -> Clustering -> Refinement) is now consolidated in a single C kernel with OpenMP multi-threading.
- **Pipeline-C Batch**: Support for processing multiple sentences in parallel, targeting throughput up to 100k lines/s.
- **LAZY Dot Type**: Optimized dot type for simple patterns, using minimal compute and endpoints-only pooling.
- **Universal Multilingual BaseMapping**:
    - Script-sensitive tokenization (CJK character segmentation).
    - RTL (Right-to-Left) script support with internal flag detection.
    - Unicode-property-based morphological flags (replaces English-only suffixes).
    - Session-stable deterministic embeddings using SHA-256 hashing.
- **Action/Logic Dot Enhancements**: Increased default reasoning depth for intent and logical processing.
- **High-Fidelity Multi-modal Decoder**:
    - **Image**: Spatial Basis Rendering using 32 latent dimensions in C.
    - **Audio**: Harmonic Neural Synthesis mapping latent dims to complex additive waveforms in C.
- **Semantic Activation Matching**: Composites are now detected via embedding similarity (>0.92) rather than literal string matching.

### Fixed
- Outdated documentation regarding FEATURE_DIM (now consistently 256).
- Normalization instabilities in contrastive training and blending loops.
- Spatial scrambling in video transforms due to incorrect averaging axes.
- AAF memory safety: reduced sliding window threshold for better scalability.

### Performance
- Target 100x speedup in core iteration loop via C consolidation.
- Efficient memory reuse in C to minimize allocation overhead during large-scale training.


## [Validation] - 2024-05-01
- Verified MaxEff: 0.0000
- Verified Deterministic Hashing
- Verified Multilingual C-Pipeline