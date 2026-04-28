# IECNN — Iterative Emergent Convergent Neural Network (V5 SOTA)

A novel neural architecture where independent **Neural Dots** produce competing predictions that **converge by agreement**.

IECNN is gradient-free, strictly additive, and uses evolutionary selection instead of backpropagation. Version 5 (SOTA) introduces **Autoregressive Agency**, **Recursive Reasoning**, and **Multi-Modal Neural Rendering**.

---

## The Core Concept

Traditional networks pass signals through fixed layers and train all weights together via gradient descent. IECNN is fundamentally different:

| Feature | Traditional NN (Transformer) | IECNN (V5 SOTA) |
|---|---|---|
| **Unit** | Neurons (tiny signal-passers) | Neural Dots (independent predictors) |
| **Logic** | Fixed Layered DAG | Dynamic Pool of Free Agents |
| **Learning** | Backpropagation | Emergent Agreement + Selection |
| **Output** | One prediction per forward pass | Consensus of thousands of candidates |
| **Memory** | Fixed KV-Cache (Weights-based) | Hierarchical Surprise-Driven Graph |

The key question: *what do independent agents agree on?* The answer that emerges is the output.

---

## V5 SOTA Architecture — 12 Layers

```
INPUT (Text, Image, Audio, Video)
  ↓
Layer 1:  Sensory BaseMapping  — Multi-scale patches, Motion vectors, IDF weights
Layer 2:  AAF Alignment        — Attention Allocation Field pre-structures the input
Layer 3:  Dot Generation       — 128 dots × 8 types × 4-8 heads (Dynamic Allocation)
Layer 4:  Prediction           — Independent prediction + Recursive Internal Reasoning
Layer 5:  AIM Inversion        — 11 inversions run in parallel with originals
Layer 6:  Pruning (S1)         — Confidence filter, duplicates, AIM cap
Layer 7:  Convergence          — Two-level hierarchical clustering (Micro + Macro)
Layer 8:  Pruning (S2+S3)      — Cluster compression + dynamic hard selection
Layer 9:  AGI Control (CSV)    — Meta-cognition, Dynamic Thinking Policy, Energy scaling
Layer 10: Reflection & Veto    — Specialized dots veto consensus; Repellent Convergence
Layer 11: Memory Consolidation — Surprise-driven World Graph update + Plasticity (F23)
Layer 12: Output & Decoding    — Tournament Decoding, Patch-Based Neural Rendering
```

---

## New V5 SOTA Features

### 1. Autoregressive Agency & Chat
IECNN now supports token-by-token predictive generation.
- **Causal Slicing**: Dots are biased toward the end of sequences (Beta distribution).
- **Causal Reinforcement (C-DRP)**: Dots are rewarded for predicting the actual next token.
- **Global Narrative Base**: History is compressed into a persistent latent vector to maintain long-term conversation context.

### 2. Deep Recursive Reasoning
- **Inner Monologue**: Neural dots perform internal micro-iterations to resolve contradictions before outputting candidates.
- **Learnable Reasoning Depth**: Dots evolve their complexity based on the difficulty of their assigned niche.
- **Cognitive Veto**: LOGIC dots can veto a weak consensus, triggering a search for non-contradictory alternatives.

### 3. Multi-Modal Neural Rendering
- **Patch-Based Convergence**: Images are reconstructed by iteratively finding visual patches that align with the global latent vector.
- **Additive Spectral Synthesis**: Complex audio is generated using latent dimensions as harmonic weights.
- **Motion-Vector Encoding**: Video frames are linked via 8-dim motion signatures.

### 4. Surprise-Driven Memory (M_long)
- **World Knowledge Graph**: Permanent fact-based memory that consolidates stable patterns.
- **Surprise Filter**: Prevents redundant memory growth (explosive memory) by only recording conceptually novel information.

---

## Documentation Suite

For detailed technical explanations, see the `/docs` directory:

- [**Architecture**](docs/ARCHITECTURE.md) — Neural Dots, Convergence, and Thinking Loops.
- [**Memory**](docs/MEMORY.md) — Hierarchical systems, Plasticity, and the World Graph.
- [**Multi-Modal**](docs/MULTIMODAL.md) — Multi-scale BaseMapping and Neural Rendering.
- [**AGI Control**](docs/AGI_CONTROL.md) — Meta-cognition, CSV, and the Thinking Policy.
- [**Formulas**](docs/FORMULAS_V5.md) — Complete derivation of F1–F35.

---

## Usage

```bash
# Interactive Chatbot Mode (New in V5)
python main.py chat

# Encode text into 256-dim Latent
python main.py encode "neural convergence"

# Train on a large corpus (streaming batches)
python main.py train data/corpus.txt --evolve --prune-every 500

# Full Demo (6 SOTA examples)
python main.py demo
```

---

## Design Principles

1. **Strictly Gradient-Free** — No backpropagation or global loss functions.
2. **Emergent Agreement** — Meaning arises from the consensus of independent agents.
3. **No Explosive Memory** — Knowledge is consolidated via novelty-driven pruning.
4. **C-Acceleration** — Performance-critical paths are implemented in shared C libraries.
