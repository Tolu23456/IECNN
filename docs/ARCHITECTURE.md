# IECNN Architecture: The Consensus-Driven Engine

This document provides a deep dive into the core mechanisms of the Iterative Emergent Convergent Neural Network (IECNN).

---

## 1. Neural Dots (The Agents)

The fundamental unit of IECNN is the **Neural Dot**. Unlike a neuron, which is a simple mathematical function, a Neural Dot is a complete, independent predictor.

### Structure of a Dot
```python
NeuralDot {
    Weights: W (transform), P (projections), Q (attention)
    BiasVector: [attention, granularity, abstraction, inversion, temperature, reasoning_depth]
    Type: SEMANTIC | LOGIC | STRUCTURAL | ...
}
```

### Operation
1.  **Causal Slicing**: The dot selects a contiguous slice of the input BaseMap. In predictive mode, it uses a Beta distribution to bias its focus toward the end of the sequence.
2.  **Recursive Internal Reasoning (Inner Monologue)**: Before outputting, the dot performs $N$ micro-iterations (determined by `reasoning_depth`). It projects its own state, applies "Cognitive Peer Pressure" from the previous consensus, and gates the result to resolve internal contradictions.
3.  **Multi-Head Prediction**: Each dot generates 4-8 diverse candidates using head-specific projection matrices.

---

## 2. Convergence (Hierarchical Clustering)

Meaning in IECNN is not calculated; it **emerges**. The Convergence Layer identifies what the independent dots agree on.

### The Two-Level Process
-   **Level 1 — Micro-Clustering**: Groups very similar predictions into tight, high-confidence micro-clusters.
-   **Level 2 — Macro-Clustering**: Groups micro-cluster centroids. This hierarchical approach allows the system to capture broad semantic consensus while maintaining fine structural detail.

### Convergence Score (F2)
A cluster's score is a function of its internal cohesion (how much members agree) and its mean confidence.

---

## 3. AIM (Attention Inverse Mechanism)

AIM is IECNN's "imagination" or counterfactual engine. It runs in parallel with original predictions.

### Inversion Types
IECNN uses 11 inversion types (e.g., Feature Flip, Relational Inversion, Compositional SVD) to challenge its own assumptions.
If an inversion gains consensus, it signals a representational breakthrough.

---

## 4. Reflection & Veto System

The final safety layer of the pipeline.

1.  **Veto Detection**: After the main convergence loop, specialized **LOGIC** and **SEMANTIC** dots examine the output.
2.  **Repellent Convergence**: If dots strongly disagree (Veto), the system triggers a new convergence pass where the current output acts as a "repellent," forcing the system to find a non-contradictory alternative.

---

## 5. Iteration Control

The system loop continues until:
1.  **Dominance** is achieved (one cluster holds > 70% weight).
2.  **Novelty Gain (EUG)** drops below a threshold.
3.  **Global Energy** stabilizes.

If the final round is worse than a previous one, the **Rollback** mechanism restores the best-known state.
