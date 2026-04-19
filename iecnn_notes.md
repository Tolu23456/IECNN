# IECNN (Iterative Emergent Convergent Neural Network)

## Core Concept

IECNN functions by generating many "neural dots". Each dot independently produces a prediction.

## Process

1. Neural dots are generated in large numbers.
2. Each dot makes its own independent prediction.
3. The system evaluates all predictions.
4. The most common or convergent results are identified.
5. Convergent results are merged.
6. Non-convergent predictions are discarded.
7. The process repeats iteratively.
8. Iteration continues until a stable pattern emerges.

## Key Characteristics

- Iterative refinement process
- Emergent behavior from multiple simple units
- Convergence-based validation mechanism
- Discarding of inconsistent outputs
- Stability as the stopping condition

## Capabilities

Designed to handle multiple data types:

- Text
- Images
- Video
- Other data formats

## Goal

To serve as a potential foundation for Artificial General Intelligence (AGI).

## Associated Mechanisms

### AIM (Attention Inverse Mechanism)

- Dot predictions are transformed or inverted.
- Attention is applied to focus on the most relevant parts of the prediction or context.
- Generates novel, context-aware candidate solutions.

### BaseMapping

A dynamic AI input representation system. Converts text into maps representing:

- **Bases** — characters, words, or multi-word phrases
- **Modifiers**
- **Features:**
  - Compression of sequences
  - Phrase-level base handling
  - Efficient handling of repeated words
  - Ability to discover new bases after deployment
- **Output:** Compact, structured matrices for IECNN processing
