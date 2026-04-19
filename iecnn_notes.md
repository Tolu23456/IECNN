# IECNN (Iterative Emergent Convergent Neural Network)

## Overview

IECNN is an AI architecture — similar in role to the Transformer, but fundamentally different in design. Instead of traditional neural networks, IECNN uses **neural dots**: independent units that each make their own prediction. The system then identifies the most common result, discards the rest, and repeats this process through a series of layers until a stable output emerges.

## Core Concept

Each neural dot receives data and independently produces a prediction. The system evaluates all predictions, finds convergence (the most common or dominant result), merges the agreeing predictions, and prunes the rest. This loop repeats until the output stabilises.

## IECNN Layers

### 1. Input Layer
- Receives raw data (text, image, video, etc.)
- Passes data into BaseMapping for structuring

### 2. BaseMapping Layer
- Converts input into maps of bases and modifiers
- Produces compact structured representations

### 3. Neural Dot Generation Layer
- Splits mapped data into many neural dots
- Each dot is assigned a portion or perspective of the data

### 4. Prediction Layer
- Each neural dot independently generates a prediction

### 5. AIM Layer (Attention Inverse Mechanism)
- Transforms/inverts predictions
- Applies attention to focus on relevant parts
- Produces refined candidate predictions

### 6. Convergence Layer
- Compares all predictions in the **unified BaseMapping representation space**
- Uses a single similarity metric over structured maps — consistent across all data types
- Detects common or dominant patterns using **similarity + agreement strength** (not purely frequency)
- Identifies convergence points

### 7. Merge Layer
- Merges convergent predictions into a unified output

### 8. Pruning Layer
- Discards non-converging or inconsistent predictions

### 9. Iteration Controller Layer
- Feeds merged output back into the system
- Repeats process until stability is reached

### 10. Output Layer
- Produces the final stable result

## Neural Dots vs. Neurons

| | Neuron (ANN) | Neural Dot (IECNN) |
|---|---|---|
| Role | Tiny signal-passer in a fixed layered system | Complete mini-predictor on its own |
| Scope | Part of a larger computation | Produces one full candidate answer |
| Interaction | Passes signals forward to the next layer | Competes with other dots, then converges |
| State | Stateless (activation only) | Stateless by default — input → prediction |

The system's "memory" comes from the **iteration and convergence process**, not from individual dots carrying state. Each dot receives a slice or perspective of the input and independently produces a candidate answer.

## Key Characteristics

- Neural dots instead of traditional neural networks
- Each dot is a complete, stateless mini-predictor
- Many dots run in parallel on the same input without interference
- Convergence-based validation — only the most common results survive
- Iterative refinement until stability is reached
- Emergent behavior from many simple independent units
- Memory emerges from iteration, not from individual dot state

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
- AIM outputs enter the **Convergence Layer in parallel with the original dot predictions** — they compete and reinforce each other rather than replacing the originals.
  - Convergence sees a richer "idea space"
  - Useful inversions gain support and survive
  - Unhelpful inversions are pruned naturally
  - Preserves the principle of emergent agreement throughout

#### Visual Inversion Types

| Type | Description | Example |
|---|---|---|
| **Feature inversion** | Flip attributes, not pixels | "bright edge" → "dark edge"; "convex" → "concave" |
| **Context inversion** | Reinterpret the role of a patch | "foreground object" → "background texture" |
| **Spatial inversion** | Mirror, rotate, or flip inside↔outside relationships | Same content, different spatial meaning |
| **Scale inversion** | Reinterpret local detail as global structure, or vice versa | Catches misclassification of local vs. global features |
| **Abstraction inversion** | Flip between levels of understanding | Patches → object hypothesis ("face"), or object → decomposed patches |
| **Noise/absence inversion** | "What if this feature isn't actually there?" | Suppress a dominant feature and generate alternative predictions |

### BaseMapping

A dynamic AI input representation system. Converts input into maps representing:

- **Bases** — the fundamental unit of any input type
- **Modifiers** — contextual properties that describe or qualify each base
- **Features:**
  - Compression of sequences
  - Phrase-level base handling
  - Efficient handling of repeated elements
  - Ability to discover new bases after deployment
- **Output:** Compact, structured matrices for IECNN processing

BaseMapping is **data-type-aware** — the base-discovery process adapts to the input type, but the output format remains consistent, allowing the Convergence Layer to apply a single similarity metric regardless of modality:

| Data Type | Bases | Modifiers |
|---|---|---|
| Text | Characters, words, multi-word phrases | Grammatical role, position, frequency |
| Images | Patches (small tiles of pixels) | Position, color distribution, edges, texture |
| Video | Patches + frame index | Position, motion, temporal change |

This treats images as "sentences of visual tokens" and video as "sentences of visual tokens over time" — consistent with the text-based design.
