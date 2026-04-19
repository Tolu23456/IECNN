# IECNN — Iterative Emergent Convergent Neural Network

## Overview
CLI-based Python + C implementation of IECNN. Novel AI architecture using independent
"neural dots" instead of neurons, with convergence-based learning and no backpropagation.

## Usage
```bash
python main.py                    # interactive demo + REPL
python main.py encode "text"      # encode text → 128-dim vector
python main.py sim "a" "b"        # similarity score between two texts
python main.py build              # compile C extensions manually
```

## Project Structure — Separate Folder Per Component

| Folder | Purpose | Languages |
|---|---|---|
| `formulas/` | All 9 IECNN custom formulas | Python + C |
| `basemapping/` | BaseMapper — converts text to structured matrices | Python + C |
| `neural_dot/` | NeuralDot, BiasVector, DotGenerator | Python |
| `aim/` | AIM — 6 inversion types + attention refinement | Python + C |
| `convergence/` | ConvergenceLayer — cluster predictions by similarity | Python + C |
| `pruning/` | PruningLayer — 3-stage candidate/cluster pruning | Python |
| `iteration/` | IterationController — 3 stopping conditions | Python |
| `pipeline/` | IECNN — full pipeline wiring all 10 layers | Python |

Each C module compiles to a `_c.so` shared library loaded via ctypes.
Python fallbacks are always available if C libs haven't been built.

## BaseMapping Design
- **Pre-seeded primitives**: a-z, 0-9, punctuation (always available from the start)
- **Known bases**: words/phrases discovered by frequency become named bases
- **Each token = ONE row**: words are NEVER split into character rows
  - Known words → `word` type, stable blend embedding
  - Unknown words → `composed` type, embedding built from character primitives
  - Frequent bigrams/trigrams → `phrase` type, single row per phrase
- **Feature vector** (128 dims): `[96 embed | 8 position | 4 freq | 16 flags | 4 context]`

## Key Architecture Details
- **Neural dots**: 64 stateless mini-predictors, each with a 5-dim bias vector
- **AIM**: 6 inversions (feature, context, spatial, scale, abstraction, noise) in parallel
- **Convergence**: C-accelerated pairwise similarity matrix, then cluster by agreement
- **Pruning**: soft filter → cluster compression → hard selection
- **Stopping**: iteration budget → dominance threshold → low novelty gain
- **Learning**: bias vector shift toward winning cluster patterns (no backprop)

## C Build
```bash
bash build.sh
# Outputs: formulas/formulas_c.so  basemapping/basemapping_c.so
#          aim/aim_c.so             convergence/convergence_c.so
```

## Workflow
- **Start application**: `python main.py` (console output, interactive REPL)
