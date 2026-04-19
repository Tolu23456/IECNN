# IECNN Project

## Overview
A full Python implementation of IECNN (Iterative Emergent Convergent Neural Network) — a novel AI architecture that uses neural dots and convergence-based selection instead of traditional neural networks and backpropagation.

## Architecture

### Python Package: `iecnn/`
| File | Purpose |
|---|---|
| `formulas.py` | All 9 custom IECNN formulas as Python functions |
| `basemapping.py` | BaseMapper + BaseMap — converts text to structured matrices |
| `neural_dot.py` | NeuralDot, BiasVector, DotGenerator |
| `aim.py` | AIMLayer with all 6 inversion types + attention |
| `convergence.py` | ConvergenceLayer — clusters predictions by similarity + agreement strength |
| `pruning.py` | PruningLayer — 3-stage pruning (soft, cluster compression, hard) |
| `iteration.py` | IterationController — 3 stopping conditions (budget, dominance, novelty) |
| `pipeline.py` | IECNN — full pipeline orchestrating all layers |
| `__init__.py` | Package exports |

### Application Files
- `server.py` — Flask web app (port 5000, host 0.0.0.0) with live IECNN demo UI
- `main.py` — Standalone CLI demo script
- `iecnn_notes.md` — Full architecture notes
- `formulas.md` — All 9 custom IECNN formulas (mathematical notation)

## Key Design Decisions
- Neural dots are stateless mini-predictors (not neurons)
- BaseMapping uses hash-based stable embeddings (128-dim feature vectors)
- Feature vector layout: [96 base embed | 8 position | 4 freq | 16 flags | 4 context]
- AIM runs in parallel with originals — both enter Convergence together
- Three pruning stages: soft filter → cluster compression → hard selection
- Three stopping conditions: budget → dominance → low novelty gain
- Learning via bias vector update (Formula 8), not backpropagation
- AIM-assisted learning signal: winning inversions shift inversion_bias

## Dependencies
- numpy — all tensor/matrix operations
- flask — web server

## Running
```bash
python server.py      # Web demo on port 5000
python main.py        # CLI demo
```

## Workflow
- **Start application**: `python server.py` on port 5000 (webview)
