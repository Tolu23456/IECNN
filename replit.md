# IECNN — Replit Setup and Architecture Notes

## What this project is

IECNN (Iterative Emergent Convergent Neural Network) is a CLI-only research project
implementing a novel neural architecture. It runs entirely in the terminal
(`python main.py`), has no web server, and requires no external API keys.

---

## Running the App

```bash
python main.py                          # silent interactive REPL (just a "> " prompt)
python main.py demo                     # original full showcase output
python main.py train corpus.txt --limit N   # train on text file (one sent/line)
python main.py generate "prompt"        # encode prompt → decode output text
python main.py encode "text"            # encode → 256-dim latent
python main.py sim "text A" "text B"    # cosine similarity
python main.py compare "a" "b" "c"      # n×n similarity table
python main.py memory                   # dot memory + evolution state
python main.py build                    # (re)compile C extensions
```

The workflow `Start application` runs `python main.py` (silent REPL).

## Persistent Brain (Learning State)

The model now persists its learning across runs. On startup, `IECNN` auto-loads
brain files if present; on every interactive command (`encode`, `sim`, `generate`,
`train`, …) it auto-saves them.

Files (next to `global_brain.pkl`):

| file | contents |
|---|---|
| `global_brain.pkl` | vocabulary / base mapper |
| `global_brain.pkl.dots.pkl` | live neural dot pool (features + head projections) |
| `global_brain.pkl.dotmem.pkl` | per-dot effectiveness, success/total counts, prediction windows |
| `global_brain.pkl.clustmem.pkl` | learned pattern library + round snapshots |
| `global_brain.pkl.evo.pkl` | evolution generation counter + history |
| `global_brain.pkl.meta.pkl` | misc state (cumulative trained sentence count, etc.) |

To train more, run `python main.py train corpus_10k.txt --limit 30` repeatedly —
each call advances `evolution.generation`, expands `cluster_memory`, and
reinforces dots that landed in winning clusters.

### Known training-cost caveat
Each evolved generation adds ~40 new dot IDs to `dot_memory`/`dot_pool`
(40% kept + 60% replaced from the live pool of 64). Around ~50 sentences,
total tracked dots reach ~2k and process memory exceeds ~1 GB, which can
get the process OOM-killed in constrained environments. Practical training
batch in this Replit env is **~30 sentences per `train` invocation**;
re-invoke to continue. Long-term fix: prune `dot_memory` of dots with
zero recorded outcomes (TODO).

---

## Dependencies

**System:** Python 3.10+, gcc (for C extensions)
**Python:** numpy, Pillow (image and video decoding)
**C:** compiled automatically by `build.sh` on first run

Install missing packages if needed:
```bash
pip install numpy Pillow
```

Compile C extensions:
```bash
bash build.sh
```
or:
```bash
python main.py build
```

---

## Project Structure

```
/                           Workspace root = the IECNN project
├── main.py                 CLI entry point
├── build.sh                gcc compilation script
├── README.md               Full architecture documentation
├── formulas.md             All 20 formulas with derivations
├── CHANGELOG.md            Version history
├── replit.md               This file
│
├── neural_dot/             The core prediction unit
│   └── neural_dot.py       NeuralDot, BiasVector, DotGenerator, DotType (8 types)
│
├── basemapping/            Input → 256-dim structured matrix
│   ├── basemapping.py      BaseMapper, BaseMap
│   ├── basemapping.c       C acceleration
│   └── basemapping_c.so    Compiled C library
│
├── aim/                    Attention Inverse Mechanism (9 inversions)
│   ├── aim.py              AIMLayer, InversionType, 9 inversion functions
│   ├── aim.c               C acceleration
│   └── aim_c.so
│
├── convergence/            Two-level hierarchical clustering
│   ├── convergence.py      MicroCluster, Cluster, ConvergenceLayer
│   ├── convergence.c
│   └── convergence_c.so
│
├── pruning/                3-stage candidate filtering
│   └── pruning.py          PruningLayer (dynamic thresholds)
│
├── iteration/              Iteration control and feedback
│   └── iteration.py        IterationController (5 stops, adaptive LR, rollback)
│
├── pipeline/               Integration of all layers
│   └── pipeline.py         IECNN (main model class), fit_file(), generate()
│
├── decoding/               Output reconstruction
│   └── decoder.py          IECNNDecoder — latent → text / image / audio / video
│
├── memory/                 Persistent state across iterations and calls
│   ├── dot_memory.py       DotMemory — per-dot effectiveness and hints
│   └── cluster_memory.py   ClusterMemory — round timeline, pattern library
│
├── evolution/              Genetic dot pool evolution
│   └── dot_evolution.py    DotEvolution, EvolutionConfig
│
├── evaluation/             Quality metrics
│   └── metrics.py          IECNNMetrics, RunMetrics
│
└── formulas/               Mathematical engine (Python + C)
    ├── formulas.py          All 20 formulas + F18/F19/F20 (Python-only)
    ├── formulas.c           C implementations (F1–F15)
    ├── formulas.h           C header
    └── formulas_c.so        Compiled
```

---

## Architecture Notes

### Neural Dots
- 8 types: SEMANTIC, STRUCTURAL, CONTEXTUAL, RELATIONAL, TEMPORAL, GLOBAL, LOGIC, MORPH
- 4 prediction heads per dot (per-head projection matrices); 128 dots total
- Bias vector (5-dim): attention, granularity, abstraction, inversion, temperature
- Memory-guided attention: recent centroid from DotMemory used as query hint

### Formulas
F1–F17 implemented in Python (fallback) and C (primary via ctypes).
F18 (Cross-Modal Binding), F19 (Semantic Drift), F20 (Vocab Coverage) — Python-only.
C .so files use `_c.so` suffix to avoid colliding with Python module names.

### BaseMapping
- Primitives (a-z, 0-9, punct) are pre-seeded and always available
- Each token = exactly ONE 256-dim row (words never split into char rows)
- Unknown words → 'composed' type (weighted char-base combination)
- Feature layout: [224 embed | 8 pos | 4 freq | 16 flags (incl. 4 modality) | 4 ctx]
- Modality flags at dims 248:252 (text/image/audio/video — one-hot)

### Multimodal Transforms
- **Image**: lossless 8×8 patches + global stats → stacked feature vector (numpy-only, PIL for I/O)
- **Audio**: numpy FFT-based MFCC approximation (no librosa)
- **Video**: PIL ImageSequence frame-by-frame processing (no cv2)

### Decoder (IECNNDecoder)
- Two-stage greedy text decoding: Stage 1 ranks all vocab by embedding cosine (cheap),
  Stage 2 selects tokens greedily by average embedding similarity (no full pipeline calls)
- Image decode: latent dims mapped to pixel pattern via sinusoidal gradient
- Audio decode: latent dims mapped to frequency components via numpy synthesis

### Iteration Loop
- 5 stopping conditions (budget, dominance, novelty gain, stability, decline)
- Adaptive LR via Formula 14: eta(t) = eta_0 * (1 - 0.8 * dom²)
- Rollback to best-scoring round if final round regressed
- Thresholds recalibrated for 256-dim: micro=0.25, macro=0.15, dom=0.35

### Evolution
- Runs between calls (not during a call)
- Uses DotMemory effectiveness scores to rank dots
- Tournament selection + mutation + crossover + random injection

### Large Dataset Training
- `fit_file(path, verbose=True)` — stream one sentence per line, batch train
- `generate(prompt, max_tokens=8, iterations=20)` — prompt → latent → decoded text

---

## Constraints and Conventions

- **CLI only** — no web server, no flask, no HTTP
- **Max 10,000 lines per file**
- **Separate folder per major component** at workspace root
- **C `.so` files** named with `_c.so` suffix
- **numpy only** — no pytorch, tensorflow, or other ML frameworks
- **No backpropagation** — only bias vector update (Formula 8) and dot evolution

---

## Version

Current: **v0.8.0** — see CHANGELOG.md for full history.

---

## F16 + F17: EUG and Dot Reinforcement Pressure

Added in v0.4.0 (EUG) and v0.5.0 (DRP).

### F16: Emergent Utility Gradient

`U(t) = E[C_{t+1}(p)] - C_t(p)`

Estimated via recency-weighted score delta:
- 2 rounds: `U = C_t - C_{t-1}`
- 3+ rounds: `U = 0.7*(C_t - C_{t-1}) + 0.3*(C_{t-1} - C_{t-2})`

**Stopping:** if `U ≤ eug_threshold (0.001)` after ≥ 3 rounds, the system stops.

**Instability injection:** when `|U| < 0.01`, Gaussian noise (σ=0.05) is added to the refined vector before blending, pushing the system out of flat convergence basins.

### F17: Dot Reinforcement Pressure (DRP)

`R_d(t) = λ1·C_d + λ2·S_d + λ3·U_norm·(1 + β·ΔU_norm) − λ4·N_d`

- `C_d` = effectiveness (hit rate in winning cluster)
- `S_d` = specialization score (prediction consistency)
- `U_norm` = `tanh(U × 5)` — normalized EUG
- `ΔU_norm` = `tanh(ΔU × 5)` — normalized utility acceleration
- `N_d` = failure rate = `1 − effectiveness`

Applied within each call after `_record_dot_outcomes`:
1. **Floor pressure:** dots with `R_d < 0.05` have `success_count` decayed by 0.90
2. **Competition decay:** bottom 30% of dots by DRP score are decayed by 0.90

This creates within-call selection pressure that amplifies high-performing dots before `DotEvolution` runs between calls.

**Why the old novelty check was broken:** cluster IDs reset to `0, 1, 2, …` each round, so `cur - prev` was always empty and `novelty_gain` was always `0.0`, triggering a stop at round 2 every time.
