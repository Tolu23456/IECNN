# IECNN — Replit Setup and Architecture Notes

## What this project is

IECNN (Iterative Emergent Convergent Neural Network) is a CLI-only research project
implementing a novel neural architecture. It runs entirely in the terminal
(`python main.py`), has no web server, and requires no external API keys.

## User Preferences

- **No web server.** This project is intentionally CLI-only. Do not add Flask, Gunicorn, or any other web framework or HTTP server.

---

## Bug Fixes Applied (May 2026)

### Critical
1. **`pipeline/pipeline.py` — 5 missing methods** — `compare()`, `generate()`, `memory_status()`, `fit_file()`, and `prune_dots()` were called from `main.py` but never defined in the class. All five are now implemented.
2. **`pipeline/pipeline.py` — `train_pass()` missing `prune_every` parameter** — `main.py` called `model.train_pass(..., prune_every=N)` which crashed with `TypeError`. Parameter added and wired to periodic `prune_dots()` calls.
3. **`pipeline/pipeline.py` — `chat()` imported non-existent modules** — `cognition.router.AGIRouter` and `utils.tools.ToolExecutor` don't exist; chat crashed on every invocation. Replaced with a clean context-window + decoder implementation.
4. **`memory/dot_memory.py` — `reset_all()` crashed** — referenced `self._var_sums` and `self._var_counts` which don't exist (correct attribute is `self._var_stats`). Also missing `_surprise_history`, `_exemplars`, `_semantic_grounding` clears.

### Moderate
5. **`pipeline/pipeline.py` — Python-fallback `run()` returned empty summary `{}`** — `cmd_encode` accessed `res.summary.get('rounds', ...)` which always fell back to an empty list `[]`. Now returns `iter_ctrl.summary()` like all other paths.
6. **`neural_dot/neural_dot.py` — `local_update()` row-index collision** — two separate `_rng.randint()` calls were used as the read-index and write-index for the same W row update, so they picked *different* rows. Fixed to capture a single `row` variable used for both.
7. **`pipeline/pipeline.py` — `import time` at bottom of file** — used inside `train_pass()` but placed after the class definition. Moved to the standard top-of-file imports.

### Minor (silent data loss)
8. **`pipeline/pipeline.py` `_blend()` — ComplexWarning on blend** — cast a complex centroid to float32 without taking `np.real()` first, silently dropping the imaginary part. Fixed.
9. **`decoding/decoder.py` `_score_emb()` / `_generative_reconstruction_text()` — ComplexWarning on decode** — same silent imaginary-part drop when scoring or slicing a complex latent. Fixed with explicit `np.real()` before cast.

---

## Features Added (May 2026)

### 1. Beam Search Decoding (`decoding/decoder.py`)
`IECNNDecoder.decode()` now accepts `beam_width: int = 1`. When `beam_width > 1`,
`_beam_search_text()` is used instead of the greedy decoder. It keeps `beam_width` partial
sequences alive at each token step, uses incremental running-average embeddings for O(FEATURE_DIM)
cost per extension, and stops early when no beam improves by more than `1e-4`. Default for
`model.generate()` and `model.chat()` is `beam_width=4`.

### 2. Causal Next-Token Prediction Training (`pipeline/pipeline.py`)
`IECNN.causal_train_pass(sentences, max_pos=6, verbose=True, prune_every=0)` slides a context
window across each sentence, encodes each prefix with `causal=True` through the Python path
(preserving phase), looks up the actual next-token base embedding, scores the output against it,
then calls `dot.local_update(target, lr=0.005)` on every dot. Dots are evolved every 10 sentences.
Available on the CLI via `python main.py train <file> --causal`.

### 3. Dot Weight Cache Invalidation (`neural_dot/neural_dot.py`)
`DotGenerator` now has a `_dot_version: int` counter and a `bust_cache()` method.
`_ensure_caches()` checks both `id(dots)` (list identity) and `_dot_version` (mutation version),
so in-place weight mutations that don't replace the list still force a cache rebuild.
`causal_train_pass` calls `bust_cache()` after every `local_update` batch.

### 4. Phase-Coding End-to-End (`pipeline/pipeline.py`)
When `phase_coding=True`, the C-pipeline path is bypassed (`not self.phase_coding` guard
on the `if use_c_pipeline ...` branch). The Python path already produces complex64 predictions
throughout. This means phase-coherence binding is actually functional at inference time.

### 5. Cluster Memory Per-Call Scoping (`pipeline/pipeline.py`, `memory/cluster_memory.py`)
`cluster_memory.reset_call()` is now called at the start of every `run()` (both C and Python
paths), so `temporal_stability()` measures the current call only. The Python fallback loop
calls `cluster_memory.record_round()` at every iteration. After convergence, the winning
centroid is committed to the cross-call pattern library via `cluster_memory.commit_pattern()`.

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

### Brain-size controls (April 2026)
Two mechanisms now cap on-disk and in-memory growth:

1. **Float16 weight storage.** `NeuralDot.__getstate__` / `__setstate__`
   serialize `W`, `Q_basis`, `b_offset` and all `head_projs` as float16 and
   restore them as float32 at load time. Runtime math is unchanged. This
   alone takes the dot pool from ~641 MB → ~161 MB on disk (~4× shrink
   vs the legacy float64 layout).
2. **Joint pruner.** `IECNN.prune_dots(min_outcomes=2, min_age_gens=2)` runs
   automatically after every `evolution.evolve(...)` call. A dot is removed
   when it is BOTH old enough (current_gen - birth_generation >= min_age_gens)
   AND has fewer than `min_outcomes` recorded predictions. Surviving ids
   are then used to prune orphan history out of `dot_memory` so
   `.dotmem.pkl` no longer grows unboundedly across generations.

`NeuralDot` instances now carry a `birth_generation` attribute, set by
`DotEvolution._mutate`, `_crossover`, and the random-diversity branch of
`evolve()`. Legacy pickles without this field default to generation 0.

The previous "~30 sentences per `train` invocation" cap should now be
substantially relaxed; re-measure before pushing further.

To compact the brain manually without running an encode:

```bash
python main.py prune                                  # apply default prune
python main.py prune --dry-run                        # preview only, no writes
python main.py prune --min-outcomes 5 --min-age 3     # stricter culling
```

`python main.py memory` also prints per-file brain sizes and a dry-run
prune preview so you can see how much would be reclaimed.

## Phase-coded dots (experimental, opt-in)

Pass `--phase-coding` on any CLI command (or before any interactive
session) to enable IECNN-native phase-coherence binding. Each prediction
gets a `phase ∈ [0, 2π)` derived from the slice center within the
current basemap. Cluster memory accumulates per-pattern circular
statistics (mean phase + concentration) and uses a phase-aware
similarity (`formulas.phase_aware_similarity`) for both pattern matching
on commit and nearest-neighbor lookup. The intent is to let the model
distinguish patterns that share content but differ in position
("dog bites man" vs "man bites dog") without bolting on softmax
attention.

Backward compatibility:

- Existing 162 MB brains load unchanged with phase coding disabled.
- The flag is persisted in `meta.pkl`. Once you enable it, subsequent
  runs without the flag will still load in phase-coded mode (the saved
  state wins).
- Phase data is stored as a parallel list of `(re_sum, im_sum, count)`
  tuples (one per pattern slot, or None for legacy/disabled patterns).
  ClusterMemory pickle format is otherwise unchanged.

Files touched: `neural_dot/neural_dot.py` (predict info), `convergence/
convergence.py` (`Cluster.mean_phase`), `formulas/formulas.py`
(`phase_aware_similarity`), `memory/cluster_memory.py` (per-pattern
phase tracking + phase-aware match), `memory/dot_memory.py` (per-dot
circular phase accumulator + fitness bonus, see below),
`pipeline/pipeline.py` (constructor flag, `commit_pattern` call, phase
forwarded into `dot_memory.record`, meta persistence).

### Phase-narrowness fitness bonus

When `--phase-coding` is on, `DotMemory.phase_bonus_weight` is set to
`0.30`. Each prediction's slice phase is forwarded into
`dot_memory.record(...)` and accumulated as a circular distribution
`(re_sum, im_sum, count)` per dot. The dot's phase concentration is the
resultant length `R/n ∈ [0, 1]` — `1.0` means the dot has only ever
fired from a single positional slice, `0.0` means uniformly spread or
no samples.

In `all_fitness_scores`, the F24 fitness is multiplied by
`(1 + phase_bonus_weight * concentration)`. So a fully phase-specialist
dot gets up to a 30% fitness boost at selection time, which propagates
into `DotEvolution.evolve()` (elites, tournament selection, mutation
parents). This is the selection pressure that turns phase coding from a
tracked metadata channel into an emergent positional binding mechanism:
dots that consistently fire from the same slot survive and reproduce.

Backward compat for the bonus: dots with no phase samples have
concentration `0.0`, so the multiplier is `1.0` — legacy dots are
unaffected. Brains saved without the flag load with
`phase_bonus_weight=0.0`; if you re-launch them with `--phase-coding`,
the pipeline restores it to `0.30`.

---

For long full-pipeline training runs, periodic pruning is built in:

```bash
python main.py train corpus.txt --evolve --prune-every 100
```

`--evolve` switches from vocab-only `fit_file` to `train_pass` (which
actually evolves the dot pool); `--prune-every N` runs an explicit
prune every N sentences and again right before each periodic save, so
the brain file size stays bounded no matter how long training goes.

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
