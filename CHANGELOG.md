# CHANGELOG

All notable changes to IECNN are documented here.

---

## [0.3.0] — 2026-04-19  Major expansion

### Architecture
- **Neural Dot expansion**: Added 6 specialization types (SEMANTIC, STRUCTURAL,
  CONTEXTUAL, RELATIONAL, TEMPORAL, GLOBAL), each with type-specific pooling strategies
  and bias presets
- **Multi-head prediction**: Each dot now generates N_HEADS=3 predictions using
  separate head projection matrices, tripling candidate diversity per dot
- **Memory-guided attention**: Dots receive recent centroid hints from DotMemory
  to bias their attention toward historically successful regions
- **AIM expansion**: Added 3 new inversion types — RELATIONAL (block correlation
  inversion), TEMPORAL (sequential reversal), COMPOSITIONAL (SVD decompose/permute/
  recompose) — total of 9 inversion types
- **Hierarchical convergence**: ConvergenceLayer now performs two-level clustering
  (micro-clusters then macro-clusters) for better precision
- **Consensus centroid**: Macro-clusters use confidence-weighted centroids rather
  than plain means
- **Cross-type agreement bonus** (Formula 13): Clusters where multiple dot types
  agree receive a score bonus

### New Formulas (F10–F15)
- **F10 Dot Specialization Score** — mean pairwise similarity of a dot's predictions
- **F11 Cluster Entropy** — normalized entropy of cluster score distribution
- **F12 Temporal Stability** — similarity of consecutive round centroids
- **F13 Cross-Type Agreement** — consensus metric across dot type centroids
- **F14 Adaptive Learning Rate** — eta(t) = eta_0 * (1 - 0.8 * dom²)
- **F15 Hierarchical Convergence Score** — macro-level cluster scoring

### New Components
- **`memory/` folder**: DotMemory + ClusterMemory tracking systems
  - Per-dot effectiveness tracking (success rate in winning clusters)
  - Specialization scoring (consistency of dot outputs)
  - Cluster timeline with temporal stability tracking
  - Cross-call pattern library for persistent learning
- **`evolution/` folder**: Genetic-style dot pool evolution
  - Tournament selection, mutation, crossover, random injection
  - Evolution kicks in after sufficient data is collected
  - EvolutionConfig for fine-grained tuning
- **`evaluation/` folder**: Quality metrics for IECNN runs
  - IECNNMetrics: cluster_entropy, temporal_stability, prediction_diversity,
    cross_type_agreement, agreement_rate, dot_specialization, convergence_quality
  - RunMetrics dataclass attached to every IECNNResult

### Improvements
- **Iteration Controller**: Added 2 new stopping conditions (temporal stability,
  score decline over N rounds); adaptive learning rate via Formula 14; rollback
  to best-scoring round if final round is worse
- **Pruning Layer**: Dynamic thresholds that adapt to dominance level; minimum
  survivor guarantee; near-miss tracking in Stage 3
- **Pipeline**: Integrates memory, evolution, evaluation, adaptive LR, rollback,
  memory hints for attention, and dot outcome recording
- **main.py CLI**: Added `compare`, `memory` commands; richer verbose output
  with per-round entropy, stability, LR columns

### C Extensions
- Expanded `formulas.c` with C implementations of F10–F15
- Updated `formulas.h` with all 15 function signatures

---

## [0.2.0] — 2026-04-19  C + Python restructure

### Added
- Separate top-level folder per component: `aim/`, `basemapping/`,
  `convergence/`, `formulas/`, `iteration/`, `neural_dot/`, `pipeline/`, `pruning/`
- C shared libraries for performance-critical modules:
  - `formulas/formulas_c.so` — cosine sim, agreement strength, pairwise matrix, attention
  - `basemapping/basemapping_c.so` — character composition, positional encoding
  - `aim/aim_c.so` — six original inversion functions
  - `convergence/convergence_c.so` — similarity matrix computation
- `build.sh` compile script (gcc, `-O2 -shared -fPIC -lm`)
- CLI-only entry point (`python main.py`)
- BaseMapping redesign: a-z/0-9 pre-seeded primitives; one row per token (never
  split words into character rows); `word`, `phrase`, `composed` token types

### Removed
- Flask web app (`server.py`)
- `iecnn/` sub-package (workspace root IS the IECNN project)

---

## [0.1.0] — 2026-04-18  Initial implementation

### Added
- Full Python implementation of the IECNN 10-layer pipeline
- 9 custom formulas (F1–F9)
- AIM layer with 6 inversion types
- 3-stage pruning
- 3 stopping conditions
- BiasVector (5-dim) with Formula 8 update
- Flask web demo
- Architecture notes (`iecnn_notes.md`) and formula reference (`formulas.md`)
