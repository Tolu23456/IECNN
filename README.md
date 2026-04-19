# IECNN — Iterative Emergent Convergent Neural Network

A novel neural architecture where independent "neural dots" produce competing predictions
that **converge by agreement** — no backpropagation, no fixed layers, no shared weights.

---

## Concept

Traditional networks pass signals through fixed layers and train all weights together
via gradient descent. IECNN is architecturally different:

| Traditional NN | IECNN |
|---|---|
| Neurons — tiny signal-passers | Neural Dots — complete independent predictors |
| Fixed layered graph | Dynamic pool of free agents |
| Global loss, shared gradients | Local agreement — emergent convergence |
| One prediction per forward pass | Hundreds of competing candidates |
| Backpropagation | Cluster dominance + bias vector update |

The key question: *what do independent agents agree on?*
The answer that emerges is the output.

---

## Architecture — 10 Layers

```
INPUT
  ↓
Layer 1:  Input              Raw text string
Layer 2:  BaseMapping        Token → one 128-dim row  (structured, composable, enriched)
Layer 3:  Dot Generation     64 dots × 6 types × 3 heads
Layer 4:  Prediction         Each dot independently predicts
Layer 5:  AIM                9 inversions run in parallel with originals
Layer 6:  Pruning (S1)       Soft filter: confidence, near-duplicates, AIM cap
Layer 7:  Convergence        Two-level hierarchical clustering (micro + macro)
Layer 8:  Pruning (S2+S3)    Cluster compression + dynamic hard selection
Layer 9:  Iteration Control  5 stopping conditions + adaptive LR + rollback
Layer 10: Output             Normalized 128-dim vector (stable and reproducible)
```

---

## Neural Dots

The fundamental unit. A neural dot is NOT a neuron:

```
NeuralDot {
  DotType:    SEMANTIC | STRUCTURAL | CONTEXTUAL | RELATIONAL | TEMPORAL | GLOBAL
  BiasVector: [attention, granularity, abstraction, inversion, temperature]
  Weights:    W (main transform), P_head × N_HEADS (per-head projections), Q (query basis)
  Predict:    slice → pool → abstract → N_HEADS predictions
}
```

**Six specialization types:**
- `SEMANTIC` — attends to embedding dims [0:64], fine granularity
- `STRUCTURAL` — attends to structural/positional features [64:128]
- `CONTEXTUAL` — global context view across the full sequence
- `RELATIONAL` — detects cross-token interaction patterns
- `TEMPORAL` — position-weighted sequential pooling
- `GLOBAL` — uniform broad overview of the entire input

**Multi-head prediction:** Each dot generates 3 predictions using different
head-specific projection matrices. With 64 dots × 3 heads = 192 base predictions,
plus AIM variants (up to 9 inversions each), the convergence layer sees hundreds of
competing candidates per round.

**Memory-guided attention:** DotMemory tracks which dots' predictions historically
land in the winning cluster. Their recent centroids are fed back as attention hints.

---

## Attention Inverse Mechanism (AIM) — 9 Types

AIM runs IN PARALLEL with original predictions. Both originals and inversions
enter Convergence together, competing fairly.

| # | Type | Description | Formula |
|---|---|---|---|
| 1 | Feature | Negate dims > mean absolute value | $p' = \text{flip}(\text{dominant}(p))$ |
| 2 | Context | Swap high/low activation groups | $p' = \text{swap}(\text{top half}, \text{bot half})$ |
| 3 | Spatial | Reverse dimension group ordering | $p' = \text{mirror}(\text{groups}(p))$ |
| 4 | Scale | Rescale quarters by [4, 2, 0.5, 0.25] | $p' = \text{scale}(p) \cdot \|p\|/\|p'\|$ |
| 5 | Abstraction | Flip patch↔object level via context projection | $p' = p - 2\,(p\cdot\hat{c})\hat{c}$ |
| 6 | Noise | Suppress dominant dims, add structured noise | $p' = \text{suppress}(p) + \epsilon$ |
| 7 | Relational | Invert dominant block correlations | $p' = p - 2\text{proj}_{\text{dom block}}$ |
| 8 | Temporal | Reverse sequential position in dims | $p'[t] = p[T-t]$ |
| 9 | Compositional | SVD → permute singular vectors → recompose | $p' = U' \Sigma V'^\top$ |

---

## Convergence — Two-Level Hierarchical Clustering

**Level 1 — Micro-clustering** (threshold 0.60):
Groups very similar predictions. Produces tight, high-confidence micro-clusters.

**Level 2 — Macro-clustering** (threshold 0.40):
Groups micro-cluster centroids. Scored via Formula 15 (Hierarchical Convergence Score).

**Centroid:** Confidence-weighted mean of all predictions in the macro-cluster.

**Cross-type bonus** (Formula 13): If predictions from multiple dot types agree on
a cluster, its score is boosted by up to 15%.

---

## Formulas (F1–F17)

| # | Name | Purpose |
|---|---|---|
| F1  | Similarity Score | $\alpha$·cos + $(1-\alpha)$·agreement |
| F2  | Convergence Score | Cluster cohesion × mean confidence |
| F3  | Attention | softmax(QKᵀ/√d)V — context grounding |
| F4  | AIM Transform | Attention over inverted prediction |
| F5  | Pruning Threshold | Dynamic keep/drop threshold |
| F6  | Prediction Confidence | tanh(‖p‖/√d) |
| F7  | Sampling Temperature | Adaptive diversity per entropy |
| F8  | Bias Vector Update | b += η(w − b) |
| F9  | Dominance Score | C(k*)/ΣC(k) — convergence signal |
| F10 | Dot Specialization | Mean pairwise S of a dot's predictions |
| F11 | Cluster Entropy | −Σ p_k log p_k (normalized) |
| F12 | Temporal Stability | S(centroid_t, centroid_{t−1}) |
| F13 | Cross-Type Agreement | Mean S across type centroids |
| F14 | Adaptive Learning Rate | η₀·(1 − 0.8·dom²) |
| F15 | Hierarchical Conv. Score | mean_C·(1 + γ·cross_sim) |
| F16 | Emergent Utility Gradient | U(t) = E[C_{t+1}] − C_t  (EUG) |
| F17 | Dot Reinforcement Pressure | R_d = λ1·C_d + λ2·S_d + λ3·U_n·(1+β·ΔU) − λ4·N_d |

Full derivations and parameter tables: see `formulas.md`.

---

## Stopping Conditions (5)

The iteration loop terminates when any condition is met:

1. **Budget** — round count reaches `T_max` (default 12)
2. **Dominance** — top cluster holds ≥ 70% of total score weight (F9)
3. **No utility gain** — EUG (F16) ≤ threshold after ≥ 3 rounds
4. **Temporal stability** — centroid barely moved: S(c_t, c_{t-1}) ≥ 0.99 (F12)
5. **Score decline** — top score fell for 3 consecutive rounds

**Rollback:** If the final round produced worse clusters than a previous round,
the iteration controller reverts to the best-scoring round.

---

## Selection Pressure (F16 + F17)

Selection pressure was the missing engine before v0.4. EUG and DRP together form a
closed feedback loop: evaluation drives selection, selection drives structural change.

### F16 — Emergent Utility Gradient (EUG)

```
U(t) = E[C_{t+1}(p)] - C_t(p)   estimated via recency-weighted score delta
```

- Measures whether the system is still improving between rounds
- Replaces the broken cluster-ID novelty_gain check (cluster IDs reset each round,
  making the old check always return 0)
- Drives the stopping condition and the instability injection trigger

### F17 — Dot Reinforcement Pressure (DRP)

```
R_d = λ1·effectiveness + λ2·specialization + λ3·U_norm·(1+β·ΔU_norm) − λ4·failure_rate
```

Applied within each call (after every round):

| Step | What happens |
|---|---|
| Floor pressure | Dots with R_d < 0.05 have success_count × 0.90 |
| Amplification | Scores stretched: sign(R)·\|R\|^1.5 — extremes move further apart |
| Competition decay | Bottom 30% by DRP score × 0.90 |
| Hard selection | Bottom 40% by DRP score × 0.50 (structural change, not just nudge) |
| Inline mutation | Dots with effectiveness < 0.10 get Gaussian weight + bias noise; 20% chance of type switch |
| Diversity constraint | If Simpson diversity < 0.60, underrepresented types get temperature boost |

**Adaptive exploration:** When \|U\| < 0.01 (EUG stagnant), the refined vector gets
Gaussian noise and context_entropy is raised by 0.30 for the next round.

---

## Memory & Evolution

### DotMemory
- Tracks per-dot effectiveness (fraction of predictions in the winner)
- Rolling window of recent predictions → recent centroid for attention hints
- Specialization score: how consistent each dot is
- DRP scores and hard selection applied within each call

### ClusterMemory
- Records cluster snapshots per round (centroids, scores, stop reason)
- Temporal stability computation (F12) across rounds
- Cross-call pattern library: stores stable patterns that recur across inputs

### DotEvolution (between calls)
- **Tournament selection** of elite dots (top 40%)
- **Mutation**: Gaussian noise on bias vector + weight matrix
- **Crossover**: average bias, uniform row-wise mix of W matrices
- **Random injection** (10% of pool) maintains diversity
- Evolution disabled until enough data exists (default 3 full calls)

---

## BaseMapping

One 128-dimensional row per token. Words are **never** split into character rows.

```
Feature vector layout (128 dims):
  [0  : 96 ]  Base embedding      (stable hash + char composition + cooccurrence enrichment)
  [96 : 104]  Position encoding   (8 sinusoidal sin/cos pairs)
  [104: 108]  Frequency features  (relative, log, tanh, sigmoid)
  [108: 124]  Modifier flags      (type, structural, morphological suffix markers)
  [124: 128]  Context summary     (semantic cohesion, balance, phrase/vocab density)
```

**Token types:**
- `primitive` — a–z, 0–9, punctuation (pre-seeded, always available)
- `word` — corpus vocabulary (hash-stable + cooccurrence-enriched embedding)
- `phrase` — bigram/trigram detected in corpus (single-row embedding)
- `composed` — unknown word (unigram + bigram character-primitive composition)

**v0.7.0 improvements:**

| Improvement | What it does |
|---|---|
| Cooccurrence enrichment | Words that appear together in the corpus get embeddings smoothed toward each other, providing distributional semantic grounding |
| Character bigrams | Composed embeddings use consecutive character pairs (in addition to unigrams) to capture subword patterns like prefixes and suffixes |
| Morphological flags | Flag dims 14–15 detect verb (-ing, -ed, -ize), noun (-tion, -ness, -ity), adjective (-ous, -ful, -able), and adverb (-ly) suffixes — POS-style signal without a tagger |
| Semantic context | Context summary dim 0 is now the mean cosine similarity between the token and its neighbors (not just a token density count) |
| IDF-weighted pooling | `basemap.pool("idf")` weights rare tokens more heavily, suppressing stop words without a stop-word list |

---

## Project Structure

```
iecnn/
├── main.py                 CLI entry point
├── build.sh                Compile all C extensions
├── README.md               This file
├── formulas.md             Full formula derivations (F1–F17)
├── CHANGELOG.md            Version history
├── plans.md                Project TODOs and roadmap
├── replit.md               Setup and architecture notes
│
├── neural_dot/             Neural Dot (6 types, 3 heads, memory-guided)
├── basemapping/            Token → BaseMap (primitives, cooccurrence, bigrams)
├── aim/                    AIM (9 inversions, attention refinement)
├── convergence/            Two-level hierarchical clustering
├── pruning/                3-stage candidate & cluster filtering
├── iteration/              Iteration control (5 stops, rollback, adaptive LR, EUG)
├── pipeline/               Full IECNN pipeline (all layers + selection pressure)
│
├── memory/                 Per-dot and cluster memory systems
│   ├── dot_memory.py       Effectiveness tracking, DRP, hard selection
│   └── cluster_memory.py   Round timeline, pattern library
│
├── evolution/              Genetic dot pool evolution
│   └── dot_evolution.py    Tournament selection, mutation, crossover, inline mutate
│
├── evaluation/             Quality metrics
│   └── metrics.py          RunMetrics, IECNNMetrics (F10–F15)
│
└── formulas/               Mathematical engine
    ├── formulas.py         Python + ctypes bindings (F1–F17 + amplify_pressure)
    ├── formulas.c          C implementations (F1–F15)
    └── formulas.h          C header with all signatures
```

---

## Running

```bash
# Full interactive demo (recommended first run)
python main.py

# Compile C extensions manually
python main.py build

# Encode text to vector
python main.py encode "neural convergence"

# Similarity between two texts
python main.py sim "neural networks learn" "deep learning trains"

# Pairwise similarity table
python main.py compare "text A" "text B" "text C"

# Memory and evolution state
python main.py memory
```

---

## Design Principles

1. **No shared weights** — dots are fully independent; no communication during prediction
2. **One row per token** — the basemap never splits words into character rows
3. **Emergent agreement** — the output is what independent agents converge on
4. **Primitive foundation** — a–z/0–9 are always available as fallback bases
5. **C acceleration** — all hot paths are compiled for speed; pure Python fallback always available
6. **Max 10k lines per file** — each component stays in its own folder
7. **Numpy only** — no pytorch, tensorflow, or external ML frameworks
8. **Selection without gradients** — dots evolve by effectiveness, not by loss minimization
