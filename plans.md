# IECNN — Project Plans & TODOs

Tracked ideas, fixes, and features. Mark items done with [x] as work completes.

---

## Done

- [x] Migrate to Replit (Python 3.11, C extensions compiled via build.sh)
- [x] F16 Emergent Utility Gradient — replaces broken cluster-ID novelty_gain check
- [x] F17 Dot Reinforcement Pressure (DRP) — within-call selection pressure
- [x] Instability injection — Gaussian noise on refined vector when EUG stagnant
- [x] Pressure amplification — sign(R)|R|^1.5 nonlinear stretch on DRP scores
- [x] Hard selection — bottom 40% of dots penalised 50% per round
- [x] Inline dot mutation — weak dots transform structurally within a call
- [x] Diversity constraint — Simpson index check; boost underrepresented dot types
- [x] Adaptive exploration trigger — raise context_entropy when EUG flat
- [x] BaseMapping v2: cooccurrence-based semantic enrichment
- [x] BaseMapping v2: character bigram composition (subword structure)
- [x] BaseMapping v2: morphological suffix flags (verb/noun/adj detection)
- [x] BaseMapping v2: semantic context summary (embedding cosine to neighbors)
- [x] BaseMapping v2: IDF-weighted pooling option

---

## Active / Near-term

### Stopping condition tuning
- [ ] Dynamic EUG threshold: scale threshold with round number
      (early rounds should tolerate lower EUG before stopping)
- [ ] Hard selection warmup: don't apply in round 0; ramp penalty from 0→0.5
      over the first 3 rounds of a call to let dots accumulate signal first

### DRP parameter learning
- [ ] Auto-tune λ1–λ4 and β in F17 based on EUG history across calls
      (if EUG consistently declines, increase λ4 failure penalty)
- [ ] Per-dot-type λ weights: semantic dots may need different pressure than
      temporal dots

### Memory system
- [ ] Persistent dot state across CLI sessions (pickle DotMemory + evolution gen)
- [ ] Make dot_id truly stable across evolution rounds (not positional index)
      so effectiveness history survives crossover/replacement
- [ ] ClusterMemory pattern matching: use stored patterns to seed round 0
      centroid hints instead of zero-initialised vectors

---

## Medium-term

### BaseMapping
- [ ] Larger corpus: expose a --corpus flag so users can feed a text file
      for richer cooccurrence and vocabulary discovery
- [ ] Subword decomposition for morphologically rich words (e.g. "unbelievably"
      → un + believ + able + ly) using a simple morpheme split heuristic
- [ ] C acceleration for cooccurrence smoothing pass (currently Python loop)
- [ ] Attention-based pooling in BaseMap.pool() — weight tokens by their
      query-key similarity to the sentence centroid

### Dot architecture
- [ ] Expand dot pool from 64 → 128 for longer / denser inputs
- [ ] Add a 7th dot type: MORPHOLOGICAL — specifically attends to the
      morphological flag dims (108:124) to specialise on word structure
- [ ] Head count from 3 → 4 (adds ~33% candidate diversity at low cost)

### Pipeline
- [ ] Multi-pass refinement: run a second lightweight convergence pass on the
      top 3 cluster centroids after the main loop exits
- [ ] Output ensemble: if multiple calls on similar inputs produce close outputs,
      average them for a more stable embedding

---

## Research / Long-term

### Evaluation
- [ ] Add a benchmark suite: cosine similarity correlation against human
      similarity judgements (STS-B subset, license-free)
- [ ] Track quality score distribution across 100+ inputs; plot histogram
- [ ] Compare IECNN vs random-projection baseline on retrieval@k

### Architecture experiments
- [ ] FEATURE_DIM = 256 variant: double the embedding space, test whether
      richer input representation improves convergence
- [ ] Replace AIM's 9 fixed inversions with learned inversion types driven
      by dot type — semantic dots propose different inversions than temporal
- [ ] Hierarchical input: allow multi-sentence input where each sentence is
      a separate BaseMap, and a top-level convergence layer fuses them

### Publication / sharing
- [ ] Write a short technical paper describing the IECNN architecture
- [ ] Publish benchmark results vs. random-projection and simple TF-IDF baseline
- [ ] Add an interactive web demo (separate from CLI) using the encode / sim API
