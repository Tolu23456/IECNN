# IECNN — Project Plans & TODOs

Tracked ideas, fixes, and features. Mark items done with [x] as work completes.

---

## Done

- [x] Migrate to Replit (Python 3.11, C extensions compiled via build.sh)
- [x] Pillow installed (required by `decoding/decoder.py`; was crashing `generate`)
- [x] Silent interactive REPL: `python main.py` shows only `> ` prompt; original showcase moved to `python main.py demo`
- [x] Persistent brain across sessions: `save_brain()` / `load_brain()` on IECNN
      - `global_brain.pkl` (vocab) + companion `.dots.pkl` / `.dotmem.pkl` /
        `.clustmem.pkl` / `.evo.pkl` / `.meta.pkl`
      - state_dict / load_state on DotMemory, ClusterMemory, DotEvolution
      - auto-load on `IECNN.__init__`, auto-save after every interactive command
- [x] `train_pass()` method on IECNN — real learning sweep over a sentence list
      (records dot outcomes, runs evolution per sentence, updates cluster memory)
- [x] `fit_file()` now runs vocab fit AND a learning pass (not just vocab)
- [x] CLI `train <file> [--limit N]` subcommand
- [x] Downloaded WikiText-2 corpus (`corpus_10k.txt`, 10 000 lines)
- [x] First real learned brain saved: 30 sentences trained,
      gen=30, active_dots=1156, max_eff=0.0472, cluster_memory ≈ 62 KB
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
- [x] FEATURE_DIM upgrade: 128 → 256 (EMBED=224, POS=8, FREQ=4, FLAGS=16, CTX=4)
- [x] 8 dot types: added LOGIC and MORPH specializations
- [x] 4 prediction heads per dot (up from 3), 128 dots (up from 64)
- [x] F18 Cross-Modal Binding — measures latent alignment across modalities
- [x] F19 Semantic Drift — measures representational shift between modality encodings
- [x] F20 Vocabulary Coverage — fraction of input tokens with known corpus bases
- [x] Multimodal image transform: lossless 8×8 patches + stats (numpy-only, no cv2)
- [x] Multimodal audio transform: pure numpy FFT (no librosa)
- [x] Multimodal video transform: PIL ImageSequence (no cv2)
- [x] fit_file(): streaming large-dataset training (one sentence per line)
- [x] generate(): prompt → latent → fast greedy text decoding
- [x] IECNNDecoder: latent → text (two-stage cheap greedy), image, audio, video
- [x] train / generate CLI commands (python main.py train <file>, python main.py generate "prompt")
- [x] Convergence threshold recalibration for 256-dim/128-dot regime (micro 0.25, macro 0.15, dom 0.35)
- [x] generate interactive command inside demo loop

---

## [V6 SOTA Completed]
- [x] Consolidated Pipeline-C with OpenMP
- [x] Universal Multilingual BaseMapping (CJK/Arabic)
- [x] Deterministic SHA-256 Embeddings
- [x] High-Fidelity Multi-modal Decoder kernels in C
- [x] LAZY Dot Type and Logic/Action enhancements
- [x] AAF Memory Safety and fixed Video reshape

## Active / Near-term

### Decoder quality
- [ ] Beam search in Stage 2 of text decoder (width 3–5): keep top-N partial sequences instead of greedy argmax to reduce first-token errors
- [ ] Add common function words to `_base_vocab` as guaranteed entries so short frequent words are always decodable even with small corpora
- [x] Expose `model.save()` / `model.load()` so trained vocab persists to disk between CLI sessions
- [ ] Order-aware decoding: current text decoder picks tokens by embedding similarity only, with no notion of word order, so generations come out as thematic word salad. Consider adding a simple bigram/trigram statistic from the training corpus to bias next-token choice.

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
- [x] Persistent dot state across CLI sessions (pickle DotMemory + evolution gen)
- [ ] Make dot_id truly stable across evolution rounds (not positional index)
      so effectiveness history survives crossover/replacement
- [ ] ClusterMemory pattern matching: use stored patterns to seed round 0
      centroid hints instead of zero-initialised vectors

### Training scalability (BLOCKER for full 10k-line training)
- [ ] **Prune `dot_memory` of dots with zero recorded outcomes.** Currently each
      generation adds ~40 new dot IDs; after ~50 sentences the tracked pool
      reaches ~2000 dots and the process gets OOM-killed mid-training.
      Practical batch is ~30 sentences per `train` invocation right now.
- [ ] Cap `dot_pool` save size — `global_brain.pkl.dots.pkl` is currently
      ~336 MB after 30 sentences because every active dot serializes its
      4× (256×256) head_proj matrices. Either persist only the live 64 dots
      or quantize the head_projs.
- [ ] Resume metadata: track cumulative trained-sentence count in `.meta.pkl`
      so repeated `train` calls can skip already-seen lines.

---

## Medium-term

### BaseMapping
- [ ] Subword decomposition for morphologically rich words (e.g. "unbelievably"
      → un + believ + able + ly) using a simple morpheme split heuristic
- [ ] C acceleration for cooccurrence smoothing pass (currently Python loop)
- [ ] Attention-based pooling in BaseMap.pool() — weight tokens by their
      query-key similarity to the sentence centroid

### Pipeline
- [ ] Multi-pass refinement: run a second lightweight convergence pass on the
      top 3 cluster centroids after the main loop exits
- [ ] Output ensemble: if multiple calls on similar inputs produce close outputs,
      average them for a more stable embedding

### Multimodal
- [ ] Image generation: improve `_decode_image` with pattern-based rendering using
      more latent dims (currently only dims 0–4 are used)
- [ ] Audio generation: use latent dims to drive harmonic envelope and beat frequency
      instead of two fixed sinusoids
- [ ] Multimodal training: `fit_file` for image/audio datasets (one path per line)

---

## Research / Long-term

### Evaluation
- [ ] Add a benchmark suite: cosine similarity correlation against human
      similarity judgements (STS-B subset, license-free)
- [ ] Track quality score distribution across 100+ inputs; plot histogram
- [ ] Compare IECNN vs random-projection baseline on retrieval@k
- [ ] Use F20 (vocab coverage) as a proxy for OOV rate; plot vs training size

### Architecture experiments
- [ ] Replace AIM's 9 fixed inversions with learned inversion types driven
      by dot type — semantic dots propose different inversions than temporal
- [ ] Hierarchical input: allow multi-sentence input where each sentence is
      a separate BaseMap, and a top-level convergence layer fuses them
- [ ] Use F18/F19 as a training signal: if cross-modal binding is low, increase
      modality-specific dot specialization

### Publication / sharing
- [ ] Write a short technical paper describing the IECNN architecture
- [ ] Publish benchmark results vs. random-projection and simple TF-IDF baseline
- [ ] Add an interactive web demo (separate from CLI) using the encode / sim API
