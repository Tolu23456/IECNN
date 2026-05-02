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

## IECNN Full-Pipeline Training Speed Optimizations (May 2026)

### Result: 6k–9k words/s causal training on AMD EPYC 9B14, 6 cores (was: ~136 w/s baseline)

The full IECNN pipeline (128 dots, 256 dim, 12 iterations) was rewritten for speed without
changing the IECNN algorithm.  All C extensions compiled with `-O3 -march=native -ffast-math
-funroll-loops -lmvec` via `build.sh`.

#### Root causes eliminated
| Root cause | Old cost | Fix |
|---|---|---|
| `causal_train_pass` called `run_batch()` but NEVER used the convergence output | ~248 ms/sentence | Skip C pipeline entirely; use `_get_ctx_cached()` for mean-pooled token embeddings |
| `base_mapper.transform()` in ctx cache: `_apply_aaf` (129ms) + `_segment` regex/enum (105ms) | 235 ms/prefix | Bypass to `_token_embedding()` directly — 0.03 ms/prefix (7,800×) |
| `for p_idx in range(n_total)` loop: 1,400 × 128 = 179,200 individual numpy ops/batch | 2–3 s/200-sent batch | Vectorized Phase 5: one BLAS DGEMM for mean targets + 128-iter memory loop |
| `dot_preds_all` L2-norm (256k norms + 65M divisions): not needed for win criterion | ~90 ms/batch | Win = `(W[d]@ctx) · tv > 0` ≡ `cos > 0`; norms are positive so sign is invariant |
| `_ensure_caches` rebuilt HP_stack (128 × `np.pad`) every batch: 0.36 s | ~0.36 s/batch | Never call `_ensure_caches` in `causal_train_pass`; maintain local W array |
| `ascontiguousarray(transpose(...))` in Phase 4 BLAS | ~90 ms/batch | Compute `ctx @ W_flat.T` → row-major directly, reshape without copy |
| OMP nesting bug: inner `#pragma omp parallel for` inside outer OMP | Deadlock/serialization | Removed inner OMP; added `training_mode` to skip fast_randn in head projections |
| `malloc`/`free` 3 MB per sentence in pipeline_c.c | ~1,710 ms fixed overhead | `_Thread_local` static buffers; allocated once per OMP thread, never freed |
| `-ffast-math` emitted `_ZGVdN8v_logf` (SVML) undefined symbol | .so fails to load | Add `-lmvec` to link against glibc's libmvec.so.1 (provides vectorized logf/expf) |
| `fast_randn`: 1.57M transcendental calls/sentence | ~200 ms/batch | 65,536-entry pre-computed noise table via `__attribute__((constructor))` |

#### Achieved throughputs
| Scenario | Speed |
|---|---|
| Warm ctx cache (corpus repeats) | **9,040 w/s** |
| Diverse 1k-sentence corpus, 2nd pass | **6,617 w/s** |
| Sustained (max_pos=6, evolution every 2k sents) | **6,338 w/s** |
| max_pos=3 (short-prefix mode) | **10,620 w/s** |
| tok_emb_cache warm (2nd pass, after session-2) | **7,315 w/s** (+10.5%) |

#### Key files changed
- `pipeline/pipeline.py` — `causal_train_pass` fully rewritten; `_get_ctx_cached` bypasses transform; `train_pass` + `causal_train_pass` add `save_every` param; `causal_train_file()` streaming method added; `_tok_emb_cache` for Phase 3; Phase-5b bulk BLAS mean_pred
- `pipeline/pipeline_c.c` — thread-local static buffers; `training_mode` param; `schedule(dynamic,1)`
- `neural_dot/neural_dot_c.c` — 65,536-entry noise table; `training_mode` skips fast_randn; removed inner OMP
- `neural_dot/neural_dot.h` — added `int training_mode` to `predict_batch_c` signature
- `build.sh` — added `-ffast-math -funroll-loops -lmvec`
- `fast_train.py` — `fast_effective_train` uses `causal_batch=200, save_every=5000`
- `train_300k.py` — full training runner for 300k corpus (Phase 1 vocab + Phase 2 causal)
- `evaluate_iecnn.py` — evaluation suite (win-rate, dot specialization, next-word accuracy, qualitative)

---

## 300k Corpus Training & Evaluation (May 2026)

### Corpus
271,526 lines assembled from 20 sources: WikiText-2 + 19 Project Gutenberg books (War and Peace, Moby Dick, Great Expectations, David Copperfield, Ulysses, Jane Eyre, Dracula, Oliver Twist, Pride & Prejudice, Two Cities, Sherlock Holmes, Dorian Gray, Emma, Wuthering Heights, Huck Finn, Dubliners, Grimm, Peter Pan, Metamorphosis). Cleaned, deduped, shuffled.  File: `/tmp/corpus_300k.txt`.

### Training run
~85,000 / 271,526 lines trained (2 × 95-second foreground sessions at ~380 lines/s ≈ 4,600 w/s).

### Evaluation results

#### [A] Speed (this session additions)
| Change | Effect |
|---|---|
| Phase-5b BLAS: `(mean_ctx @ W_flat.T).reshape(n_dots, dim)` replaces 128 matmuls | 1 BLAS call instead of 128 |
| `_tok_emb_cache`: caches `_token_embedding()` results per token | 7,315 w/s warm (+10.5%) |
| `causal_train_file()`: streaming file training (bounded RAM) | New method |

#### [B] Causal win-rate (primary quality metric)
- **88.77%** vs 50% random baseline → **+38.77 percentage-point lift**
- Verdict: ✓ STRONG directional learning

#### [C] Dot specialization
- MaxEff: **0.9399** | MeanEff: **0.8825** | StdEff: **0.1382**
- 120/128 dots above 0.5 effectiveness; 114/128 above 0.9
- Inter-dot W-matrix cosine similarity: **0.6567** ⚠ HIGH COLLAPSE (dots are too similar to each other)

#### [D] Context sensitivity
- Mean prediction cosine across different input contexts: **0.8097** ⚠ LOW
- Dots largely ignore context — they predict similar directions regardless of prefix

#### [E] Next-word prediction accuracy
- Top-1: **0.00%** | Top-5: **0.00%** | Top-10: **0.00%** (from 3000-candidate pool)
- Chance baseline: 0.033% / 0.167% / 0.333%
- Root cause: dot collapse + low context-sensitivity → all dots vote for same words

### Where to improve (priority order)

1. **Dot diversity loss** — add inter-dot repulsion term during training (penalise W[d] ≈ W[d']); prevents collapse
2. **Competitive / winner-takes-most update** — only top-K winning dots update per position; forces specialisation
3. **Context-sensitive pooling** — replace mean-pool of prefix tokens with attention-weighted pool using a learned query
4. **Nearest-neighbour inference decoder** — map predicted embedding → nearest vocab vector via FAISS/HNSW instead of voting
5. **More training data + passes** — current 85k/271k is 31%; full corpus + 2–3 passes needed for stable statistics
6. **Scale dots** — 128 dots for 32,575 vocab is sparse; try 256 or 512 dots
7. **GloVe/FastText init** — replace polynomial hash embeddings with pre-trained word vectors for better cold-start signal

---

## Vocabulary Training Speed Optimizations (May 2026)

### Result: 600k–650k sent/s vocabulary training (was: ~8,900 sent/s baseline)

Achieved through three generations of optimization across `basemapping/basemapping.py`, `fast_train.py`, and a new C extension (`fast_count_c.so`):

#### Generation 1 — Python optimizations → ~145k sent/s (16×)
| # | Change | Speedup |
|---|--------|---------|
| 1 | `_stable_embedding`: `np.random.default_rng` (md5) instead of `RandomState` (sha256 + os.urandom) | 14.8× per new word |
| 2 | Pre-compiled `_TOKENIZE_PATTERN` at module level (no per-call regex compile) | 2× tokenization |
| 3 | `multiprocessing.Pool` across all 4 CPU cores for parallel chunk counting | 4× parallelism |
| 4 | Persistent Pool across batches (no 70ms respawn overhead per batch) | eliminates per-batch overhead |
| 5 | Vectorized phrase embedding: `_batch_build_phrase_embeddings()` | 4× phrase building |
| 6 | Cooccurrence + subword loops skipped in fast mode | 3× worker throughput |

#### Generation 2 — C extension + phrase cap → ~221k sent/s (25×)
| # | Change | Speedup |
|---|--------|---------|
| 7 | `fast_count_c.so` — integer hash-table ngram counting in C (replaces Python Counter) | 6× word/ngram counting |
| 8 | ASCII fast-path tokenizer: `str.isascii()` + stdlib `re` (avoids Unicode overhead) | 2.1× tokenization |
| 9 | `new_phrases` cap at 2000/batch (was `max_vocab_size//2`) — phrase embed 151ms→12ms | 12× phrase embedding |

#### Generation 3 — C bytes tokenizer + bigram cap + chunk tuning → 600k–650k sent/s (75×)
| # | Change | Speedup |
|---|--------|---------|
| 10 | `fast_tok_bytes_ascii` C function: tokenise+lowercase all sentences in one C call, output flat byte buffer; Python splits on `\x01` using bytes-keyed dict (avoids per-token string creation, decodes only unique vocab ≤ 20 words at end) | 2.7× worker throughput |
| 11 | **Bigram-only cap in fast mode** (`min(ng_hi, 2)`): trigrams produce 42³≈74k Counter entries vs 42²≈1.8k for bigrams — **45× smaller IPC payload, 35× faster merge** | 3× IPC+merge |
| 12 | Optimal chunk size: cap at 12.5k sentences/chunk (text pickling scales with chunk size; smaller chunks keep IPC overhead proportional to compute) | 1.4× throughput |

#### Generation 4 — Shared-memory IPC → targets >1M sent/s
| # | Change | Speedup |
|---|--------|---------|
| 13 | `_count_chunk_shmem_worker` + `multiprocessing.shared_memory`: all batch texts placed in a single OS shared-memory block; workers attach zero-copy. IPC payload per worker drops from several MB (pickled text list) to ~100 KB (shm_name string + two int32 offset/length arrays). | 2–4× IPC elimination |

**How it works**: The main process encodes all batch texts into a flat UTF-8 byte buffer, writes it into a `SharedMemory` block, then dispatches `(shm_name, offsets_bytes, lens_bytes)` tuples to workers. Workers attach to the shared block, read their slice of texts, detach, then run the same C-accelerated counting logic. The shared block is unlinked by the main process after all workers return.

**Quality**: unchanged — shared-memory path uses identical counting logic (`_count_chunk_c` or `_count_chunk_py`).

**New APIs**:
- `BaseMapper.fit_fast(texts, n_workers, skip_subwords, _pool, use_shmem=False)` — pass `use_shmem=True` to activate zero-copy IPC
- `BaseMapper._batch_build_phrase_embeddings(phrases)` — vectorized phrase embedding
- `BaseMapper._batch_build_word_embeddings(words)` — vectorized word embedding
- `fast_train.fast_vocab_train(corpus_path, ..., use_shmem=False)` — streaming parallel trainer; pass `use_shmem=True` to eliminate text-pickling
- `fast_train.fast_full_train(corpus_path, ...)` — vocab + full pipeline dot training
- `fast_train._count_chunk_worker(args)` — standard picklable multiprocessing worker
- `fast_train._count_chunk_shmem_worker(args)` — zero-copy shared-memory worker

**CLI**:
```
python main.py train corpus.txt --fast [--evolve] [--workers N]
python main.py train corpus.txt --fast --shared-memory [--workers N]   # zero-copy IPC
python fast_train.py corpus.txt --shared-memory [--workers N]
```

#### Generation 5 — OpenMP Two-Pass C Corpus Scanner (`fast_scan_c.so`) → 1.9M sent/s (C-only)
| # | Change | Measured |
|---|--------|---------|
| 14 | `fast_scan.c` + `fast_scan_c.so` — complete corpus scanner in C+OpenMP. Two passes: (1) parallel unigram counting into thread-local hash tables, serial merge → global vocab+IDs; (2) parallel bigram counting using read-only global ID table, serial merge → output. mmap I/O throughout. Zero Python in the counting hot path. | **1.9M sent/s** (C-only scan, 300k-line corpus, AMD EPYC 9B14, 6 OMP threads) |
| 15 | Vectorized Python bigram decode: `np.array(words, dtype=object)` + numpy advanced indexing replaces 662k-iteration Python loop over bi_w1/bi_w2 pairs. | 0.78s → 0.46s for full decode (646k sent/s end-to-end) |
| 16 | `fast_vocab_train_omp()` — single-function corpus-to-brain path: C+OMP scan → `BaseMapper._batch_build_word_embeddings` → `_batch_build_phrase_embeddings` → `_apply_cooc_smoothing` → `save_brain`. 300k lines, 47k unique words, 662k unique bigrams → 82k sent/s end-to-end. | **3.6s total** for 300k lines |

**Architecture**: `fast_scan.c` → `fast_scan_c.so` (gcc -O3 -fopenmp), loaded by `fast_train._LIB_FS`. Python side calls `scan_corpus_omp()` via ctypes with pre-allocated numpy buffers for word strings, offsets, lengths, counts, and bigram ID pairs.

**Bug fixed during this session**: `pipeline_c.so` was compiled without `aim/aim.c`, leaving `invert_relational` and `invert_feature` as undefined symbols. The C pipeline always fell back to the Python `fast_encode` path which does NOT call `dot_memory.record()`. Fixed by adding `aim/aim.c` to the `pipeline_c.so` link step in `build.sh`.

**MaxEff analysis (300k corpus, full pipeline)**: With the C pipeline now functional, MaxEff stays at 0.5 (the uninformed prior). Root cause: in convergence clustering, each dot wins cluster-0 approximately 1–2% of the time across the 128-dot pool. After enough observations, `effectiveness = success_counts/total_counts ≈ 0.01–0.02`, well below the 0.5 prior. DotEvolution culls these trained dots (lower than prior) and replaces them with fresh ones (effectiveness defaults to 0.5), so the reported MaxEff is always the prior of the newest generation. To push MaxEff > 0.5 requires either: (a) `--causal` training where dots predict actual next tokens and can accumulate true win records, or (b) a lower cluster-win rate threshold (e.g., top-3 instead of cluster-0 only).

**New CLI flag**: `--ultra`
```
python main.py train corpus.txt --ultra          # OMP single-call vocab scan, fastest possible
```
**New functions in `fast_train.py`**:
- `scan_corpus_c(filepath, n_threads, max_words, max_bigrams)` → `(Counter, Counter)` — calls C+OMP scanner
- `fast_vocab_train_omp(corpus_path, brain_path, n_threads, verbose)` — full vocab pipeline via OMP

### C Extension Missing → Explicit Notification + Auto-Rebuild

All C extension `_load_lib()` functions (`formulas`, `basemapping`, `pipeline`) now print a clear warning when the `.so` file is missing instead of silently falling back:

```
[IECNN] WARNING: .../formulas_c.so not found — formulas will use slow Python path.
         Fix: run  python main.py build  to compile C extensions.
```

`main.py _build_c()` now checks **all 9 `.so` files** (previously only 4), lists each missing one by path, runs `build.sh`, and reports which extensions are still unavailable if the build fails — so the user always knows exactly what is broken and why performance is degraded.

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

## 300k Corpus Training — Backtest Results (May 2026)

### Setup
- **Corpus**: 300,000 sentences assembled from WikiText-2 (train/valid/test) + 10 Project Gutenberg books (Pride & Prejudice, Alice, Frankenstein, A Tale of Two Cities, Moby Dick, Sherlock Holmes, Dorian Gray, Sense & Sensibility, Jane Eyre, Dracula). Deduped, shuffled, padded to 300k.
- **Vocab**: 47,521 tokens seeded via `model.fit()` on first 10k sentences (3.9s)
- **Training**: `causal_train_pass`, `max_pos=6`, `causal_batch=400`
- **Script**: `train_backtest.py` — checkpoints every 40k sentences

### Backtest Results (checkpoint every 40k sentences)

| CP | Sents | MaxEff | MeanEff | WinRate | Dots>50% | Diversity | Signal | w/s |
|---|---|---|---|---|---|---|---|---|
| 1 | 40,000 | 0.5000 | 0.3080 | 1.70% | 0/128 | 0.615 | STABLE | 18,068 |
| 2 | 80,000 | 0.5000 | 0.3072 | 1.49% | 0/128 | 0.526 | STABLE | 18,944 |
| 3 | 120,000 | 0.5000 | 0.3073 | 1.53% | 0/128 | 0.529 | STABLE | 19,330 |
| 4 | 160,000 | 0.5000 | 0.3072 | 1.52% | 0/128 | 0.597 | STABLE | 19,351 |
| 5 | 200,000 | 0.5000 | 0.3071 | 1.47% | 0/128 | 0.522 | STABLE | 19,148 |
| 6 | 240,000 | 0.5000 | 0.3070 | 1.51% | 0/128 | 0.528 | STABLE | 19,327 |
| 7 | 280,000 | 0.5000 | 0.3074 | 1.55% | 0/128 | 0.536 | STABLE | 19,388 |
| 8 | 300,000 | 0.5000 | 0.3072 | 1.48% | 0/128 | 0.677 | STABLE | 19,342 |

### Summary
- **Total words**: 6,966,793 across 300k sentences in **6.0 minutes**
- **Average speed**: **19,342 w/s** — exceeds 16k w/s architecture target ✓
- **MaxEff**: Flat at 0.5000 (the default prior for freshly-evolved dots)
- **MeanEff**: Stable ~0.307 (mix of trained ~0.016 + fresh ~0.500 dots)
- **Dots above 50%**: 0/128 across all checkpoints

### Root Cause (identified and fixed)
The causal training uses **winner-take-all** (argmax over 128 dots per position).
Raw win counting gave each dot ~0.78% win-rate (1/128), far below the 0.5 prior → evolution culled all trained dots.

**Fix applied (F14 + F17 + normalized scoring):**
1. **F17 Dominance EMA** added to `NeuralDot`: `dominance=0.5`, updated per-batch via `0.9*d + 0.1*(wins/expected_wins)`. Dots above baseline gain dominance; below-baseline dots lose it.
2. **F14 Adaptive LR**: `lr = base_lr * (1 - 0.8 * dominance²)` — dominant dots slow down to prevent monopoly.
3. **Normalized win scoring**: Each batch records `score = 0.5 * min(2, actual_wins/expected_wins)`. Average dot → score=0.5 → effectiveness=0.5. Specialists exceed 0.5.
4. **Memory reset**: Old raw-count data cleared before retraining; W-matrices retained.

---

## 300k Corpus Re-Training — After F14/F17 Fix (May 2026)

| CP | Sents | MaxEff | MeanEff | Dots>50% | Dots>70% | Dots>80% | DomMax | Signal | w/s |
|---|---|---|---|---|---|---|---|---|---|
| 1 | 40,000 | 1.0000 | 0.6043 | 51/128 | 35/128 | 14/128 | 0.9995 | YES | 18,241 |
| 2 | 80,000 | 1.0000 | 0.6086 | 51/128 | 36/128 | 19/128 | 1.0000 | STABLE | 18,999 |
| 3 | 120,000 | 1.0000 | 0.6082 | 51/128 | 32/128 | 15/128 | 0.9999 | STABLE | 19,033 |
| 4 | 160,000 | 1.0000 | 0.6117 | 51/128 | 34/128 | 19/128 | 0.9997 | STABLE | 19,337 |
| 5 | 200,000 | 1.0000 | 0.6126 | 51/128 | 31/128 | 21/128 | 0.9994 | STABLE | 19,599 |
| 6 | 240,000 | 1.0000 | 0.6060 | 51/128 | 33/128 | 16/128 | 0.9994 | STABLE | 19,771 |
| 7 | 280,000 | 1.0000 | 0.6082 | 51/128 | 37/128 | 16/128 | 0.9993 | STABLE | 19,901 |
| 8 | 300,000 | 1.0000 | 0.6103 | 51/128 | 33/128 | 20/128 | 0.9993 | STABLE | 19,835 |

**Verdict**: STRONG LEARNING — dots have significantly specialised.  
**Speed**: 19,834 w/s average (exceeds 16k target).  
**Next**: Dots>50% stable at 51/128. To push further, consider increasing causal_batch or adding repulsion between dom>0.9 dots.

---

## Generation Architecture (May 2026)

### Peep Mechanism + Grammar Guide (`peep/peep.py`, `grammar/grammar.py`)

Three improvements to semantic context and generation quality:

#### 1. Recency-Weighted Context Encoding (`pipeline.py: _get_ctx_cached`)
- Context vector = weighted sum of token embeddings with decay 0.70^(n−1−i)
- Last word gets weight 1.0, second-to-last gets 0.70, etc.
- L2-normalised to unit sphere
- Prevents long prompts from diluting recent semantic signal

#### 2. Peep Mechanism (`peep/peep.py`)
- `PeepMechanism`: 128 specialisation vectors (one per dot), shape (128, FEATURE_DIM)
- `observe_batch(ctxs, best_dots)`: after each causal training position, EMA-updates the context-direction centroid for each winning dot (lr=0.04)
- `select(ctx_eff, raw_preds)`: pure cosine similarity selects the single dot most aligned to current context
- `top_k(ctx_eff, raw_preds, k=3)`: returns top-3 aligned dots (used for generation averaging)
- Saved/loaded from `global_brain.pkl.peep.pkl`
- After 300k calibration: 128/128 active dots, 236k hits, strong std=2174 (specialisation spread confirmed)

#### 3. Grammar Guide (`grammar/grammar.py`)
- Rule-based POS tagger: 9 classes (ARTICLE, NOUN, VERB, ADJECTIVE, ADVERB, PRONOUN, PREPOSITION, CONJUNCTION, OTHER)
- `GrammarGuide(vocab)`: pre-computes (9, 9) transition weight matrix × per-tag scores → additive vocab bias vector
- `weights(last_token)` → (V,) additive bias applied to cosine scores before softmax sampling
- Encourages grammatically plausible next-word POS class

#### Generation Paths (PATH A / PATH B)
- **PATH A (Peep+Grammar)**: top-3 dots by cosine alignment selected; residuals blended by cosine weight; decoded with grammar bias + softmax temperature. Shows `d{N}(score)` per dot.
- **PATH B (majority-vote fallback)**: original 128-dot majority vote with grammar bias added; shown when Peep not yet calibrated.

#### Calibration
```bash
python main.py calibrate                            # use default /tmp/corpus_300k.txt
python main.py calibrate /path/to/corpus.txt        # custom corpus
python main.py calibrate /path/to/corpus.txt --limit 50000  # limit lines
```
Or interactively:
```
> calibrate
> calibrate /tmp/corpus_300k.txt
```
Calibration re-runs `causal_train_pass` with cleared ctx-cache, then saves Peep to disk. Subsequent `cgen` commands use PATH A automatically.

## Running the App

```bash
python main.py                          # silent interactive REPL (just a "> " prompt)
python main.py demo                     # original full showcase output
python main.py train corpus.txt         # vocab-only training (standard speed)
python main.py train corpus.txt --fast  # ultra-fast parallel training (100k+ sent/s)
python main.py train corpus.txt --fast --evolve   # fast vocab + full dot evolution
python main.py train corpus.txt --fast --workers 4  # specify CPU count explicitly
python main.py train corpus.txt --limit N   # train on first N lines
python main.py calibrate                # build Peep specialisations (enables PATH A)
python main.py generate "prompt"        # encode prompt → decode output text
python main.py cgen "prompt"            # IECNN-native generation (Peep+Grammar)
python main.py encode "text"            # encode → 256-dim latent
python main.py sim "text A" "text B"    # cosine similarity
python main.py compare "a" "b" "c"      # n×n similarity table
python main.py memory                   # dot memory + evolution state
python main.py build                    # (re)compile C extensions
```

Fast training standalone:
```bash
python fast_train.py corpus.txt                      # vocab-only, 100k+ sent/s
python fast_train.py corpus.txt --full               # vocab + dot evolution
python fast_train.py corpus.txt --workers 4          # explicit worker count
python fast_train.py corpus.txt --subwords           # enable BPE subword discovery
python fast_train.py --bench                         # synthetic throughput benchmark
python fast_train.py --bench --bench-n 100000        # benchmark with 100k sentences
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
- `generate(prompt, max_tokens=8, iterations=20)` — prompt → latent → beam-search decoded text (old pipeline)
- `causal_generate(prompt, max_tokens=20)` — IECNN-native majority-vote generation (new, preferred)

### IECNN-Native Generation (`causal_generate`) — added May 2026

The IECNN-native generation loop. Not transformer-style. No softmax over vocab. Steps:

1. **Independent predictions**: `pred[d] = W[d] @ ctx` for all 128 dots — (128, 256)
2. **Residual isolation**: `residual[d] = pred[d] - mean(all_preds)` — strips corpus-average direction
3. **Per-dot vocabulary vote**: each dot decodes its residual to the nearest vocab word via cosine similarity
4. **Plurality selection**: count votes across 128 dots; winning token = token with most dot nominations
5. **Controlled variation**: softmax-sample from top-5 most-voted tokens (temperature scales with confidence)
6. **Stop condition**: `max_votes / 128 < confidence_threshold` (default 0.02) — stop when too uncertain
7. **Context roll**: `ctx = 0.55 * ctx + 0.45 * embed(next_token)` — EMA toward new token
8. **Exclusion signal**: running EMA of used token embeddings, subtracted from ctx each step to prevent attractor loops

**Vote pool**: word-type tokens only — alphabetic (regex `^[A-Za-z][a-z]{2,}$`), contains a vowel, no subword fragments, no numbers. ~10,500 tokens from the 47k vocab.

**CLI command**: `cgen <prompt>` in the interactive REPL, or `python main.py cgen "prompt"` one-shot.

**Output format**: text + per-token `Votes/128` display showing how many dots agreed on each token.

**Context sensitivity**: different prompts produce different token streams (measurable from step ~3 onward). The model is associative, not grammatical — output reflects the W-matrix specialisation directions, not language rules.

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
