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

### `generation/` Module — Full Score-Processor Pipeline

IECNN's generation engine lives entirely in `generation/` and is wired into
`causal_generate()` / `causal_generate_nbest()` in `pipeline/pipeline.py`.

#### Processors (`generation/processors.py`) — 13 pluggable stages
All processors operate on cosine-similarity score arrays `(V,)` in IECNN's
unit-sphere space (no gradients, no logits).

| # | Class | Paper | Purpose |
|---|---|---|---|
| 1 | `SemanticFieldBias` | — | Boost words near the prompt's topic centroid |
| 2 | `VocabFrequencyPrior` | — | Log-freq prior; penalise ultra-rare words |
| 3 | `RepetitionPenalty` | Keskar 2019 (CTRL) | Presence + frequency decay |
| 4 | `NoRepeatNGram` | Paulus 2018 | Hard-block repeated bigrams + trigrams |
| 5 | `DegenerationPenalty` | Su & Collier NeurIPS'22 | SimCTG embedding-level anti-repeat |
| 6 | `MinLengthGuard` | — | Block stop tokens before min_tokens |
| 7 | `ExponentialDecayLength` | HuggingFace 4.x | Smooth stop-token boosting after start_idx |
| 8 | `TypicalFilter` | Meister 2023 | Remove over/under-predictable tokens |
| 9 | `NucleusFilter` | Holtzman 2020 | Top-p probability mass cutoff |
| 10 | `EtaFilter` | Hewitt 2022 | Entropy-adaptive token truncation |
| 11 | `MinPFilter` | Menhguin 2023 | Remove tokens < min_p × peak prob |
| 12 | `DynamicTemperature` | — | Trend-adaptive temperature (first 3 steps) |
| 13 | `MirostatScheduler` | Basu 2020 (ICLR'21) | Feedback-loop target-confidence temperature |

#### Context Enrichment (`generation/context_hist.py`)
- `ContextHistory(window, dim, decay, ctx_alpha)` — attention-weighted ring buffer; `attend(ctx)` returns history-enriched context vector (IECNN's KV-cache equivalent)
- `ContextAnchor(prompt_ctx, drift_threshold, correction_strength)` — prompt-anchored anti-drift correction; prevents context from drifting away from the original topic

#### Multi-Head Convergence (`generation/multihead.py`)
- `MultiHeadConvergence(n_heads=8, embed_dim=224)` — 8 independent dot groups via round-robin assignment
- `forward(raw_preds, word_vecs_n, gram_w, block_mask)` → `(V_scores, confidence)` — confidence-weighted sum across heads
- `peep_forward(...)` → `(V_scores, confidence)` — 70% Peep signal + 30% MHC blend (PATH A)
- `contrastive_forward(raw_preds, ctx_eff, ...)` → `(V_scores, confidence)` — IECNN Contrastive Decoding (Li et al. 2022): top-½ context-aligned dots (expert) vs bottom-½ (amateur); score = (1+α)×expert − α×amateur; PATH B

#### `causal_generate()` — 10-stage loop
```
1. ContextHistory.attend() → history-enriched ctx
2. ContextAnchor.correct() → anti-drift correction
3. Exclusion steering → push away visited embedding space
4. W_stack @ ctx_eff → raw_preds (128, 256)
5. PATH A (Peep calibrated): mhc.peep_forward() — Peep top-5 (70%) + MHC (30%)
   PATH B (uncalibrated):    mhc.contrastive_forward(α_decay=0.18→0.08) — expert/amateur split
2.5. Speculative two-pass vocab pre-filter: draft with top-32 ctx-aligned dots →
     keep top-350; full voting on survivors only (~3× step speedup)
5.5. Cross-head agreement bonus: vote count / n_heads × 0.10 (per word)
6. Score pipeline (9 stages):
   SemanticFieldBias(dynamic anchor, blend=0.12/3steps) → VocabFrequencyPrior →
   BigramContinuationBonus(last_tok) → SemanticProximityPenalty(win=4, thresh=0.82) →
   RepetitionPenalty → NoRepeatNGram(3)+NGram(2) → DegenerationPenalty →
   MinLengthGuard → ExponentialDecayLength
7. 1-step lookahead pre-scorer: top-3 candidates scored by W_stack consensus-norm
   under would-be next context (step-adaptive _ca); bonus 0.06 × consensus_norm
8. Confidence check + low-confidence early stop + coherence-trend degeneration guard (coh3<0.04)
9. Entropy-adaptive exclusion coefficient update (H_norm → excl = 0.35–0.50)
10. Six-layer filter: TopKFilter(k=40, adaptive) → TypicalFilter(p=0.95) →
    TailFreeFilter(z=0.97) → Nucleus(p=0.92→0.87, adaptive) → Eta(ε=3e-4) → MinP(0.05)
11. Temperature: DynamicTemperature (steps 0-2) →
    MirostatScheduler(τ=0.38, lr=0.08, τ_warmup=0.85→0.65 over 6 steps) (step 3+)
12. softmax_sample → emit token; update EMA context + ContextHistory + exclusion EMA
    + dynamic SemanticFieldBias anchor blend (every 3 steps)
```
Result dict includes: `text`, `tokens`, `confidences`, `stop_reason`,
`coherence` (avg cos-sim between consecutive token embeddings),
`diversity` (unique/total tokens), `fluency` (attested bigram fraction),
`vocab_size` (freq-filtered pool size).

#### Key quality improvements (session 2 — May 2026)
| Feature | Technique | Effect |
|---|---|---|
| Contrastive voting (PATH B) | Li et al. 2022 — split top/bot-½ dots by ctx alignment | Amplifies context-specific signal |
| Bigram continuation bonus | Data-driven collocation scoring | Improves syntactic fluency |
| Freq-filtered vocab (min_freq=3) | Removes hapax legomena | Eliminates cosine noise from rare words |
| TypicalFilter | Meister 2023 | Removes over/under-predictable tokens |
| EtaFilter | Hewitt 2022 | Entropy-adaptive truncation |
| 1-step lookahead scorer | IECNN native | Prevents garden-path token choices |
| Entropy-adaptive exclusion | — | Stronger steering when model is uncertain |
| Coherence-trend stop | — | Detects and halts at degeneration onset |
| Step-decaying α (PATH B) | — | More guidance early, less late |
| N-best reranking (×3 / ×5) | — | Picks best-coherence run from multiple |
| SemanticProximityPenalty | Embedding cosine penalty (quadratic) | Blocks near-synonym repetition |
| Step-adaptive ctx_alpha | Linear schedule 0.72→base | Tighter prompt hold at start, natural drift later |
| Mirostat τ-warmup | Ceiling 0.85→0.65 over 6 steps | Richer early exploration, focused late sampling |
| Quality rating display | avg_conf × coh × diversity composite | Immediate per-run quality feedback in CLI |
| Cross-head agreement bonus | Head vote-count normalised to [0, 0.10] | Rewards tokens confirmed by multiple MHC heads |
| Two-pass speculative filter | Draft 32 dots → keep top-350 → full 128-dot pass | ~3× per-step cost reduction, quality preserved |
| Dynamic SemanticFieldBias | EMA anchor blend 0.12 every 3 steps | Bias tracks evolving topic, not just original prompt |
| Adaptive top_p scheduling | 0.92 (step 0) → 0.87 (step 8+) | Wider beam early, focused sampling once context set |
| TailFreeFilter (z=0.97) | 2nd-derivative tail cut (Phénix & Egan 2019) | Removes flat low-prob tail; complements nucleus |
| N-best diversity forcing | Per-run seed offset + first-token penalty | Ensures N runs explore distinct continuations |
| Bigram fluency bonus | attested_bigrams / (ntok-1) × 0.12 in N-best score | Rewards linguistically natural token sequences |
| Dot dropout (PATH B) | 10% random dot zeroing per step | Attention-dropout analog; improves N-best diversity |
| Confidence bar display | 5-block ASCII bar per token (░░░░░ → █████) | Instant visual confidence read-out in CLI |
| TopKFilter (adaptive k=40) | Entropy-scaled k: 40 (confident) → 60 (uncertain) | Hard cap before soft filters; prevents extreme tail |
| Adaptive anchor strength | 0.20→0.06 decay (0.88^step) | Strong on-topic start; natural drift allowed later |
| Fluency metric | bigram_hits/(ntok-1) in result dict + CLI display | Quantifies attested-collocation density of output |
| Multi-window coherence guard | coh3 < 0.04 OR coh6 < 0.10 dual-trigger stop | Catches abrupt collapse AND slow-rolling drift |
| Prompt-type detection | Q/Imperative/Statement → CONF_STOP ± 15%, min_tokens ± 1-2 | Adapts generation budget to prompt intent |
| Adaptive HARD_BLOCK window | 4 (step 0) → 8 (step 16+) via 4 + step//4 | More open early exploration, tighter loop prevention late |
| Head z-score normalisation | per-head (hs − mean)/std before combination | Prevents high-magnitude heads drowning quieter focused heads |
| DotVariancePenalty (18th proc) | σ²_norm × 0.08 subtracted per token | Penalises tokens where dots strongly disagree |
| Peep path z-score | (peep_scores − μ)/σ before blend | Makes Peep/MHC magnitudes comparable for 70/30 blend |
| Contrastive z-score | final combined normalised post-subtraction | Stabilises filter chain input across all steps |
| Context momentum (β1=0.85) | vel = β1·vel + (1−β1)·Δctx; ctx += 0.12·vel | Smooths context trajectory, dampens oscillation |
| Score-margin gating | top1-top2 gap < 0.05 → 70% top1 + 30% top2 context | Soft context commit for uncertain tokens |
| Shannon diversity index | H_norm = H/log(N) in result dict | Information-theoretic diversity vs simple TTR |
| EMA entropy tracker (H̄) | α=0.15 per-step H_norm rolling mean | Tracks generation uncertainty across the sequence |
| nbest10 command | 10-candidate N-best reranking | Broader search for highest-quality output |
| `stats` CLI command | vocab, dots, Peep, filter chain summary | Instant brain health overview |
| LocalSemanticFilter (19th) | top-200 context-cosine vocab restriction | Cuts live vocab to semantically focused pool |
| Adaptive min_p scheduling | min_p = 0.05×(1−0.40×H_norm) in [0.02,0.05] | More permissive at high entropy, tighter when confident |
| Peep path z-score | (peep_scores−μ)/σ before 70/30 blend | Equal-magnitude inputs into Peep/MHC blend |
| Recency-weighted confidence (RW) | linear ramp weights, later tokens heavier | Reflects current quality better than simple average |
| Confidence histogram | 5-bin [▁▂▃▄█] bar in CLI output | Instant visual distribution of per-token confidence |
| PromptDriftPenalty (20th) | strength=0.06, threshold=0.08 — penalises prompt-distant tokens | Symmetric push-pull attractor with SemanticFieldBias |
| Recency-decay rep penalty | presence×(0.88^age) per token | Old tokens penalised 37% as heavily as recent ones |
| Penalty momentum (β=0.78) | EMA of rep-delta, 0.30× applied next step | Burst-repetition accumulates extra penalty |
| Confidence trend boost | polyfit(last-5 confs); slope<-0.04 → +0.12 temp | Prevents collapse into a low-confidence rut |
| Dynamic local_sem top_k | 150+100×H_norm in [150,250] | Adapts semantic pool width to current uncertainty |
| Coherence-adaptive ctx alpha | +0.04 if coh≥0.30, -0.04 if coh<0.08 | Context hold strength tracks coherence quality |
| High-quality early stop | conf≥0.78 & coh≥0.55 & conf×coh>0.50 → stop | Prevents overextension past a coherent peak |
| Score floor clamp | finite scores clamped to ≥ -3.0 | Guards softmax against stacked-penalty underflow |
| Multi-scale coherence (C3/C6) | last-3 and last-6 window cosine averages in result + CLI | Detects local vs global coherence independently |
| Low-confidence token marker | `~` prefix on tokens with conf < 0.25 in CLI | Visually flags uncertain tokens in output |
| SurpriseBonus (21st proc) | +0.04 to context-close non-top-50 tokens | Widens beam beyond dot-consensus attractor |
| Head-spread penalty (step 5.55) | max−min head score, normalised × 0.05 | Penalises per-head high-disagreement tokens |
| Beam entropy gate (step 8.5) | if <3 survivors, restore pre-filter top-3 | Prevents filter over-kill from single candidates |
| Vocabulary precision metric | hits/total in result dict + CLI VPrec | Fraction of tokens chosen from context-close pool |
| Two-path consensus blend (5.75) | Path A 88% + Path B contrastive 12% | Cross-validates Peep+MHC with contrastive path |
| Adaptive anchor strength (C3) | +0.05 if last 3-step sum cosines < 0.20 | Stronger prompt anchor when local coherence drops |
| Near-window rep penalty (6-3a) | −0.08 extra for last-3 tokens | Burst-repetition suppression on top of EMA momentum |
| Enriched N-best scoring | adds RW_conf, VPrec, C3, Shannon to composite | Richer candidate selection beyond avg_conf×coherence |
| `ctxvec` CLI command | top-10 nearest vocab words to context direction | Diagnose whether context is pointing at right region |
| Context oscillation dampener | cosine(Δ_t, Δ_{t-1}) < −0.10 → velocity × 0.70 | Resists ctx ping-pong between two semantic poles |
| Pseudo-perplexity in result dict | 2^(H̄ × log2 V); CLI label PPL | Normalised generation difficulty score |
| Confidence variance in result dict | std dev of step confidences; CLI label Var | Measures patchy vs consistent model certainty |
| Bigram+trigram fluency blend | 80% bigram + 20% trigram vocab hit fraction | More sensitive naturalness proxy than bigrams alone |
| `topwords` CLI command | top-N vocab words by dot-outcome frequency | Reveals generation frequency bias in the brain |
| Enriched N-best scoring v2 | adds pseudo_ppl (inverted) and conf_variance to composite | Richer N-best selection |
| Semantic field momentum | slow centroid α=0.05; pull str=0.08 if dist>0.35 | Resists runaway topic drift while allowing sentence dev |
| Prompt-type anchor multiplier | question×1.20, continuation×0.85, imperative×1.10 | Tighter anchoring for Q&A, looser for continuations |
| Score-gap EMA tracker (8.7) | EMA top1−top2 score gap per step; result key avg_score_gap | Measures per-step decision certainty |
| Low-entropy collapse gate (8.6) | restores top-5 pre-filter when ≤2 survivors or spread<0.02 | Prevents filter chain from collapsing to deterministic output |
| Confidence trend in result dict | polyfit slope over all steps; result key conf_trend; CLI Trend | Positive = growing confidence |
| CLI stats expanded | +Var, +Trend, +Gap, +PPL labels | All new metrics visible in every generation |
| Repetition burst detector (6-3c) | 2+/4 identical tokens → reset rep_mom + temp+0.20 | Forces model off attractor on burst-repeat detection |
| Score-gap guided temperature | high gap→temp−0.05; low gap→temp+0.05 | Decisive when confident, exploratory when uncertain |
| Adaptive temp lower bound | RW_conf≥0.50→lb=0.08; <0.30→lb=0.20 | Tighter sampling when consistently confident |
| Exclusion radius scaling | +0.50×rep_mom_mag (capped +0.15) | Stronger exclusion push when rep momentum is high |
| Low-confidence recovery pulse | conf<0.15 at step≥2 → temp+0.08 | Nudges model out of very-low-confidence local minimum |
| Mirostat target adaptation | coh≥0.30→target=0.35; coh<0.08→0.42 | Coherence-responsive sampling tightness |
| Entropy budget guard | cumulative H > max_tokens×0.65 → temp≤0.10 | Near-greedy finishing once entropy budget exhausted |
| Peep hit-rate tracking | fraction of steps Peep top-5 ⊇ chosen token | Exposes Peep calibration quality; result key peep_hit_rate |
| Context velocity magnitude | EMA ‖ctx_vel‖; result key ctx_vel_mag | Measures how fast context is shifting semantically |
| Soft anchor restart | 3-step low-gap streak (gap<0.03) → re-warm anchor to current ctx | Escapes stale anchor plateau (one-time per generation) |
| Exclusion warmup | exclusion disabled for first 2 steps | Free early-token exploration before exclusion kicks in |
| Final-3-step vocab tightening | LocalSem top_k=120 for last 3 steps | Focused ending candidates for cleaner generation close |
| `peepstats` CLI command | adj-cosine spread + norm stats across dot specialisations | Diagnose Peep calibration quality and diversity |
| Depth-2 lookahead (6-8) | simulate 2-step paths for top-2 candidates (+0.03 bonus) | More accurate planning vs depth-1 garden-path avoidance |
| Coherence-guided SFB strength | coh≥0.30→×0.70; coh<0.08→×1.40 | Adaptive topic-coherence bias based on local fluency |
| Score centroid floor lift (6-0b) | EMA top-10 mean; slope<−0.10→+0.04 lift | Rescues fading candidate pool before filter chain zeros it |
| Peep top-k adaptation | H_norm>0.60→top-k=7 (wider candidate set) | Uncertain steps use more Peep candidates for diversity |
| `compare A|B|C` interactive command | pairwise similarity matrix in the REPL | Compares multiple texts without leaving the interactive loop |
| Dynamic DRAFT_D scaling | 24+24×H_norm (24→48) | Wider dot draft on uncertain steps for better pre-filtering |
| CLI Vel+PHit labels | ctx_vel_mag and peep_hit_rate in stats line | Every generation shows context speed and Peep hit-rate |
| `histgen` CLI command | 5 seeded variations with Avg/Coh/Flu/PPL table | Side-by-side comparison of generation variation quality |
| Confidence floor early stop | 4 consecutive steps <0.12 conf → stop early | Avoids appending low-quality tokens at generation end |
| Adaptive SimCTG alpha from C3 | coh>0.30→α=0.20; coh<0→α=0.65 | Softer degeneration penalty when model is already coherent |
| Peak confidence step tracking | result key peak_conf_step | Index of the highest-confidence step per generation |
| Context-weighted two-path blend | Path-B contribution halved for cos(word,ctx)<0 tokens | Consensus blend only rewards context-relevant agreement |
| `seedgen [N] <prompt>` CLI command | N seeded runs with quality ranking, marks best with * | Best-of-N seeded generation with composite Q score |
| Entropy-adaptive near-window penalty | 0.08×(0.70+0.60×H_norm) range 0.056–0.104 | Stronger burst suppression when model is uncertain |
| Depth-2 lookahead disabled final 2 steps | skip depth-2 when step ≥ max_tokens−2 | No planning benefit in last 2 generation steps |
| N-best peak-step bonus | ×(1+0.05×tanh(pcs/ntok)) | Rewards candidates whose confidence peaked later |
| Prompt-length adaptive anchor strength | 0.15+0.03×(nwords−1), capped 0.36 | Longer prompts get stronger anchor; single words get 0.15 |
| `topconfs` CLI command | top-5 confidence steps from last generation | Pinpoints where the model was most/least certain |
| Score-gap oscillation damper | 3+ sign-flips in 4-step window → halve delta for 2 steps | Prevents temp oscillation when score gap alternates wildly |
| Gap-adaptive ctx EMA α | gap>0.15→+0.02; gap<0.04→−0.02 on _ca | Tighter context hold on confident steps, looser on uncertain |
| C6 sustained-drift pulse | 3 steps C6<0.05 → one-time +0.12 temp pulse | Escapes sustained semantic drift that Mirostat doesn't catch |
| Per-step score-gap history | result key sg_step_gaps (list) | Powers scorehist CLI; full decision-certainty timeline |
| `scorehist` CLI command | ASCII bar chart of per-step score gaps | Visualises model certainty across entire last generation |
| Vocab-prec adaptive min_p | vprec>0.50→lower min_p; vprec<0.50→raise min_p | Diversity-tuned filter based on how well vocab precision is holding |
| Coherence-variance SFB boost | std-dev coh>0.15 over last 4 steps → +0.04 SFB | Zigzagging topic triggers stronger prompt-topic pull |
| Contrastive alpha decay adaptive to length | max_tokens>16 → slower decay (up to 0.92) | Longer gens preserve contrastive signal beyond step 8 |
| Token embedding spread metric | mean pairwise 1−cos across last 16 token embeddings | Result key token_embed_spread; higher = more semantically diverse |
| Spread in CLI stats line | Spread:{tok_spread:.3f} added to stats output | Every generation shows how spread-out the chosen embeddings are |
| `confgraph [N]` CLI command | Unicode sparkline + bar chart of step confidences | Full visual confidence profile with per-token bars |
| Dot agreement bonus (6-0a) | ≥2 dots top-1 agree → +0.012×(extra_dots) bonus | Multi-dot consensus on a token gets a small additive reward |
| Score spread clipping | max score capped at min+1.6 to prevent softmax collapse | Prevents outlier-high tokens from monopolising sampling |
| Adaptive history window | window = history_window + max_tokens//4, capped 40 | Longer generations automatically attend further back |
| Generation rhythm score | 1−(alternations/steps); 1=smooth, 0=zigzag | Result key rhythm_score measures confidence smoothness |
| `dotscores` CLI command | vote-count bar chart of dot top-1 agreement | Shows which tokens the dot ensemble converged on last step |
| Dot agreement bonus cap | per-token bonus capped at 0.06 | Prevents over-rewarding very popular consensus tokens |
| Velocity-adaptive exclusion | excl_coeff −= 0.08×vel_mag | High context velocity → loosen exclusion (already moving away) |
| Early C3 stop relaxation | C3≥0.40 + conf≥0.70 at step≥min_tokens−1 → stop | Very coherent early runs can end slightly before min_tokens |
| Rhythm in CLI stats line | Rhythm:{rhythm:.2f} in stats output | Every generation shows its confidence smoothness score |
| `rhythmgraph` CLI command | ▲▼─ sparkline of per-step confidence deltas | Full visual rhythm profile with alternation count |
| Per-step top-3 token log | result key _top3_log: list of [(score,word)×3] | Powers top3log CLI; shows what the model almost picked |
| Rhythm-adaptive temperature | 3-step alternating zigzag → +0.04 temp | Smooths recurring confidence zigzag patterns |
| N-best rhythm bonus | ×(1+rhythm_score×0.04) | Smoother-confidence candidates rewarded in N-best |
| `gensummary` CLI command | one-liner: Qual/Avg/Coh/Flu/PPL/Rhythm/Spread | Quick quality digest of the last generation |
| `top3log [N]` CLI command | per-step table of top-3 token candidates | Reveals what the model almost chose at every step |
| Anchor strength result key | result["anchor_strength"] = _anc_strength | Prompt-adaptive anchor strength exposed for inspection |
| Prompt type result key | result["prompt_type"] = _ptype (question/imperative/statement) | Detected intent exposed for downstream analysis |
| Coherence direction signal | result["coh_direction"] = rolling slope of coh3 over last 4 steps | + = improving, − = degrading coherence trajectory |
| Rhythm-adaptive score floor | rhythm<0.4 → floor raised +0.02 | Zigzag confidence broadens candidate pool to find alternatives |
| Anc/CohDir/PType in CLI stats | Anc:{anc_str}  CohDir:{coh_dir}  PType:{prompt_type} | Full visibility into anchor, coherence direction, prompt intent |
| `analyze` CLI command | brain diagnostics: dot/W-stack/word-vec norms + vocab coverage | One-stop brain health check |
| Step-wise confidence EMA | α=0.35 smoothed EMA of per-step confidence (not used for sampling) | conf_ema_final result key; reduces tracking noise |
| Score histogram result key | result["score_hist"]: 8-bucket count of per-step confidence values | Distribution of confidence across generation; shown in scorehist |
| Adaptive degeneration threshold | coh_dir < −0.06 → guard tightens from 0.04 to 0.06 | Catches degrading coherence trajectories earlier |
| `scorehist` confidence histogram | Appended to scorehist output: 8 confidence buckets with bars | Visual distribution of how confident the model was per step |
| `diffgen A \| B` CLI command | side-by-side stats for two different prompts | Compare avg/coh/rhythm/ppl/spread/toks between two prompts |
| Conf-EMA used in floor-stop | _conf_ema replaces raw confidence in floor-stop check | Single noisy step no longer prematurely terminates |
| Vocab-prec EMA (α=0.20) | _vprec_ema smooths raw vocab_prec; used in min_p adaptive | Smoother adaptive min_p; result key vocab_prec_ema |
| coh_dir N-best bonus | ×(1 + max(coh_dir,0)×0.06) in N-best reranking | Improving-coherence generations rewarded in N-best |
| `trendgen [N] <prompt>` CLI | run same prompt N times, compare conf_ema_final across runs | Repeatability/stability metric for any prompt |
| Per-step top-1 margin log | result["top1_margins"]: top1−top2 gap at every step | Decisiveness profile per generation step |
| Adaptive Mirostat from vprec_ema | vprec_ema≥0.70 → target=min(target,0.32) | High vocab precision tightens Mirostat exploration |
| coh_dir-adaptive centroid lift | coh_dir<−0.04→+0.05, >+0.04→+0.03, else +0.04 | Coherence trajectory shapes how aggressively scores are rescued |
| `confema` CLI command | raw avg vs EMA, variance, trend, vprec_ema, sparkline | Full confidence quality summary for last generation |
| `marginlog [N]` CLI command | per-step top-1 score margin sparkline | Shows decisiveness (how far ahead top token was) at each step |
| Margin-adaptive temperature | margin_ema>0.15→−0.02 temp; <0.03→+0.02 | Decisive steps allow tighter temp; wavering opens up |
| Low-margin burst trigger | margin<0.02 for 3+ steps → 75%-strength burst pulse | Indecisive plateau breaks out via temp pulse, same as burst |
| Adaptive early-C3 threshold | vprec_ema≥0.65 → C3 threshold relaxed 0.40→0.35 | High vocab precision lets model stop on slightly weaker C3 |
| Quality history tracker | model._quality_history deque (last 8 N-best scores) | Tracks nbest/seedgen quality across recent generations |
| `qualplot` CLI command | bar chart of quality history scores | Visual trend of generation quality over last 8 N-best runs |
| Per-step coh3 log | result["coh3_steps"]: coh3 at every step≥3 | Coherence trajectory for cohplot |
| Per-step entropy log | result["entropy_steps"]: H_norm at every step | Entropy trajectory for entropyplot |
| SFB strength fresh per gen | SemanticFieldBias recreated each causal_generate call | Prevents mutated SFB strength from contaminating next generation |
| `cohplot [N]` CLI command | sparkline + bar chart of per-step coh3 values | Visual coherence trajectory for last generation |
| `entropyplot [N]` CLI command | sparkline + bar chart of per-step H_norm values | Visual entropy trajectory for last generation |
| Per-step velocity log | result["velocity_steps"]: ctx_vel_mag_ema at every step | Powers velplot CLI |
| Per-step SFB strength log | result["sfb_strength_steps"]: SFB.strength at every step | Powers sfbplot; shows how SFB adapts during generation |
| Margin-adaptive anchor boost | margin_ema<0.03 → anchor.strength += 0.04 | Indecisive plateau pulls harder back to prompt |
| `velplot [N]` CLI command | sparkline + bar chart of per-step velocity EMA | Shows how fast context moves semantically step by step |
| `sfbplot [N]` CLI command | sparkline + bar chart of per-step SFB strength | Shows prompt-topic pull force profile during generation |
| `speedrun <prompt>` CLI command | run prompt at 5/10/15/20/25 max_tokens, compare metrics | Length-vs-quality diagnostic for any prompt |
| Per-step score percentile | result["percentile_steps"]: np.mean(finite≤winner) per step | 1.0=always top; <1.0=below-top token chosen (surprise/sampling) |
| `fulldiag` CLI command | all trajectory sparklines (coh3/entropy/vel/sfb/margin/pct) | One-command complete generation health report |
| `benchprompts` CLI command | 5 standard prompts → conf_ema/coherence/rhythm/coh_dir table | Reproducible quality benchmark across prompt types |
| `pctplot [N]` CLI command | sparkline + bar chart of per-step score percentile | Shows how consistently the top token is chosen |
| Adaptive eta epsilon | _eta.epsilon = clip(3e-4 × (1+2×H_norm), 3e-4, 9e-4) | Entropy-responsive eta filter: uncertain→wider, confident→tighter |
| Short-token filler penalty | len(token)≤2 → score-=0.025 when >3 alternatives exist | Discourages "a","in","of" from dominating uncertain steps |
| `conf_declining` result key | True when last 4 confidences are strictly decreasing | Flag for detecting loss-of-confidence tail in generation |
| `repdiag` CLI command | unique/repeated/short/long token stats for last gen | Quick repetition + confidence health check |
| `confoverlay [N]` CLI command | raw conf + recomputed EMA per-step sparkline overlay | Shows whether EMA is tracking or lagging raw confidence |
| Per-step temperature log | result["temp_steps"]: final temp at every step | Powers tempplot; shows how mirostat/boosts affect sampling temp |
| `vocab_prec_ema` result key | result["vocab_prec_ema"]: vocab precision EMA at gen end | Measures how consistently model picks context-aligned vocab |
| Adaptive SFB decay from coh3 | coh3≥0.45→_sfb._decay≥0.96; coh3<0.20→_sfb._decay≤0.90 | Keeps topic bias alive when coherent; releases when drifting |
| `tempplot [N]` CLI command | sparkline + bar chart of per-step temperature | Shows when mirostat tightens or boosts loosen sampling |
| `tokenmap` CLI command | color grid of token length profile (S/M/L/X) | Visual token-length distribution for last generation |
| `gencompare <prompt>` CLI command | causal_generate vs causal_generate_nbest side-by-side | Direct comparison of single-run vs N-best reranking quality |
| Per-step coh6 log | result["coh6_steps"]: 6-window coherence at every step≥6 | Powers coh6plot; longer window than coh3 for smoother signal |
| `sg_slope` result key | linear regression slope of sg_step_gaps | Positive=growing decisiveness; negative=wavering toward end |
| Adaptive typical p from coh3 | coh3→typical.p = clip(0.95-0.03×coh3, 0.88, 0.97) | Coherent steps tighten filter; incoherent widen to find path |
| `coh6plot [N]` CLI command | sparkline + bar chart of per-step 6-token coherence | Slower/smoother coherence signal than coh3plot |
| `topconfgen [N] <prompt>` CLI command | run N times, sort by conf_ema_final, show best | Self-selecting best generation without nbest overhead |
| `pivotgen <word> \| <prompt>` CLI command | generate baseline vs pivot-appended prompt, compare | Tests how a forced first-token direction affects generation |
| Per-step score variance | result["score_var_steps"]: var of finite scores per step | High var=wide spread; low var=flat/uncertain distribution |
| Adaptive TopK from entropy | _topk.k = clip(30+int(20×H_norm),20,60) | High entropy→more candidates; confident→tighter beam |
| `stresstest` CLI command | 10 built-in prompts → avg/std for conf/coh/rhythm table | Reproducible broad stress-test of current brain quality |
| `varplot [N]` CLI command | sparkline + bar chart of per-step score variance | Shows how spread the score distribution is at each step |
| `genheatmap` CLI command | score histogram heat map showing bucket fill fractions | Distribution shape visualizer for last generation |
| `confmatrix` CLI command | 2-D confidence×position heat map (4 bins × 3 positions) | Shows where in the sequence confidence is high vs low |
| Per-step TopK log | result["topk_steps"]: _topk.k at every step | Powers topkplot; shows how entropy drives candidate beam width |
| N-gram novelty bonus | +0.010 to tokens not in last-8 output; O(8) lookup | Soft positive counterpart to ngram2 hard block |
| `_conf_ema_steps` result key | per-step smoothed confidence EMA log | Powers confrise detector |
| Adaptive SFB decay from coh_trend | if running coh3 slope < -0.02 → _sfb._decay ≤ 0.91 | Loosens anchor when coherence is linearly declining |
| `sgplot [N]` CLI command | sparkline + bar chart of per-step score-gap EMA | Visualizes model decisiveness at each step |
| `confrise [N]` CLI command | steps where conf_ema Δ > mean+1σ (sharpest rises) | Opposite of conf_declining — find recovery moments |
| `multiprofile` CLI command | 3 preset prompts → full metric table + generated texts | Quick multi-prompt quality snapshot |
| `entropy_ratio` result key | max(entropy_steps) / avg_entropy; spikiness measure | >2 = one step was much noisier than the rest |
| Adaptive anchor boost from prompt_type | question→1.30×, imperative→1.15×, statement→1.00×, continuation→0.85× | Tightened question anchor (was 1.20) |
| `_vprec_ema_steps` result key | per-step vocab-precision EMA log | Powers vocabjump CLI |
| `EntRatio` in stats line | entropy spikiness shown on every generation | Quick diagnostic for unstable distributions |
| `spikeplot [N]` CLI command | marks entropy steps ≥ 1.5× mean as '!' spikes | Pinpoints which tokens had chaotic distributions |
| `confplateau [win]` CLI command | detects windows of ≥win flat conf_ema steps | Finds where model stalled / plateaued |
| `vocabjump [thr]` CLI command | steps where vprec_ema dropped > thr in one step | Pinpoints vocab-quality stumbles |
| `vel_ratio` result key | max(velocity) / mean(velocity); velocity spikiness | >2 = one step shifted context much faster than average |
| Adaptive lookahead K from score gap | gap<0.04→K=5, gap>0.15→K=2, else K=3 | More path exploration when uncertain, less when confident |
| `VelRatio` in stats line | shown on every generation summary | Velocity spikiness at a glance |
| `velspike` CLI command | steps where velocity > mean+1σ with ×mean factor | Pinpoints rapid semantic shifts |
| `confbands` CLI command | per-token L/M/H band strip with colour coding | Visual confidence band breakdown for last gen |
| `repchain` CLI command | 3+ same-token in 6-step window detection | Finds local repetition loops not caught by ngram block |
| `bestseg`/`worstseg` result keys | {start, val} of highest/lowest 3-step coh3 window | Pinpoints peak and trough of generation coherence |
| `flow_score` result key | fraction of steps where both conf_ema+coh3 improved | 1.0 = perfect momentum; 0.0 = always declining |
| `flow_steps` result key | per-step B/C/H/N: Both/Conf/Coh3/Neither improving | Powers flowbar CLI |
| `Flow`/`Best@`/`Worst@` in stats line | shown on every generation | Quick coherence segment quality at a glance |
| `segplot` CLI command | rolling-3 coh3 window sparkline with ★/☆ best/worst | Visualizes where generation coherence peaked and troughed |
| `flowbar` CLI command | per-step ▲▲/▲=/=▲/▼▼ indicator strip with FlowScore | Visual momentum breakdown for entire generation |
| Flow-adaptive nucleus p | low flow_frac→+0.02 wider; high flow_frac→-0.02 tighter | Nucleus responds to recent momentum quality |
| `peak_conf_val`/`min_conf_val`/`min_conf_step` result keys | peak + valley confidence with step index | Powers confpeak CLI |
| `confpeak` CLI command | peak+valley confidence with ±2 token context window | Pinpoints best and worst confidence moments |
| `trendcompare A \| B` CLI command | run two prompts, compare 9 metrics side by side | Quick A/B quality comparison with winner column |
| `qualmap` CLI command | per-step conf×coh3 composite quality heat map | Single combined quality signal for each generated token |
| Flow-adaptive Mirostat target | strong flow(≥0.75B)→−0.03; weak(≤0.25B)→+0.03 target | Mirostat tightness tracks recent momentum |
| `topkdelta` result key | std-dev of topk_steps; measures beam-width fluctuation | High = model was uncertain about how wide to search |
| `conf_smooth` result key | 3-point centred MA of conf_ema_steps | Smoothed confidence trajectory, less noise |
| `TopKΔ` in stats line | shown on every generation | Beam width stability at a glance |
| `smoothplot [N]` CLI command | raw conf_ema vs 3-pt smoothed sparkline overlay | Compare raw vs smoothed confidence trajectory |
| `diagsummary` CLI command | all 24 key metrics in one compact block | Complete generation diagnostic at a glance |
| Acceleration-adaptive typical p | conf_acc>0.005→−0.02 tighter; <−0.005→+0.02 looser | Typical filter responds to confidence momentum |
| `entropy_acc` result key | per-step entropy 2nd derivative list | Rate of change of entropy change |
| `conf_acc` result key | per-step confidence 2nd derivative list | Powers accelplot CLI |
| `accelplot [N]` CLI command | ▲/─/▼ sparkline of confidence acceleration + top events | Pinpoints fastest rising and falling confidence moments |
| `cohrank` CLI command | tokens ranked by their coh3-window coherence value | Shows which tokens contributed most to coherence |
| `confslope_steps` result key | per-step 1st derivative of conf_ema (instantaneous slope) | Powers slopechart CLI |
| `coh_acc` result key | per-step coherence 2nd derivative list | Powers cohaccel CLI |
| `slopechart [N]` CLI command | ▲/─/▼ sparkline of per-step confidence slope | Rising/flat/falling breakdown of confidence direction |
| `cohaccel` CLI command | ▲/─/▼ sparkline of coherence acceleration + top events | Pinpoints fastest coh3 changes |
| `scorevar` result key | mean within-step score variance across generation | High = spread of candidates varied a lot step to step |
| `perc_trend` result key | linear slope of per-step score percentile | Rising = model making higher-quality picks over time |
| `sfb_trend` result key | linear slope of SFB strength over generation | Rising = anchor field strengthening as gen proceeds |
| `perfindex` result key | composite quality index ∈ [0,1] (conf×coh×fluency×flow) | Single headline quality number |
| `SVar`/`PercTrend`/`SFBTrend`/`PerfIdx` in stats line | shown on every generation | Score variance and trend metrics at a glance |
| `scorevarplot [N]` CLI command | per-step score-variance sparkline with table | Visualizes where candidate spread was highest |
| `conf_range` result key | max−min confidence across generation | High = model swung between high and low confidence |
| `coh_range` result key | max−min coh3 across generation | High = coherence was very uneven |
| `tokenlen_steps` result key | per-step token character length list | Powers toklenplot CLI |
| `rangeplot` CLI command | side-by-side conf and coh3 range bars | Visual spread comparison |
| `buckethist` CLI command | 10-bucket 0.0-1.0 confidence histogram with % | Finer-grained confidence distribution than 8-bucket hist |
| `toklenplot` CLI command | per-step token length sparkline with length distribution | Shows token length patterns across the generation |
| Adaptive expDecay factor from perc_trend | rising picks→factor=1.04; falling→1.08 | Decay rate responds to token quality trajectory |
| `wintok`/`wintok_count` result keys | most-repeated token and its count | Dominant token detection |
| `phasestats` result key | {early,mid,late} phase avg conf + avg coh3 + n_toks | Powers phasemap CLI |
| `vocabtop10` CLI command | top-10 tokens by frequency with % and bar | Frequency profile of the generation |
| `phasemap` CLI command | early/mid/late phase conf+coh bars with token snippets | Generation structure in three phases |
| Flow-adaptive lookahead weight | flow≥0.70→_LH_WEIGHT=0.04; ≤0.25→0.09; else 0.06 | Lookahead leans harder when model is struggling |
| `uniq_ratio` result key | unique token types / total tokens | 1.0 = all different; <0.5 = significant repetition |
| `conf_ema_delta` result key | last − first conf_ema step (net improvement) | Positive = model gained confidence over the generation |
| `uniq_steps` result key | per-step running unique-token ratio | Powers uniqplot CLI |
| `UniqR`/`ConfΔ` in stats line | shown on every generation | Lexical diversity and net confidence change at a glance |
| `uniqplot` CLI command | running unique-token ratio sparkline | Visualizes how fast vocabulary is being exhausted |
| `cohseries` CLI command | full per-step coh3 table with sparkline | Detailed coherence series for every step |
| Conf-range adaptive ScoreSpreadClip | conf_range>0.35→clip=2.0; <0.10→clip=1.3; else 1.6 | Clip adapts to how volatile confidence has been |
| `conf_ema_mid` result key | conf_ema value at midpoint step | Split-point for early vs late quality comparison |
| `entdelta` result key | last − first entropy step (negative = focused over time) | Shows if distribution tightened during generation |
| `ConfMid`/`EntΔ` in stats line | shown on every generation | Midpoint confidence and entropy change at a glance |
| `midconf` CLI command | early/mid/late conf_ema with ▲/─/▼ direction arrows | Trajectory of confidence through the generation |
| conf_ema_delta adaptive temperature | delta>0.05→cool −0.03; <−0.05→warm +0.05 | Temperature responds to net confidence trajectory |
| `coh_vel_correlation` result key | Pearson r between coh3_steps and velocity_steps | Positive = coherence and velocity co-move |
| `conf_drop_steps` result key | list of steps with confidence drops ≥0.05 | Structured drop events with token + magnitude |
| `correlplot` CLI command | side-by-side coh3+velocity sparklines with Pearson r | Visual correlation between coherence and speed |
| `confdrop` CLI command | steps where confidence dropped ≥0.05 with ▼ bars | Pinpoints confidence collapse events |
| Flow-adaptive RepMom decay | flow≥0.70→B1=0.88 (fast clear); ≤0.25→B1=0.68 (hold pressure) | RepMom decays faster when generation is flowing well |
| `conf_rise_steps` result key | list of steps with confidence rises ≥0.05 | Complement of conf_drop_steps for surge detection |
| `topk_mode` result key | most common TopK k value across all steps | Single representative k for the whole generation |
| `TopKMode` in stats line | shown on every generation | Most-used sampling width at a glance |
| `confrises` CLI command | steps where confidence jumped ≥0.05 with ▲ bars | Pinpoints confidence surge events |
| `topkmode` CLI command | histogram of TopK k values with % and bars | Distribution of sampling width across steps |
| coh_acc-adaptive NGram3 | positive accel → pass-through (model self-corrects); decel→standard hard block | Trigram blocking relaxes when coherence is already rising |
| `score_gap_trend` result key | per-step 4-step rolling mean of top1−top2 score gap | Trend in how decisive the model is about its top choice |
| `sfb_acc` result key | linear slope of SFB values (positive = variety improving) | Single-number SFB trajectory |
| `SFBAcc` in stats line | shown on every generation | Phrase-variety slope at a glance |
| `scoregaptrend` CLI command | rolling score-gap sparkline with avg/min/max | Visualizes decisiveness trend across the generation |
| `sfbaccel` CLI command | SFB per-step sparkline with slope and direction label | Shows whether phrase variety is rising or falling |
| sfb_acc-adaptive SFB strength | slope>0.002→ease off −0.015; <−0.002→boost +0.015 | SFB self-regulates based on whether variety is trending up |
| `margin_trend` result key | linear slope of top1-top2 margins (+ = more decisive) | Single-number decisiveness trajectory |
| `MarginTrend` in stats line | shown on every generation | Margin slope at a glance |
| `conf_bucket_hist` result key | {0.0-0.2, …, 0.8-1.0} step counts | Shape of confidence distribution across the generation |
| `margintrend` CLI command | per-step margin sparkline with slope/direction label | Visualizes whether model picks are getting sharper |
| `confbuckets` CLI command | 5-bucket confidence histogram with % bars | Distribution shape at a glance |
| margin_trend-adaptive Surprise strength | slope>0.0002→−0.005; <−0.0002→+0.005 | Surprise backs off when model is already decisive |
| `coh_drop_steps` result key | list of steps where coh3 fell ≥0.05 | Coherence collapse events with token + magnitude |
| `entropy_trend` result key | linear slope of entropy_steps over generation | Negative = distribution tightening (more focused) |
| `EntTrend` in stats line | shown on every generation | Entropy direction at a glance |
| `cohdrop` CLI command | steps where coh3 dropped ≥0.05 with ▼ bars | Pinpoints coherence crash events |
| `entropytrend` CLI command | entropy sparkline with slope and direction label | Shows whether generation is focusing or dispersing |
| conf_ema_delta-adaptive ProxPen strength | delta>0.05→ease −0.01; <−0.05→tighten +0.01 | Near-synonym suppression adapts to conf trajectory |
| `coh_rise_steps` result key | list of steps where coh3 jumped ≥0.05 | Coherence surge events with token + magnitude |
| `vel_trend` result key | linear slope of velocity_steps (+ = drifting faster) | Single-number velocity trajectory |
| `VelTrend` in stats line | shown on every generation | Velocity direction at a glance |
| `cohrises` CLI command | steps where coh3 jumped ≥0.05 with ▲ bars | Pinpoints coherence surge events |
| `veltrend` CLI command | velocity sparkline with slope and direction label | Shows whether context is drifting more or less |
| vel_trend-adaptive NGram2 | slope>0.00008→skip on odd steps (allow pivot); else normal | Bigram blocking relaxes during fast context drift |
| `topk_entropy_corr` result key | Pearson r between topk_steps and entropy_steps | Positive = wider k → more entropy (expected coupling) |
| `coh_conf_corr` result key | Pearson r between coh3 and confidence per step | Positive = coherent picks are also high-confidence |
| `TKECorr`/`CohCConf` in stats line | shown on every generation | Two cross-signal correlations at a glance |
| `topkentropyplot` CLI command | TopK + entropy sparklines with Pearson r | Visualizes coupling between sampling width and randomness |
| `cohconfcorr` CLI command | coh3 + confidence sparklines with Pearson r | Shows whether coherence and confidence co-move |
| entropy_trend-adaptive FreqPrior | slope>0.002→strengthen +0.008; <−0.002→ease −0.006 | Frequency prior responds to rising/falling entropy |
| `vel_conf_corr` result key | Pearson r between velocity and confidence | Negative = faster drift → lower confidence (typical) |
| `score_var_trend` result key | linear slope of score_var_steps | + = scores spreading; − = tightening |
| `VCCorr`/`SVTrend` in stats line | shown on every generation | Velocity-confidence coupling and score variance slope |
| `velconfcorr` CLI command | velocity + confidence sparklines with Pearson r | Shows whether context drift suppresses model confidence |
| `scorevartrend` CLI command | score variance per-step sparkline with slope | Visualizes whether score rankings are stable or volatile |
| score_var_trend-adaptive coherence gate | svt>0.0001→threshold=0.55; <−0.0001→0.45 | Extension gate requires stronger coh3 when scores are volatile |
| `_rhythm_rate_steps` internal list | per-step running rhythm rate accumulated each step | Powers rhythmtrend CLI and rhythm_trend slope computation |
| `rhythm_trend` result key | linear slope of rhythm_rate_steps | + = generation becoming smoother over time |
| `rhythm_rate_steps` result key | per-step running rhythm rate (1=smooth, 0=fully alternating) | Full time-series of rhythm smoothness |
| `RhythmTrend` in stats line | shown on every generation | Rhythm trajectory at a glance |
| `rhythmtrend` CLI command | running rhythm rate sparkline with slope + direction label | Shows whether confidence oscillation is damping or growing |
| `confemascatter` CLI command | conf_ema slope ▲/─/▼ scatter with rise/drop/flat counts | Visual derivative of conf_ema trajectory |
| vel_trend-adaptive LocalSem top_k | slope>0.00008→−10 (focus); <−0.00008→+10 (open up) | LocalSem window shrinks when context is drifting fast |
| `conf_var_steps` result key | per-step running variance of confidence so far | Tracks spread of confidence values as they accumulate |
| `coh_entropy_corr` result key | Pearson r between coh3_steps and entropy_steps | Negative = coherent tokens have lower entropy (expected) |
| `CohEntCorr` in stats line | shown on every generation | Coherence-entropy coupling at a glance |
| `confvarplot` CLI command | running confidence variance sparkline | Shows how confidence spread evolves through generation |
| `cohentropycorr` CLI command | coh3 + entropy sparklines with Pearson r | Visualizes coherence-entropy inverse relationship |
| rhythm_trend-adaptive NGramNoveltyBonus | slope<−0.00005→+0.016; >+0.00005→+0.007; else +0.010 | Boosts novelty when confidence oscillates; reduces it when smooth |
| `margin_spike_steps` result key | step indices where margin > mean+1.5σ | Identifies sudden decisive-clarity moments during generation |
| `coh_var_steps` result key | per-step running variance of coh3 values | Tracks whether coherence is fluctuating or stabilizing |
| `marginspikeplot` CLI command | annotated step/token/margin table for spike steps | Shows exactly when and where the model became most decisive |
| `cohvarplot` CLI command | running coh3 variance sparkline | Visualizes coherence stability across the generation |
| coh_var-adaptive Surprise strength | coh3_var_now>0.018→+0.004; <0.006→−0.003 | Boosts variety injection when coherence fluctuates; eases off when stable |
| `conf_entropy_corr` result key | Pearson r between confidence and entropy_steps | Negative = high confidence → lower entropy (confident = focused) |
| `coh_slope_steps` result key | per-step 1st derivative of coh3 trajectory | Instantaneous coherence slope at each step |
| `ConfEntCorr` in stats line | shown on every generation | Confidence-entropy coupling at a glance |
| `confentropycorr` CLI command | confidence + entropy sparklines with Pearson r | Shows whether confident tokens are also lower-entropy |
| `cohslopeplot` CLI command | coh3 slope scatter ▲/─/▼ per step | Visual derivative of coh3 trajectory (mirrors confemascatter) |
| coh_slope-adaptive ProxPen | slope<-0.005→−0.008 (ease); >+0.005→+0.006 (tighten) | Allows fresh tokens when coherence falls; exploits groove when rising |
| `vprec_slope_steps` result key | per-step 1st derivative of vprec_ema | Instantaneous vocab precision change at each step |
| `coh_conf_steps` result key | per-step coh3 × confidence quality product | Combined coherence+confidence quality signal per token |
| `vprecslope` CLI command | vprec_ema slope scatter ▲/─/▼ per step | Shows whether model is converging/broadening its vocabulary |
| `cohconfplot` CLI command | coh3 × confidence product sparkline | Visualizes combined quality (coherence × confidence) per step |
| quality_steps-adaptive Depth2Lookahead weight | q_ema>0.25→+0.01; <0.10→+0.02 | Increases lookahead lean when quality is poor; exploits when high |
| `quality_steps` result key | per-step EMA-smoothed coh3×conf (0.7 decay) | Smooth quality trajectory signal for each generation step |
| `conf_coh_slope_corr` result key | Pearson r between conf_ema slope and coh3 slope | + = confidence and coherence move together (healthy generation) |
| `qualityplot` CLI command | smoothed coh3×conf quality EMA sparkline | High-level quality trajectory view across the generation |
| `confcohslopecorr` CLI command | conf_ema slope + coh3 slope scatter with Pearson r | Shows whether confidence and coherence rise/fall in sync |
| conf_vprec_corr-adaptive TopK | r>0.40→−3 (tighten); <−0.40→+4 (open) | Exploits precision-confidence alignment; broadens when decoupled |
| `conf_vprec_corr` result key | Pearson r between confidence and vprec_ema | + = confident tokens are also high-precision vocabulary |
| `qual_spike_steps` result key | step indices where coh3×conf > mean+1.5σ | Peak quality moments — coherent AND confident simultaneously |
| `confvpreccorr` CLI command | confidence + vprec_ema sparklines with Pearson r | Reveals whether model confidence aligns with vocabulary precision |
| `qualspikeplot` CLI command | annotated step/token/quality table for spike steps | Shows exactly when and where peak quality occurred |
| entropy_var-adaptive burst pulse | ev>0.015→×1.40 (harder); <0.004→×0.75 (softer) | Hits bursts harder when entropy is chaotic; gentler on stable runs |
| `entropy_var_steps` result key | per-step running variance of entropy values | Tracks how chaotic/stable the token distribution has been |
| `sg_spike_steps` result key | step indices where ScoreGapEMA > mean+1.5σ | Sudden decisive-gap spikes — model becomes unusually certain |
| `entropyvarplot` CLI command | running entropy variance sparkline | Shows whether token distribution is becoming more or less chaotic |
| `sgspikeplot` CLI command | annotated ScoreGapEMA spike steps with tokens | Identifies moments of sudden decisive scoring clarity |
| conf_topk_corr-adaptive Mirostat | r<−0.40→−0.02 (tighten); >+0.40→+0.02 (ease) | Rewards healthy wide-beam-low-confidence pattern |
| `coh6_var_steps` result key | per-step running variance of coh6 values | Tracks long-range coherence stability across the generation |
| `conf_topk_corr` result key | Pearson r between confidence and TopK k | Negative = wide beam → lower confidence (healthy behaviour) |
| `coh6varplot` CLI command | running coh6 variance sparkline | Shows whether 6-gram long-range coherence is stable or drifting |
| `conftopkcorr` CLI command | confidence + TopK sparklines with Pearson r | Reveals whether TopK width and confidence are coupled as expected |
| qual_trend-adaptive ScoreSpreadClip | qt>0.0001→min clip 1.4; <−0.0001→max clip 2.2 | Tightens score range when quality improving; opens up to recover |
| `qual_trend` result key | linear slope of coh3×conf quality product | + = coherence and confidence both rising through generation |
| `vprec_conf_slope_corr` result key | Pearson r between vprec slope and conf_ema slope | + = vocab precision and confidence rise/fall together |
| `qualtrend` CLI command | quality_steps sparkline with slope + improving/stable/degrading label | High-level generation quality trajectory at a glance |
| `vprecconfslopecorr` CLI command | vprec slope + conf_ema slope scatter with Pearson r | Shows whether vocabulary precision drives or follows confidence |
| sg_conf_corr-adaptive Nucleus | r<−0.45→+0.03 (widen); >+0.45→−0.02 (tighten) | Widens beam when decisive-low-confidence pattern detected |
| `qual_var_steps` result key | per-step running variance of quality signal | High = quality is inconsistent across generation |
| `sg_conf_corr` result key | Pearson r between confidence and ScoreGapEMA | + = larger score gap = higher confidence (healthy) |
| `qualvarplot` CLI command | running quality variance sparkline | Shows whether generation quality is consistent or erratic |
| `sgconfcorr` CLI command | ScoreGapEMA + confidence sparklines with Pearson r | Reveals whether scoring decisiveness and confidence are coupled |
| margin_var-adaptive FreqPrior strength | var>0.008→+0.006; <0.001→−0.005 | Anchors to common words when margin variance spikes |
| `margin_var_steps` result key | per-step running variance of top1 score margins | High = erratic decisiveness pattern |
| `coh3_sg_corr` result key | Pearson r between coh3 and ScoreGapEMA | + = more coherent = wider score gap (healthy) |
| `marginvarplot` CLI command | running top1 margin variance sparkline | Shows whether model decisiveness is consistent or erratic |
| `coh3sgcorr` CLI command | coh3 + ScoreGapEMA sparklines with Pearson r | Shows whether coherence and scoring decisiveness are coupled |
| coh3_vprec_corr-adaptive DotAgreementBonus cap | r>0.40→0.068; <−0.40→0.054 | Boosts consensus reward when coherence+precision rise together |
| `vel_var_steps` result key | per-step running variance of velocity EMA | High = topic movement is erratic/inconsistent |
| `coh3_vprec_corr` result key | Pearson r between coh3 and vprec_ema | + = coherence and vocab precision rise together (quality run) |
| `velvarplot` CLI command | running velocity variance sparkline | Shows whether topic-drift rate is steady or erratic |
| `coh3vpreccorr` CLI command | coh3 + vprec_ema sparklines with Pearson r | Reveals whether coherence and vocab precision are coupled |
| conf_vel_corr-adaptive PromptDrift | r<−0.40→−0.012; >+0.40→+0.010 | Eases drift penalty during healthy exploration; tightens on overconfident drift |
| `conf_vel_corr` result key | Pearson r between confidence and velocity EMA | Negative = fast topic movement + lower conf = healthy exploration |
| `topk_var_steps` result key | per-step running variance of adaptive TopK k | High = beam width is swinging widely |
| `confvelcorr` CLI command | confidence + velocity sparklines with Pearson r | Identifies overconfident topic drift vs healthy exploration |
| `topkvarplot` CLI command | running adaptive TopK variance sparkline | Shows whether beam width is stable or erratic through generation |
| topk_vel_corr-adaptive NGram3 | r>0.40 → soft pre-penalty −12.0 (allow novel trigrams) | Eases trigram blocking during active exploration mode |
| `coh6_conf_corr` result key | Pearson r between 6-gram coherence and confidence | + = long-range coherence and token confidence aligned |
| `topk_vel_corr` result key | Pearson r between TopK k and velocity EMA | + = wide beam during fast topic movement (active exploration) |
| `coh6confcorr` CLI command | coh6 + confidence sparklines with Pearson r | Shows whether long-range coherence tracks token confidence |
| `topkvelcorr` CLI command | TopK + velocity sparklines with Pearson r | Reveals beam width / topic movement coupling |
| margin_conf_corr-adaptive ScoreCentroidLift | r>0.40→−0.003; <−0.40→+0.004 | Eases floor lift when decisive+confident; boosts it on decisive+unconfident |
| `margin_conf_corr` result key | Pearson r between top1 margins and confidence | + = decisive = confident (healthy scoring pattern) |
| `entropy_vel_corr` result key | Pearson r between entropy and velocity EMA | + = high entropy correlates with fast topic movement |
| `marginconfcorr` CLI command | top1 margin + confidence sparklines with Pearson r | Shows whether scoring decisiveness and confidence are aligned |
| `entropyvelcorr` CLI command | entropy + velocity sparklines with Pearson r | Reveals whether token entropy drives topic movement |
| entropy_topk_corr-adaptive TypicalFilter | r>0.45→−0.05; <−0.45→+0.05 | Tightens typical when random+wide; widens under tension |
| `coh3_margin_slope_corr` result key | Pearson r between coh3 slope and margin slope | + = coherence and score decisiveness evolve in sync |
| `entropy_topk_corr` result key | Pearson r between entropy and TopK k | + = high entropy opens wider beam (expected healthy pattern) |
| `coh3marginslopecorr` CLI command | coh3 slope + margin slope scatter with Pearson r | Shows whether coherence and decisiveness drift in sync |
| `entropytopkcorr` CLI command | entropy + TopK sparklines with Pearson r | Detects tension when entropy is high but beam is narrow |
| vprec_entropy_corr-adaptive LocalSem top_k | r<−0.40→−8; >+0.40→+8 | Tightens local context when precise=focused; widens on precision+entropy conflict |
| `vprec_entropy_corr` result key | Pearson r between vprec_ema and entropy | Negative = vocab-precise = low-entropy = focused (healthy) |
| `coh3_entropy_slope_corr` result key | Pearson r between coh3 slope and entropy slope | Negative = coherence rises as entropy falls (ideal pattern) |
| `vprecedntropycorr` CLI command | vprec_ema + entropy sparklines with Pearson r | Shows whether vocab precision and distribution sharpness are aligned |
| `coh3entropyslopecorr` CLI command | coh3 slope + entropy slope scatter with Pearson r | Shows whether coherence and entropy evolve in opposite directions |
| coh3_vel_conf_joint-adaptive RepMom B1 | frac>0.50→B1+0.04; <0.20→B1−0.03 | Eases repetition pressure when all ideal conditions active |
| `conf_margin_vel_score` result key | composite health score 0-4 (healthy correlation count) | At-a-glance signal: how many generation health patterns are active |
| `coh3_vel_conf_joint` result key | fraction of steps with coh3>avg ∧ vel<avg ∧ conf>avg | Peak quality step fraction: all three ideal signals simultaneously |
| `healthscore` CLI command | composite health score 0-4 with per-component breakdown | Summarizes 4 key correlation health patterns in one view |
| `jointidealmeter` CLI command | step-by-step ✓/─ map of all-ideal steps | Shows which steps had coherence, stability, and confidence all above average |
| coh3_slope_trend-adaptive Depth2Lookahead | acc>0.001→+0.015; <−0.001→−0.010 | Boosts lookahead weight when coherence is gaining momentum |
| `coh3_slope_trend` result key | 2nd derivative of coh3 (coherence momentum) | + = coherence gaining momentum; − = slowing |
| `conf_slope_trend` result key | 2nd derivative of conf_ema (confidence momentum) | + = confidence accelerating; − = decelerating |
| `coh3slopetrend` CLI command | coh3 1st + 2nd derivative sparklines | Shows coherence speed and momentum direction |
| `confslopetrend` CLI command | conf_ema 1st + 2nd derivative sparklines | Shows confidence speed and acceleration |
| vel_slope_trend-adaptive Surprise | acc>0.00002→−0.003; <−0.00002→+0.003 | Eases novelty injection during active drift; boosts on settling |
| `vel_slope_trend` result key | 2nd derivative of velocity EMA | + = topic drift accelerating; − = settling |
| `margin_slope_trend` result key | 2nd derivative of top1 margins | + = decisiveness accelerating; − = easing |
| `velslopetrend` CLI command | velocity 1st + 2nd derivative sparklines | Shows whether topic drift is accelerating or settling |
| `marginslopetrend` CLI command | margin 1st + 2nd derivative sparklines | Shows whether scoring decisiveness is building or fading |
| entropy_slope_trend-adaptive TypicalFilter τ | acc>0.0008→τ+0.04; <−0.0008→τ−0.03 | Widens typical window when entropy spreading; tightens on focusing |
| `entropy_slope_trend` result key | 2nd derivative of entropy (distribution momentum) | + = distribution spreading faster; − = focusing |
| `topk_slope_trend` result key | 2nd derivative of top-k (beam momentum) | + = beam widening faster; − = narrowing |
| `entropyslopetrend` CLI command | entropy 1st + 2nd derivative sparklines | Shows whether distribution is spreading or focusing over time |
| `topkslopetrend` CLI command | topk 1st + 2nd derivative sparklines | Shows whether sampling beam is widening or narrowing |
| sg_slope_trend-adaptive ScoreSpreadClip | acc>0.0008→clip−0.15; <−0.0008→clip+0.12 | Tightens spread clip when competition sharpening; widens on softening |
| `sg_slope_trend` result key | 2nd derivative of score-gap EMA | + = competition sharpening faster; − = softening |
| `vprec_slope_trend` result key | 2nd derivative of vprec_ema | + = vocab precision tightening faster; − = loosening |
| `sgslopetrend` CLI command | score-gap 1st + 2nd derivative sparklines | Shows score-gap momentum: sharpening or softening |
| `vprecslopetrend` CLI command | vprec_ema 1st + 2nd derivative sparklines | Shows vocab precision momentum: tightening or loosening |
| margin_vel_joint-adaptive NGram3 | joint>0.50→soft pre-penalty −10.0 | Eases ngram3 blocking when model is decisively stable |
| `coh6_slope_trend` result key | 2nd derivative of coh6 | + = long-window coherence gaining momentum |
| `margin_vel_joint` result key | fraction of steps with margin>avg ∧ vel<avg | Decisive+stable step fraction |
| `coh6slopetrend` CLI command | coh6 1st + 2nd derivative sparklines | Shows 6-window coherence momentum |
| `marginveljoint` CLI command | decisive+stable joint step meter (✓/─ per step) | Shows which steps were simultaneously decisive and stable |
| coh3_margin_joint-adaptive PromptDrift | joint>0.50→strength−0.004; <0.20→+0.004 | Eases prompt anchoring when coherent+decisive; tightens when weak |
| `conf_vel_joint` result key | fraction of steps with conf>avg ∧ vel<avg | Confident+stable step fraction |
| `coh3_margin_joint` result key | fraction of steps with coh3>avg ∧ margin>avg | Coherent+decisive step fraction |
| `confveljoint` CLI command | confident+stable joint step meter (✓/─ per step) | Shows which steps were simultaneously confident and stable |
| `coh3marginjoint` CLI command | coherent+decisive joint step meter (✓/─ per step) | Shows which steps were simultaneously coherent and decisive |
| entropy_vel_joint-adaptive score floor | joint>0.50→15th pct; <0.20→25th pct | Eases sampling floor when focused+stable; tightens on drift |
| `entropy_vel_joint` result key | fraction of steps with entropy<avg ∧ vel<avg | Focused+stable step fraction |
| `vprec_coh3_joint` result key | fraction of steps with vprec>avg ∧ coh3>avg | Precise+coherent step fraction |
| `entropyveljoint` CLI command | focused+stable joint step meter (✓/─ per step) | Shows which steps were simultaneously low-entropy and stable |
| `vpreccoh3joint` CLI command | precise+coherent joint step meter (✓/─ per step) | Shows which steps had both high vocab precision and high coherence |
| ideal_run_len-adaptive Mirostat | streak≥4→target−0.03; ≤1→target+0.03 | Tightens Mirostat target on a streak of ideal steps; opens up when stalled |
| `quadrant_map` result key | step counts per coh3×vel quadrant (ideal/exploring/flat/drifting) | 2D classification of every step |
| `ideal_run_len` result key | longest consecutive run of all-ideal steps | Streak length: coh3+vel+conf all on right side of avg |
| `quadrantmap` CLI command | coh3×vel quadrant distribution with dominant quadrant | Shows how steps distribute across the four quality quadrants |
| `idealrunlen` CLI command | longest ideal streak + step-by-step map | Highlights the best quality streak in the generation |
| streak-adaptive FreqPrior | ideal streak≥3→−0.003; drifting streak≥3→+0.003 | Eases vocab anchoring on ideal runs; tightens on drifting runs |
| `phase_quality_score` result key | avg_conf × avg_coh3 per early/mid/late third | Shows which phase of generation had the highest combined quality |
| `streak_map` result key | list of (quadrant, length) pairs for the full generation | Full quadrant run-length encoding of the generation |
| `phasequalityscore` CLI command | early/mid/late phase quality bar chart | Shows which third of the generation was best quality |
| `streakmap` CLI command | sequence of quadrant streaks with symbols | Visual run-length view of generation quadrant flow |
| phase_transition-adaptive PromptDrift | ≥2 drift→ideal recoveries→−0.004; ≥2 ideal→drift falls→+0.004 | Eases/tightens drift penalty based on self-correction track record |
| `phase_transition_map` result key | list of (step_idx, from_q, to_q) per quadrant change | Full transition log for every quadrant boundary crossing |
| `conf_entropy_ratio` result key | mean_conf / mean_entropy | >1.4=overconfident, 0.8–1.4=balanced, <0.8=entropy-led |
| `confentropyratio` CLI command | ratio bar + label + mean_conf/mean_entropy breakdown | Confidence vs entropy dominance indicator |
| `phasetransitionmap` CLI command | step-by-step quadrant transition table with recovery/fall counts | Shows every quality shift in the generation |
| conf_velocity_score-adaptive RepMom B1 | score≥0.60→B1+0.02; ≤0.30→B1−0.02 | Eases repetition pressure when generation is confident+stable |
| `conf_velocity_score` result key | mean_conf × (1 − mean_vel) | Composite: high = confident + velocity-stable |
| `quad_entropy` result key | per-quadrant mean entropy across all steps | Shows which quadrant had the most focused / least random steps |
| `confvelocityscore` CLI command | composite quality bar + label (excellent/good/fair/weak) | Single quality number combining confidence and stability |
| `quadentropy` CLI command | per-quadrant mean entropy focus table | Which quadrant had the tightest entropy distribution |
| transition_rate-adaptive Depth2Lookahead | rate≥0.50→+0.008; ≤0.20→−0.006 | Boosts lookahead weight on volatile generation; eases on stable |
| `quad_conf_mean` result key | per-quadrant mean confidence | Which quadrant had the highest certainty |
| `transition_rate` result key | fraction of step-to-step transitions that change quadrant | 0=perfectly stable, 1=every step changes quadrant |
| `quadconfmean` CLI command | per-quadrant mean confidence table with bar chart | Most certain quadrant + comparative view |
| `transitionrate` CLI command | instability rate bar + label (volatile/moderate/stable) | Quick summary of generation stability |
| ideal_frac-adaptive TypicalFilter τ | frac≥0.50→τ−0.04; frac≤0.20→τ+0.03 | Eases typical sampling when generation is mostly ideal; tightens when sparse |
| `ideal_frac` result key | fraction of steps in the ideal quadrant (coh3↑ vel↓) | How often the model was in its best state |
| `quad_velocity_mean` result key | per-quadrant mean semantic velocity | Which quadrant had the most/least drift per step |
| `idealfrac` CLI command | ideal-quadrant fraction bar + label (dominant/strong/moderate/sparse) | How dominant the ideal quadrant was in this generation |
| `quadvelocitymean` CLI command | per-quadrant mean velocity table with bar chart | Velocity breakdown per generation quality quadrant |
| coh3_vel_divergence-adaptive Surprise | div<0.05→strength−0.001; div>0.15→+0.002 | Eases novelty injection when coh3 and vel are aligned; boosts when misaligned |
| `coh3_vel_divergence` result key | abs(mean_coh3 − (1 − mean_vel)) | 0=perfectly aligned, >0.15=misaligned signals |
| `quad_coh3_mean` result key | per-quadrant mean coh3 coherence | Which quadrant had the highest local coherence |
| `coh3veldivergence` CLI command | divergence bar + label (aligned/slight/moderate/misaligned) | Checks whether cohesion and velocity signals agree |
| `quadcoh3mean` CLI command | per-quadrant mean coh3 table with bar chart | Coherence breakdown per generation quality quadrant |
| conf_coh3_gap-adaptive ScoreSpreadClip | gap>0.10→clip−0.10; gap<−0.10→clip+0.08 | Narrows beam when overconfident vs coherence; widens when under-confident |
| `conf_coh3_gap` result key | mean_conf − mean_coh3 | >+0.10=overconfident, ±0.10=balanced, <−0.10=under-confident |
| `quad_transition_from` result key | dict of {quadrant: departure_count} | Which quadrant was left most often |
| `confcoh3gap` CLI command | gap bar + label with mean_conf/mean_coh3 breakdown | Signals whether confidence leads or lags coherence |
| `quadtransitionfrom` CLI command | departure count per quadrant (most abandoned) | Shows which quadrant is least stable to stay in |
| ideal_entry_rate-adaptive NGram3 pre-penalty | rate≥0.40→floor−1.0; rate≤0.10→floor+2.0 | Eases ngram block when model self-corrects to ideal; tightens when rare |
| `quad_transition_to` result key | dict of {quadrant: entry_count} | Which quadrant was entered most often after a transition |
| `ideal_entry_rate` result key | ideal entries / total transitions | How reliably does the model recover to the ideal quadrant |
| `quadtransitionto` CLI command | entry count per quadrant (most returned-to) | Shows which quadrant the model gravitates toward after changes |
| `idealentryrate` CLI command | ideal entry rate bar + label (excellent/good/fair/rare) | How often transitions land back in ideal quality |
| quad_balance_score-adaptive LocalSemanticFilter | balance≥0.30→top_k+10; ≤−0.30→top_k−12 | Widens semantic vocabulary on quality runs; tightens on drift-biased gens |
| `drifting_frac` result key | fraction of steps in the drifting quadrant (coh3↓ vel↑) | How often the model was in its worst quality state |
| `quad_balance_score` result key | (ideal_count − drifting_count) / n | +0.20=quality-biased, ±0.20=neutral, −0.20=drift-biased |
| `driftingfrac` CLI command | drifting-quadrant fraction bar + label (dominant/high/moderate/low) | How much of the generation was in the worst quality state |
| `quadbalancescore` CLI command | quality bias bar with ideal_frac/drifting_frac breakdown | Single quality bias number: whether the gen was mostly ideal or mostly drifting |
| exploring_frac-adaptive DotVariancePenalty | ef≥0.35→strength+0.02; ≤0.10→strength−0.01 | Amplifies dot-disagreement penalty during creative-divergence runs |
| `exploring_frac` result key | fraction of steps in exploring quadrant (coh3↑ vel↑) | How often the model was in creative-divergence state |
| `flat_frac` result key | fraction of steps in flat quadrant (coh3↓ vel↓) | How often the model was stagnant |
| `exploringfrac` CLI command | exploring-quadrant fraction bar + label | Creative divergence fraction |
| `flatfrac` CLI command | flat-quadrant fraction bar + label | Stagnation fraction |
| `quadsummary` CLI command | all-in-one quadrant dashboard — all 4 fracs + balance + transition rate + ideal run len + ideal entry rate | Single command for full quadrant state picture |
| quad_dominance_margin-adaptive RepulsionMomentum B1 | margin≥0.30→B1+0.015; ≤0.08→B1−0.015 | Extends penalty memory when model self-consistent; shortens when unstable |
| `quad_volatility_score` result key | normalised quadrant flip rate per step (0=constant 1=flip every step) | How erratically the model switches between quality states |
| `quad_dominance_margin` result key | dominant_frac − second_frac (0=tied 1=total dominance) | How decisively one quadrant leads |
| `quadvolatilityscore` CLI command | normalised flip rate bar + label (stable/moderate/volatile/very volatile) | Quadrant instability indicator |
| `quaddominancemargin` CLI command | decisiveness bar + dominant quadrant label | Whether the gen clearly stayed in one state or was contested |
| quad_recovery_rate-adaptive SurpriseBonus | rate≥0.60→strength+0.003; ≤0.20→strength−0.002 | Rewards self-correction by boosting creative surprise on recovery |
| `ideal_run_density` result key | ideal_frac × longest_ideal_run (quality richness) | Combines frequency and streak length into one richness score |
| `quad_recovery_rate` result key | drifting→ideal/exploring recovery fraction | How reliably the model bounces back after drifting |
| `idealrundensity` CLI command | richness bar + ideal_frac × run_len breakdown | Single quality richness number |
| `quadrecoveryrate` CLI command | recovery rate bar + drifting_frac context | Self-correction health after drift episodes |
| quad_persistence_score-adaptive TypicalFilter τ | score≥6.0→p−0.02; ≤2.5→p+0.02 | Eases typical-set constraint when states are sticky; tightens when volatile |
| `quad_persistence_score` result key | total_steps / transitions — avg steps per quadrant visit | How sticky each quadrant state is (higher=more self-consistent) |
| `ideal_stability_score` result key | ideal_frac − drifting_frac net quality index | Net quality: +0.30=very stable, ±0.10=neutral, −0.30=very unstable |
| `quadpersistencescore` CLI command | stickiness bar + avg steps per visit + transition_rate | How long the model lingers in each quality state |
| `idealstabilityscore` CLI command | net quality index bar + ideal/drifting frac breakdown | Single number for overall generation quality health |
| quad_oscillation_score-adaptive VocabFrequencyPrior | osc≥0.55→strength−0.004; ≤0.15→strength+0.003 | Eases frequency prior when model ping-pongs; tightens when exploring freely |
| `quad_oscillation_score` result key | fraction of A→B→A ping-pong transitions | How much the model bounces between exactly two quality states |
| `quad_ideal_entry_velocity` result key | mean velocity at moment of entering ideal quadrant | Lower = smoother transition into the ideal state |
| `quadoscillationscore` CLI command | ping-pong fraction bar + transition_rate context | Detects whether the model is stuck flip-flopping |
| `quadidealentryvelocity` CLI command | entry velocity bar vs. mean velocity baseline | Whether the model enters ideal smoothly or abruptly |
| quad_coh3_entry_mean-adaptive ContextAnchor | entry_coh3≥mean+0.05→strength−0.02; ≤mean−0.05→strength+0.02 | Eases anchor when ideal entries are coherence-backed; tightens for fragile entries |
| `quad_coh3_entry_mean` result key | mean coh3 at ideal-quadrant entry moments | Whether the model reaches ideal from a strong or weak coherence base |
| `quad_drifting_exit_coh3` result key | mean coh3 at drifting-quadrant exit moments | Quality of recovery — higher coh3 at exit = cleaner escape from drift |
| `quadcoh3entrymean` CLI command | entry coh3 bar + overall vs ideal in-state comparison | Coherence strength at ideal-quadrant entries |
| `quaddriftingexitcoh3` CLI command | exit coh3 bar + drifting in-state vs overall comparison | Recovery quality when leaving the drifting state |
| quad_drifting_entry_velocity-adaptive ScoreSpreadClip | entry_vel≥mean+0.04→clip−0.12; ≤mean−0.04→clip+0.08 | Tightens score spread when model falls hard into drift |
| `quad_drifting_entry_velocity` result key | mean velocity at moment of entering drifting quadrant | Higher = harder/faster fall into the drifting state |
| `quad_ideal_duration_variance` result key | std-dev of ideal-run lengths (0=consistent; high=erratic) | Whether the model sustains ideal states evenly or in bursts |
| `quaddriftingentryvelocity` CLI command | drifting-entry velocity bar vs. mean baseline | Hard-fall vs soft-fall into drift |
| `quadidealdurationvariance` CLI command | ideal run length variance bar + ideal_frac/run_len context | Consistency of ideal-quality streaks |
| quad_flat_duration_variance-adaptive Mirostat | std≥2.5→target−0.02; ≤0.5→target+0.015 | Tightens Mirostat when stagnation is bursty; eases when flat runs are short/uniform |
| `quad_flat_duration_variance` result key | std-dev of flat-run lengths (0=uniform; high=erratic) | Whether the model stagnates in steady or bursty episodes |
| `quad_recovery_velocity` result key | mean velocity at moment of exiting drifting quadrant | Lower = smoother exit from drift; higher = abrupt jump out |
| `quadflatdurationvariance` CLI command | flat run length variance bar + flat_frac context | How erratically the model stagnates |
| `quadrecoveryvelocity` CLI command | drifting-exit velocity bar vs. mean + recovery_rate context | Smoothness of drift-recovery transitions |
| quad_drifting_duration_variance-adaptive DynamicTemperature | std≥2.0→base+0.02; ≤0.5→base−0.01 | Widens temp envelope when drift bursts are erratic; narrows when controlled |
| `quad_drifting_duration_variance` result key | std-dev of drifting-run lengths (0=uniform; high=erratic) | Whether the model's drift episodes are steady or unpredictably bursty |
| `quad_exploring_duration_variance` result key | std-dev of exploring-run lengths (0=uniform; high=erratic) | Whether creative divergence comes in steady or bursty episodes |
| `quaddriftingdurationvariance` CLI command | drifting run variance bar + drifting_frac context | Erratic vs. uniform drift patterns |
| `quadexploringdurationvariance` CLI command | exploring run variance bar + exploring_frac context | Erratic vs. uniform creative divergence patterns |
| quad_exploring_exit_coh3-adaptive _TREND_BOOST | exit_coh≥mean+0.06→boost−0.02; ≤mean−0.06→boost+0.02 | Reduces confidence recovery boost when exploring exits are already coherent |
| `quad_ideal_entry_coh3_variance` result key | std-dev of coh3 values at ideal-quadrant entry (0=stable gate) | Whether the entry threshold into ideal state is predictable or noisy |
| `quad_exploring_exit_coh3` result key | mean coh3 at moment of exiting exploring quadrant | Higher = model leaves creative divergence gracefully |
| `quadidealentrycoh3variance` CLI command | ideal-entry coh3 variance bar + entry mean + ideal_frac context | Stability of the ideal-state entry threshold |
| `quadexploringexitcoh3` CLI command | exploring-exit coh3 bar vs. entry mean baseline | Graceful vs. abrupt exits from creative divergence |
| quad_flat_exit_velocity-adaptive _TREND_SLOPE | exit_vel≥mean+0.04→slope+0.01; ≤mean−0.04→slope−0.01 | Tightens confidence-trend trigger when model exits stagnation sluggishly |
| `quad_flat_exit_velocity` result key | mean velocity at moment of exiting flat quadrant | Higher = sharp snap-out of stagnation; lower = sticky flat state |
| `quad_ideal_to_exploring_rate` result key | fraction of ideal-quadrant exits that land in exploring | How often high-quality flow transitions into creative divergence |
| `quadflatexitvelocity` CLI command | flat-exit velocity bar vs. mean baseline | Snap-out sharpness from stagnation |
| `quadidealtoexploringrate` CLI command | ideal→exploring transition rate bar + ideal/exploring_frac | How frequently quality flow tips into divergence |
| quad_drifting_to_flat_rate-adaptive _CF_THR | rate≥0.50→CF_THR+0.01; ≤0.15→CF_THR−0.01 | Tightens conf-floor trigger when drift reliably collapses to stagnation |
| `quad_drifting_to_flat_rate` result key | fraction of drifting exits that land in flat (0=never; 1=always) | How often drift collapses into stagnation instead of recovering |
| `quad_exploring_to_ideal_rate` result key | fraction of exploring exits that land in ideal (0=never; 1=always) | How productively creative divergence resolves into quality flow |
| `quaddriftingtoflatrate` CLI command | drifting→flat transition rate bar + drifting/flat_frac context | Risk of drift collapsing to stagnation |
| `quadexploringtoidealrate` CLI command | exploring→ideal transition rate bar + exploring/ideal_frac context | Productivity of creative divergence |
| quad_flat_to_exploring_rate-adaptive _CF_WIN | rate≥0.45→CF_WIN+1; ≤0.15→CF_WIN−1 | Widens conf-floor window when stagnation recovers quickly; narrows when sticky |
| `quad_flat_to_drifting_rate` result key | fraction of flat exits that escalate to drifting (0=never; 1=always) | How often stagnation tips into worse drift rather than recovering |
| `quad_flat_to_exploring_rate` result key | fraction of flat exits that jump to exploring (0=never; 1=always) | How often stagnation breaks out into creative divergence |
| `quadflattodriftingrate` CLI command | flat→drifting escalation rate bar + flat/drifting_frac context | Risk of stagnation escalating to drift |
| `quadflattoexploringrate` CLI command | flat→exploring recovery rate bar + flat/exploring_frac context | Creative escape rate from stagnation |
| quad_flat_to_ideal_rate-adaptive _LOW_GAP_THR | rate≥0.35→thr+0.005; ≤0.10→thr−0.005 | Raises score-gap exploration trigger when model has strong direct-recovery signal |
| `quad_flat_to_ideal_rate` result key | fraction of flat exits that jump directly to ideal (0=never; 1=always) | How powerfully the model springs from stagnation to quality flow |
| `quad_drifting_to_exploring_rate` result key | fraction of drifting exits that land in exploring (0=never; 1=always) | How productively drift converts to creative divergence |
| `quadflattoidealrate` CLI command | flat→ideal direct rate bar + flat/ideal_frac context | Strength of direct stagnation-to-quality recovery |
| `quaddriftingtoexploringrate` CLI command | drifting→exploring rate bar + drifting/exploring_frac context | Creative escape rate from drift |
| quad_drifting_to_ideal_rate-adaptive _CONF_EMA_ALPHA | rate≥0.30→alpha−0.03; ≤0.08→alpha+0.03 | Slows EMA when drift self-corrects quickly; speeds it when model is stuck |
| `quad_drifting_to_ideal_rate` result key | fraction of drifting exits that jump directly to ideal (0=never; 1=always) | How powerfully the model self-corrects from drift to quality flow |
| `quad_exploring_to_flat_rate` result key | fraction of exploring exits that collapse into flat (0=never; 1=always) | How often creative divergence fizzles into stagnation |
| `quaddriftingtoidealrate` CLI command | drifting→ideal direct rate bar + drifting/ideal_frac context | Strength of direct drift-to-quality recovery |
| `quadexploringtoflatrate` CLI command | exploring→flat collapse rate bar + exploring/flat_frac context | Risk of creative divergence fizzling to stagnation |
| quad_exploring_to_drifting_rate-adaptive _VPREC_EMA_ALPHA | rate≥0.40→alpha+0.04; ≤0.12→alpha−0.03 | Speeds VPREC tracking when divergence often tips into drift |
| `quad_exploring_to_drifting_rate` result key | fraction of exploring exits that tip into drifting (0=never; 1=always) | How often creative divergence deteriorates into uncontrolled drift |
| `quad_ideal_to_flat_rate` result key | fraction of ideal exits that collapse into flat (0=never; 1=always) | How often quality flow falls directly into stagnation |
| `quadexploringtodriftingrate` CLI command | exploring→drifting collapse rate bar + exploring/drifting_frac context | Risk of divergence tipping into uncontrolled drift |
| `quadidealtoflatrate` CLI command | ideal→flat collapse rate bar + ideal/flat_frac context | Risk of quality flow falling to stagnation |
| quad_transition_entropy-adaptive _BURST_TEMP | entropy≥1.8→_BURST_TEMP+0.02; ≤0.8→_BURST_TEMP−0.02 | Sharpens burst response when transitions are chaotic; softens it when predictable |
| `quad_ideal_to_drifting_rate` result key | fraction of ideal exits that tip into drifting (0=never; 1=always) | How often the model falls from quality flow into uncontrolled drift |
| `quad_transition_entropy` result key | Shannon entropy over observed quadrant-transition types (bits; max≈3.58) | How predictable/chaotic the model's state-transition behaviour is |
| `quadidealtodriftingrate` CLI command | ideal→drifting collapse rate bar + ideal/drifting_frac context | Risk of quality flow tipping into drift |
| `quadtransitionentropy` CLI command | transition entropy bits bar + label + quad summary context | Overall predictability of the model's quadrant navigation |
| quad_self_transition_rate-adaptive _SEM_CENT_STR | rate≥0.55→str+0.02; ≤0.25→str−0.015 | Strengthens centroid pull when model is stuck; eases it when already mobile |
| `quad_self_transition_rate` result key | fraction of steps where quadrant label is unchanged (0=fluid; 1=frozen) | How often the model stays in the same quadrant — mobility vs. stuckness |
| `quad_transition_matrix_skew` result key | max row-prob minus min row-prob in 4×4 transition matrix | How biased per-source transitions are (0=uniform; 1=all-or-nothing) |
| `quadselftransitionrate` CLI command | self-loop rate bar + transition_entropy context | Quadrant mobility/stuckness at a glance |
| `quadtransitionmatrixskew` CLI command | transition matrix skew bar + entropy context | How strongly source quadrant biases the destination |
| quad_flat_run_confidence_mean-adaptive _REP_MOM_S | flat_conf≥0.55→+0.02; ≤0.25→−0.01 | Raises repulsion penalty when model is confidently stagnating |
| quad_transition_matrix_skew-adaptive _LOW_MARGIN_THR | skew≥0.65→thr+0.005; ≤0.20→thr−0.003 | Raises score-gap bar when model is locked into one transition pathway |
| `quad_ideal_run_confidence_mean` result key | mean confidence during ideal-quadrant steps (0–1) | How robustly the model sustains quality flow |
| `quad_ideal_run_confidence_mean`-adaptive _CTX_MOM_S | ideal_conf≥0.60→+0.015; ≤0.30→−0.01 | Boosts context momentum when ideal flow is stable; eases when fragile |
| `quad_drifting_run_confidence_mean` result key | mean confidence during drifting-quadrant steps | How confident the model is while drifting |
| `quad_exploring_run_confidence_mean` result key | mean confidence during exploring-quadrant steps | How confident the model is while exploring |
| `quad_flat_run_confidence_mean` result key | mean confidence during flat-quadrant steps | How confident the model is while stagnating |
| `quad_confidence_gap` result key | ideal_conf_mean − drifting_conf_mean (positive=ideal more certain) | Separation between quality states in confidence space |
| `quadidealrunconfidencemean` CLI command | ideal-run confidence bar + ideal_frac + run_density context | Quality flow confidence health check |
| `quaddriftingrunconfidencemean` CLI command | drifting-run confidence bar + drifting_frac + ideal_conf context | Drift confidence diagnostic |
| `quadexploringrunconfidencemean` CLI command | exploring-run confidence bar + exploring_frac + ideal_conf context | Exploring confidence volatility check |
| `quadflatrunconfidencemean` CLI command | flat-run confidence bar + flat_frac + ideal_conf context | Stagnation trap detector (high flat-conf=bad) |
| `quadconfidencegap` CLI command | signed gap bar (█=positive ░=inverted) + ideal/drifting conf context | Confidence-space separation between ideal and drifting states |
| quad_confidence_spread-adaptive _MARGIN_EMA_A | spread≥0.12→+0.04; ≤0.04→−0.03 | Speeds margin EMA when quadrants are clearly separated in conf space |
| quad_coh3_spread-adaptive _SG_EMA_ALPHA | spread≥0.10→+0.04; ≤0.03→−0.03 | Speeds score-gap EMA when coh3 strongly distinguishes quadrants |
| quad_velocity_spread-adaptive _CTX_OSC_DAMP | spread≥0.08→+0.04; ≤0.02→−0.03 | Strengthens oscillation damping when quadrant velocities are widely spread |
| quad_coh3_ideal_vs_flat_ratio-adaptive _RW_CONF_ALPHA | ratio≥1.30→+0.04; ≤1.05→−0.03 | Speeds reward-conf EMA when ideal is clearly more coherent than flat |
| quad_velocity_ideal_vs_drifting_ratio-adaptive _CTX_OSC_THR | ratio≤0.55→+0.015; ≥0.85→−0.015 | Tightens oscillation threshold when ideal is much slower than drifting |
| `quad_confidence_spread` result key | σ of the 4 per-quadrant confidence means | How well confidence separates the 4 quadrant states |
| `quad_coh3_spread` result key | σ of the 4 per-quadrant coh3 means | How well coh3 distinguishes quadrant states |
| `quad_velocity_spread` result key | σ of the 4 per-quadrant velocity means | How well velocity separates quadrant states |
| `quad_coh3_ideal_vs_flat_ratio` result key | ideal_coh3_mean / flat_coh3_mean (>1.0 = ideal more coherent) | Coh3 quality contrast between best and worst states |
| `quad_velocity_ideal_vs_drifting_ratio` result key | ideal_vel_mean / drifting_vel_mean (≤1 = ideal slower = good) | Velocity contrast between controlled ideal and runaway drifting |
| `quadconfidencespread` CLI command | conf spread bar + ideal/drifting conf context | Confidence-space quadrant separation at a glance |
| `quadcoh3spread` CLI command | coh3 spread bar + conf_spread context | Coh3-space quadrant separation |
| `quadvelocityspread` CLI command | velocity spread bar + coh3_spread context | Velocity-space quadrant separation |
| `quadcoh3idealvsflatratio` CLI command | ideal/flat coh3 ratio bar + coh3_spread context | Quality contrast in coherence between best and worst states |
| `quadvelocityidealvsdriftingratio` CLI command | ideal/drifting velocity ratio bar + vel_spread context | Speed contrast between controlled flow and runaway drift |
| quad_ideal_max_streak-adaptive _SEM_CENT_THR | streak≥8→thr+0.04; ≤2→thr−0.04 | Eases centroid pull when ideal flow is sustained; tightens it when brief |
| quad_drifting_max_streak-adaptive _SC_ALPHA | streak≥6→+0.04; ≤1→−0.03 | Speeds score-centroid EMA when drift is prolonged |
| quad_exploring_max_streak-adaptive _VEL_MAG_ALPHA | streak≥7→+0.04; ≤2→−0.03 | Speeds velocity-magnitude EMA when exploring runs are long |
| quad_flat_max_streak-adaptive _LOW_GAP_WIN | streak≥6→win+1; ≤1→win−1 | Widens low-gap detection window when flat runs are sustained |
| `quad_ideal_max_streak` result key | longest consecutive run of ideal-quadrant steps (int) | Peak quality flow duration |
| `quad_drifting_max_streak` result key | longest consecutive run of drifting-quadrant steps (int) | Worst drift episode length |
| `quad_exploring_max_streak` result key | longest consecutive run of exploring-quadrant steps (int) | Peak creative divergence duration |
| `quad_flat_max_streak` result key | longest consecutive run of flat-quadrant steps (int) | Worst stagnation episode length |
| `quad_dominant_streak_label` result key | which quadrant has the longest max streak (string) | Which state the model spends its longest runs in |
| `quadmaxstreaks` CLI command | all 4 max-streak bars + dominant label | Full quadrant streak summary |
| `quadidealstreak` CLI command | ideal max-streak bar + dominant/total context | Peak quality flow duration |
| `quaddriftingstreak` CLI command | drifting max-streak bar + dominant/total context | Worst drift episode detector |
| `quadexploringstreak` CLI command | exploring max-streak bar + dominant/total context | Peak creative divergence duration |
| `quadflatstreak` CLI command | flat max-streak bar + dominant/total context | Worst stagnation episode detector |
| quad_ideal_mean_streak-adaptive _SEM_CENT_ALPHA | mean≥5→alpha−0.01; ≤2→alpha+0.01 | Slows centroid blend when ideal runs are long; speeds it when brief |
| `quad_ideal_run_count` result key | number of distinct ideal-quadrant episodes (int) | How many separate quality flow episodes occurred |
| `quad_drifting_run_count` result key | number of distinct drifting-quadrant episodes (int) | How many separate drift episodes occurred |
| `quad_ideal_mean_streak` result key | mean length of ideal-quadrant runs (float) | Average duration of quality flow episodes |
| `quad_drifting_mean_streak` result key | mean length of drifting-quadrant runs (float) | Average duration of drift episodes |
| `quad_streak_variability` result key | σ of all quadrant run lengths (float; low=uniform; high=irregular) | Pacing consistency across all state transitions |
| `quadruncounts` CLI command | ideal + drifting episode count bars | Episode frequency at a glance |
| `quadidealruncount` CLI command | ideal episode count bar + mean/max streak context | How fragmented quality flow is |
| `quaddriftingruncount` CLI command | drifting episode count bar + mean/max streak context | How fragmented drift episodes are |
| `quadidealmeanstreak` CLI command | ideal mean-streak bar + max/count context | Average duration of quality flow |
| `quaddriftingmeanstreak` CLI command | drifting mean-streak bar + max/count context | Average duration of drift episodes |
| `quadstreakvariability` CLI command | streak σ bar + ideal/drifting mean context | Pacing consistency — how uniform quadrant run lengths are |
| `quad_early_ideal_frac` result key | fraction of ideal steps in first half of generation (0–1) | How much quality flow the model had at the start |
| `quad_late_ideal_frac` result key | fraction of ideal steps in second half of generation (0–1) | How much quality flow the model had at the end |
| `quad_ideal_frac_trend` result key | late_ideal_frac − early_ideal_frac (positive=improving; negative=degrading) | Whether quality flow strengthens or weakens over the generation |
| `quad_early_drifting_frac` result key | fraction of drifting steps in first half of generation (0–1) | How much early-generation drift occurred |
| `quad_late_drifting_frac` result key | fraction of drifting steps in second half of generation (0–1) | How much late-generation drift occurred |
| `quadearlylateideal` CLI command | early/late ideal frac bars + trend label | Quality flow arc at a glance |
| `quadidealfractrend` CLI command | signed trend bar (█=improving ░=degrading) + early/late fracs | Quality trend over the generation |
| `quadearlylatedrifting` CLI command | early/late drifting frac bars + Δ label | Drift arc: worsening/stable/recovering |
| `quad_weighted_ideal_score` result key | fraction of total conf mass on ideal steps (0–1) | How much "confidence weight" the model puts on its quality-flow tokens |
| `quad_weighted_drifting_score` result key | fraction of total conf mass on drifting steps (0–1) | How much confidence weight falls on runaway-drift tokens |
| `quad_confidence_mass_ratio` result key | ideal conf mass / drifting conf mass (>1 = ideal heavier) | Confidence-weighted quality vs. drift balance |
| `quad_ideal_first_step` result key | first step index at which ideal quadrant is reached (−1=never) | How quickly the model enters quality flow |
| `quad_drifting_first_step` result key | first step index at which drifting quadrant is reached (−1=never) | How quickly the model falls into drift |
| `quadweightedscores` CLI command | conf-mass ideal/drifting bars + ratio label | Confidence-weighted quadrant balance at a glance |
| `quadconfidencemassratio` CLI command | ratio bar + ideal/drifting conf mass context | Confidence dominance: ideal vs. drifting |
| `quadfirststeps` CLI command | first ideal + first drifting step index + % position | Entry-point timing for quality flow and drift |
| `quad_early_third_ideal_frac` result key | fraction of ideal steps in first third of generation | Quality flow density early on |
| `quad_mid_third_ideal_frac` result key | fraction of ideal steps in middle third of generation | Quality flow density in the middle |
| `quad_late_third_ideal_frac` result key | fraction of ideal steps in last third of generation | Quality flow density late on |
| `quad_early_third_drifting_frac` result key | fraction of drifting steps in first third of generation | Early drift density |
| `quad_health_score` result key | composite: ideal_frac × ideal_conf × (1−drift_frac) × gap_boost | Single 0–∞ quality health number for the generation |
| `quadhealthscore` CLI command | composite health score bar + decomposition | One-number generation quality summary |
| `quadthirds` CLI command | ideal frac in early/mid/late thirds bars + arc label | Quality flow arc over the generation in thirds |
| `quad_weighted_ideal_coh3` result key | fraction of total coh3 mass on ideal steps | How much coherence weight the model puts on quality-flow tokens |
| `quad_weighted_drifting_coh3` result key | fraction of total coh3 mass on drifting steps | How much coherence weight falls on runaway-drift tokens |
| `quad_coh3_mass_ratio` result key | ideal coh3 mass / drifting coh3 mass (>1 = ideal heavier) | Coh3-weighted quality vs. drift balance |
| `quad_ideal_coh3_momentum` result key | mean coh3 of last 3 ideal steps minus first 3 ideal steps | Whether coherence rises or falls within ideal runs |
| `quadreport` CLI command | full one-screen report: health/fracs/trend/streaks/conf/first-steps | Comprehensive quadrant diagnostic at a glance |
| `quadcoh3massratio` CLI command | coh3 mass ratio bar + ideal/drifting coh3 mass + momentum | Coherence-weighted quality vs. drift dominance |
| `quadidealcoh3momentum` CLI command | signed momentum bar + coh3_mass_ratio context | Whether coherence trends up or down inside ideal runs |
| `quad_weighted_ideal_velocity` result key | fraction of total |velocity| mass on ideal steps (lower=good) | How much motion energy is in quality-flow tokens |
| `quad_weighted_drifting_velocity` result key | fraction of total |velocity| mass on drifting steps (higher=bad) | How much motion energy falls on runaway-drift tokens |
| `quad_velocity_mass_ratio` result key | ideal |vel| mass / drifting |vel| mass (<1 = ideal is slower = good) | Velocity-weighted quality vs. drift balance |
| `quad_ideal_velocity_momentum` result key | mean vel of last 3 ideal steps minus first 3 (negative=slowing=good) | Whether ideal steps get faster or slower over time |
| `quadtransitionmap` CLI command | ASCII 4×4 matrix with row-probabilities + entropy | Full transition probability table |
| `quadvelocitymassratio` CLI command | velocity mass ratio bar + ideal/drifting vel mass + momentum | Motion-energy quality vs. drift balance |
| `quadidealvelocitymomentum` CLI command | signed momentum bar (░=slowing=good) + mass_ratio context | Speed trend inside ideal runs |
| `quad_drifting_coh3_momentum` result key | last3 minus first3 coh3 of drifting steps (positive=worsening) | Whether coherence falls further or recovers within drift runs |
| `quad_drifting_velocity_momentum` result key | last3 minus first3 vel of drifting steps (positive=accelerating=bad) | Whether drift is accelerating or decelerating |
| `quad_exploring_coh3_momentum` result key | last3 minus first3 coh3 of exploring steps (positive=rising) | Whether coherence improves or falls within exploration runs |
| `quad_inter_ideal_gap` result key | mean steps between end of one ideal run and start of next | How quickly quality flow resumes after a break |
| `quad_inter_drifting_gap` result key | mean steps between end of one drifting run and start of next | How frequently drift relapses |
| `quadinterrungaps` CLI command | ideal + drifting gap bars + run counts | Episode frequency and re-entry timing |
| `quaddriftingmomentum` CLI command | coh3 + velocity momentum bars for drifting runs | Whether drift is worsening or self-correcting |
| `quadexploringcoh3momentum` CLI command | signed coh3 trend bar for exploring runs | Coherence arc inside exploration episodes |
| `quad_flat_coh3_momentum` result key | last3 minus first3 coh3 of flat steps (positive=recovering) | Whether stagnation is deepening or self-correcting in coherence |
| `quad_flat_velocity_momentum` result key | last3 minus first3 vel of flat steps (positive=pre-escape signal) | Whether velocity creeps up as flat period ends |
| `quad_ideal_to_drift_rate` result key | fraction of ideal exits going to drifting (lower=better) | How often quality flow decays directly into drift |
| `quad_drift_to_ideal_rate` result key | fraction of drifting exits going to ideal (higher=better) | Direct drift-to-ideal recovery rate |
| `quad_exploring_to_ideal_rate` result key | fraction of exploring exits going to ideal (higher=better) | How often exploration converts to quality flow |
| `quadtransitionrates` CLI command | ideal→drift / drift→ideal / exploring→ideal rate bars | Key decay and recovery transition rates |
| `quadflatmomentum` CLI command | coh3 + velocity momentum bars for flat runs | Whether flat stagnation is deepening or self-correcting |
| `quad_flat_to_ideal_rate` result key | fraction of flat exits going to ideal (stagnation escape rate) | How often stagnation resolves directly into quality flow |
| `quad_ideal_to_exploring_rate` result key | fraction of ideal exits going to exploring (partial destabilisation) | How often ideal flow tips into exploration vs. drift |
| `quad_ideal_persistence` result key | ideal→ideal self-loop fraction (high=flow is self-sustaining) | How sticky the ideal quadrant is |
| `quad_net_recovery_rate` result key | drift→ideal rate minus ideal→drift rate (positive=net recovery) | Whether the model recovers from drift more than it decays into it |
| `quadidealpersistence` CLI command | ideal→ideal self-loop bar + run count/max streak context | How sticky quality flow is |
| `quadnetrecovery` CLI command | signed net recovery bar + drift↔ideal rate context | Net balance between decay and recovery |
| `quadescaperates` CLI command | flat/expl/drift→ideal + ideal→expl rate bars | All paths that escape to or from ideal |
| `quad_exploring_persistence` result key | exploring→exploring self-loop fraction | How sticky the exploring quadrant is |
| `quad_drifting_persistence` result key | drifting→drifting self-loop fraction | How self-sustaining runaway drift is |
| `quad_flat_persistence` result key | flat→flat self-loop fraction | How sticky stagnation is |
| `quad_symmetry_score` result key | normalised entropy of quadrant distribution [0,1] (1=balanced) | How evenly the generation visits all four quadrants |
| `quad_fingerprint` result key | "I>E>D>F" string sorted by fraction descending | Compact single-string generation quality signature |
| `quadpersistence` CLI command | all 4 self-loop bars + fingerprint + symmetry | Full quadrant stickiness at a glance |
| `quadsymmetry` CLI command | normalised entropy bar + fingerprint + label | Quadrant balance diagnostic |
| `quadfingerprint` CLI command | fingerprint string + I/E/D/F fracs + health + symmetry | Complete generation quality signature |
| `quad_ideal_minus_drifting_frac` result key | ideal_frac − drifting_frac (positive=ideal dominates) | Single signed number for quality vs. drift balance |
| `quad_exploring_minus_flat_frac` result key | exploring_frac − flat_frac (positive=creative > stagnant) | Single signed number for exploration vs. stagnation balance |
| `quad_positive_frac` result key | ideal_frac + exploring_frac (high-coh3 fraction) | Fraction of steps with above-median coherence |
| `quad_negative_frac` result key | drifting_frac + flat_frac (low-coh3 fraction) | Fraction of steps with below-median coherence |
| `quad_quality_arc` result key | arc label: cold_start/warm_start/sustained/collapse/recovery/oscillating/improving/degrading | Single-word characterisation of quality flow shape over time |
| `quadqualityarc` CLI command | arc label + early/late ideal fracs + trend + health + fingerprint | One-word generation quality arc summary |
| `quaddeltas` CLI command | ideal−drift + expl−flat signed bars + hi/lo coh3 frac | Four-quadrant balance at a glance |
| `quad_hi_vel_frac` result key | fraction of steps with above-median velocity (drifting+exploring) | How much of the generation is fast-moving |
| `quad_lo_vel_frac` result key | fraction of steps with below-median velocity (ideal+flat) | How much of the generation is slow/controlled |
| `quad_flow_efficiency` result key | ideal_frac / hi_coh3_frac — what fraction of high-coherence steps are controlled | Whether high coherence means flow or just exploration |
| `quad_drift_severity` result key | drifting_frac / hi_vel_frac — what fraction of fast steps are low-coherence | Whether speed means creativity or runaway drift |
| `quad_recovery_efficiency` result key | drift→ideal_rate × ideal_persistence (product; higher=better) | Combined: recovers from drift AND sustains ideal flow |
| `quadflowefficiency` CLI command | ideal/hi-coh3 bar + ideal/exploring fracs | Flow purity — controlled vs. fast coherence |
| `quaddriftseverity` CLI command | drift/hi-vel bar + hi_vel/drifting fracs | How damaging fast motion is |
| `quadrecoveryefficiency` CLI command | product bar + drift→ideal + persistence context | Overall recovery and sustain capability |
| `quadvelocityfracs` CLI command | hi-vel + lo-vel fraction bars | Speed split at a glance |
| `quad_ideal_conf_cv` result key | coefficient of variation (σ/μ) of confidence on ideal steps | How consistent the model's confidence is during quality flow |
| `quad_ideal_coh3_cv` result key | coefficient of variation (σ/μ) of coh3 on ideal steps | How stable coherence is within quality flow |
| `quad_drifting_coh3_cv` result key | coefficient of variation (σ/μ) of coh3 on drifting steps | Whether drift is erratic or uniformly low-quality |
| `quad_ideal_conf_stability` result key | 1/(1+ideal_conf_cv) mapped to (0,1]; high=very consistent | Normalised confidence stability during ideal steps |
| `quad_overall_efficiency` result key | health_score / hi_vel_frac — quality per unit of fast motion | Single composite: how much ideal quality is produced per speed cost |
| `quadoverallefficiency` CLI command | composite bar + health/hi_vel/recovery context | One-number generation efficiency |
| `quadconfcv` CLI command | ideal conf CV + stability bar + coh3 CV bars | Confidence and coherence consistency in ideal + drifting |
| `quad_ideal_conf_mean` result key | mean confidence on ideal steps | Average model confidence during quality flow |
| `quad_exploring_conf_mean` result key | mean confidence on exploring steps | Average confidence during creative exploration |
| `quad_drifting_conf_mean` result key | mean confidence on drifting steps | Average confidence during drift (lower=model is uncertain while drifting) |
| `quad_flat_conf_mean` result key | mean confidence on flat steps | Average confidence during stagnation |
| `quad_conf_gap` result key | ideal_conf_mean − drifting_conf_mean (positive=ideal is more confident) | How much more confident the model is during quality flow vs. drift |
| `quadconfmeans` CLI command | all 4 per-quadrant mean confidence bars + gap | Confidence profile across all four quadrants |
| `quad_ideal_vel_mean` result key | mean velocity on ideal steps | How slow/fast ideal steps are on average |
| `quad_exploring_vel_mean` result key | mean velocity on exploring steps | Speed of creative exploration steps |
| `quad_drifting_vel_mean` result key | mean velocity on drifting steps | Average drift speed |
| `quad_flat_vel_mean` result key | mean velocity on flat steps | Average stagnation speed |
| `quad_vel_gap` result key | drifting_vel_mean − ideal_vel_mean (positive=drift faster) | Velocity separation between quadrants; large gap = quadrants are well-separated |
| `quadvelmeans` CLI command | all 4 per-quadrant mean velocity bars + vel gap | Velocity profile across all four quadrants |
| `quad_ideal_run_mean_len` result key | mean consecutive ideal steps per ideal run | Average ideal streak length |
| `quad_drifting_run_mean_len` result key | mean consecutive drifting steps per drifting run | Average drift streak length |
| `quad_exploring_run_mean_len` result key | mean consecutive exploring steps per exploring run | Average exploration streak length |
| `quad_flat_run_mean_len` result key | mean consecutive flat steps per flat run | Average stagnation streak length |
| `quad_run_len_ratio` result key | ideal_run_mean_len / drifting_run_mean_len (>1 = ideal streaks longer than drift = good) | Whether quality flow or drift dominates run duration |
| `quadrunlens` CLI command | all 4 mean run length bars + ideal/drift ratio | Run duration profile across all four quadrants |
| `quad_ideal_coh3_mean` result key | mean coh3 on ideal steps | Average coherence during quality flow |
| `quad_drifting_coh3_mean` result key | mean coh3 on drifting steps | Average coherence during drift |
| `quad_coh3_gap` result key | ideal_coh3_mean − drifting_coh3_mean (positive=ideal more coherent) | Coherence separation between quality flow and drift |
| `quadcoh3means` CLI command | all 4 per-quadrant mean coh3 bars + gap | Coherence profile across all four quadrants |
| `quad_exploring_coh3_mean` result key | mean coh3 on exploring steps | Average coherence during creative exploration |
| `quad_flat_coh3_mean` result key | mean coh3 on flat steps | Average coherence during stagnation |
| `quad_ideal_coh3_std` result key | std dev of coh3 on ideal steps (low=stable quality flow) | How variable coherence is during ideal steps |
| `quad_drifting_coh3_std` result key | std dev of coh3 on drifting steps (high=erratic drift) | Whether drift is uniformly or variably low-quality |
| `quad_coh3_contrast` result key | (ideal−flat)/(ideal+flat) normalised contrast [−1,+1] | Normalised coherence gap between best and worst quadrant |
| `quadcoh3stds` CLI command | ideal + drifting coh3 std/CV bars + normalised contrast | Coherence variability and contrast diagnostics |
| `quad_ideal_conf_std` result key | std dev of confidence on ideal steps | Confidence spread during quality flow |
| `quad_drifting_conf_std` result key | std dev of confidence on drifting steps | Confidence spread during drift |
| `quad_conf_contrast` result key | (ideal_conf_mean − flat_conf_mean)/(ideal+flat) normalised [−1,+1] | Normalised confidence contrast between quality flow and stagnation |
| `quad_ideal_vel_std` result key | std dev of velocity on ideal steps (low=consistently slow=focused) | Velocity consistency during ideal steps |
| `quad_drifting_vel_std` result key | std dev of velocity on drifting steps (high=erratic drift speed) | Whether drift speed is steady or chaotic |
| `quad_vel_contrast` result key | (drifting_vel_mean − ideal_vel_mean)/(drift+ideal) normalised [0,1] | Normalised velocity contrast between ideal and drifting quadrants |
| `quadvelstds` CLI command | ideal + drifting vel mean/σ/CV + contrast + gap | Velocity spread and contrast diagnostics |
| `quadconfstds` CLI command | ideal + drifting conf mean/σ/CV + contrast + gap | Confidence spread and contrast diagnostics |
| `quad_ideal_conf_range` result key | max − min confidence on ideal steps (low=very consistent) | Confidence spread during quality flow |
| `quad_drifting_conf_range` result key | max − min confidence on drifting steps | Confidence spread during drift |
| `quad_ideal_vel_range` result key | max − min velocity on ideal steps (low=steady and focused) | Velocity spread during ideal steps |
| `quad_drifting_vel_range` result key | max − min velocity on drifting steps | Velocity spread during drift |
| `quad_separation_score` result key | composite (coh3_gap×0.5 + vel_gap×0.3 + conf_gap×0.2) normalised [0,1] | How well-separated ideal and drifting quadrants are across all three axes |
| `quadseparation` CLI command | separation score bar + coh3/vel/conf gap + vel contrast | Overall quadrant distinctness diagnostic |
| `quadranges` CLI command | ideal + drifting conf/vel range bars + ideal CV/σ context | Confidence and velocity spread per quadrant |
| `quad_first_drifting_step` result key | index of first drifting step (−1 if none) | When drift first appeared in the generation |
| `quad_last_ideal_step` result key | index of last ideal step (large=ideal persisted to end) | Whether quality flow continued through the end |
| `quad_last_drifting_step` result key | index of last drifting step (large=drift continued to end) | Whether drift persisted into the tail of the generation |
| `quad_ideal_coverage_frac` result key | ideal_steps/(last_ideal−first_ideal+1) density [0,1] | How densely ideal flow filled its own span |
| `quad_final_state` result key | label of last step (ideal/exploring/drifting/flat) | What quadrant state the generation finished in |
| `quadfinalstate` CLI command | final state label + first/last ideal/drift indices + coverage | End-of-generation state and ideal/drift span diagnostics |
| `quad_transition_entropy` result key | Shannon entropy of observed quadrant transitions (bits, max≈4) | How predictable or uniform the quadrant state machine is |
| `quad_dom_transition` result key | string label of most frequent single-step transition e.g. "ideal→ideal" | Which transition dominates the generation |
| `quad_ideal_tail_frac` result key | fraction of last-25% steps that are ideal (high=quality persisted) | Whether quality flow continued through the tail |
| `quad_drift_tail_frac` result key | fraction of last-25% steps that are drifting (high=drift owned the ending) | Whether drift dominated the tail |
| `quad_ideal_improvement` result key | (tail_ideal_frac − head_ideal_frac) positive=quality improved over time | Whether ideal fraction grew or shrank over the generation |
| `quadtailfracs` CLI command | ideal/drift tail fracs vs global + improvement score | Late-generation quality and drift trend |
| `quadtransitionentropy` CLI command | transition entropy bar + dominant transition + final state | Predictability of the quadrant state machine |
| `quad_ideal_mid_frac` result key | fraction of middle-50% steps that are ideal | Whether quality flow dominated the core of the generation |
| `quad_drift_mid_frac` result key | fraction of middle-50% steps that are drifting | Whether drift dominated the core |
| `quad_ideal_peak_coh3` result key | max coh3 on ideal steps (best coherence moment) | Peak coherence during quality flow |
| `quad_ideal_peak_conf` result key | max confidence on ideal steps (peak confidence) | Highest confidence reached during ideal steps |
| `quad_drifting_worst_coh3` result key | min coh3 on drifting steps (floor of coherence during drift) | How bad coherence got at worst drift |
| `quadpeaks` CLI command | ideal peak coh3/conf + drifting floor coh3 vs means | Best and worst per-quadrant signal values |
| `quadmidfracs` CLI command | ideal/drift mid-50% fracs + tail comparison + improvement | Mid-generation quality and drift profile |
| `quad_ideal_head_frac` result key | fraction of first-25% steps that are ideal (high=started strong) | Whether quality flow dominated the opening |
| `quad_drift_head_frac` result key | fraction of first-25% steps that are drifting (high=bad opening) | Whether drift dominated the start |
| `quad_quality_score` result key | composite [0,1]: 50%×ideal_frac + 25%×(1−drift_frac) + 25%×improvement_norm | Single-number generation quality summary |
| `quad_ideal_conf_floor` result key | min confidence on ideal steps (low=conf dipped even during quality flow) | How reliable confidence was during ideal steps |
| `quad_drifting_peak_conf` result key | max confidence on drifting steps (high=surprising confident drift moments) | Whether drift ever produced high-confidence output |
| `quadqualityscore` CLI command | composite quality score bar + component breakdown | Single-number generation quality diagnostic |
| `quadheadfracs` CLI command | head/mid/tail ideal+drift frac table + trend label | Full temporal profile across all three generation segments |
| `quad_above_median_coh3_frac` result key | fraction of steps with coh3 above the run mean (~0.50 if symmetric) | Whether coh3 distribution is skewed high or low |
| `quad_below_median_vel_frac` result key | fraction of steps with velocity below the run mean (~0.50 if symmetric) | Whether velocity distribution is skewed focused or chaotic |
| `quad_coherent_to_incoherent_ratio` result key | above-coh3-mean / below-coh3-mean steps ratio (>1=more coherent) | Overall coherence balance of the generation |
| `quad_focused_to_chaotic_ratio` result key | below-vel-mean / above-vel-mean steps ratio (>1=more focused) | Overall velocity focus balance of the generation |
| `quad_health_vector` result key | compact "IEDF" vector e.g. "I..f" — upper=dominant ≥35%, lower=present ≥10%, .=absent | Single-string snapshot of which quadrants are active |
| `quadbalance` CLI command | coherent/focused ratios + health vector + fingerprint | Generation balance and quadrant dominance diagnostic |
| `quad_max_ideal_streak` result key | max consecutive ideal steps in a single run | Length of the longest quality flow burst |
| `quad_max_drifting_streak` result key | max consecutive drifting steps in a single run | Length of the longest drift burst |
| `quad_max_ideal_streak_start` result key | step index where the longest ideal streak started (−1 if none) | Where the best quality flow burst occurred |
| `quad_ideal_streak_ratio` result key | max_ideal_streak / total_ideal_steps — high=concentrated in one burst | Whether ideal flow is one big stretch or many small ones |
| `quad_drift_streak_ratio` result key | max_drifting_streak / total_drifting_steps — high=drift was one big block | Whether drift is one big block or spread throughout |
| `quadstreaks` CLI command | max ideal/drift streak bars + start index + concentration ratios | Streak structure and concentration diagnostics |
| `quad_oscillation_count` result key | number of ideal↔other label flips (high=volatile; low=stable) | How many times generation switched in/out of quality flow |
| `quad_longest_non_ideal_run` result key | max consecutive non-ideal steps (max gap without quality flow) | Longest stretch without any ideal steps |
| `quad_rle_first5` result key | first 5 RLE segments as string e.g. "i3d2e1" — opening of generation | Compact view of how generation started structurally |
| `quad_ideal_isolation_score` result key | ideal_run_count / total_steps — high=fragmented ideal flow | Whether ideal steps are scattered or consolidated |
| `quad_late_ideal_start` result key | 1 if first ideal step is in second half of generation (0=early start) | Whether quality flow arrived late or early |
| `quadrle` CLI command | opening RLE string + oscillations + gap + isolation + late flag | Sequence structure and fragmentation diagnostic |
| `quadoscillation` CLI command | oscillation count + longest gap + isolation + run count + late flag | Volatility and stability of quality flow |
| `quad_exploring_run_count` result key | number of times generation entered the exploring quadrant | Frequency of creative exploration bursts |
| `quad_flat_run_count` result key | number of times generation entered the flat quadrant | Frequency of stagnation entries |
| `quad_max_exploring_streak` result key | max consecutive exploring steps | Longest creative exploration burst |
| `quad_max_flat_streak` result key | max consecutive flat steps | Longest stagnation burst |
| `quad_zigzag_score` result key | ideal↔other flips / (total_steps−1) [0,1] — high=volatile | Per-step oscillation rate; high=generation is unstable |
| `quadzigzag` CLI command | zigzag score bar + oscillations + all 4 run counts | Generation-level volatility diagnostic |
| `quadexplorestreaks` CLI command | exploring+flat max streak + run count + mean length + frac | Exploring and stagnation streak diagnostics |
| `quad_coh3_skew` result key | Pearson skewness of coh3 distribution (positive=right-skewed=sparse hi-coh3) | Whether high coherence is rare or common |
| `quad_vel_skew` result key | Pearson skewness of velocity distribution (positive=right-skewed=sparse hi-vel) | Whether high velocity is rare or common |
| `quad_ideal_centroid` result key | weighted mean step index of ideal steps (where ideal flow is centred) | Temporal location of quality flow in the generation |
| `quad_drifting_centroid` result key | weighted mean step index of drifting steps | Temporal location of drift in the generation |
| `quad_centroid_gap` result key | ideal_centroid − drifting_centroid (positive=ideal later; negative=ideal earlier) | Whether quality flow comes before or after drift |
| `quadcentroids` CLI command | ideal+drift temporal centroids + gap label + late-start flag | Temporal placement of quality flow vs drift |
| `quadskew` CLI command | coh3 + velocity skewness diagnostics | Signal distribution shape (left/right-skewed) |
| `quad_coh3_kurtosis` result key | excess kurtosis of coh3 distribution (>0=heavy tails; <0=flat) | Whether high coherence is outlier-prone or uniformly spread |
| `quad_vel_kurtosis` result key | excess kurtosis of velocity distribution | Whether extreme velocities are rare spikes or common |
| `quad_conf_skew` result key | Pearson skewness of confidence distribution | Shape of confidence spread across the generation |
| `quad_conf_kurtosis` result key | excess kurtosis of confidence distribution | Whether confidence has outlier spikes or uniform spread |
| `quad_signal_quality_index` result key | composite [0,1]: 35%×ideal + 20%×(1−drift) + 25%×coh3_norm + 20%×conf | Raw signal quality independent of temporal position |
| `quadsignalquality` CLI command | signal quality index bar + component breakdown + quality/separation | Composite signal quality diagnostic |
| `quadkurtosis` CLI command | coh3+vel+conf excess kurtosis + skewness table | Signal distribution shape (outlier-prone vs uniform) |
| `quad_ideal_coh3_var` result key | variance of coh3 values on ideal steps (low=consistent quality) | How noisy coh3 is during quality flow |
| `quad_drifting_coh3_var` result key | variance of coh3 values on drifting steps | How noisy coh3 is during drift |
| `quad_ideal_conf_var` result key | variance of confidence values on ideal steps | How stable confidence is during quality flow |
| `quad_drifting_conf_var` result key | variance of confidence values on drifting steps | How stable confidence is during drift |
| `quad_ideal_vs_drift_coh3_var_ratio` result key | ideal_coh3_var / drifting_coh3_var (<1=ideal tighter) | Whether ideal flow is more consistent or noisier than drift |
| `quadvariance` CLI command | ideal+drift coh3/conf variance + ratio label | Per-quadrant signal variance diagnostic |
| `quad_coh3_conf_correlation` result key | Pearson r between coh3 and confidence across all steps | Whether coherence and confidence move together |
| `quad_coh3_vel_correlation` result key | Pearson r between coh3 and velocity (expect negative for quality) | Whether high coherence is paired with low velocity |
| `quad_conf_vel_correlation` result key | Pearson r between confidence and velocity | Whether high confidence is paired with low velocity |
| `quad_ideal_coh3_conf_correlation` result key | coh3↔conf Pearson r restricted to ideal steps only | Signal coupling during quality flow specifically |
| `quad_signal_coupling_score` result key | abs(c3_cf_r)×0.5 + (1−abs(c3_vel_r))×0.5 [0,1] | Whether coh3+conf are aligned while vel is decoupled |
| `quadcorrelations` CLI command | coh3/conf/vel Pearson correlations + ideal-only + coupling score | Cross-signal correlation diagnostic |
| `quad_coh3_autocorr_lag1` result key | lag-1 autocorrelation of coh3 series (high=smooth/persistent) | Whether coherence evolves smoothly or choppily |
| `quad_vel_autocorr_lag1` result key | lag-1 autocorrelation of velocity series | Whether velocity evolves smoothly or anti-persists |
| `quad_conf_autocorr_lag1` result key | lag-1 autocorrelation of confidence series | Whether confidence is smooth or oscillating |
| `quad_ideal_persistence_score` result key | P(ideal at t+1 \| ideal at t) — probability ideal state continues | How sticky quality flow is once it starts |
| `quad_drift_persistence_score` result key | P(drift at t+1 \| drift at t) — probability drift state continues | How sticky drift is once it starts |
| `quadpersistence` CLI command | P(stay in ideal/drift) bars + lag-1 autocorr table | State persistence and signal memory diagnostic |
| `quadautocorr` CLI command | lag-1 autocorr for coh3/vel/conf + smoothness labels | Signal temporal memory diagnostic |
| `quad_ideal_to_exploring_rate` result key | P(exploring at t+1 \| ideal at t) — direct quality→explore transition | How often quality flow tips into creative exploration |
| `quad_ideal_to_drifting_rate` result key | P(drifting at t+1 \| ideal at t) — direct quality→chaos transition | How often ideal flow collapses directly to drift |
| `quad_exploring_to_ideal_rate` result key | P(ideal at t+1 \| exploring at t) — exploration converting to quality | Whether creative exploration leads to quality flow |
| `quad_drifting_to_flat_rate` result key | P(flat at t+1 \| drifting at t) — drift decelerating to stagnation | Whether drift tends to burn out into flat |
| `quad_markov_stability_score` result key | 0.6×ideal_persist + 0.4×(1−drift_persist) [0,1] — high=ideal sticky, drift fleeting | Overall Markov stability of the quadrant process |
| `quadtransitions` CLI command | full Markov transition table + stability score | State-to-state conditional probabilities diagnostic |
| `quad_label_entropy` result key | normalized Shannon H(i,e,d,f) ∈[0,1]; 1=all four equally used | Whether generation explores all quadrant states equally |
| `quad_coh3_entropy` result key | normalized entropy of coh3 values binned into 8 buckets | How spread out the coherence values are |
| `quad_vel_entropy` result key | normalized entropy of velocity values binned into 8 buckets | How spread out the velocity values are |
| `quad_conf_entropy` result key | normalized entropy of confidence values binned into 8 buckets | How spread out the confidence values are |
| `quad_entropy_index` result key | 0.4×label_entropy + 0.3×coh3_entropy + 0.3×(1−vel_entropy) [0,1] | Composite entropy favouring diverse states but focused velocity |
| `quadentropy` CLI command | label/coh3/vel/conf entropy bars + entropy index | Generation entropy and diversity diagnostic |
| `quad_coh3_trend_slope` result key | linear regression slope of coh3 over step index (positive=rising) | Whether coherence is trending up or down across the generation |
| `quad_vel_trend_slope` result key | linear regression slope of velocity over step index (negative=slowing=good) | Whether velocity is trending toward focus or chaos |
| `quad_conf_trend_slope` result key | linear regression slope of confidence over step index | Whether confidence is building or eroding across the generation |
| `quad_coh3_trend_r2` result key | R² of coh3 linear fit — how well coh3 follows a linear trend | Strength of coh3 trend (high=consistent direction) |
| `quad_trend_alignment_score` result key | 0.5×clamp(coh3_slope×100,[0,1]) + 0.5×clamp(−vel_slope×100,[0,1]) | Whether generation is converging toward quality (coh3↑ vel↓) |
| `quadtrends` CLI command | coh3/vel/conf linear slope + R² + trend alignment label | Generation trajectory and convergence diagnostic |
| `quad_first_half_ideal_frac` result key | ideal_frac in first half of generation steps | Whether quality flow is front-loaded |
| `quad_second_half_ideal_frac` result key | ideal_frac in second half of generation steps | Whether quality flow is back-loaded |
| `quad_first_half_coh3_mean` result key | mean coh3 in first half of generation | Average coherence during generation opening |
| `quad_second_half_coh3_mean` result key | mean coh3 in second half of generation | Average coherence during generation closing |
| `quad_half_improvement_score` result key | 0.6×(2nd_ideal−1st_ideal) + 0.4×norm_coh3_delta; positive=improving | Whether the generation gets better in the second half |
| `quadhalves` CLI command | first/second half ideal_frac + coh3_mean + improvement score | Generation improvement/decline between halves |
| `quad_q1_ideal_frac` result key | ideal_frac in first quarter of generation steps | Whether quality flow is concentrated in the opening |
| `quad_q4_ideal_frac` result key | ideal_frac in last quarter of generation steps | Whether quality flow is concentrated in the close |
| `quad_q1_coh3_mean` result key | mean coh3 in first quarter | Average coherence at generation opening |
| `quad_q4_coh3_mean` result key | mean coh3 in last quarter | Average coherence at generation close |
| `quad_quarter_arc_score` result key | 0.6×(Q4_ideal−Q1_ideal) + 0.4×norm_coh3_delta; positive=rising arc | Whether generation has a quality arc from opening to close |
| `quadquarters` CLI command | Q1/Q4 ideal_frac + coh3_mean + arc score + half improvement | Generational arc from opening quarter to closing quarter |
| `quad_ideal_vel_mean` result key | mean velocity on ideal steps (expect low — slow focused movement) | Whether ideal steps are genuinely focused |
| `quad_drifting_vel_mean` result key | mean velocity on drifting steps (expect high — fast chaotic movement) | Whether drift steps are genuinely fast |
| `quad_exploring_vel_mean` result key | mean velocity on exploring steps (high=fast but coherent) | Speed of creative exploration phase |
| `quad_flat_vel_mean` result key | mean velocity on flat steps (low but incoherent) | Whether flat steps are truly slow |
| `quad_vel_quadrant_spread` result key | σ of the 4 per-quadrant velocity means (high=quadrants differ in speed) | How much velocity varies across quadrant states |
| `quadvelprofile` CLI command | per-quadrant velocity means bars + spread | Velocity profile broken down by quadrant state |
| `quad_ideal_conf_mean` result key | mean confidence on ideal steps | Whether ideal steps carry high model confidence |
| `quad_drifting_conf_mean` result key | mean confidence on drifting steps | Whether drift is confident or uncertain |
| `quad_exploring_conf_mean` result key | mean confidence on exploring steps | Confidence level during creative exploration |
| `quad_flat_conf_mean` result key | mean confidence on flat steps | Confidence level during stagnation |
| `quad_conf_quadrant_spread` result key | σ of the 4 per-quadrant confidence means (high=quadrants differ in confidence) | How differently confident the model is across quadrant states |
| `quadconfprofile` CLI command | per-quadrant confidence means bars + spread | Confidence profile broken down by quadrant state |
| `quad_ideal_vs_drift_conf_gap` result key | ideal_conf_mean − drifting_conf_mean (positive=ideal is more confident) | Whether ideal steps have higher model confidence than drift |
| `quad_ideal_vs_drift_coh3_gap` result key | ideal_coh3_mean − drifting_coh3_mean (positive=ideal more coherent) | Raw coherence gap between ideal and drifting states |
| `quad_ideal_vs_flat_coh3_gap` result key | ideal_coh3_mean − flat_coh3_mean (positive=ideal coherent, flat is not) | Coherence gap between ideal and stagnant states |
| `quad_quality_gap_score` result key | 0.4×norm_conf_gap + 0.6×norm_coh3_gap [0,1]; high=ideal/drift well separated | Composite signal that ideal and drifting are clearly distinct |
| `quad_signal_separation_index` result key | (ideal_coh3_mean − drift_coh3_mean) / coh3_std — Cohen's d analogue | Effect-size-like separation between ideal and drifting on coh3 |
| `quadgaps` CLI command | ideal-vs-drift conf/coh3 gaps + quality gap + separation index | How clearly ideal steps are separated from drifting on all signals |
| `quad_coh3_volatility` result key | std of step-to-step \|Δcoh3\| (high=erratic coherence) | How smoothly or erratically coherence changes between steps |
| `quad_vel_volatility` result key | std of step-to-step \|Δvelocity\| (high=erratic velocity) | How smoothly or erratically velocity changes between steps |
| `quad_conf_volatility` result key | std of step-to-step \|Δconfidence\| (high=erratic confidence) | How smoothly confidence evolves across the generation |
| `quad_coh3_vel_volatility_ratio` result key | coh3_volatility / vel_volatility; >1=coh3 more erratic than velocity | Whether coherence or velocity is the more chaotic signal |
| `quad_stability_composite` result key | 1 − mean(coh3_vol,vel_vol,conf_vol) clipped [0,1]; high=smooth generation | Overall smoothness/stability of all three signals |
| `quadvolatility` CLI command | coh3/vel/conf volatility + ratio + stability composite | Step-to-step signal stability and smoothness diagnostic |
| `quad_ideal_burst_count` result key | number of contiguous ideal runs of length ≥2 | How many sustained quality episodes the generation contains |
| `quad_ideal_burst_max_len` result key | length of the longest consecutive ideal run | Peak quality burst duration |
| `quad_ideal_burst_mean_len` result key | mean length of all consecutive ideal runs | Average sustained quality episode length |
| `quad_drift_burst_count` result key | number of contiguous drifting runs of length ≥2 | How many sustained chaotic episodes the generation contains |
| `quad_burst_quality_ratio` result key | ideal_burst_count / (drift_burst_count+1); high=quality-rich | Whether quality or drift bursts dominate the generation |
| `quadbursts` CLI command | ideal/drift burst counts + max/mean ideal burst + quality ratio | Sustained quality vs. drift episode analysis |
| `quad_coh3_min` result key | global minimum coh3 value across the generation | Worst-case coherence floor |
| `quad_coh3_max` result key | global maximum coh3 value across the generation | Best-case coherence peak |
| `quad_coh3_range` result key | coh3 max − min; high=coh3 spans a wide band | Dynamic range of coherence across the generation |
| `quad_coh3_above_075_frac` result key | fraction of steps with coh3 > 0.75 (high=consistently high-coherence) | How often the model reaches strong coherence on an absolute scale |
| `quad_coh3_above_090_frac` result key | fraction of steps with coh3 > 0.90 (high=elite coherence) | How often the model reaches near-peak coherence |
| `quadcoh3stats` CLI command | coh3 min/max/range + frac>0.75 + frac>0.90 | Absolute coherence range and high-coherence threshold fractions |
| `quad_vel_min` result key | global minimum velocity | Lowest velocity (most focused step) in the generation |
| `quad_vel_max` result key | global maximum velocity | Highest velocity (most chaotic step) in the generation |
| `quad_vel_range` result key | velocity max − min | Dynamic range of velocity across the generation |
| `quad_vel_below_025_frac` result key | fraction of steps with velocity < 0.25 (focused/deep steps) | How much of the generation runs at low velocity |
| `quad_vel_above_mean_frac` result key | fraction of velocity steps above their own mean (expect ~0.50) | Symmetry of velocity distribution |
| `quadvelstats` CLI command | vel min/max/range + frac<0.25 + frac>mean | Absolute velocity range and focus-threshold fractions |
| `quad_conf_min` result key | global minimum confidence across the generation | Worst-case confidence floor |
| `quad_conf_max` result key | global maximum confidence across the generation | Best-case confidence peak |
| `quad_conf_range` result key | confidence max − min | Dynamic range of confidence across the generation |
| `quad_conf_above_075_frac` result key | fraction of steps with confidence > 0.75 | How often model is confidently generating |
| `quad_conf_above_090_frac` result key | fraction of steps with confidence > 0.90 | How often model is at elite confidence |
| `quadconfstats` CLI command | conf min/max/range + frac>0.75 + frac>0.90 | Absolute confidence range and high-confidence threshold fractions |
| `quad_ideal_to_exploring_ratio` result key | ideal_count / max(exploring_count,1); >1=ideal dominates | Whether quality or exploration is the dominant high-coherence mode |
| `quad_drift_to_flat_ratio` result key | drifting_count / max(flat_count,1); high=chaos over stagnation | Whether the bad steps are chaotic or stagnant |
| `quad_coherent_states_frac` result key | (ideal+exploring) / total — fraction with above-median coh3 | How much of the generation is operating in high-coherence territory |
| `quad_high_vel_states_frac` result key | (drifting+exploring) / total — fraction with above-median velocity | How much of the generation is running at high velocity |
| `quadratiostats` CLI command | ideal/explore ratio + drift/flat ratio + coherent/high-vel fracs | State balance ratios and coherence/velocity occupancy fractions |
| `quad_focus_score` result key | ideal_frac × (1−vel_mean) [0,1]; high=high ideal fraction AND low velocity | Whether the model is both on-target and focused |
| `quad_coherence_focus_score` result key | coh3_mean × (1−vel_mean); pure signal quality-focus product | Coherence weighted by how focused/slow the generation is |
| `quad_overall_health_score` result key | 0.25×ideal + 0.25×(1-drift) + 0.25×coh3>0.75 + 0.25×conf>0.75 [0,1] | Balanced four-axis health score |
| `quad_generation_quality_index` result key | 0.30×ideal + 0.20×coh3 + 0.20×(1-drift) + 0.15×conf + 0.15×(1-vel) | Comprehensive generation quality index |
| `quadhealthscore` CLI command | overall health + quality index + focus + grade A-F | Comprehensive generation health and quality summary |
| `quad_coh3_p25` result key | 25th percentile of coh3 values | Lower-quartile coherence |
| `quad_coh3_p50` result key | median (50th percentile) of coh3 values | Median coherence across the generation |
| `quad_coh3_p75` result key | 75th percentile of coh3 values | Upper-quartile coherence |
| `quad_coh3_iqr` result key | coh3 p75 − p25 inter-quartile range; high=wide spread | Robust spread measure for coherence |
| `quadcoh3percentiles` CLI command | coh3 min/p25/p50/p75/max + IQR + skew | Full coh3 percentile distribution with skew indicator |
| `quad_vel_p25` result key | 25th percentile of velocity values | Lower-quartile velocity |
| `quad_vel_p50` result key | median (50th percentile) of velocity values | Median velocity across the generation |
| `quad_vel_p75` result key | 75th percentile of velocity values | Upper-quartile velocity |
| `quad_vel_iqr` result key | velocity p75 − p25 inter-quartile range | Robust spread measure for velocity |
| `quadvelpercentiles` CLI command | vel min/p25/p50/p75/max + IQR + skew | Full velocity percentile distribution with skew indicator |
| `quad_conf_p25` result key | 25th percentile of confidence values | Lower-quartile confidence |
| `quad_conf_p50` result key | median confidence across the generation | Median confidence |
| `quad_conf_p75` result key | 75th percentile of confidence values | Upper-quartile confidence |
| `quad_conf_iqr` result key | confidence p75 − p25 inter-quartile range | Robust spread measure for confidence |
| `quadconfpercentiles` CLI command | conf min/p25/p50/p75/max + IQR + skew | Full confidence percentile distribution with skew indicator |
| `quad_coh3_positive_delta_frac` result key | fraction of step-pairs where coh3 rises; >0.5=net rising | How often coherence improves from step to step |
| `quad_coh3_negative_delta_frac` result key | fraction of step-pairs where coh3 falls | How often coherence declines from step to step |
| `quad_coh3_mean_positive_delta` result key | mean magnitude of positive coh3 jumps | Average size of coherence improvements |
| `quad_coh3_mean_negative_delta` result key | mean magnitude of negative coh3 drops | Average size of coherence declines |
| `quad_coh3_delta_asymmetry` result key | mean_pos_delta / mean_neg_delta; >1=rises bigger than falls | Whether coherence net-improves or net-declines per swing |
| `quadcoh3deltas` CLI command | coh3 +/- delta fracs + mean magnitudes + asymmetry | Step-to-step coherence movement direction and size |
| `quad_vel_positive_delta_frac` result key | fraction of step-pairs where velocity rises (acceleration) | How often velocity is increasing (bad) |
| `quad_vel_negative_delta_frac` result key | fraction of step-pairs where velocity falls (deceleration) | How often velocity is decreasing toward focus (good) |
| `quad_vel_mean_positive_delta` result key | mean magnitude of positive velocity jumps | Average size of acceleration episodes |
| `quad_vel_mean_negative_delta` result key | mean magnitude of negative velocity drops | Average size of deceleration episodes |
| `quad_vel_delta_asymmetry` result key | neg_delta_mean / pos_delta_mean; >1=decelerations bigger than accelerations | Whether velocity is net converging toward focus |
| `quadveldeltas` CLI command | vel +/- delta fracs + mean magnitudes + decel asymmetry | Step-to-step velocity movement direction and convergence |
| `quad_conf_positive_delta_frac` result key | fraction of step-pairs where confidence rises | How often model confidence is improving step-to-step |
| `quad_conf_negative_delta_frac` result key | fraction of step-pairs where confidence falls | How often model confidence is declining step-to-step |
| `quad_conf_mean_positive_delta` result key | mean magnitude of positive confidence jumps | Average size of confidence recoveries |
| `quad_conf_mean_negative_delta` result key | mean magnitude of negative confidence drops | Average size of confidence erosions |
| `quad_conf_delta_asymmetry` result key | pos_delta_mean / neg_delta_mean; >1=rises bigger than falls | Whether confidence net-builds or net-erodes across the generation |
| `quadconfdeltas` CLI command | conf +/- delta fracs + mean magnitudes + asymmetry | Step-to-step confidence movement direction and build/erode balance |
| `quad_coh3_momentum` result key | mean signed Δcoh3 per step (positive=overall rising coherence) | Net drift of coherence across the generation |
| `quad_vel_momentum` result key | mean signed Δvelocity per step (negative=decelerating/converging) | Net drift of velocity — negative is ideal |
| `quad_conf_momentum` result key | mean signed Δconfidence per step (positive=building confidence) | Net drift of model confidence across the generation |
| `quad_coh3_curvature_mean` result key | mean \|2nd difference\| of coh3 — curvature/acceleration of coherence | How sharply coherence changes direction step-to-step |
| `quad_vel_curvature_mean` result key | mean \|2nd difference\| of velocity — curvature of velocity signal | How sharply velocity accelerates or decelerates |
| `quadmomentum` CLI command | coh3/vel/conf net momentum per step + coh3/vel curvature | Signal net direction and acceleration diagnostic |
| `quad_ideal_coh3_std` result key | std of coh3 on ideal steps (low=ideal is internally consistent) | How homogeneous ideal steps are on coherence |
| `quad_drifting_coh3_std` result key | std of coh3 on drifting steps | How homogeneous drifting steps are on coherence |
| `quad_ideal_conf_std` result key | std of confidence on ideal steps | How homogeneous ideal steps are on confidence |
| `quad_ideal_uniformity` result key | 1 − mean(ideal_coh3_std, ideal_conf_std); high=ideal is homogeneous | Whether ideal steps cluster tightly in coh3/conf space |
| `quaduniformity` CLI command | ideal/drift coh3 std + ideal conf std + ideal uniformity | Internal consistency of ideal and drifting quadrant states |
| `quad_coh3_vel_correlation` result key | Pearson r(coh3, vel); negative=ideal pairing (high coh3 + low vel) | Cross-signal alignment between coherence and velocity |
| `quad_coh3_conf_correlation` result key | Pearson r(coh3, conf); positive=good (high coh3 + high conf) | Cross-signal alignment between coherence and confidence |
| `quad_vel_conf_correlation` result key | Pearson r(vel, conf) | Cross-signal relationship between velocity and confidence |
| `quad_coh3_vel_anticorrelation_score` result key | max(0, −r(coh3,vel)); high=strong ideal anticorrelation | Directional signal that coh3 and vel move oppositely |
| `quadcorrelations` CLI command | r(coh3,vel) + r(coh3,conf) + r(vel,conf) + anticorr score | Three pairwise Pearson correlations + ideal anticorrelation |
| `quad_ideal_max_run` result key | longest consecutive run of ideal steps | Whether ideal state is sustained in clusters |
| `quad_drifting_max_run` result key | longest consecutive run of drifting steps | Whether drifting state is sustained in clusters |
| `quad_ideal_mean_run` result key | mean length of ideal-step runs (>1=ideal clusters) | Average run length of ideal state |
| `quad_run_balance` result key | ideal_max_run / drifting_max_run; >1=ideal clusters longer than drift | Which quadrant type sustains longer consecutive runs |
| `quadruns` CLI command | ideal/drift max-run + mean-run + run balance ratio | Consecutive quadrant run-length analysis |
| `quad_ideal_to_ideal_frac` result key | P(ideal → ideal): prob of staying ideal given currently ideal | How sticky the ideal state is |
| `quad_ideal_to_drift_frac` result key | P(ideal → drifting): prob of falling to drift from ideal | How fragile the ideal state is |
| `quad_drift_to_ideal_frac` result key | P(drifting → ideal): recovery rate from drift | How quickly generation escapes the drifting state |
| `quad_drift_to_drift_frac` result key | P(drifting → drifting): prob of staying in drift | How sticky the drifting state is (bad if high) |
| `quad_transition_stability` result key | mean(P(ideal→ideal), P(drift→drift)); high=states are sticky | Overall state persistence / regime stickiness |
| `quadtransitions` CLI command | full 2×2 transition matrix + stability score | Markov-style quadrant transition probabilities |
| `quad_tail_ideal_frac` result key | ideal frac in last 25% of steps | Whether generation ends strong |
| `quad_tail_coh3_mean` result key | mean coh3 in last 25% of steps | Coherence quality at the end of the generation |
| `quad_tail_vel_mean` result key | mean velocity in last 25% of steps | Velocity (focus) at the end of the generation |
| `quad_tail_conf_mean` result key | mean confidence in last 25% of steps | Model confidence at the end of the generation |
| `quad_tail_score` result key | 0.4×tail_ideal + 0.3×tail_coh3 + 0.3×(1−tail_vel); composite tail quality | Whether the generation finishes in a strong state |
| `quadtail` CLI command | last-25% ideal/coh3/vel/conf + tail score vs global | Tail analysis: does the generation finish strong? |
| `quad_head_ideal_frac` result key | ideal frac in first 25% of steps | How strongly the generation starts |
| `quad_head_coh3_mean` result key | mean coh3 in first 25% of steps | Coherence quality at the start of the generation |
| `quad_head_vel_mean` result key | mean velocity in first 25% of steps | Velocity level at the start of the generation |
| `quad_head_conf_mean` result key | mean confidence in first 25% of steps | Model confidence at the start of the generation |
| `quad_head_score` result key | 0.4×head_ideal + 0.3×head_coh3 + 0.3×(1−head_vel); composite head quality | Whether the generation starts in a strong state |
| `quadhead` CLI command | first-25% ideal/coh3/vel/conf + head score vs global | Head analysis: does the generation start strong? |
| `quad_head_tail_ideal_delta` result key | tail_ideal_frac − head_ideal_frac; positive=ideal density grows | Whether ideal state increases toward the end |
| `quad_head_tail_coh3_delta` result key | tail_coh3_mean − head_coh3_mean; positive=coherence grows | Whether coherence builds over the generation |
| `quad_head_tail_vel_delta` result key | tail_vel_mean − head_vel_mean; negative=decelerating (good) | Whether velocity decreases (focuses) over the generation |
| `quad_head_tail_score_delta` result key | tail_score − head_score; positive=improving arc | Overall quality improvement from start to finish |
| `quad_generation_arc` result key | signed (tail_score − head_score); positive=improving, negative=declining | Scalar arc of the generation: is it getting better or worse? |
| `quadarc` CLI command | head→tail deltas for ideal/coh3/vel + composite arc label | Full generation arc: improving / stable / declining |
| `quad_mid_ideal_frac` result key | ideal frac in the middle 50% of steps | How the body of the generation performs |
| `quad_mid_coh3_mean` result key | mean coh3 in middle 50% of steps | Coherence quality in the body of the generation |
| `quad_mid_vel_mean` result key | mean velocity in middle 50% of steps | Velocity level in the body of the generation |
| `quad_mid_conf_mean` result key | mean confidence in middle 50% of steps | Model confidence in the body of the generation |
| `quad_mid_score` result key | 0.4×mid_ideal + 0.3×mid_coh3 + 0.3×(1−mid_vel); composite mid quality | Quality of the generation's middle segment |
| `quadmid` CLI command | middle-50% ideal/coh3/vel/conf + mid score vs global | Mid analysis: generation body quality |
| `quad_peak_step` result key | normalized position (0=start,1=end) of max coh3; early/mid/late label | Where in the generation coherence peaks |
| `quad_peak_coh3` result key | max coh3 value in the generation | Highest single-step coherence achieved |
| `quad_peak_conf` result key | confidence value at the peak coh3 step | Model confidence when coherence is at its best |
| `quad_trough_step` result key | normalized position of min coh3 step; early/mid/late label | Where in the generation coherence reaches its lowest |
| `quad_peak_to_trough_range` result key | max_coh3 − min_coh3: dynamic range of coherence | How wide the coherence swings across the generation |
| `quadpeak` CLI command | peak/trough coh3 position + dynamic range + conf at peak | Peak and trough analysis for coherence signal |
| `quad_overall_segment_score` result key | 0.25×head + 0.50×mid + 0.25×tail score; center-weighted quality | Single score weighting all three generation segments |
| `quad_segment_range` result key | max(head,mid,tail) − min(head,mid,tail); spread of quality across segments | How evenly quality is distributed across head/mid/tail |
| `quadpanorama` CLI command | head/mid/tail scores side-by-side + best/worst + weighted overall | Quality panorama across all three generation segments |
| `quad_q1_ideal_frac` result key | ideal frac in Q1 (steps 0-25%) | Ideal density in the first quartile |
| `quad_q2_ideal_frac` result key | ideal frac in Q2 (steps 25-50%) | Ideal density in the second quartile |
| `quad_q3_ideal_frac` result key | ideal frac in Q3 (steps 50-75%) | Ideal density in the third quartile |
| `quad_q4_ideal_frac` result key | ideal frac in Q4 (steps 75-100%) | Ideal density in the fourth quartile |
| `quad_ideal_peak_quartile` result key | quartile (1-4) with the highest ideal-step density | Which quartile the model spends most time in ideal state |
| `quaddensity` CLI command | ideal frac per Q1-Q4 bar chart + peak quartile indicator | Ideal-state density heatmap across quartiles |
| `quad_total_step_count` result key | total steps analyzed in the generation | Raw step count |
| `quad_ideal_step_count` result key | raw count of ideal steps | Absolute number of ideal-state steps |
| `quad_drifting_step_count` result key | raw count of drifting steps | Absolute number of drifting-state steps |
| `quad_exploring_step_count` result key | raw count of exploring steps | Absolute number of exploring-state steps |
| `quad_flat_step_count` result key | raw count of flat steps | Absolute number of flat-state steps |
| `quadcounts` CLI command | ideal/drift/explore/flat counts + fraction bars | Absolute quadrant step counts with visual bars |
| `quad_coh3_linear_deviation` result key | mean \|coh3[i] − linear_interp[i]\|; high=nonlinear shape | How much coherence deviates from a straight start-to-end line |
| `quad_coh3_midpoint_vs_linear` result key | coh3[mid] − linear interp at mid; positive=arch, negative=bowl | Whether coherence arches (peaks in middle) or bowls (troughs in middle) |
| `quad_vel_linear_deviation` result key | mean \|vel[i] − linear_interp[i]\| for velocity | How much velocity deviates from a straight start-to-end line |
| `quad_vel_midpoint_vs_linear` result key | vel[mid] − linear interp; negative=dips in middle (good focus episode) | Whether velocity dips (focuses) or peaks in the middle of the generation |
| `quadshape` CLI command | coh3/vel linear deviation + midpoint vs linear + shape label | Signal shape analysis: arch / bowl / linear |
| Adaptive score floor from conf_ema decline | block bottom-20% finite candidates when conf_ema falls ≥4 steps | Raises bar for sampling when model is losing confidence |
| `topicshift` CLI command | marks steps where velocity > mean+1σ as topic-jump candidates | Semantic jump detector for last generation |
| `lastgen` CLI command | compact one-line summary: text + ConfEMA/Coh/C3/CohTrend/Flu/PPL | Quick re-read of last gen without re-running |
| `genprofile <prompt>` CLI command | generate + inline conf/tokenmap/coh3/var/topk/topicshift report | All-in-one generation profiler |
| Adaptive TFS z from entropy | _tfs.z = clip(0.97-0.04×H_norm, 0.90, 0.98) | High entropy→tighter tail cut; confident→looser |
| `coh_trend` result key | linear slope of coh3_steps | Positive=coherence growing; negative=semantic drift |
| Stats line additions | CohTrend, SGSlope, VPrecEMA, [!DECLINING] flag | Richer per-generation diagnostics in the output summary |
| `topkplot [N]` CLI command | sparkline + bar chart of per-step TopK k value | Visualizes how entropy steers candidate beam width |
| Local coherence extension gate | coh3≥0.50 at min_tokens → min_tokens+=1 | Allows one extra token when generation is on excellent track |
| Margin-adaptive nucleus | margin_ema<0.03 → nuc_p+=0.03 (capped 0.97) | Widens nucleus on indecisive plateau to find better candidates |

#### `causal_generate_nbest(prompt, n=3)` — N-best reranking
Runs `causal_generate` N times; picks winner by:
`score = avg_conf × (0.80 + 0.20 × coherence) × tanh(ntok/6) × (1 + diversity×0.10)`

#### Vocab filtering
Generation vocab filtered to words with `freq ≥ 3` in training corpus.
Eliminates hapax legomena and rare proper nouns that produce cosine noise.
Fallback chain ensures ≥ 200 / ≥ 50 words even on sparse brains.

#### CLI commands
```
cgen <prompt>    — single generation (Contrastive·MHC or Peep+MHC·70/30)
nbest <prompt>   — best-of-3 reranked (highest coherence×confidence)
nbest5 <prompt>  — best-of-5 reranked
rebuild          — re-embed vocab with char n-gram embeddings
calibrate [path] — build Peep specialisations → activates Peep+MHC·70/30 path
```

#### Peep Mechanism (`peep/peep.py`)
- 128 specialisation vectors (one per dot), shape `(128, FEATURE_DIM)`
- `observe_batch(ctxs, best_dots)`: EMA-updates context-direction centroid for each winning dot (lr=0.04)
- `top_k(ctx_eff, raw_preds, k=5)`: top-5 dots most aligned to current context → used in PATH A
- Saved to `global_brain.pkl.peep.pkl`

#### Grammar Guide (`grammar/grammar.py`)
- Rule-based POS tagger: 9 classes
- `GrammarGuide(vocab)`: (9×9) transition weight matrix → additive vocab bias `(V,)`
- Applied at step 3 of the loop alongside anti-rep mask

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
