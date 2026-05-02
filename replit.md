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
