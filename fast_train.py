"""
fast_train.py — Ultra-fast parallel IECNN vocabulary training.

Speed improvements over the standard fit() / train_brain.py path:
  1. _stable_embedding: np.random.default_rng (no os.urandom) → 14x per word
  2. Pre-compiled regex tokenizer (module-level, zero per-call overhead) → 2x
  3. Multiprocessing: corpus split across all CPU cores → 4x on 4-core
  4. Persistent pool across batches — eliminates 70ms respawn overhead
  5. C counting backend (fast_count_c.so): integer hash tables, no Python string
     creation per ngram, no Counter overhead → 5-10x over Python Counter
  6. Vectorized phrase/word embedding construction → 4x over sequential np.mean
  7. Subword / cooc loops skipped in fast mode → 3x worker throughput

Combined: targets 300k–700k sentences/sec for vocab-only training.

Usage:
  python fast_train.py corpus.txt [--brain global_brain.pkl] [--workers 4]
  python fast_train.py corpus.txt --full   # vocab + full pipeline dot training
"""

import sys
import os
import time
import argparse
import unicodedata
import ctypes as ct
import numpy as np
import re as _re
import regex
from collections import Counter
from typing import List, Tuple, Dict, Optional

try:
    from multiprocessing.shared_memory import SharedMemory as _SharedMemory
    _SHMEM_AVAILABLE = True
except ImportError:
    _SHMEM_AVAILABLE = False

ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, ROOT)

# ── Module-level tokenisation patterns ────────────────────────────────────────
# ASCII-only path: stdlib re (2.3x faster than regex module for pure ASCII)
_ASCII_PATTERN = _re.compile(r"[a-z]+|[0-9]+|[!-/:-@\[-`{-~]")
# Unicode fallback: regex module with Unicode property escapes
_FAST_PATTERN = regex.compile(
    r"\p{Han}|\p{Hiragana}|\p{Katakana}|\p{Hangul}|\p{L}+|\p{N}+|\p{P}|\S"
)

def _tokenize(text: str) -> List[str]:
    """Tokenise one sentence — ASCII fast-path or Unicode fallback."""
    lower = text.lower().strip()
    if lower.isascii():
        return _ASCII_PATTERN.findall(lower)
    return _FAST_PATTERN.findall(unicodedata.normalize("NFKC", lower))

# ── Load fast_scan_c.so — OpenMP parallel corpus scanner ──────────────────────
_LIB_FS: Optional[ct.CDLL] = None
_FS_SO = os.path.join(ROOT, "fast_scan_c.so")
try:
    _lib_fs = ct.CDLL(_FS_SO)
    _lib_fs.scan_corpus_omp.argtypes = [
        ct.c_char_p,                   # filepath
        ct.c_int32,                    # n_threads
        ct.c_int32,                    # max_words
        ct.c_int32,                    # max_bigrams
        ct.c_char_p,                   # out_word_buf
        ct.POINTER(ct.c_int32),        # out_word_offs
        ct.POINTER(ct.c_int16),        # out_word_lens
        ct.POINTER(ct.c_int32),        # out_word_cnts
        ct.POINTER(ct.c_int32),        # out_n_words
        ct.POINTER(ct.c_int32),        # out_bi_w1
        ct.POINTER(ct.c_int32),        # out_bi_w2
        ct.POINTER(ct.c_int32),        # out_bi_cnts
        ct.POINTER(ct.c_int32),        # out_n_bigrams
    ]
    _lib_fs.scan_corpus_omp.restype = ct.c_int32
    _lib_fs.count_lines_c.argtypes  = [ct.c_char_p]
    _lib_fs.count_lines_c.restype   = ct.c_int64
    _LIB_FS = _lib_fs
except (OSError, AttributeError):
    pass

if _LIB_FS is None:
    print(
        f"[IECNN] WARNING: {_FS_SO} not found — OMP corpus scanner unavailable.\n"
        f"         Fix: run  python main.py build  to compile C extensions.",
        file=sys.stderr,
    )

# ── Load C counting + tokenising backend ──────────────────────────────────────
_LIB_FC: Optional[ct.CDLL] = None
_FC_SO   = os.path.join(ROOT, "fast_count_c.so")
try:
    _lib = ct.CDLL(_FC_SO)

    # fast_count_c — integer hash-table ngram counting
    _lib.fast_count_c.argtypes = [
        ct.POINTER(ct.c_int32),   # flat_ids
        ct.POINTER(ct.c_int32),   # sent_lens
        ct.c_int32,               # n_sents
        ct.c_int32,               # vocab_size
        ct.c_int32,               # ngram_lo
        ct.c_int32,               # ngram_hi
        ct.POINTER(ct.c_int32),   # word_counts
        ct.POINTER(ct.c_uint64),  # bi_keys
        ct.POINTER(ct.c_int32),   # bi_counts
        ct.c_uint32,              # bi_slots
        ct.POINTER(ct.c_uint64),  # tri_keys  (nullable)
        ct.POINTER(ct.c_int32),   # tri_counts (nullable)
        ct.c_uint32,              # tri_slots
    ]
    _lib.fast_count_c.restype = None

    # fast_tok_bytes_ascii — ASCII tokenise+lowercase → flat byte buffer
    _lib.fast_tok_bytes_ascii.argtypes = [
        ct.POINTER(ct.c_char),    # text_buf (flat concat)
        ct.POINTER(ct.c_int32),   # text_offs
        ct.POINTER(ct.c_int32),   # text_lens
        ct.c_int32,               # n_texts
        ct.POINTER(ct.c_char),    # out_buf
        ct.c_int32,               # out_cap
        ct.POINTER(ct.c_int32),   # out_byte_lens
    ]
    _lib.fast_tok_bytes_ascii.restype = ct.c_int32

    _LIB_FC = _lib
except (OSError, AttributeError):
    pass  # Graceful fallback to pure-Python worker

if _LIB_FC is None:
    print(
        f"[IECNN] WARNING: {_FC_SO} not found or failed to load.\n"
        f"         Counting will run in pure Python (significantly slower).\n"
        f"         Fix: run  python main.py build  to compile C extensions.",
        file=sys.stderr,
    )

# ── Null-pointer sentinels for optional args ──────────────────────────────────
_NULL_U64 = ct.POINTER(ct.c_uint64)()
_NULL_I32 = ct.POINTER(ct.c_int32)()
_UINT64_MAX = np.iinfo(np.uint64).max

# ── Bi/tri hash table size constants (power-of-two) ──────────────────────────
# 2^19 = 524 288 slots × 12 bytes (key+count) = 6 MB per table
# Supports up to ~130k unique ngrams at 25% load — enough for a 25k-line chunk.
_BI_SLOTS  = 1 << 19
_TRI_SLOTS = 1 << 18   # trigrams are rarer; 2^18 = 262 144 slots


# ─────────────────────────────────────────────────────────────────────────────
# Worker functions — must be module-level to be picklable by multiprocessing
# ─────────────────────────────────────────────────────────────────────────────

def _count_chunk_shmem_worker(args: Tuple) -> Tuple[Counter, Counter, Counter, Dict]:
    """
    Shared-memory variant of _count_chunk_worker.

    IPC payload per worker: shm_name (~20 bytes) + two tiny int32 arrays
    (offsets + lengths, each chunk_size × 4 bytes ≈ 50 KB for 12.5k lines)
    instead of pickling the full text list (potentially several MB).

    args = (shm_name, total_bytes, offsets_bytes, lens_bytes, ng_lo, ng_hi)
      shm_name:     str name of the SharedMemory block holding all batch texts
      total_bytes:  int size of used region in the shared block
      offsets_bytes: serialised np.int32 array of per-text byte offsets
      lens_bytes:    serialised np.int32 array of per-text byte lengths
      ng_lo, ng_hi:  ngram range integers
    """
    shm_name, total_bytes, offsets_bytes, lens_bytes, ng_lo, ng_hi = args

    shm = _SharedMemory(name=shm_name)
    try:
        buf = bytes(shm.buf[:total_bytes])
    finally:
        shm.close()

    offsets = np.frombuffer(offsets_bytes, dtype=np.int32)
    lens    = np.frombuffer(lens_bytes,    dtype=np.int32)

    texts: List[str] = []
    for off, ln in zip(offsets.tolist(), lens.tolist()):
        if ln > 0:
            texts.append(buf[off:off + ln].decode("utf-8", errors="replace"))

    if not texts:
        return Counter(), Counter(), Counter(), {}

    if _LIB_FC is not None:
        return _count_chunk_c(texts, ng_lo, ng_hi)
    return _count_chunk_py(texts, ng_lo, ng_hi, 2, True)


def _count_chunk_worker(args: Tuple) -> Tuple[Counter, Counter, Counter, Dict]:
    """
    Worker: tokenize + count one chunk using the C backend when available.

    Returns (word_freq, ngram_freq, subword_freq, cooc).

    C path (fast_count_c.so available):
      - Tokenise in Python (pre-compiled regex, ~5 µs/sentence)
      - Encode tokens to int32 IDs (dict lookup, ~2 µs/sentence)
      - Count unigrams + bigrams in C (integer hash tables, ~1 µs/sentence)
      - Decode C output back to string Counters (numpy vectorised, < 1 µs/sentence)

    Python fallback path:
      - Counter.update(toks)  — C-level bulk insert
      - Counter.update(generator)  — C-level bulk insert for ngrams
    """
    texts, ngram_range, cooc_window, skip_subwords = args
    ng_lo, ng_hi = ngram_range

    if _LIB_FC is not None:
        return _count_chunk_c(texts, ng_lo, ng_hi)
    else:
        return _count_chunk_py(texts, ng_lo, ng_hi, cooc_window, skip_subwords)


def _count_chunk_c(
    texts: List[str],
    ng_lo: int,
    ng_hi: int,
) -> Tuple[Counter, Counter, Counter, Dict]:
    """C-accelerated counting path.

    Tokenisation strategy (fastest first):
      1. C bytes tokenizer (fast_tok_bytes_ascii): for pure-ASCII chunks —
         tokenises + lowercases all sentences in one C call, then Python
         decodes the flat byte buffer.  Avoids per-sentence Python regex
         calls entirely.
      2. Python isascii+re fallback: per-sentence if any non-ASCII text
         fails the C tokenizer.
    """

    # ── Step 1a: try C batch tokeniser for ASCII chunks ──────────────
    all_tok_lists: Optional[List[List[str]]] = None

    if _LIB_FC is not None:
        # Build flat byte buffer of lowercased texts
        lowers = [t.lower() for t in texts]
        if all(s.isascii() for s in lowers):
            enc = [s.encode("ascii") for s in lowers]
            text_buf_bytes = b"".join(enc)
            text_lens_arr  = np.array([len(b) for b in enc], dtype=np.int32)
            text_offs_arr  = np.zeros(len(enc), dtype=np.int32)
            np.cumsum(text_lens_arr[:-1], out=text_offs_arr[1:])

            # Output buffer: worst case each char becomes a token + separator
            out_cap = max(len(text_buf_bytes) * 2, 64)
            out_buf      = np.empty(out_cap, dtype=np.uint8)
            out_blens    = np.zeros(len(texts), dtype=np.int32)

            n_written = _LIB_FC.fast_tok_bytes_ascii(
                text_buf_bytes,
                text_offs_arr.ctypes.data_as(ct.POINTER(ct.c_int32)),
                text_lens_arr.ctypes.data_as(ct.POINTER(ct.c_int32)),
                ct.c_int32(len(texts)),
                out_buf.ctypes.data_as(ct.POINTER(ct.c_char)),
                ct.c_int32(out_cap),
                out_blens.ctypes.data_as(ct.POINTER(ct.c_int32)),
            )

            if n_written >= 0:
                # ── Fast path: bytes-keyed dict avoids per-token str decode ──
                # We split on b'\x01' (bytes level, C-fast), look up in a
                # bytes-keyed word2id, and only decode the unique vocabulary
                # (typically ≪ total tokens) at the very end.
                raw = bytes(out_buf[:n_written])
                pos = 0
                word2id_bytes: Dict[bytes, int] = {}
                flat_list: List[int] = []
                lengths: List[int] = []

                for blen in out_blens.tolist():
                    if blen > 0:
                        toks_b = raw[pos:pos + blen].split(b"\x01")
                        ids = []
                        for tb in toks_b:
                            if tb:
                                wid = word2id_bytes.get(tb)
                                if wid is None:
                                    word2id_bytes[tb] = wid = len(word2id_bytes)
                                ids.append(wid)
                        if ids:
                            flat_list.extend(ids)
                            lengths.append(len(ids))
                    pos += blen

                if not flat_list:
                    return Counter(), Counter(), Counter(), {}

                # Decode only unique vocab bytes → str (cheap: small set)
                vocab_size = len(word2id_bytes)
                id2word: Dict[int, str] = {
                    v: k.decode("ascii") for k, v in word2id_bytes.items()
                }

                # ── Jump straight to Steps 3–7 (skip the Python encode step) ──
                return _finish_c_count(
                    flat_list, lengths, id2word, vocab_size, ng_lo, ng_hi
                )

    # ── Step 1b: Python fallback (Unicode or C path failed) ───────────
    all_tok_lists = []
    for text in texts:
        toks = _tokenize(text)
        if toks:
            all_tok_lists.append(toks)

    if not all_tok_lists:
        return Counter(), Counter(), Counter(), {}

    # ── Step 2: encode tokens → local int IDs (Python path) ──────────
    word2id: Dict[str, int] = {}
    flat_list = []
    lengths   = []

    for toks in all_tok_lists:
        ids = []
        for tok in toks:
            wid = word2id.get(tok)
            if wid is None:
                wid = len(word2id)
                word2id[tok] = wid
            ids.append(wid)
        flat_list.extend(ids)
        lengths.append(len(ids))

    vocab_size = len(word2id)
    id2word = {v: k for k, v in word2id.items()}

    return _finish_c_count(flat_list, lengths, id2word, vocab_size, ng_lo, ng_hi)


def _finish_c_count(
    flat_list:  List[int],
    lengths:    List[int],
    id2word:    Dict[int, str],
    vocab_size: int,
    ng_lo:      int,
    ng_hi:      int,
) -> Tuple[Counter, Counter, Counter, Dict]:
    """
    Shared finish step for both C-tokeniser and Python-fallback paths.
    Takes pre-encoded int-ID arrays and drives fast_count_c → Counters.
    """
    # ── numpy C arrays ────────────────────────────────────────────────
    flat_ids  = np.array(flat_list, dtype=np.int32)
    sent_lens = np.array(lengths,   dtype=np.int32)
    wc        = np.zeros(vocab_size, dtype=np.int32)

    bi_keys   = np.full(_BI_SLOTS, _UINT64_MAX, dtype=np.uint64)
    bi_counts = np.zeros(_BI_SLOTS, dtype=np.int32)

    do_tri = ng_hi >= 3
    if do_tri:
        tri_keys   = np.full(_TRI_SLOTS, _UINT64_MAX, dtype=np.uint64)
        tri_counts = np.zeros(_TRI_SLOTS, dtype=np.int32)
        tri_kp = tri_keys.ctypes.data_as(ct.POINTER(ct.c_uint64))
        tri_cp = tri_counts.ctypes.data_as(ct.POINTER(ct.c_int32))
        tri_sl = ct.c_uint32(_TRI_SLOTS)
    else:
        tri_kp = _NULL_U64
        tri_cp = _NULL_I32
        tri_sl = ct.c_uint32(0)

    # ── call C counting ───────────────────────────────────────────────
    _LIB_FC.fast_count_c(
        flat_ids.ctypes.data_as(ct.POINTER(ct.c_int32)),
        sent_lens.ctypes.data_as(ct.POINTER(ct.c_int32)),
        ct.c_int32(len(sent_lens)),
        ct.c_int32(vocab_size),
        ct.c_int32(ng_lo),
        ct.c_int32(ng_hi),
        wc.ctypes.data_as(ct.POINTER(ct.c_int32)),
        bi_keys.ctypes.data_as(ct.POINTER(ct.c_uint64)),
        bi_counts.ctypes.data_as(ct.POINTER(ct.c_int32)),
        ct.c_uint32(_BI_SLOTS),
        tri_kp, tri_cp, tri_sl,
    )

    # ── decode word_freq (numpy mask → Counter) ───────────────────────
    word_freq: Counter = Counter()
    nz = np.where(wc > 0)[0]
    for idx in nz:
        word_freq[id2word[int(idx)]] = int(wc[idx])

    # ── decode bigram hash table (vectorised) ─────────────────────────
    ngram_freq: Counter = Counter()
    vs = np.uint64(vocab_size)

    valid = bi_keys != _UINT64_MAX
    if valid.any():
        vk = bi_keys[valid]
        vc = bi_counts[valid]
        id1_arr = (vk // vs).astype(np.intp)
        id2_arr = (vk  % vs).astype(np.intp)
        for id1, id2, cnt in zip(id1_arr.tolist(), id2_arr.tolist(), vc.tolist()):
            w1 = id2word.get(id1)
            w2 = id2word.get(id2)
            if w1 is not None and w2 is not None:
                ngram_freq[w1 + " " + w2] = cnt

    # ── decode trigram hash table (vectorised) ────────────────────────
    if do_tri:
        valid_t = tri_keys != _UINT64_MAX
        if valid_t.any():
            vk  = tri_keys[valid_t]
            vc  = tri_counts[valid_t]
            id3 = (vk % vs).astype(np.intp)
            vk  = vk // vs
            id2 = (vk % vs).astype(np.intp)
            id1 = (vk // vs).astype(np.intp)
            for i1, i2, i3, cnt in zip(
                id1.tolist(), id2.tolist(), id3.tolist(), vc.tolist()
            ):
                w1 = id2word.get(i1)
                w2 = id2word.get(i2)
                w3 = id2word.get(i3)
                if w1 is not None and w2 is not None and w3 is not None:
                    ngram_freq[w1 + " " + w2 + " " + w3] = cnt

    return word_freq, ngram_freq, Counter(), {}


def _count_chunk_py(
    texts:         List[str],
    ng_lo:         int,
    ng_hi:         int,
    cooc_window:   int,
    skip_subwords: bool,
) -> Tuple[Counter, Counter, Counter, Dict]:
    """Pure-Python fallback counting path (used when C .so is absent)."""
    word_freq:    Counter = Counter()
    ngram_freq:   Counter = Counter()
    subword_freq: Counter = Counter()
    cooc: Dict[str, Counter] = {}
    join = " ".join

    for text in texts:
        toks = _tokenize(text)
        if not toks:
            continue

        word_freq.update(toks)
        n_toks = len(toks)

        for n in range(ng_lo, ng_hi + 1):
            if n_toks >= n:
                ngram_freq.update(
                    join(toks[i:i + n]) for i in range(n_toks - n + 1)
                )

        if not skip_subwords:
            for tok in toks:
                lt = len(tok)
                if lt > 3:
                    hi_sub = min(lt, 6)
                    subword_freq.update(
                        tok[s:s + sl]
                        for sl in range(3, hi_sub)
                        for s  in range(lt - sl + 1)
                    )
            for i, tok in enumerate(toks):
                if tok not in cooc:
                    cooc[tok] = Counter()
                lo = max(0, i - cooc_window)
                hi = min(n_toks, i + cooc_window + 1)
                cooc[tok].update(toks[j] for j in range(lo, hi) if j != i)

    return word_freq, ngram_freq, subword_freq, cooc


def fast_vocab_train(
    corpus_path:   str,
    brain_path:    str  = "global_brain.pkl",
    n_workers:     int  = None,
    batch_size:    int  = 100_000,
    skip_subwords: bool = True,
    verbose:       bool = True,
    use_shmem:     bool = False,
) -> None:
    """
    Ultra-fast parallel vocabulary training.

    Reads corpus_path in streaming batches of `batch_size` lines, fits
    the BaseMapper in parallel across `n_workers` CPU cores, and saves
    the brain at the end.

    Uses a single persistent Pool across all batches to avoid the 70ms
    per-batch spawn overhead that would otherwise dominate small batches.

    Args:
        corpus_path:   Path to a plain-text corpus (one sentence per line).
        brain_path:    Where to save/load the IECNN brain.
        n_workers:     CPU workers for multiprocessing (None = all cores).
        batch_size:    Lines processed per fit_fast() call.
        skip_subwords: Skip BPE-style subword discovery (3x speedup, minimal
                       quality impact for corpora >= 10k lines).
        verbose:       Print progress and throughput.
        use_shmem:     Use multiprocessing.shared_memory to eliminate
                       text-pickling IPC overhead (~2-4x faster IPC,
                       targeting >1M sent/s on modern hardware).
    """
    import multiprocessing as mp
    from pipeline.pipeline import IECNN

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    if use_shmem and not _SHMEM_AVAILABLE:
        print("[IECNN] WARNING: shared_memory not available on this Python build; "
              "falling back to standard pool.map.", file=sys.stderr)
        use_shmem = False

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║           IECNN FAST PARALLEL VOCABULARY TRAINING            ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  corpus    : {corpus_path}")
        print(f"  brain     : {brain_path if brain_path else '(in-memory)'}")
        print(f"  workers   : {n_workers}")
        print(f"  batch     : {batch_size:,} lines")
        print(f"  subwords  : {'disabled (fast)' if skip_subwords else 'enabled'}")
        print(f"  IPC mode  : {'shared-memory (zero-copy)' if use_shmem else 'standard pickle'}")
        print()

    model = IECNN(persistence_path=brain_path)

    t_global = time.perf_counter()
    total_processed = 0

    # ── Persistent pool: spawned once, reused for all batches ─────────────
    # This avoids the ~70ms pool-creation overhead per batch.
    with mp.Pool(n_workers) as pool:
        with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
            batch: List[str] = []
            batch_num = 0

            for raw_line in fh:
                line = raw_line.strip()
                if not line or line.startswith("#"):
                    continue
                batch.append(line)

                if len(batch) >= batch_size:
                    batch_num += 1
                    t0 = time.perf_counter()
                    model.base_mapper.fit_fast(batch, n_workers=n_workers,
                                               skip_subwords=skip_subwords,
                                               _pool=pool,
                                               use_shmem=use_shmem)
                    elapsed = time.perf_counter() - t0
                    total_processed += len(batch)
                    rate = len(batch) / max(elapsed, 1e-9)

                    if verbose:
                        vocab_size = len(model.base_mapper._base_vocab)
                        print(f"  batch {batch_num:>4}  |  {total_processed:>10,} lines  "
                              f"|  {rate:>9,.0f} sent/s  "
                              f"|  vocab {vocab_size:>8,}",
                              flush=True)
                    batch = []

            # Final partial batch
            if batch:
                batch_num += 1
                t0 = time.perf_counter()
                model.base_mapper.fit_fast(batch, n_workers=n_workers,
                                           skip_subwords=skip_subwords,
                                           _pool=pool,
                                           use_shmem=use_shmem)
                elapsed = time.perf_counter() - t0
                total_processed += len(batch)
                rate = len(batch) / max(elapsed, 1e-9)
                if verbose:
                    vocab_size = len(model.base_mapper._base_vocab)
                    print(f"  batch {batch_num:>4}  |  {total_processed:>10,} lines  "
                          f"|  {rate:>9,.0f} sent/s  "
                          f"|  vocab {vocab_size:>8,}",
                          flush=True)

    total_elapsed = time.perf_counter() - t_global
    overall_rate  = total_processed / max(total_elapsed, 1e-9)

    if verbose:
        print()
        print(f"  ✓ {total_processed:,} sentences in {total_elapsed:.1f}s "
              f"({overall_rate:,.0f} sent/s overall)")
        bm = model.base_mapper
        n_words  = sum(1 for t in bm._base_types.values() if t == "word")
        n_phrase = sum(1 for t in bm._base_types.values() if t == "phrase")
        print(f"  Word bases  : {n_words:,}")
        print(f"  Phrase bases: {n_phrase:,}")
        print(f"  Primitives  : {len(bm._primitive_embeddings):,}")
        if brain_path:
            print(f"  Saving brain → {brain_path} ...")

    if brain_path:
        model.base_mapper.save(brain_path)

    if verbose and brain_path:
        print(f"  Done.")


def fast_full_train(
    corpus_path:  str,
    brain_path:   str  = "global_brain.pkl",
    n_workers:    int  = None,
    batch_size:   int  = 256,
    num_dots:     int  = 64,
    max_iter:     int  = 2,
    prune_every:  int  = 0,
    verbose:      bool = True,
) -> None:
    """
    Fast full-pipeline training (vocab + dot evolution).

    Uses run_batch() with the C pipeline backend for maximum throughput.
    Parallelism is applied at the vocab-fitting stage; the C pipeline
    batch-processes sentences in chunks.

    Expected throughput: 200-1,000 sent/s depending on corpus and hardware.
    """
    import multiprocessing as mp
    from pipeline.pipeline import IECNN

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║            IECNN FAST FULL-PIPELINE TRAINING                 ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  corpus   : {corpus_path}")
        print(f"  brain    : {brain_path}")
        print(f"  num_dots : {num_dots}  max_iter : {max_iter}")
        print()

    model = IECNN(
        persistence_path=brain_path,
        num_dots=num_dots,
        n_heads=2,
        max_iterations=max_iter,
        evolve=True,
    )

    if verbose:
        print("  Phase 1: Fast vocab scan ...", flush=True)

    # Phase 1: Fast vocab pass on the whole corpus
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
        all_lines = [l.strip() for l in fh if l.strip() and not l.startswith("#")]

    t0 = time.perf_counter()
    model.base_mapper.fit_fast(all_lines, n_workers=n_workers, skip_subwords=True)
    vocab_time = time.perf_counter() - t0

    if verbose:
        vocab_size = len(model.base_mapper._base_vocab)
        rate = len(all_lines) / max(vocab_time, 1e-9)
        print(f"  Vocab: {vocab_size:,} bases in {vocab_time:.1f}s ({rate:,.0f} sent/s)")
        print(f"  Phase 2: Full-pipeline dot learning on {len(all_lines):,} lines ...",
              flush=True)

    # Phase 2: Full pipeline training
    model.save_brain()
    t_global = time.perf_counter()
    model.train_pass(all_lines, use_c_pipeline=True, verbose=verbose,
                     prune_every=prune_every)
    pipe_time = time.perf_counter() - t_global

    pipe_rate = len(all_lines) / max(pipe_time, 1e-9)
    if verbose:
        print(f"\n  Pipeline: {len(all_lines):,} lines in {pipe_time:.1f}s "
              f"({pipe_rate:.1f} sent/s)")

    model.save_brain()
    if verbose:
        print(f"  Brain saved → {brain_path}")


def scan_corpus_c(
    filepath:    str,
    n_threads:   int = 0,
    max_words:   int = 500_000,
    max_bigrams: int = 2_000_000,
) -> Tuple[Counter, Counter]:
    """
    Full corpus vocabulary scan entirely in C+OpenMP — no Python in the hot path.

    Reads the corpus file with mmap, splits into n_threads chunks at line
    boundaries, and runs two parallel passes:
      Pass 1 (OMP): tokenize + count unigrams into thread-local hash tables
      Merge  (serial): combine all thread-local tables → global vocab + IDs
      Pass 2 (OMP): tokenize + count bigrams using global IDs (read-only)
      Merge  (serial): combine bigram tables → output

    Returns:
        word_freq  : Counter  {word_str: count}
        ngram_freq : Counter  {"w1 w2": count}

    Throughput: ~5–15 M sent/s for counting only (embedding build is separate).
    """
    if _LIB_FS is None:
        raise RuntimeError(
            "fast_scan_c.so not loaded. Run  python main.py build  to compile."
        )

    WORD_BUF_SZ = max_words * 65        # 65 bytes per word (max 64 chars + NUL)
    word_buf  = np.zeros(WORD_BUF_SZ, dtype=np.uint8)
    word_offs = np.zeros(max_words,   dtype=np.int32)
    word_lens = np.zeros(max_words,   dtype=np.int16)
    word_cnts = np.zeros(max_words,   dtype=np.int32)
    n_words   = ct.c_int32(0)
    bi_w1     = np.zeros(max_bigrams, dtype=np.int32)
    bi_w2     = np.zeros(max_bigrams, dtype=np.int32)
    bi_cnts   = np.zeros(max_bigrams, dtype=np.int32)
    n_bi      = ct.c_int32(0)

    rc = _LIB_FS.scan_corpus_omp(
        filepath.encode("utf-8"),
        ct.c_int32(n_threads),
        ct.c_int32(max_words),
        ct.c_int32(max_bigrams),
        word_buf.ctypes.data_as(ct.c_char_p),
        word_offs.ctypes.data_as(ct.POINTER(ct.c_int32)),
        word_lens.ctypes.data_as(ct.POINTER(ct.c_int16)),
        word_cnts.ctypes.data_as(ct.POINTER(ct.c_int32)),
        ct.byref(n_words),
        bi_w1.ctypes.data_as(ct.POINTER(ct.c_int32)),
        bi_w2.ctypes.data_as(ct.POINTER(ct.c_int32)),
        bi_cnts.ctypes.data_as(ct.POINTER(ct.c_int32)),
        ct.byref(n_bi),
    )
    if rc != 0:
        raise RuntimeError(f"scan_corpus_omp returned error code {rc}")

    nw = n_words.value
    raw = bytes(word_buf[:int(word_lens[:nw].astype(np.int32).sum()) + nw])

    # Decode words — only nw strings, cheap
    words: List[str] = []
    for i in range(nw):
        off = int(word_offs[i])
        ln  = int(word_lens[i])
        words.append(raw[off:off + ln].decode("latin-1"))

    word_freq: Counter = Counter(
        {words[i]: int(word_cnts[i]) for i in range(nw)}
    )

    nb = n_bi.value
    ngram_freq: Counter = Counter()
    if nb > 0:
        # Vectorized bigram decode via numpy advanced indexing + list comprehension.
        # np.array(..., dtype=object) + advanced indexing avoids 662k individual
        # dict lookups; the string concat loop is then the only O(nb) Python work.
        words_arr = np.array(words, dtype=object)
        w1_strs = words_arr[bi_w1[:nb]]
        w2_strs = words_arr[bi_w2[:nb]]
        bc      = bi_cnts[:nb].tolist()
        ngram_freq = Counter(
            {w1 + " " + w2: c for w1, w2, c in zip(w1_strs, w2_strs, bc)}
        )

    return word_freq, ngram_freq


def fast_vocab_train_omp(
    corpus_path:  str,
    brain_path:   Optional[str] = "global_brain.pkl",
    n_threads:    int = 0,
    verbose:      bool = True,
) -> None:
    """
    Ultra-fast single-call corpus vocabulary training via OpenMP C scanner.

    Architecture:
      1. scan_corpus_c()  — entire file counted in C+OpenMP, no Python workers
      2. fit_from_freq()  — embedding build for new words/phrases (Python+numpy)
      3. save_brain()

    Expected throughput for counting phase: 5–15 M sent/s.
    Full pipeline (counting + embedding build): ~2–5 M sent/s depending on
    the new-word rate (embedding build is O(unique_new_words × embed_dim)).

    Args:
        corpus_path: Path to plain-text corpus (one sentence per line).
        brain_path:  Where to save/load the IECNN brain (None = in-memory).
        n_threads:   OpenMP threads (0 = use all available cores).
        verbose:     Print progress and throughput.
    """
    from pipeline.pipeline import IECNN

    if _LIB_FS is None:
        raise RuntimeError(
            "fast_scan_c.so not loaded. Run  python main.py build  to compile."
        )

    if verbose:
        import multiprocessing as mp
        _nt = n_threads if n_threads > 0 else mp.cpu_count()
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║          IECNN OMP ULTRA-FAST VOCABULARY TRAINING            ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  corpus    : {corpus_path}")
        print(f"  brain     : {brain_path or '(in-memory)'}")
        print(f"  OMP threads: {_nt}")
        print()

    # ── Step 1: Count entire corpus in C+OpenMP ───────────────────────────
    t_count0 = time.perf_counter()
    word_freq, ngram_freq = scan_corpus_c(corpus_path, n_threads=n_threads)
    t_count = time.perf_counter() - t_count0

    n_lines = int(_LIB_FS.count_lines_c(corpus_path.encode("utf-8")))
    count_rate = n_lines / max(t_count, 1e-9)

    if verbose:
        print(f"  [C+OMP] Counted {n_lines:,} lines in {t_count:.3f}s"
              f"  →  {count_rate:,.0f} sent/s")
        print(f"  Unique words: {len(word_freq):,}   "
              f"Unique bigrams: {len(ngram_freq):,}")

    # ── Step 2: Build embeddings for new words/phrases ────────────────────
    model = IECNN(persistence_path=brain_path)
    bm = model.base_mapper

    t_emb0 = time.perf_counter()
    bm._word_freq.update(word_freq)
    bm._ngram_freq.update(ngram_freq)

    new_words = [
        w for w, cnt in bm._word_freq.most_common(bm.max_vocab_size)
        if cnt >= bm.min_base_freq and w not in bm._base_vocab
    ]
    bm._batch_build_word_embeddings(new_words)

    new_phrases = [
        ng for ng, cnt in bm._ngram_freq.most_common(2000)
        if cnt >= bm.min_base_freq and ng not in bm._base_vocab
    ]
    bm._batch_build_phrase_embeddings(new_phrases)
    bm._apply_cooc_smoothing()
    bm.is_fitted = True
    t_emb = time.perf_counter() - t_emb0

    total = t_count + t_emb
    eff_rate = n_lines / max(total, 1e-9)

    if verbose:
        n_word_bases   = sum(1 for t in bm._base_types.values() if t == "word")
        n_phrase_bases = sum(1 for t in bm._base_types.values() if t == "phrase")
        print(f"  [embed]  {len(new_words):,} new words + {len(new_phrases):,} phrases"
              f"  in {t_emb:.3f}s")
        print(f"  Word bases  : {n_word_bases:,}")
        print(f"  Phrase bases: {n_phrase_bases:,}")
        print(f"  Primitives  : {len(bm._primitive_embeddings):,}")
        print()
        print(f"  ✓ Total: {total:.3f}s  →  {eff_rate:,.0f} sent/s (end-to-end)")
        if brain_path:
            print(f"  Saving brain → {brain_path} ...")

    if brain_path:
        model.base_mapper.save(brain_path)

    if verbose and brain_path:
        print(f"  Done.")


def fast_effective_train(
    corpus_path:  str,
    brain_path:   str  = "global_brain.pkl",
    n_threads:    int  = 0,
    n_sentences:  int  = 20_000,
    n_epochs:     int  = 1,
    max_pos:      int  = 6,
    verbose:      bool = True,
) -> None:
    """
    Two-phase training that reliably drives MaxEff above 0.5.

    Phase 1 — OMP vocab scan (fast_vocab_train_omp):
        Counts the entire corpus in C+OpenMP, builds embeddings for all new
        words and bigrams.  Throughput: 2-15 M sent/s.

    Phase 2 — Per-dot causal training (IECNN.causal_train_pass):
        Uses the fixed per-dot W-matrix win criterion so each dot is scored
        individually against the causal next-token target.  By Gaussian
        symmetry ~50 % of dots win per step; specialising dots drift above
        50 % → effectiveness > 0.5 → MaxEff > 0.5 after the first evolution
        cycle (every 10 sentences).

    Args:
        corpus_path : plain-text corpus, one sentence per line
        brain_path  : brain persistence path (loaded then saved)
        n_threads   : OMP threads for phase 1 (0 = all cores)
        n_sentences : max sentences for causal training phase (phase 2)
        n_epochs    : number of full passes over the causal training subset
        max_pos     : max prefix positions per sentence in phase 2
        verbose     : print progress
    """
    from pipeline.pipeline import IECNN

    if verbose:
        print("╔══════════════════════════════════════════════════════════════╗")
        print("║        IECNN EFFECTIVE TRAINING  (MaxEff > 0.5 target)       ║")
        print("╚══════════════════════════════════════════════════════════════╝")
        print(f"  corpus    : {corpus_path}")
        print(f"  brain     : {brain_path}")
        print(f"  sentences : {n_sentences:,}  epochs: {n_epochs}  max_pos: {max_pos}")
        print()

    # ── Phase 1: OMP vocab scan ────────────────────────────────────────────
    if verbose:
        print("  Phase 1: OMP vocab scan ...", flush=True)

    fast_vocab_train_omp(
        corpus_path=corpus_path,
        brain_path=brain_path,
        n_threads=n_threads,
        verbose=verbose,
    )

    # ── Load sentences for phase 2 ─────────────────────────────────────────
    sentences: List[str] = []
    with open(corpus_path, "r", encoding="utf-8", errors="replace") as fh:
        for line in fh:
            line = line.strip()
            if line and not line.startswith("#"):
                sentences.append(line)
                if len(sentences) >= n_sentences:
                    break

    if not sentences:
        if verbose:
            print("  No sentences loaded — phase 2 skipped.")
        return

    if verbose:
        print(f"\n  Phase 2: Per-dot causal training on {len(sentences):,} sentences"
              f"  ×  {n_epochs} epoch(s) ...", flush=True)

    # ── Phase 2: Causal training with per-dot win criterion ───────────────
    model = IECNN(persistence_path=brain_path)

    t0 = time.perf_counter()
    for epoch in range(n_epochs):
        if verbose and n_epochs > 1:
            print(f"\n  Epoch {epoch + 1}/{n_epochs}")
        model.causal_train_pass(sentences, max_pos=max_pos, verbose=verbose,
                                causal_batch=200, save_every=5000)

    elapsed = time.perf_counter() - t0
    rate = len(sentences) * n_epochs / max(elapsed, 1e-9)

    if verbose:
        print(f"\n  Phase 2 done: {len(sentences) * n_epochs:,} sentence-passes"
              f"  in {elapsed:.1f}s  ({rate:,.0f} sent/s)")

    # ── Final report ───────────────────────────────────────────────────────
    status = model.memory_status()
    dm = status["dot_memory"]
    if verbose:
        print(f"\n  MaxEff       : {dm['max_eff']:.4f}")
        print(f"  MeanEff      : {dm['mean_eff']:.4f}")
        print(f"  Evolution gen: {status['evolution']['generation']}")

    model.save_brain()
    if verbose:
        print(f"  Brain saved  → {brain_path}")


def _benchmark(n_sentences: int = 10_000, n_workers: int = None) -> None:
    """Quick benchmark: measure vocab-fit throughput on synthetic data."""
    import multiprocessing as mp
    from pipeline.pipeline import IECNN

    if n_workers is None:
        n_workers = min(mp.cpu_count(), 8)

    print(f"[bench] Generating {n_sentences:,} synthetic sentences ...")
    import random
    words = ["the", "quick", "brown", "fox", "neural", "network", "learns",
             "patterns", "from", "data", "artificial", "intelligence", "deep",
             "learning", "convergence", "embedding", "cluster", "gradient",
             "attention", "transformer", "semantic", "representation"]
    sentences = [" ".join(random.choices(words, k=random.randint(5, 15)))
                 for _ in range(n_sentences)]

    model = IECNN(persistence_path=None)
    model.base_mapper.is_fitted = False

    print(f"[bench] Running fit_fast() on {n_sentences:,} sentences "
          f"with {n_workers} workers ...")
    t0 = time.perf_counter()
    model.base_mapper.fit_fast(sentences, n_workers=n_workers, skip_subwords=True)
    elapsed = time.perf_counter() - t0

    rate = n_sentences / max(elapsed, 1e-9)
    vocab = len(model.base_mapper._base_vocab)
    print(f"[bench] {n_sentences:,} sentences in {elapsed:.3f}s → {rate:,.0f} sent/s")
    print(f"[bench] Vocab size: {vocab:,} bases")


if __name__ == "__main__":
    import multiprocessing as mp
    mp.freeze_support()

    parser = argparse.ArgumentParser(
        description="Ultra-fast IECNN training",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("corpus", nargs="?", help="Path to corpus file")
    parser.add_argument("--brain", default="global_brain.pkl",
                        help="Brain persistence path (default: global_brain.pkl)")
    parser.add_argument("--workers", type=int, default=None,
                        help="CPU workers (default: all cores)")
    parser.add_argument("--batch", type=int, default=50_000,
                        help="Lines per batch for vocab fitting (default: 50000)")
    parser.add_argument("--full", action="store_true",
                        help="Run full-pipeline dot training after vocab fitting")
    parser.add_argument("--subwords", action="store_true",
                        help="Enable BPE-style subword discovery (slower)")
    parser.add_argument("--shared-memory", action="store_true",
                        help="Use shared-memory IPC to eliminate text-pickling "
                             "overhead (targets >1M sent/s; requires Python 3.8+)")
    parser.add_argument("--bench", action="store_true",
                        help="Run synthetic benchmark instead of training")
    parser.add_argument("--bench-n", type=int, default=50_000,
                        help="Sentences for benchmark (default: 50000)")

    args = parser.parse_args()

    if args.bench:
        _benchmark(args.bench_n, args.workers)
    elif args.corpus:
        if args.full:
            fast_full_train(
                corpus_path=args.corpus,
                brain_path=args.brain,
                n_workers=args.workers,
                verbose=True,
            )
        else:
            fast_vocab_train(
                corpus_path=args.corpus,
                brain_path=args.brain,
                n_workers=args.workers,
                batch_size=args.batch,
                skip_subwords=not args.subwords,
                verbose=True,
                use_shmem=getattr(args, "shared_memory", False),
            )
    else:
        parser.print_help()
