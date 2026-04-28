"""
BaseMapping — Structured input representation for IECNN.

Design principles:
  - a-z (and 0-9) are PRE-SEEDED as primitive bases — always available
  - Each TOKEN (word or phrase) maps to exactly ONE row in the BaseMap matrix
  - Known bases (words/phrases discovered by frequency) get their own embedding
  - Unknown words get ONE row whose embedding is composed from their
    constituent character (a-z) embeddings — never split into separate rows
  - Phrases (frequent bigrams/trigrams) are detected and treated as single bases

Improvements (v0.7.0):
  1. Cooccurrence-based semantic enrichment — words that appear together in
     the corpus get their embeddings smoothed toward each other, giving
     distributional semantic grounding without a large pretrained model.
  2. Character bigram composition — in addition to unigram character features,
     consecutive character pairs contribute to the composed embedding, capturing
     subword structure (prefixes, suffixes, letter patterns).
  3. Morphological suffix flags — two flag dimensions are repurposed to detect
     verb/noun/adjective suffixes (-ing, -tion, -ous, etc.), giving POS-style
     signal without a tagger.
  4. Semantic context summary — the 4-dim context window now includes actual
     embedding cosine similarity to neighbors rather than just token statistics.
  5. IDF-weighted pooling — the pool() method offers an 'idf' mode that weights
     rare tokens more heavily (common stop words contribute less).

Feature vector layout (128 dims):
  [0  : 96 ]  base_embedding    — stable/composed embedding for the base token
  [96 : 104]  position_enc      — sinusoidal position encoding (8 dims)
  [104: 108]  freq_features     — frequency statistics (4 dims)
  [108: 124]  modifier_flags    — structural/linguistic/morphological flags (16 dims)
  [124: 128]  context_summary   — semantic 4-dim context window summary
"""

import numpy as np
import re
import math
import ctypes
import os
import pickle
from collections import Counter
from typing import List, Dict, Tuple, Optional, Any
try:
    from PIL import Image
    import librosa
    import cv2
except ImportError:
    pass

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "basemapping_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            # Define argtypes and restypes for safety
            _lib.compose_from_chars.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
            ]
            _lib.compose_from_chars.restype = None

            _lib.sinusoidal_position_enc.argtypes = [
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
            ]
            _lib.sinusoidal_position_enc.restype = None

            _lib.normalize_vector.argtypes = [ctypes.POINTER(ctypes.c_float), ctypes.c_int]
            _lib.normalize_vector.restype = None

            _lib.mean_pool.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.c_int, ctypes.c_int, ctypes.POINTER(ctypes.c_float)
            ]
            _lib.mean_pool.restype = None

            _lib.attention_pool.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_float, ctypes.POINTER(ctypes.c_float)
            ]
            _lib.attention_pool.restype = None

            _lib.cooccurrence_smooth.argtypes = [
                ctypes.POINTER(ctypes.c_float), ctypes.POINTER(ctypes.c_int), ctypes.POINTER(ctypes.c_float),
                ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_float
            ]
            _lib.cooccurrence_smooth.restype = None
        except Exception:
            _lib = None
    return _lib


# ── Primitive bases: a-z, 0-9, basic punctuation, and ACTION tokens ──
_PRIMITIVES = list("abcdefghijklmnopqrstuvwxyz0123456789") + [
    ".", ",", "!", "?", "'", "-", "_", "/", "(", ")",
    "WRITE_CODE", "RUN_CODE", "SEARCH", "GENERATE_IMAGE", "SPEAK"
]

EMBED_DIM   = 224
POS_DIM     = 8
FREQ_DIM    = 4
FLAG_DIM    = 16
CTX_DIM     = 4
FEATURE_DIM = EMBED_DIM + POS_DIM + FREQ_DIM + FLAG_DIM + CTX_DIM  # = 256

# Modality flag indices in modifier_flags (16 dims total)
# Dims 12-15 used for modality identification
MOD_TEXT  = 12
MOD_IMAGE = 13
MOD_AUDIO = 14
MOD_VIDEO = 15

# Morphological suffix sets for flag dims 10 and 11
_VERB_SUFFIXES = ("ing", "ed", "ize", "ise", "ify", "ate")
_NOUN_SUFFIXES = ("tion", "sion", "ness", "ity", "ment", "er", "or", "ist", "ism")
_ADJ_SUFFIXES  = ("ful", "ous", "ive", "able", "ible", "al", "ic", "ent", "ant")
_ADV_SUFFIX    = "ly"

_PREFIXES = (
    "un", "re", "in", "im", "dis", "pre", "post", "anti", "mis", "non",
    "pro", "over", "under", "trans", "inter", "sub", "ex", "de"
)
_SUFFIXES = _VERB_SUFFIXES + _NOUN_SUFFIXES + _ADJ_SUFFIXES + (_ADV_SUFFIX,)


def _stable_embedding(token: str, dim: int) -> np.ndarray:
    """Stable hash-based unit-sphere embedding for a token string."""
    seed = abs(hash(token)) % (2 ** 31)
    rng = np.random.RandomState(seed)
    v = rng.randn(dim).astype(np.float32)
    n = np.linalg.norm(v)
    return v / n if n > 1e-10 else v


class BaseMap:
    """
    Structured representation of one input produced by BaseMapper.

    Attributes:
      matrix      — (num_tokens, 128) float32 feature matrix
      tokens      — list of base token strings (one per row)
      token_types — list of type tags: 'primitive'|'word'|'phrase'|'composed'
      modifiers   — dict of modifier arrays
      metadata    — info dict
    """

    def __init__(self, matrix, tokens, token_types, modifiers, metadata):
        self.matrix      = matrix
        self.tokens      = tokens
        self.token_types = token_types
        self.modifiers   = modifiers
        self.metadata    = metadata

    def __len__(self):
        return len(self.tokens)

    def pool(self, method: str = "mean", query: Optional[np.ndarray] = None) -> np.ndarray:
        """
        Pool the token rows into a single vector.

        Methods:
          "mean"      — uniform average of all rows (default)
          "max"       — element-wise maximum
          "idf"       — IDF-style: down-weight high-frequency tokens.
          "attention" — weight tokens by similarity to a query vector (or centroid).
        """
        if method == "mean":
            return np.mean(self.matrix, axis=0)
        if method == "max":
            return np.max(self.matrix, axis=0)
        if method == "idf":
            freq_col = self.matrix[:, EMBED_DIM + POS_DIM]
            idf_w    = np.clip(1.0 - freq_col * 0.7, 0.3, 1.0)
            weighted = self.matrix * idf_w[:, None]
            denom    = idf_w.sum()
            return (weighted.sum(axis=0) / max(float(denom), 1e-10)).astype(np.float32)
        if method == "attention":
            q = query if query is not None else np.mean(self.matrix, axis=0)
            # Scaled dot-product attention scores
            dim = self.matrix.shape[1]
            scores = (self.matrix @ q) / np.sqrt(dim)
            scores -= np.max(scores)
            weights = np.exp(scores * 2.0)
            weights /= weights.sum() + 1e-10
            return (weights[:, None] * self.matrix).sum(axis=0).astype(np.float32)

        return np.mean(self.matrix, axis=0)

    def slice(self, start: int, end: int) -> np.ndarray:
        return self.matrix[start:end]

    def __repr__(self):
        types = Counter(self.token_types)
        return (f"BaseMap(tokens={len(self.tokens)}, shape={self.matrix.shape}, "
                f"types={dict(types)})")


class BaseMapper:
    """
    Converts text input into a BaseMap of structured token representations.

    BaseMapping logic:
      1. Start with a-z, 0-9, punctuation as pre-seeded primitives
      2. On fit(): discover word-level and phrase-level bases from a corpus
         (tokens appearing >= min_base_freq times become new named bases),
         and collect cooccurrence counts within a context window.
         After building initial embeddings, apply a cooccurrence smoothing
         pass so words that appear together get similar embeddings.
      3. On transform(text):
         a. Tokenize into words (and detect known phrases)
         b. For each token (one row per token):
            - If it's a known base → use that base's (possibly enriched) embedding
            - If it's an unknown word → compose embedding from its characters
              using unigram + bigram character primitive combinations
            - Either way: ONE row per token
         c. Attach modifiers: position, frequency, morphological flags, context
            (context now includes actual embedding cosine similarity to neighbors)
         d. Output BaseMap matrix
    """

    def __init__(
        self,
        feature_dim:   int   = FEATURE_DIM,
        min_base_freq: int   = 2,
        max_vocab_size: int  = 50_000,
        ngram_range:   Tuple[int, int] = (2, 3),
        context_window: int  = 3,
        cooc_window:   int   = 3,
        cooc_alpha:    float = 0.15,
    ):
        self.feature_dim    = feature_dim
        self.min_base_freq  = min_base_freq
        self.max_vocab_size = max_vocab_size
        self.ngram_range    = ngram_range
        self.context_window = context_window
        self.cooc_window    = cooc_window
        self.cooc_alpha     = cooc_alpha

        # Pre-seed primitive bases (a-z, 0-9, punctuation)
        self._primitive_embeddings: Dict[str, np.ndarray] = {
            p: _stable_embedding(p, EMBED_DIM) for p in _PRIMITIVES
        }

        # Discovered bases (words + phrases learned from corpus)
        self._word_freq:  Counter = Counter()
        self._ngram_freq: Counter = Counter()
        self._subword_freq: Counter = Counter()        # Dynamic subwords (BPE-like)
        self._composite_freq: Counter = Counter()      # Recursive concepts (v3)
        self._base_vocab: Dict[str, np.ndarray] = {}  # token → embedding
        self._base_types: Dict[str, str] = {}          # token → 'word'|'phrase'|'subword'|'composite'

        # Cooccurrence: token → {neighbor_token: count}
        # neighbor_token can be a word string or a specialized sensory key
        self._cooc: Dict[Any, Counter] = {}

        self._embed_cache: Dict[str, np.ndarray] = {}
        self.is_fitted = False
        self.persistence_path: Optional[str] = None

    # ── Persistence ──────────────────────────────────────────────────

    def save(self, filepath: str):
        """Save mapper state to a pickle file."""
        state = {
            "word_freq":  self._word_freq,
            "ngram_freq": self._ngram_freq,
            "subword_freq": self._subword_freq,
            "composite_freq": self._composite_freq,
            "base_vocab": self._base_vocab,
            "base_types": self._base_types,
            "cooc":       self._cooc,
            "is_fitted":  self.is_fitted,
        }
        with open(filepath, "wb") as f:
            pickle.dump(state, f)

    def load(self, filepath: str):
        """Load mapper state from a pickle file."""
        if not os.path.exists(filepath):
            return
        with open(filepath, "rb") as f:
            state = pickle.load(f)
        self._word_freq  = state.get("word_freq", Counter())
        self._ngram_freq = state.get("ngram_freq", Counter())
        self._subword_freq = state.get("subword_freq", Counter())
        self._composite_freq = state.get("composite_freq", Counter())
        self._base_vocab = state.get("base_vocab", {})
        self._base_types = state.get("base_types", {})
        self._cooc       = state.get("cooc", {})
        self.is_fitted   = state.get("is_fitted", False)
        self._embed_cache.clear()

    # ── Fitting ──────────────────────────────────────────────────────

    def fit(self, texts: List[str]) -> "BaseMapper":
        """
        Discover word, phrase, and subword bases from a corpus, build embeddings,
        then apply cooccurrence smoothing to give distributional grounding.
        """
        for text in texts:
            toks = self._tokenize(text)
            self._word_freq.update(toks)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                self._ngram_freq.update(self._ngrams(toks, n))

            # Dynamic subword discovery: find frequent internal character sequences
            for tok in toks:
                if len(tok) > 3:
                    for sublen in range(3, min(len(tok), 6)):
                        for start in range(len(tok) - sublen + 1):
                            sub = tok[start:start+sublen]
                            self._subword_freq[sub] += 1

            # Collect cooccurrence within cooc_window
            for i, tok in enumerate(toks):
                if tok not in self._cooc:
                    self._cooc[tok] = Counter()
                start = max(0, i - self.cooc_window)
                end   = min(len(toks), i + self.cooc_window + 1)
                for j in range(start, end):
                    if j != i:
                        self._cooc[tok][toks[j]] += 1

        # Register word bases (initial embeddings)
        for w, cnt in self._word_freq.most_common(self.max_vocab_size):
            if cnt >= self.min_base_freq and w not in self._base_vocab:
                self._base_vocab[w] = self._build_embedding(w, "word")
                self._base_types[w] = "word"

        # Register phrase bases
        for ng, cnt in self._ngram_freq.most_common(self.max_vocab_size // 2):
            if cnt >= self.min_base_freq and ng not in self._base_vocab:
                self._base_vocab[ng] = self._build_embedding(ng, "phrase")
                self._base_types[ng] = "phrase"

        # Register subword bases (BPE-like discovery)
        for sub, cnt in self._subword_freq.most_common(self.max_vocab_size // 4):
            if cnt >= self.min_base_freq * 2 and sub not in self._base_vocab:
                # Subwords get their own stable embedding
                self._base_vocab[sub] = _stable_embedding(sub, EMBED_DIM)
                self._base_types[sub] = "subword"

        # Cooccurrence smoothing pass: blend each word's embedding with a
        # weighted average of its top-5 most frequent neighbor embeddings.
        # This gives distributional semantic grounding (words that appear
        # together become more similar) without large external resources.
        self._apply_cooc_smoothing()

        self.is_fitted = True
        return self

    def _apply_cooc_smoothing(self, sensory_hints: Optional[Dict[str, np.ndarray]] = None):
        """
        One-pass cooccurrence enrichment with Sensory Grounding.
        Blends word embeddings toward both textual neighbors AND sensory centroids.
        """
        lib = _load_lib()
        words = [w for w, t in self._base_types.items() if t == "word"]
        if not words: return

        if lib:
            word_to_idx = {w: i for i, w in enumerate(words)}
            n_words = len(words)
            n_neighbors = 5

            embeddings = np.ascontiguousarray(np.stack([self._base_vocab[w] for w in words]), dtype=np.float32)
            nb_indices = np.full((n_words, n_neighbors), -1, dtype=np.int32)
            nb_weights = np.zeros((n_words, n_neighbors), dtype=np.float32)

            for i, w in enumerate(words):
                neighbors = self._cooc.get(w, Counter())
                total = float(sum(neighbors.values()))
                if total < 1e-10: continue

                for k, (other, cnt) in enumerate(neighbors.most_common(n_neighbors)):
                    if other in word_to_idx:
                        nb_indices[i, k] = word_to_idx[other]
                        nb_weights[i, k] = cnt / total

            lib.cooccurrence_smooth(
                embeddings.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                nb_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                nb_weights.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n_words),
                ctypes.c_int(n_neighbors),
                ctypes.c_int(EMBED_DIM),
                ctypes.c_float(self.cooc_alpha)
            )

            # Post-C smoothing: Handle Sensory Grounding manually as it's sparse
            if sensory_hints:
                for i, w in enumerate(words):
                    if w in sensory_hints:
                        s_vec = sensory_hints[w][:EMBED_DIM]
                        if len(s_vec) < EMBED_DIM:
                            s_vec = np.pad(s_vec, (0, EMBED_DIM - len(s_vec)))
                        sn = np.linalg.norm(s_vec)
                        if sn > 1e-10:
                            # Stronger blend for sensory evidence (2.0 factor)
                            embeddings[i] = (1.0 - self.cooc_alpha) * embeddings[i] + self.cooc_alpha * (s_vec / sn)
                            en = np.linalg.norm(embeddings[i])
                            if en > 1e-10: embeddings[i] /= en

            for i, w in enumerate(words):
                self._base_vocab[w] = embeddings[i].copy()
            return

        # Python Fallback
        updates: Dict[str, np.ndarray] = {}
        for w in words:
            emb = self._base_vocab[w]
            neighbors = self._cooc.get(w, Counter())

            total = float(sum(neighbors.values()))
            delta = np.zeros(EMBED_DIM, dtype=np.float32)
            count = 0

            # 1. Textual Grounding
            if neighbors:
                for other, cnt in neighbors.most_common(5):
                    if isinstance(other, str) and other in self._base_vocab:
                        delta += (cnt / total) * self._base_vocab[other]
                        count += 1

            # 2. Sensory Grounding
            if sensory_hints and w in sensory_hints:
                s_vec = sensory_hints[w][:EMBED_DIM]
                if len(s_vec) < EMBED_DIM:
                    s_vec = np.pad(s_vec, (0, EMBED_DIM - len(s_vec)))
                sn = np.linalg.norm(s_vec)
                if sn > 1e-10:
                    delta += 2.0 * (s_vec / sn)
                    count += 2

            if count == 0:
                continue
            dn = np.linalg.norm(delta)
            if dn > 1e-10:
                blended = ((1.0 - self.cooc_alpha) * emb
                           + self.cooc_alpha * (delta / dn))
                bn = np.linalg.norm(blended)
                updates[w] = blended / bn if bn > 1e-10 else blended
        for w, new_emb in updates.items():
            self._base_vocab[w] = new_emb

    def transform(self, input_data: Any, mode: str = "text") -> BaseMap:
        """
        Convert input data into a BaseMap (one row per token/patch).
        Supports cross-modal interleaving if input_data is a List[Dict].
        """
        if isinstance(input_data, list) and mode == "fusion":
            return self._transform_fusion(input_data)

        if mode == "image":
            return self._transform_image(input_data)
        if mode == "audio":
            return self._transform_audio(input_data)
        if mode == "video":
            return self._transform_video(input_data)

        # Default text mode
        text = str(input_data)
        raw_tokens = self._tokenize(text)
        if not raw_tokens:
            raw_tokens = ["[empty]"]

        tokens, types = self._segment(raw_tokens)

        n      = len(tokens)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)

        freq_vals = self._freq_values(tokens, types)
        max_freq  = max(freq_vals) if freq_vals else 1.0

        # Pre-compute all embeddings and batch operations
        all_embeds = np.zeros((n, EMBED_DIM), dtype=np.float32)
        for i in range(n):
            all_embeds[i] = self._token_embedding(tokens[i], types[i])

        all_ctx = self._context_summary_batch(tokens, types, all_embeds)

        for i, (tok, typ) in enumerate(zip(tokens, types)):
            pos   = self._position_enc(i, n)
            freq  = self._freq_features(tok, freq_vals[i], max_freq)
            flags = self._modifier_flags(tok, typ, i, n)
            flags[MOD_TEXT] = 1.0

            matrix[i, :EMBED_DIM] = all_embeds[i]
            matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = pos
            matrix[i, EMBED_DIM+POS_DIM:EMBED_DIM+POS_DIM+FREQ_DIM] = freq
            matrix[i, EMBED_DIM+POS_DIM+FREQ_DIM:EMBED_DIM+POS_DIM+FREQ_DIM+FLAG_DIM] = flags
            matrix[i, -CTX_DIM:] = all_ctx[i]

        # ── Layer 2.1: Attention Allocation Field (AAF) ────────────
        # Pre-align tokens based on semantic relationships before dots see them.
        matrix = self._apply_aaf(matrix)

        modifiers = {
            "position":  np.array([i / max(n-1, 1) for i in range(n)], np.float32),
            "frequency": np.array(freq_vals, np.float32),
            "types":     types,
        }
        metadata = {
            "text":       text,
            "num_tokens": len(raw_tokens),
            "num_bases":  n,
            "base_types": dict(Counter(types)),
            "fitted":     self.is_fitted,
        }
        return BaseMap(matrix=matrix, tokens=tokens, token_types=types,
                       modifiers=modifiers, metadata=metadata)

    def _transform_fusion(self, input_list: List[Dict]) -> BaseMap:
        """Interleave multiple modalities into a single BaseMap stream."""
        all_matrices = []
        all_tokens = []
        all_types = []

        for item in input_list:
            mode = item.get("mode", "text")
            data = item.get("data")
            bm = self.transform(data, mode=mode)
            all_matrices.append(bm.matrix)
            all_tokens.extend(bm.tokens)
            all_types.extend(bm.token_types)

        final_matrix = np.vstack(all_matrices)
        # Update global position encodings for fused stream
        n = final_matrix.shape[0]
        for i in range(n):
            final_matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = self._position_enc(i, n)

        return BaseMap(final_matrix, all_tokens, all_types, {}, {"mode": "fusion"})

    def _transform_image(self, img_path: str) -> BaseMap:
        """
        Multi-Scale Sensory Patches (v3 SOTA):
        Treat images as sentences of hierarchical visual tokens using 4x4, 8x8,
        and 16x16 patch sizes to capture detail, structure, and context.
        """
        img = Image.open(img_path).convert("RGB")
        img = img.resize((64, 64))
        arr = np.array(img, dtype=np.float32) / 255.0  # (64, 64, 3)

        all_patches: List[np.ndarray] = []
        all_types: List[str] = []

        for ps in [4, 8, 16]: # Detail, Structure, Context
            grid_dim = 64 // ps
            for r in range(0, 64, ps):
                for c in range(0, 64, ps):
                    patch = arr[r:r+ps, c:c+ps]        # (ps, ps, 3)

                    # Statistical features (lossless compression approach)
                    # To keep dim at 224, we use statistics for larger patches
                    flat = patch.flatten()
                    if len(flat) > EMBED_DIM:
                        # For 16x16, flat is 768. Use mean/std blocks
                        b_size = ps // 4
                        flat = patch.reshape(4, b_size, 4, b_size, 3).mean(axis=(1, 3)).flatten()

                    ch_std   = patch.std(axis=(0, 1))
                    feat = np.concatenate([flat, ch_std])
                    if len(feat) < EMBED_DIM:
                        feat = np.pad(feat, (0, EMBED_DIM - len(feat)))
                    else:
                        feat = feat[:EMBED_DIM]

                    n_val = np.linalg.norm(feat)
                    if n_val > 1e-10: feat = feat / n_val

                    all_patches.append(feat.astype(np.float32))
                    all_types.append(f"visual_{ps}x{ps}")

        n = len(all_patches)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)
        for i, p in enumerate(all_patches):
            matrix[i, :EMBED_DIM] = p
            matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = self._position_enc(i, n)
            matrix[i, EMBED_DIM+POS_DIM+FREQ_DIM+MOD_IMAGE] = 1.0

        # Apply AAF to hierarchical patches
        matrix = self._apply_aaf(matrix)

        return BaseMap(matrix, [f"patch_{i}" for i in range(n)],
                       all_types, {}, {"mode": "image", "multi_scale": True})

    def _transform_audio(self, audio_path: str) -> BaseMap:
        """
        Multi-Scale Audio Spectrum (v3 SOTA):
        Treat audio as hierarchical spectral tokens using multiple window sizes
        (256, 512, 1024) to capture temporal detail and frequency precision.
        """
        import wave as _wave

        try:
            with _wave.open(audio_path, "rb") as wf:
                n_channels   = wf.getnchannels()
                sample_width = wf.getsampwidth()
                sample_rate  = wf.getframerate()
                raw = wf.readframes(sample_rate * 5)
        except Exception:
            return self.transform("[silent]", mode="text") # Fallback

        if sample_width == 2: samples = np.frombuffer(raw, dtype=np.int16).astype(np.float32) / 32768.0
        elif sample_width == 1: samples = (np.frombuffer(raw, dtype=np.uint8).astype(np.float32) / 128.0) - 1.0
        else: samples = np.frombuffer(raw, dtype=np.float32)
        if n_channels > 1: samples = samples.reshape(-1, n_channels).mean(axis=1)

        all_spectral: List[np.ndarray] = []
        all_types: List[str] = []

        for fft_size in [256, 512, 1024]:
            n_tokens = 16 if fft_size == 1024 else 32
            needed = fft_size + (n_tokens - 1) * (fft_size // 2)
            s_pad = np.pad(samples, (0, max(0, needed - len(samples))))

            hann = np.hanning(fft_size).astype(np.float32)
            hop  = (len(s_pad) - fft_size) // max(n_tokens - 1, 1)

            for i in range(n_tokens):
                frame = s_pad[i*hop : i*hop + fft_size]
                spectrum = np.abs(np.fft.rfft(frame * hann))[:EMBED_DIM]
                spectrum = np.log1p(spectrum * 100.0)
                if np.max(spectrum) > 1e-10: spectrum /= np.max(spectrum)
                if len(spectrum) < EMBED_DIM: spectrum = np.pad(spectrum, (0, EMBED_DIM - len(spectrum)))

                all_spectral.append(spectrum.astype(np.float32))
                all_types.append(f"spectral_{fft_size}")

        n = len(all_spectral)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)
        for i, t in enumerate(all_spectral):
            matrix[i, :EMBED_DIM] = t
            matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = self._position_enc(i, n)
            matrix[i, EMBED_DIM+POS_DIM+FREQ_DIM+MOD_AUDIO] = 1.0

        return BaseMap(matrix, [f"spectral_{i}" for i in range(n)],
                       all_types, {}, {"mode": "audio", "multi_scale": True})

    def _transform_video(self, video_path: str) -> BaseMap:
        """
        Treat video as a temporal sequence of visual tokens.

        Accepts animated GIF / APNG / WebP (via PIL ImageSequence) or a
        directory of image files sorted alphabetically.  Up to 16 frames
        are extracted; each frame is down-sampled to 64×64 and represented
        as an 8×8 block-mean tensor (8×8×3=192 dims), matching the image
        transform for cross-modal consistency.  Motion (mean absolute frame
        difference) is encoded in the final embedding dimension.

        No cv2 dependency — uses only Pillow + numpy.
        """
        import glob as _glob
        from PIL import ImageSequence as _IS

        frames_rgb: List[np.ndarray] = []

        try:
            img = Image.open(video_path)
            for i, frame in enumerate(_IS.Iterator(img)):
                if i >= 16:
                    break
                f = np.array(frame.convert("RGB").resize((64, 64)),
                             dtype=np.float32) / 255.0
                frames_rgb.append(f)
        except Exception:
            pass

        if not frames_rgb and os.path.isdir(video_path):
            img_files: List[str] = []
            for ext in ("*.png", "*.jpg", "*.jpeg", "*.bmp", "*.webp"):
                img_files.extend(_glob.glob(os.path.join(video_path, ext)))
            for fp in sorted(img_files)[:16]:
                try:
                    f = np.array(Image.open(fp).convert("RGB").resize((64, 64)),
                                 dtype=np.float32) / 255.0
                    frames_rgb.append(f)
                except Exception:
                    continue

        if not frames_rgb:
            frames_rgb = [np.zeros((64, 64, 3), dtype=np.float32)]

        n = len(frames_rgb)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)

        for i, frame in enumerate(frames_rgb):
            # 8×8 block means: (64,64,3) → (8,8,8,8,3).mean → (8,8,3) = 192 dims
            block = frame.reshape(8, 8, 8, 8, 3).mean(axis=(2, 4)).flatten()
            if len(block) < EMBED_DIM:
                block = np.pad(block, (0, EMBED_DIM - len(block)))
            else:
                block = block[:EMBED_DIM]

            # Motion-Vector Encoding (v5 SOTA):
            # Compute 8-dim motion deltas between frames to capture temporal dynamics.
            if i > 0:
                diff = np.abs(frames_rgb[i] - frames_rgb[i - 1])
                # 8-dim motion signature: 4 quadrants + RGB means + global std
                m_sig = np.array([
                    diff[:32, :32].mean(), diff[:32, 32:].mean(),
                    diff[32:, :32].mean(), diff[32:, 32:].mean(),
                    diff.mean(axis=(0,1))[0], diff.mean(axis=(0,1))[1], diff.mean(axis=(0,1))[2],
                    diff.std()
                ], dtype=np.float32)
                block = block.copy()
                block[-8:] = m_sig  # Encode motion in last 8 embedding dims

            n_val = np.linalg.norm(block)
            if n_val > 1e-10:
                block = block / n_val

            matrix[i, :EMBED_DIM] = block.astype(np.float32)
            matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = self._position_enc(i, n)
            matrix[i, EMBED_DIM+POS_DIM+FREQ_DIM+MOD_VIDEO] = 1.0

        return BaseMap(matrix, [f"frame_{i}" for i in range(n)],
                       ["temporal_visual"] * n, {}, {"mode": "video"})

    def fit_transform(self, texts: List[str], target: Optional[str] = None) -> BaseMap:
        self.fit(texts)
        return self.transform(target or texts[0])

    # ── Tokenization ────────────────────────────────────────────────

    def _tokenize(self, text: str) -> List[str]:
        text = text.lower().strip()
        return re.findall(r"\b[\w']+\b|[^\w\s]", text)

    def _ngrams(self, tokens: List[str], n: int) -> List[str]:
        return [" ".join(tokens[i:i+n]) for i in range(len(tokens)-n+1)]

    # ── Segmentation: phrase detection + word/char base assignment ───

    def _segment(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Greedily segment token stream into the highest-level bases available.

        Priority (highest wins):
          0. Composite concepts (Recursive Base Composition)
          1. Known phrase bases (longest match first)
          2. Known word bases
          3. Unknown word — represented as ONE token of type 'composed'
        """
        result_toks:  List[str] = []
        result_types: List[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            if self.is_fitted:
                # 0. Check for Composite Concepts first
                for comp_name, comp_type in self._base_types.items():
                    if comp_type == "composite":
                        # Greedy composite match: does the next span match a known concept?
                        # (Concepts are identified by their stable string name)
                        span = " ".join(tokens[i:i+3])
                        if comp_name in span:
                            result_toks.append(comp_name)
                            result_types.append("composite")
                            # We heuristicly skip ahead based on word count
                            i += len(comp_name.split("_")) # e.g. 'concept_123'
                            matched = True
                            break

                # 1. Phrases
                for ng_len in range(self.ngram_range[1], self.ngram_range[0]-1, -1):
                    if i + ng_len <= len(tokens):
                        candidate = " ".join(tokens[i:i+ng_len])
                        if candidate in self._base_vocab:
                            result_toks.append(candidate)
                            result_types.append(self._base_types.get(candidate, "phrase"))
                            i += ng_len
                            matched = True
                            break
            if not matched:
                tok = tokens[i]
                if self.is_fitted and tok in self._base_vocab:
                    result_toks.append(tok)
                    result_types.append("word")
                elif tok in self._primitive_embeddings:
                    result_toks.append(tok)
                    result_types.append("primitive")
                else:
                    result_toks.append(tok)
                    result_types.append("composed")
                i += 1
        return result_toks, result_types

    # ── Embedding construction ───────────────────────────────────────

    def _build_embedding(self, token: str, token_type: str) -> np.ndarray:
        """Build the EMBED_DIM embedding for a known base."""
        if token_type == "phrase":
            words       = token.split()
            word_embeds = [self._compose_word_embedding(w) for w in words]
            v = np.mean(np.stack(word_embeds), axis=0).astype(np.float32)
        else:
            stable   = _stable_embedding(token, EMBED_DIM)
            composed = self._compose_word_embedding(token)
            v = (0.6 * stable + 0.4 * composed).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    def _compose_word_embedding(self, word: str, depth: int = 0) -> np.ndarray:
        """
        Compose a word's embedding from its constituent character primitives,
        using both character unigrams, bigrams, and morpheme decomposition.

        Unigrams capture the overall letter distribution; bigrams capture
        local subword structure; morphemes capture semantic chunks.

        Returns a single unit-norm EMBED_DIM vector — ONE representation.
        """
        if word in self._embed_cache:
            return self._embed_cache[word]

        chars = list(word.lower())
        char_embeds: List[np.ndarray] = []
        weights: List[float] = []

        # 1. Morpheme Decomposition (Semantic chunks)
        # Only decompose if we are at the top level to avoid infinite recursion
        if depth == 0 and len(word) > 5:
            morphemes = self._split_morphemes(word.lower())
            if len(morphemes) > 1:
                for m in morphemes:
                    # Recursively get embedding for morpheme (at depth 1)
                    m_emb = self._compose_word_embedding(m, depth=1)
                    char_embeds.append(m_emb)
                    weights.append(2.0) # High weight for semantic chunks

        # 2. Character unigrams (primary signal)
        for k, ch in enumerate(chars):
            if ch in self._primitive_embeddings:
                char_embeds.append(self._primitive_embeddings[ch])
                weights.append(1.0 / (1.0 + k * 0.1))

        # 3. Character bigrams (secondary signal, lower weight)
        for k in range(len(chars) - 1):
            c1, c2 = chars[k], chars[k+1]
            if c1 in self._primitive_embeddings and c2 in self._primitive_embeddings:
                bigram_embed = 0.5 * (self._primitive_embeddings[c1]
                                      + self._primitive_embeddings[c2])
                char_embeds.append(bigram_embed)
                weights.append(0.5 / (1.0 + k * 0.1))

        if not char_embeds:
            v = _stable_embedding(word, EMBED_DIM)
            if depth == 0: self._embed_cache[word] = v
            return v

        lib   = _load_lib()
        char_arr = np.ascontiguousarray(np.stack(char_embeds), np.float32)
        w_arr    = np.ascontiguousarray(weights, np.float32)
        out      = np.zeros(EMBED_DIM, dtype=np.float32)

        if lib:
            lib.compose_from_chars(
                char_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(len(char_embeds)),
                ctypes.c_int(EMBED_DIM),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        else:
            w_total = sum(weights)
            for ce, wt in zip(char_embeds, weights):
                out += (wt / w_total) * ce
            n = np.linalg.norm(out)
            if n > 1e-10:
                out /= n

        if depth == 0: self._embed_cache[word] = out
        return out

    def _split_morphemes(self, word: str) -> List[str]:
        """
        Split word into known prefixes, suffixes, and discovered subwords.
        Uses a greedy longest-match approach.
        """
        res = []
        remaining = word.lower()

        # 1. Hardcoded Prefixes
        for p in _PREFIXES:
            if remaining.startswith(p) and len(remaining) > len(p) + 2:
                res.append(p)
                remaining = remaining[len(p):]
                break

        # 2. Hardcoded Suffixes (stored for later)
        suffix = ""
        for s in sorted(_SUFFIXES, key=len, reverse=True):
            if remaining.endswith(s) and len(remaining) > len(s) + 2:
                suffix = s
                remaining = remaining[:-len(s)]
                break

        # 3. Discovered Subwords (Dynamic BPE-like)
        # Greedy segment remaining part using discovered subwords
        while len(remaining) > 2:
            best_sub = ""
            for sublen in range(min(len(remaining), 6), 2, -1):
                sub = remaining[:sublen]
                if sub in self._base_types and self._base_types[sub] == "subword":
                    best_sub = sub
                    break

            if best_sub:
                res.append(best_sub)
                remaining = remaining[len(best_sub):]
            else:
                # If no subword match, take one char and continue
                res.append(remaining[0])
                remaining = remaining[1:]

        if remaining:
            res.append(remaining)
        if suffix:
            res.append(suffix)

        return res

    def _token_embedding(self, token: str, token_type: str) -> np.ndarray:
        """Return the EMBED_DIM embedding for a token based on its type."""
        if token_type == "primitive":
            return self._primitive_embeddings.get(token, _stable_embedding(token, EMBED_DIM))
        if token_type in ("word", "phrase"):
            return self._base_vocab.get(token, self._compose_word_embedding(token))
        return self._compose_word_embedding(token)

    # ── Position encoding ────────────────────────────────────────────

    def _position_enc(self, pos: int, total: int, grid_dim: Optional[int] = None) -> np.ndarray:
        """
        Enhanced Positional Encoding: supports linear and 2D relational grids.
        If grid_dim provided, pos is interpreted as a 2D index (row * grid_dim + col).
        """
        lib = _load_lib()
        out = np.zeros(POS_DIM, dtype=np.float32)

        if grid_dim:
            # 2D spatial encoding (for image patches)
            r = pos // grid_dim
            c = pos % grid_dim
            rel_r = r / max(grid_dim - 1, 1)
            rel_c = c / max(grid_dim - 1, 1)
            # Use half for row, half for col
            for k in range(POS_DIM // 4):
                out[2*k]     = math.sin(rel_r * math.pi * (k+1))
                out[2*k+1]   = math.cos(rel_r * math.pi * (k+1))
                out[POS_DIM//2 + 2*k]   = math.sin(rel_c * math.pi * (k+1))
                out[POS_DIM//2 + 2*k+1] = math.cos(rel_c * math.pi * (k+1))
            return out

        if lib:
            lib.sinusoidal_position_enc(
                ctypes.c_int(pos), ctypes.c_int(total), ctypes.c_int(POS_DIM),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float))
            )
            return out

        rel = pos / max(total - 1, 1)
        for k in range(POS_DIM // 2):
            out[2*k]   = math.sin(rel * math.pi * (k+1))
            out[2*k+1] = math.cos(rel * math.pi * (k+1))
        return out

    # ── Frequency features ───────────────────────────────────────────

    def _freq_values(self, tokens: List[str], types: List[str]) -> List[float]:
        vals = []
        for tok, typ in zip(tokens, types):
            if typ == "phrase":
                vals.append(float(self._ngram_freq.get(tok, 1)))
            elif typ in ("word", "composed"):
                vals.append(float(self._word_freq.get(tok, 1)))
            else:
                vals.append(1.0)
        return vals

    def _freq_features(self, token: str, freq: float, max_freq: float) -> np.ndarray:
        log_f   = math.log1p(freq)
        log_max = math.log1p(max_freq) if max_freq > 0 else 1.0
        return np.array([
            freq / max(max_freq, 1e-10),
            log_f / max(log_max, 1e-10),
            math.tanh(freq / 10.0),
            1.0 / (1.0 + math.exp(-freq + 3)),
        ], dtype=np.float32)

    # ── Modifier flags ───────────────────────────────────────────────

    def _modifier_flags(self, token: str, typ: str, pos: int, total: int) -> np.ndarray:
        """
        16 binary/continuous flags encoding token identity, position, and
        morphological role.
        """
        f = np.zeros(FLAG_DIM, dtype=np.float32)
        f[0]  = 1.0 if typ == "primitive" else 0.0
        f[1]  = 1.0 if typ == "word"      else 0.0
        f[2]  = 1.0 if typ == "phrase"    else 0.0
        f[3]  = 1.0 if typ == "composed"  else 0.0
        f[4]  = 1.0 if pos == 0           else 0.0
        f[5]  = 1.0 if pos == total - 1   else 0.0
        f[6]  = 1.0 if 0 < pos < total-1  else 0.0
        f[7]  = 1.0 if token.replace(" ","").isalpha() else 0.0
        f[8]  = 1.0 if token.replace(" ","").isdigit() else 0.0
        f[9]  = 1.0 if not token.replace(" ","").isalnum() else 0.0

        # Morphological suffix detection (dims 10-11)
        clean = token.replace(" ", "")
        f[10] = 1.0 if any(clean.endswith(s) for s in _VERB_SUFFIXES) else 0.0
        f[11] = 1.0 if (any(clean.endswith(s) for s in _NOUN_SUFFIXES)
                        or any(clean.endswith(s) for s in _ADJ_SUFFIXES)
                        or clean.endswith(_ADV_SUFFIX)) else 0.0

        return np.clip(f, 0.0, 1.0)

    # ── Context summary ──────────────────────────────────────────────

    def _context_summary_batch(self, tokens: List[str], types: List[str],
                               embeddings: np.ndarray) -> np.ndarray:
        """Vectorized batch context summary."""
        n = len(tokens)
        ws = self.context_window
        ctx_matrix = np.zeros((n, CTX_DIM), dtype=np.float32)

        # Dim 0: Semantic cohesion (batch dot product)
        sim_matrix = embeddings @ embeddings.T

        for i in range(n):
            left_idx = max(0, i - ws)
            right_idx = min(n, i + ws + 1)
            nbs = list(range(left_idx, i)) + list(range(i + 1, right_idx))

            if not nbs: continue

            # Dim 0: mean sim
            ctx_matrix[i, 0] = np.clip(np.mean(sim_matrix[i, nbs]) * 0.5 + 0.5, 0.0, 1.0)

            # Dim 1: balance
            ctx_matrix[i, 1] = (i - left_idx) / len(nbs)

            # Dim 2: phrase density
            ctx_matrix[i, 2] = sum(1 for j in nbs if types[j] == "phrase") / len(nbs)

            # Dim 3: vocab density
            ctx_matrix[i, 3] = sum(1 for j in nbs if types[j] in ("word", "phrase")) / len(nbs)

        return ctx_matrix

    def _context_summary(self, pos: int, tokens: List[str],
                          types: List[str]) -> np.ndarray:
        # Legacy fallback
        embeds = np.zeros((len(tokens), EMBED_DIM), dtype=np.float32)
        for i in range(len(tokens)):
            embeds[i] = self._token_embedding(tokens[i], types[i])
        return self._context_summary_batch(tokens, types, embeds)[pos]

    def fit_contrastive(self, matching_pairs: List[Tuple[str, str]],
                        mismatch_pairs: List[Tuple[str, str]],
                        lr: float = 0.05):
        """
        Ultimate SOTA Upgrade: Contrastive Anchoring.
        Explicitly pulls matching concepts closer and pushes mismatches apart
        in the base vocabulary.
        """
        for a, b in matching_pairs:
            if a in self._base_vocab and b in self._base_vocab:
                ea, eb = self._base_vocab[a], self._base_vocab[b]
                # Pull together
                self._base_vocab[a] = ea + lr * (eb - ea)
                self._base_vocab[b] = eb + lr * (ea - eb)
                # Normalize
                self._base_vocab[a] /= np.linalg.norm(self._base_vocab[a])
                self._base_vocab[b] /= np.linalg.norm(self._base_vocab[b])

        for a, b in mismatch_pairs:
            if a in self._base_vocab and b in self._base_vocab:
                ea, eb = self._base_vocab[a], self._base_vocab[b]
                # Push apart
                self._base_vocab[a] = ea - lr * (eb - ea)
                self._base_vocab[b] = eb - lr * (ea - eb)
                # Normalize
                self._base_vocab[a] /= np.linalg.norm(self._base_vocab[a])
                self._base_vocab[b] /= np.linalg.norm(self._base_vocab[b])

    def shard_vocab(self, max_shard_size: int = 10000):
        """Ultimate SOTA Scale: Vocabulary Sharding.
        Splits the massive vocabulary into manageable shards to keep search fast.
        """
        if len(self._base_vocab) <= max_shard_size:
            return [self._base_vocab]

        items = list(self._base_vocab.items())
        shards = []
        for i in range(0, len(items), max_shard_size):
            shards.append(dict(items[i:i + max_shard_size]))
        return shards

    def register_composite_base(self, name: str, embedding: np.ndarray):
        """
        Recursive Base Composition (v3 SOTA):
        Cast a winning cluster centroid back into a persistent named base.
        This allows the system to discover and name complex concepts.
        """
        if name not in self._base_vocab:
            # We strip any complex parts to store a stable float32 base
            real_emb = np.real(embedding[:EMBED_DIM]).astype(np.float32)
            n = np.linalg.norm(real_emb)
            if n > 1e-10: real_emb /= n

            self._base_vocab[name] = real_emb
            self._base_types[name] = "composite"
            self._composite_freq[name] = 1
        else:
            self._composite_freq[name] += 1
            # Update existing composite (gentle blend)
            real_emb = np.real(embedding[:EMBED_DIM]).astype(np.float32)
            n = np.linalg.norm(real_emb)
            if n > 1e-10: real_emb /= n
            self._base_vocab[name] = 0.95 * self._base_vocab[name] + 0.05 * real_emb

    def _apply_aaf(self, matrix: np.ndarray) -> np.ndarray:
        """
        Compute the Attention Allocation Field (AAF).
        Tokens 'poll' their neighbors for semantic context to pre-refine
        their embeddings. This turns the static sequence into a dynamic field.
        """
        n, dim = matrix.shape
        if n <= 1: return matrix

        # Compute semantic self-attention matrix
        # For efficiency, we use the base embeddings region [0:EMBED_DIM]
        bases = matrix[:, :EMBED_DIM]

        # O(n^2) check: for long sequences, we use a sliding window AAF
        if n > 256:
            aligned = matrix.copy()
            window = 64
            for i in range(n):
                start = max(0, i - window)
                end = min(n, i + window + 1)

                nbs = bases[start:end]
                sim = nbs @ bases[i]

                sim -= np.max(sim)
                weights = np.exp(sim * 5.0)
                weights /= weights.sum() + 1e-10

                aligned[i] = (1.0 - 0.15) * matrix[i] + 0.15 * (weights @ matrix[start:end])
            return aligned.astype(np.float32)

        # (n x n) similarity matrix
        sim = bases @ bases.T

        # Softmax over rows to get allocation weights
        sim -= np.max(sim, axis=1, keepdims=True)
        weights = np.exp(sim * 5.0) # sharpness factor
        weights /= weights.sum(axis=1, keepdims=True) + 1e-10

        # Pre-align matrix: tokens incorporate information from their field
        # We use a gentle blend (0.15) to keep the base identity strong
        aligned = (1.0 - 0.15) * matrix + 0.15 * (weights @ matrix)

        return aligned.astype(np.float32)
