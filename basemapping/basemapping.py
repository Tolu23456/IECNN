"""
BaseMapping — Structured input representation for IECNN.

Design principles:
  - a-z (and 0-9) are PRE-SEEDED as primitive bases — always available
  - Each TOKEN (word or phrase) maps to exactly ONE row in the BaseMap matrix
  - Known bases (words/phrases discovered by frequency) get their own embedding
  - Unknown words get ONE row whose embedding is composed from their
    constituent character (a-z) embeddings — never split into separate rows
  - Phrases (frequent bigrams/trigrams) are detected and treated as single bases

Feature vector layout (128 dims):
  [0:96]   base_embedding  — stable embedding for the base token
  [96:104] position_enc    — sinusoidal position encoding (8 dims)
  [104:108] freq_features  — frequency statistics (4 dims)
  [108:124] modifier_flags — structural/linguistic flags (16 dims)
  [124:128] context_summary — local context window summary (4 dims)
"""

import numpy as np
import re
import math
import ctypes
import os
from collections import Counter
from typing import List, Dict, Tuple, Optional

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
            _lib.compose_from_chars.restype     = None
            _lib.sinusoidal_position_enc.restype = None
            _lib.normalize_vector.restype       = None
            _lib.mean_pool.restype              = None
            _lib.attention_pool.restype         = None
        except Exception:
            _lib = None
    return _lib


# ── Primitive bases: a-z, 0-9, basic punctuation ────────────────────
_PRIMITIVES = list("abcdefghijklmnopqrstuvwxyz0123456789") + [
    ".", ",", "!", "?", "'", "-", "_", "/", "(", ")",
]

EMBED_DIM   = 96
POS_DIM     = 8
FREQ_DIM    = 4
FLAG_DIM    = 16
CTX_DIM     = 4
FEATURE_DIM = EMBED_DIM + POS_DIM + FREQ_DIM + FLAG_DIM + CTX_DIM  # = 128


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
      matrix    — (num_tokens, 128) float32 feature matrix
      tokens    — list of base token strings (one per row)
      token_types — list of type tags: 'primitive'|'word'|'phrase'|'composed'
      modifiers — dict of modifier arrays
      metadata  — info dict
    """

    def __init__(self, matrix, tokens, token_types, modifiers, metadata):
        self.matrix      = matrix
        self.tokens      = tokens
        self.token_types = token_types
        self.modifiers   = modifiers
        self.metadata    = metadata

    def __len__(self):
        return len(self.tokens)

    def pool(self, method: str = "mean") -> np.ndarray:
        if method == "mean": return np.mean(self.matrix, axis=0)
        if method == "max":  return np.max(self.matrix, axis=0)
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
         (tokens appearing >= min_base_freq times become new named bases)
      3. On transform(text):
         a. Tokenize into words (and detect known phrases)
         b. For each token (one row per token):
            - If it's a known base → use that base's stable embedding
            - If it's an unknown word → compose embedding from its
              characters using the pre-seeded primitive embeddings (C-accelerated)
            - Either way: ONE row per token
         c. Attach modifiers: position, frequency, structural flags, context
         d. Output BaseMap matrix
    """

    def __init__(
        self,
        feature_dim: int = FEATURE_DIM,
        min_base_freq: int = 2,
        max_vocab_size: int = 50_000,
        ngram_range: Tuple[int, int] = (2, 3),
        context_window: int = 2,
    ):
        self.feature_dim    = feature_dim
        self.min_base_freq  = min_base_freq
        self.max_vocab_size = max_vocab_size
        self.ngram_range    = ngram_range
        self.context_window = context_window

        # Pre-seed primitive bases (a-z, 0-9, punctuation)
        self._primitive_embeddings: Dict[str, np.ndarray] = {
            p: _stable_embedding(p, EMBED_DIM) for p in _PRIMITIVES
        }

        # Discovered bases (words + phrases learned from corpus)
        self._word_freq:   Counter = Counter()
        self._ngram_freq:  Counter = Counter()
        self._base_vocab:  Dict[str, np.ndarray] = {}  # token → embedding
        self._base_types:  Dict[str, str] = {}          # token → 'word'|'phrase'

        self._embed_cache: Dict[str, np.ndarray] = {}
        self.is_fitted = False

    def fit(self, texts: List[str]) -> "BaseMapper":
        """Discover word and phrase bases from a corpus."""
        for text in texts:
            toks = self._tokenize(text)
            self._word_freq.update(toks)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                self._ngram_freq.update(self._ngrams(toks, n))

        # Register word bases
        for w, cnt in self._word_freq.most_common(self.max_vocab_size):
            if cnt >= self.min_base_freq and w not in self._base_vocab:
                self._base_vocab[w] = self._build_embedding(w, "word")
                self._base_types[w] = "word"

        # Register phrase bases
        for ng, cnt in self._ngram_freq.most_common(self.max_vocab_size // 2):
            if cnt >= self.min_base_freq and ng not in self._base_vocab:
                self._base_vocab[ng] = self._build_embedding(ng, "phrase")
                self._base_types[ng] = "phrase"

        self.is_fitted = True
        return self

    def transform(self, text: str) -> BaseMap:
        """Convert text into a BaseMap (one row per token)."""
        raw_tokens = self._tokenize(text)
        if not raw_tokens:
            raw_tokens = ["[empty]"]

        # Detect phrases first (greedy, longest match)
        tokens, types = self._segment(raw_tokens)

        n = len(tokens)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)

        freq_vals  = self._freq_values(tokens, types)
        max_freq   = max(freq_vals) if freq_vals else 1.0

        for i, (tok, typ) in enumerate(zip(tokens, types)):
            embed  = self._token_embedding(tok, typ)
            pos    = self._position_enc(i, n)
            freq   = self._freq_features(tok, freq_vals[i], max_freq)
            flags  = self._modifier_flags(tok, typ, i, n)
            ctx    = self._context_summary(i, tokens, types)

            matrix[i, :EMBED_DIM] = embed
            matrix[i, EMBED_DIM:EMBED_DIM+POS_DIM] = pos
            matrix[i, EMBED_DIM+POS_DIM:EMBED_DIM+POS_DIM+FREQ_DIM] = freq
            matrix[i, EMBED_DIM+POS_DIM+FREQ_DIM:EMBED_DIM+POS_DIM+FREQ_DIM+FLAG_DIM] = flags
            matrix[i, -CTX_DIM:] = ctx

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
          1. Known phrase bases (longest match first)
          2. Known word bases
          3. Unknown word — represented as ONE token of type 'composed'
             (embedding built from character primitives; NOT split into chars)
        """
        result_toks: List[str] = []
        result_types: List[str] = []
        i = 0
        while i < len(tokens):
            matched = False
            if self.is_fitted:
                # Try longest phrase match
                for ng_len in range(self.ngram_range[1], self.ngram_range[0]-1, -1):
                    if i + ng_len <= len(tokens):
                        candidate = " ".join(tokens[i:i+ng_len])
                        if candidate in self._base_vocab:
                            result_toks.append(candidate)
                            result_types.append("phrase")
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
                    # Unknown word — one row, composed from character primitives
                    result_toks.append(tok)
                    result_types.append("composed")
                i += 1
        return result_toks, result_types

    # ── Embedding construction ───────────────────────────────────────

    def _build_embedding(self, token: str, token_type: str) -> np.ndarray:
        """Build the EMBED_DIM embedding for a known base."""
        if token_type == "phrase":
            # Phrase: blend word embeddings for each constituent word
            words = token.split()
            word_embeds = [self._compose_word_embedding(w) for w in words]
            v = np.mean(np.stack(word_embeds), axis=0).astype(np.float32)
        else:
            # Known word: give it a stable unique embedding but also
            # blend in its character composition for grounding
            stable = _stable_embedding(token, EMBED_DIM)
            composed = self._compose_word_embedding(token)
            v = (0.6 * stable + 0.4 * composed).astype(np.float32)
        n = np.linalg.norm(v)
        return v / n if n > 1e-10 else v

    def _compose_word_embedding(self, word: str) -> np.ndarray:
        """
        Compose a word's embedding from its constituent character primitives.
        Each character that is a known primitive contributes its embedding,
        weighted by its position (later chars matter slightly less).
        Returns a single EMBED_DIM vector — ONE representation for the whole word.
        """
        if word in self._embed_cache:
            return self._embed_cache[word]

        chars = list(word.lower())
        char_embeds = []
        weights = []
        for k, ch in enumerate(chars):
            if ch in self._primitive_embeddings:
                char_embeds.append(self._primitive_embeddings[ch])
                weights.append(1.0 / (1.0 + k * 0.1))  # slight positional decay
            # Unknown characters are simply skipped

        if not char_embeds:
            # Fully unknown characters — fall back to stable hash embedding
            v = _stable_embedding(word, EMBED_DIM)
            self._embed_cache[word] = v
            return v

        lib = _load_lib()
        char_arr = np.ascontiguousarray(np.stack(char_embeds), np.float32)
        w_arr    = np.ascontiguousarray(weights, np.float32)
        out      = np.zeros(EMBED_DIM, dtype=np.float32)

        if lib:
            lib.compose_from_chars(
                char_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                w_arr.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(len(chars)),
                ctypes.c_int(EMBED_DIM),
                out.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
            )
        else:
            w_total = sum(weights)
            for ce, wt in zip(char_embeds, weights):
                out += (wt / w_total) * ce
            n = np.linalg.norm(out)
            if n > 1e-10: out /= n

        self._embed_cache[word] = out
        return out

    def _token_embedding(self, token: str, token_type: str) -> np.ndarray:
        """Return the EMBED_DIM embedding for a token based on its type."""
        if token_type == "primitive":
            return self._primitive_embeddings.get(token, _stable_embedding(token, EMBED_DIM))
        if token_type in ("word", "phrase"):
            return self._base_vocab.get(token, self._compose_word_embedding(token))
        # 'composed' — unknown word, built from character primitives
        return self._compose_word_embedding(token)

    # ── Position encoding ────────────────────────────────────────────

    def _position_enc(self, pos: int, total: int) -> np.ndarray:
        lib = _load_lib()
        out = np.zeros(POS_DIM, dtype=np.float32)
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
        f = np.zeros(FLAG_DIM, dtype=np.float32)
        f[0]  = 1.0 if typ == "primitive" else 0.0
        f[1]  = 1.0 if typ == "word"      else 0.0
        f[2]  = 1.0 if typ == "phrase"    else 0.0
        f[3]  = 1.0 if typ == "composed"  else 0.0
        f[4]  = 1.0 if pos == 0           else 0.0   # sentence start
        f[5]  = 1.0 if pos == total - 1   else 0.0   # sentence end
        f[6]  = 1.0 if 0 < pos < total-1  else 0.0   # middle
        f[7]  = 1.0 if token.replace(" ","").isalpha() else 0.0
        f[8]  = 1.0 if token.replace(" ","").isdigit() else 0.0
        f[9]  = 1.0 if not token.replace(" ","").isalnum() else 0.0
        f[10] = float(len(token)) / 30.0
        f[11] = 1.0 if " " in token else 0.0          # is phrase
        f[12] = 1.0 if len(token) == 1 else 0.0
        f[13] = 1.0 if len(token) > 8  else 0.0
        f[14] = float(pos) / max(total - 1, 1)
        f[15] = 1.0 if pos < total / 3 else (0.5 if pos < 2*total/3 else 0.0)
        return np.clip(f, 0.0, 1.0)

    # ── Context summary ──────────────────────────────────────────────

    def _context_summary(self, pos: int, tokens: List[str], types: List[str]) -> np.ndarray:
        n = len(tokens)
        ws = self.context_window
        window = [(tokens[j], types[j]) for j in range(max(0, pos-ws), min(n, pos+ws+1)) if j != pos]
        ctx = np.zeros(CTX_DIM, dtype=np.float32)
        ctx[0] = len(window) / (2 * ws) if ws > 0 else 0.0
        ctx[1] = sum(1 for _, t in window if t == "phrase") / max(len(window), 1)
        ctx[2] = sum(1 for _, t in window if t in ("word","phrase")) / max(len(window), 1)
        ctx[3] = sum(len(tok) for tok, _ in window) / max(len(window) * 20, 1)
        return np.clip(ctx, 0.0, 1.0)
