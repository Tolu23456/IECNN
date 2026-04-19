import numpy as np
import re
import math
from collections import Counter, defaultdict
from typing import List, Dict, Tuple, Optional


class BaseMap:
    """
    The structured output of BaseMapping for a single input.

    Contains:
      - matrix: (num_bases, feature_dim) — the full BaseMap matrix
      - bases: list of base tokens (words, chars, phrases)
      - modifiers: dict of modifier arrays per base
      - metadata: info about the mapping
    """

    def __init__(
        self,
        matrix: np.ndarray,
        bases: List[str],
        modifiers: Dict[str, np.ndarray],
        metadata: Dict,
    ):
        self.matrix = matrix
        self.bases = bases
        self.modifiers = modifiers
        self.metadata = metadata

    def __len__(self):
        return len(self.bases)

    def __repr__(self):
        return (
            f"BaseMap(bases={len(self.bases)}, "
            f"shape={self.matrix.shape}, "
            f"types={self.metadata.get('base_types', {})})"
        )

    def slice(self, start: int, end: int) -> np.ndarray:
        """Return a slice of the BaseMap matrix."""
        return self.matrix[start:end]

    def pool(self, method: str = "mean") -> np.ndarray:
        """Pool the full BaseMap matrix into a single vector."""
        if method == "mean":
            return np.mean(self.matrix, axis=0)
        elif method == "max":
            return np.max(self.matrix, axis=0)
        elif method == "sum":
            return np.sum(self.matrix, axis=0)
        else:
            return np.mean(self.matrix, axis=0)


class BaseMapper:
    """
    Converts input into structured maps of bases and modifiers.

    Architecture:
      - Bases: fundamental units (characters, words, multi-word phrases)
      - Modifiers: contextual properties (position, frequency, role, context)
      - Output: compact structured matrix for IECNN processing

    Feature vector layout (feature_dim = 128):
      [0:96]   base_embedding  — stable hash-based vector for each base token
      [96:104] position_enc    — sinusoidal position encoding (8 dims)
      [104:108] freq_features  — frequency, log_freq, rank, normalized_freq
      [108:124] modifier_flags — linguistic/structural flags (16 dims)
      [124:128] context_summary — summary of local context window (4 dims)
    """

    FEATURE_DIM = 128
    EMBED_DIM = 96
    POS_DIM = 8
    FREQ_DIM = 4
    FLAG_DIM = 16
    CTX_DIM = 4

    def __init__(
        self,
        feature_dim: int = 128,
        min_base_freq: int = 2,
        max_vocab_size: int = 50000,
        ngram_range: Tuple[int, int] = (2, 3),
        context_window: int = 2,
    ):
        self.feature_dim = feature_dim
        self.min_base_freq = min_base_freq
        self.max_vocab_size = max_vocab_size
        self.ngram_range = ngram_range
        self.context_window = context_window

        self.word_freq: Counter = Counter()
        self.ngram_freq: Counter = Counter()
        self.char_freq: Counter = Counter()

        self.word_vocab: Dict[str, int] = {}
        self.ngram_vocab: Dict[str, int] = {}
        self.char_vocab: Dict[str, int] = {}

        self.is_fitted = False
        self._embed_cache: Dict[str, np.ndarray] = {}

    def fit(self, texts: List[str]) -> "BaseMapper":
        """
        Discover bases from a corpus of texts.
        Learns word frequencies, n-gram frequencies, and builds vocabularies.
        Works in zero-shot mode if not called (falls back to character-level).
        """
        for text in texts:
            tokens = self._tokenize(text)
            self.word_freq.update(tokens)
            self.char_freq.update(ch for t in tokens for ch in t)
            for n in range(self.ngram_range[0], self.ngram_range[1] + 1):
                ngrams = self._extract_ngrams(tokens, n)
                self.ngram_freq.update(ngrams)

        words_by_freq = [
            w for w, c in self.word_freq.most_common(self.max_vocab_size)
            if c >= self.min_base_freq
        ]
        self.word_vocab = {w: i for i, w in enumerate(words_by_freq)}

        ngrams_by_freq = [
            ng for ng, c in self.ngram_freq.most_common(self.max_vocab_size // 2)
            if c >= self.min_base_freq
        ]
        self.ngram_vocab = {ng: i for i, ng in enumerate(ngrams_by_freq)}

        chars_by_freq = [ch for ch, _ in self.char_freq.most_common()]
        self.char_vocab = {ch: i for i, ch in enumerate(chars_by_freq)}

        self.is_fitted = True
        return self

    def transform(self, text: str) -> BaseMap:
        """
        Convert text into a BaseMap.

        Process:
          1. Tokenize into words
          2. Detect known n-gram phrases
          3. Map each token to a base (word-level or char-level fallback)
          4. Extract modifiers for each base
          5. Assemble into feature matrix
        """
        tokens = self._tokenize(text)
        bases, base_types = self._extract_bases_with_types(tokens)

        if not bases:
            bases = ["[empty]"]
            base_types = ["word"]

        n = len(bases)
        matrix = np.zeros((n, self.feature_dim), dtype=np.float32)

        total_words = len(tokens)
        freq_values = self._get_freq_values(bases)
        max_freq = max(freq_values) if freq_values else 1.0

        for i, (base, btype) in enumerate(zip(bases, base_types)):
            embed = self._base_embedding(base, self.EMBED_DIM)
            pos = self._position_encoding(i, n, self.POS_DIM)
            freq = self._frequency_features(base, freq_values[i], max_freq, btype)
            flags = self._modifier_flags(base, btype, i, n, tokens)
            ctx = self._context_summary(i, bases, base_types)

            matrix[i, :self.EMBED_DIM] = embed
            matrix[i, self.EMBED_DIM:self.EMBED_DIM + self.POS_DIM] = pos
            matrix[i, self.EMBED_DIM + self.POS_DIM:self.EMBED_DIM + self.POS_DIM + self.FREQ_DIM] = freq
            matrix[i, self.EMBED_DIM + self.POS_DIM + self.FREQ_DIM:
                      self.EMBED_DIM + self.POS_DIM + self.FREQ_DIM + self.FLAG_DIM] = flags
            matrix[i, -self.CTX_DIM:] = ctx

        base_type_counts = Counter(base_types)
        modifiers = {
            "position": np.array([i / max(n - 1, 1) for i in range(n)], dtype=np.float32),
            "frequency": np.array(freq_values, dtype=np.float32),
            "type": base_types,
        }

        metadata = {
            "text": text,
            "num_tokens": total_words,
            "num_bases": n,
            "base_types": dict(base_type_counts),
            "fitted": self.is_fitted,
        }

        return BaseMap(matrix=matrix, bases=bases, modifiers=modifiers, metadata=metadata)

    def fit_transform(self, texts: List[str], target_text: Optional[str] = None) -> BaseMap:
        """Fit on texts and transform the target (or first text if none given)."""
        self.fit(texts)
        return self.transform(target_text or texts[0])

    def _tokenize(self, text: str) -> List[str]:
        """Tokenize text into cleaned word tokens."""
        text = text.lower().strip()
        tokens = re.findall(r"\b[\w']+\b|[^\w\s]", text)
        return [t for t in tokens if t]

    def _extract_ngrams(self, tokens: List[str], n: int) -> List[str]:
        """Extract n-gram strings from a token list."""
        return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]

    def _extract_bases_with_types(self, tokens: List[str]) -> Tuple[List[str], List[str]]:
        """
        Extract bases from tokens.

        Priority:
          1. Known n-gram phrases (greedy, longest first)
          2. Known words (from vocabulary)
          3. Character-level fallback for unknown tokens
        Repeated identical bases are compressed (single entry with freq modifier).
        """
        bases = []
        base_types = []
        seen_bases: Dict[str, int] = {}

        i = 0
        while i < len(tokens):
            matched = False

            if self.is_fitted:
                for n in range(self.ngram_range[1], self.ngram_range[0] - 1, -1):
                    if i + n <= len(tokens):
                        candidate = " ".join(tokens[i:i + n])
                        if candidate in self.ngram_vocab:
                            if candidate not in seen_bases:
                                bases.append(candidate)
                                base_types.append("phrase")
                                seen_bases[candidate] = len(bases) - 1
                            i += n
                            matched = True
                            break

            if not matched:
                token = tokens[i]
                if self.is_fitted and token in self.word_vocab:
                    if token not in seen_bases:
                        bases.append(token)
                        base_types.append("word")
                        seen_bases[token] = len(bases) - 1
                    i += 1
                else:
                    chars = list(token)
                    for ch in chars:
                        if ch not in seen_bases:
                            bases.append(ch)
                            base_types.append("char")
                            seen_bases[ch] = len(bases) - 1
                    i += 1

        return bases, base_types

    def _base_embedding(self, base: str, dim: int) -> np.ndarray:
        """
        Stable hash-based embedding for a base token.
        Uses a seeded RNG to ensure the same base always gets the same vector.
        Normalized to unit sphere.
        """
        if base in self._embed_cache:
            return self._embed_cache[base]
        seed = hash(base) % (2 ** 31)
        rng = np.random.RandomState(seed)
        vec = rng.randn(dim).astype(np.float32)
        norm = np.linalg.norm(vec)
        if norm > 1e-10:
            vec /= norm
        self._embed_cache[base] = vec
        return vec

    def _position_encoding(self, pos: int, total: int, dim: int) -> np.ndarray:
        """
        Sinusoidal position encoding.
        Encodes both absolute and relative position.
        """
        enc = np.zeros(dim, dtype=np.float32)
        relative_pos = pos / max(total - 1, 1)
        for k in range(dim // 2):
            freq = 1.0 / (10000 ** (2 * k / dim))
            enc[2 * k] = math.sin(relative_pos * math.pi * (k + 1))
            enc[2 * k + 1] = math.cos(relative_pos * math.pi * (k + 1))
        return enc

    def _frequency_features(
        self, base: str, freq: float, max_freq: float, btype: str
    ) -> np.ndarray:
        """
        4-dimensional frequency feature vector:
          [raw_freq, log_freq, rank_normalized, normalized_freq]
        """
        raw = freq
        log_f = math.log1p(freq)
        log_max = math.log1p(max_freq) if max_freq > 0 else 1.0
        norm_log = log_f / max(log_max, 1e-10)
        norm_raw = freq / max(max_freq, 1e-10)
        rank = 0.0
        if self.is_fitted:
            if btype == "word" and base in self.word_freq:
                rank_val = sorted(self.word_freq.values(), reverse=True).index(
                    self.word_freq[base]
                ) if self.word_freq[base] in self.word_freq.values() else 0
                rank = 1.0 / (1.0 + rank_val)
        return np.array([norm_raw, norm_log, rank, math.tanh(raw / 10.0)], dtype=np.float32)

    def _modifier_flags(
        self, base: str, btype: str, pos: int, total: int, tokens: List[str]
    ) -> np.ndarray:
        """
        16-dimensional modifier flag vector encoding structural/linguistic properties.
        """
        flags = np.zeros(self.FLAG_DIM, dtype=np.float32)
        flags[0] = 1.0 if btype == "word" else 0.0
        flags[1] = 1.0 if btype == "char" else 0.0
        flags[2] = 1.0 if btype == "phrase" else 0.0
        flags[3] = 1.0 if pos == 0 else 0.0
        flags[4] = 1.0 if pos == total - 1 else 0.0
        flags[5] = 1.0 if 0 < pos < total - 1 else 0.0
        flags[6] = 1.0 if base.replace(" ", "").isalpha() else 0.0
        flags[7] = 1.0 if base.replace(" ", "").isdigit() else 0.0
        flags[8] = 1.0 if not base.replace(" ", "").isalnum() else 0.0
        flags[9] = float(len(base)) / 20.0
        flags[10] = 1.0 if " " in base else 0.0
        flags[11] = 1.0 if len(base) == 1 else 0.0
        flags[12] = 1.0 if len(base) > 8 else 0.0
        flags[13] = float(pos) / max(total - 1, 1)
        flags[14] = 1.0 if pos < total / 3 else 0.0
        flags[15] = 1.0 if pos > 2 * total / 3 else 0.0
        return np.clip(flags, 0.0, 1.0)

    def _context_summary(
        self, pos: int, bases: List[str], base_types: List[str]
    ) -> np.ndarray:
        """
        4-dimensional summary of local context window around position pos.
        """
        n = len(bases)
        window_start = max(0, pos - self.context_window)
        window_end = min(n, pos + self.context_window + 1)
        window = [bases[j] for j in range(window_start, window_end) if j != pos]

        ctx = np.zeros(self.CTX_DIM, dtype=np.float32)
        ctx[0] = len(window) / (2 * self.context_window)
        ctx[1] = sum(1 for b in window if " " in b) / max(len(window), 1)
        ctx[2] = sum(1 for b in window if base_types[bases.index(b)] == "word") / max(len(window), 1) if window else 0.0
        ctx[3] = float(np.mean([len(b) for b in window])) / 20.0 if window else 0.0
        return np.clip(ctx, 0.0, 1.0)

    def _get_freq_values(self, bases: List[str]) -> List[float]:
        """Get frequency values for a list of bases."""
        freqs = []
        for base in bases:
            if base in self.word_freq:
                freqs.append(float(self.word_freq[base]))
            elif base in self.ngram_freq:
                freqs.append(float(self.ngram_freq[base]))
            elif base in self.char_freq:
                freqs.append(float(self.char_freq[base]))
            else:
                freqs.append(1.0)
        return freqs
