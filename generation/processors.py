"""
IECNN Score Processor Pipeline
================================
All processors operate on cosine similarity score arrays (V,) over the
generation vocabulary.  They mirror HuggingFace's LogitsProcessorList
pattern but live entirely in IECNN's unit-sphere score space — no token
IDs, no softmax, no gradient computation.

Recommended application order
------------------------------
1.  SemanticFieldBias           — add prompt-topic coherence signal
2.  VocabFrequencyPrior         — log-frequency prior (penalise ultra-rare words)
3.  RepetitionPenalty           — subtract presence + frequency penalties
4.  NoRepeatNGram               — hard-block n-gram repeats
5.  DegenerationPenalty         — SimCTG anti-degeneration (NeurIPS '22)
6.  MinLengthGuard              — block stop tokens before min_tokens
7.  ExponentialDecayLength      — gradually boost stop scores after start_idx
8.  TypicalFilter               — typical-sampling filter
9.  NucleusFilter               — top-p nucleus filter
10. MinPFilter / EtaFilter      — adaptive threshold filter (choose one)
11. DynamicTemperature          — adaptive temperature scaling
12. softmax_sample()            — draw next token

Reference papers
----------------
• Repetition penalty          : Keskar et al. 2019 (CTRL)
• No-repeat n-gram            : Paulus et al. 2018 (Abstractive Summarization)
• SimCTG / Contrastive search : Su & Collier 2022, NeurIPS spotlight
• Typical sampling            : Meister et al. 2023
• Nucleus (top-p)             : Holtzman et al. 2020 (The Curious Case of NN)
• Min-p filter                : Menhguin & Kalomaze, 2023 (ICLR 2025 Oral)
• Eta (truncation) sampling   : Hewitt et al. 2022
• Mirostat (target perplexity): Basu et al. 2020 (ICLR 2021)
• Exponential decay length    : HuggingFace Transformers 4.x
"""

from __future__ import annotations
import numpy as np
from typing import Dict, List, Optional, Sequence, Tuple


_NEG_INF = -1e18   # sentinel "blocked" score


# ─────────────────────────────────────────────────────────────────────
# Helper
# ─────────────────────────────────────────────────────────────────────

def _log_softmax(scores: np.ndarray) -> np.ndarray:
    s = scores - scores.max()
    return s - np.log(np.exp(s).sum() + 1e-30)


def softmax_sample(scores: np.ndarray, temperature: float = 1.0,
                   rng: Optional[np.random.Generator] = None) -> int:
    """Sample an index from scores via temperature softmax.

    Handles -inf (blocked) entries by treating them as zero probability.
    Returns the argmax when temperature → 0 (greedy mode).
    """
    if temperature <= 0.0:
        valid = np.where(scores > _NEG_INF / 2)[0]
        return int(valid[scores[valid].argmax()]) if len(valid) else 0

    finite = scores > _NEG_INF / 2
    if not finite.any():
        return int(scores.argmax())

    s = scores[finite]
    s = (s - s.max()) / max(temperature, 1e-6)
    probs = np.exp(s); probs /= probs.sum()

    _rng = rng or np.random.default_rng()
    chosen = int(_rng.choice(finite.sum(), p=probs))
    return int(np.where(finite)[0][chosen])


# ─────────────────────────────────────────────────────────────────────
# Base class
# ─────────────────────────────────────────────────────────────────────

class IECNNScoreProcessor:
    """Base class — receives and returns (V,) float32 score array."""
    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        raise NotImplementedError


class ScoreProcessorList:
    """Applies a sequence of score processors in order.

    Equivalent to HuggingFace's LogitsProcessorList but in IECNN score space.
    Processors are called with shared keyword-context so they can read
    any keyword that an upstream caller passes.
    """
    def __init__(self, processors: Sequence[IECNNScoreProcessor] = ()):
        self._procs: List[IECNNScoreProcessor] = list(processors)

    def append(self, p: IECNNScoreProcessor) -> "ScoreProcessorList":
        self._procs.append(p); return self

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        for p in self._procs:
            scores = p(scores, **kw)
        return scores


# ─────────────────────────────────────────────────────────────────────
# 1. Semantic Field Bias
# ─────────────────────────────────────────────────────────────────────

class SemanticFieldBias(IECNNScoreProcessor):
    """Bias scores toward vocab semantically related to the prompt.

    Pre-computed at generation start from the prompt's embedding.
    Words above ``threshold`` cosine similarity to the prompt vector
    receive a bonus proportional to their similarity × ``strength``.

    This is IECNN's inference-time equivalent of prompt-following:
    no additional training needed — pure embedding-space steering.

    Parameters
    ----------
    strength  : additive bonus scale (default 0.10)
    threshold : minimum cosine similarity to qualify (default 0.20)
    decay     : bonus decay per token step (default 1.0 = no decay)
    """

    def __init__(self, word_vecs_n: np.ndarray, prompt_embed: np.ndarray,
                 strength: float = 0.10, threshold: float = 0.20,
                 decay: float = 1.0):
        pv  = prompt_embed[:word_vecs_n.shape[1]].astype(np.float32)
        pv /= np.linalg.norm(pv) + 1e-9
        sims          = word_vecs_n @ pv                         # (V,)
        above         = sims > threshold
        self._bias    = np.where(above, sims * strength, 0.0).astype(np.float32)
        self._decay   = float(decay)
        self._step    = 0
        # Stored for update() method — topic-tracking support
        self._anchor      = pv.copy()
        self._word_vecs_n = word_vecs_n
        self._strength    = float(strength)
        self._threshold   = float(threshold)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        scale = self._decay ** self._step
        self._step += 1
        return scores + self._bias * scale

    def update(self, new_embed: np.ndarray, blend: float = 0.15) -> None:
        """Blend the topic anchor toward a new embedding.

        Called periodically (e.g., every 3 tokens) with the current context
        vector so the bias tracks the evolving topic rather than only the
        original prompt.  ``blend`` controls how fast the anchor drifts:
        0.0 = frozen, 1.0 = full replacement.

        Algorithm:
            anchor = (1−blend) × anchor + blend × unit(new_embed)
            _bias  = recomputed from updated anchor × strength

        Parameters
        ----------
        new_embed : raw (non-normalised) embedding of the new reference point
        blend     : EMA blending coefficient (default 0.15)
        """
        v = new_embed[:self._word_vecs_n.shape[1]].astype(np.float32)
        n = float(np.linalg.norm(v))
        if n < 1e-9:
            return
        v /= n
        # Blend stored anchor toward new embedding
        self._anchor = (1.0 - blend) * self._anchor + blend * v
        an = float(np.linalg.norm(self._anchor))
        if an > 1e-9:
            self._anchor /= an
        # Recompute bias from updated anchor
        sims        = self._word_vecs_n @ self._anchor              # (V,)
        above       = sims > self._threshold
        self._bias  = np.where(above, sims * self._strength, 0.0).astype(np.float32)

    def reset(self):
        self._step = 0


# ─────────────────────────────────────────────────────────────────────
# 2. Repetition Penalty  (Keskar et al. 2019, CTRL)
# ─────────────────────────────────────────────────────────────────────

class RepetitionPenalty(IECNNScoreProcessor):
    """Presence + frequency repetition penalty in cosine score space.

    Transformer version divides logits; IECNN version subtracts from
    cosine scores (additive penalty in similarity space).

    For each previously seen token index i:
      penalty[i] = presence + frequency × count[i]

    Parameters
    ----------
    presence  : flat penalty for any token seen before  (default 0.15)
    frequency : additional penalty per extra occurrence (default 0.08)
    window    : only penalise last ``window`` tokens (default 0 = all)
    """

    def __init__(self, presence: float = 0.15, frequency: float = 0.08,
                 window: int = 0):
        self.presence  = float(presence)
        self.frequency = float(frequency)
        self.window    = int(window)

    def __call__(self, scores: np.ndarray, generated_ids: Sequence[int] = (),
                 recency_decay: float = 1.0,
                 **kw) -> np.ndarray:
        """Apply presence + frequency repetition penalty.

        Parameters
        ----------
        scores        : (V,) score array
        generated_ids : sequence of vocab indices emitted so far (oldest first)
        recency_decay : per-step exponential decay for *presence* penalty only.
                        1.0 = uniform (default), 0.88 = tokens 8 steps ago
                        receive ~37 % of the presence penalty.  Frequency
                        penalty is always full-strength regardless of age.
        """
        if not generated_ids:
            return scores
        ids = list(generated_ids[-self.window:]) if self.window else list(generated_ids)
        n   = len(ids)
        # presence penalty with recency decay: age 0 (most recent) → full penalty
        #   age k (steps ago) → recency_decay^k × presence
        counts: Dict[int, int] = {}
        pres:   Dict[int, float] = {}
        for rev_i, token_id in enumerate(reversed(ids)):
            age   = rev_i            # 0 = most recently generated
            decay = float(recency_decay ** age)
            if token_id not in counts:
                counts[token_id] = 0
                pres[token_id]   = 0.0
            counts[token_id] += 1
            pres[token_id]    = max(pres[token_id], self.presence * decay)

        out = scores.copy()
        for idx, cnt in counts.items():
            out[idx] -= pres[idx] + self.frequency * cnt
        return out


# ─────────────────────────────────────────────────────────────────────
# 3. No-Repeat N-Gram  (Paulus et al. 2018)
# ─────────────────────────────────────────────────────────────────────

class NoRepeatNGram(IECNNScoreProcessor):
    """Hard-block tokens that would create a repeated n-gram.

    Tracks all ngrams of size ``n`` in the output so far.  When the
    last (n-1) tokens form a prefix that has appeared before, the
    token(s) that completed that ngram in the past are blocked.

    Transformer equivalent: ``no_repeat_ngram_size``.

    Parameters
    ----------
    n           : n-gram size (default 3 — blocks repeated trigrams)
    word_index  : dict mapping token string → vocab index
    """

    def __init__(self, n: int = 3, word_index: Optional[Dict[str, int]] = None):
        self.n          = int(n)
        self.word_index = word_index or {}

    def __call__(self, scores: np.ndarray,
                 output_tokens: Sequence[str] = (),
                 word_index: Optional[Dict[str, int]] = None,
                 **kw) -> np.ndarray:
        n  = self.n
        wi = word_index or self.word_index
        if len(output_tokens) < n - 1 or not wi:
            return scores

        hist = list(output_tokens)
        ngrams: Dict[Tuple, set] = {}
        for i in range(len(hist) - n + 1):
            prefix = tuple(hist[i : i + n - 1])
            ngrams.setdefault(prefix, set()).add(hist[i + n - 1])

        cur_prefix = tuple(hist[-(n - 1):])
        blocked    = ngrams.get(cur_prefix, set())
        if not blocked:
            return scores

        out = scores.copy()
        for tok in blocked:
            idx = wi.get(tok)
            if idx is not None:
                out[idx] = _NEG_INF
        return out


# ─────────────────────────────────────────────────────────────────────
# 4. Degeneration Penalty  (SimCTG — Su & Collier, NeurIPS '22)
# ─────────────────────────────────────────────────────────────────────

class DegenerationPenalty(IECNNScoreProcessor):
    """IECNN-native contrastive degeneration penalty (SimCTG).

    For each candidate token v:
      deg(v) = max_{j in output} cos(embed(v), embed(j))
      score(v) = (1 − α) × score(v)  −  α × deg(v)

    This penalises tokens whose embeddings are similar to any previously
    generated token's embedding, directly fighting degenerate repetition
    at the representation level — more powerful than surface-level
    n-gram blocking because it catches semantic repetition too
    ("sad" blocked if "unhappy" was already said).

    Parameters
    ----------
    alpha  : degeneration weight (default 0.12).
             0 = no penalty, 1 = pure contrastive.
    window : only consider last ``window`` output embeddings (default 16)
    """

    def __init__(self, alpha: float = 0.12, window: int = 16):
        self.alpha  = float(alpha)
        self.window = int(window)

    def __call__(self, scores: np.ndarray,
                 word_vecs_n: Optional[np.ndarray] = None,
                 prev_vecs: Optional[np.ndarray] = None,
                 **kw) -> np.ndarray:
        if (word_vecs_n is None or prev_vecs is None
                or len(prev_vecs) == 0 or self.alpha == 0.0):
            return scores

        pv = np.asarray(prev_vecs[-self.window:], dtype=np.float32)  # (N, D)
        wv = word_vecs_n.astype(np.float32)                          # (V, D)

        # Align dimensions
        D = min(wv.shape[1], pv.shape[1])
        sims = wv[:, :D] @ pv[:, :D].T   # (V, N)
        deg  = sims.max(axis=1)           # (V,) max similarity to any prev token

        return (1.0 - self.alpha) * scores - self.alpha * deg


# ─────────────────────────────────────────────────────────────────────
# 5. Minimum Length Guard
# ─────────────────────────────────────────────────────────────────────

class MinLengthGuard(IECNNScoreProcessor):
    """Block stop tokens until ``min_tokens`` have been generated.

    Prevents the model from emitting a single-token or empty response
    when early uncertainty would normally trigger a stop.

    Transformer equivalent: ``min_new_tokens``.

    Parameters
    ----------
    min_tokens : minimum output tokens before any stop is allowed
    stop_ids   : set of vocab indices that are stop tokens
    """

    def __init__(self, min_tokens: int = 5, stop_ids: Optional[set] = None):
        self.min_tokens = int(min_tokens)
        self.stop_ids   = stop_ids or set()

    def __call__(self, scores: np.ndarray, step: int = 0,
                 stop_ids: Optional[set] = None, **kw) -> np.ndarray:
        sids = stop_ids or self.stop_ids
        if step >= self.min_tokens or not sids:
            return scores
        out = scores.copy()
        for idx in sids:
            if 0 <= idx < len(out):
                out[idx] = _NEG_INF
        return out


# ─────────────────────────────────────────────────────────────────────
# 6. Typical Sampling  (Meister et al. 2023)
# ─────────────────────────────────────────────────────────────────────

class TypicalFilter(IECNNScoreProcessor):
    """Locally typical sampling filter.

    Keeps tokens whose surprisal is closest to the expected entropy of
    the distribution.  More natural than top-p because it rejects both
    over-predictable (surprisal << entropy) AND over-surprising tokens.

    Parameters
    ----------
    p        : mass cutoff (default 0.95)
    min_keep : always keep at least this many tokens
    """

    def __init__(self, p: float = 0.95, min_keep: int = 1):
        self.p        = float(p)
        self.min_keep = int(min_keep)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        log_p  = _log_softmax(scores)              # (V,)
        probs  = np.exp(log_p)
        H      = float(-(probs * log_p).sum())     # entropy of distribution
        # Typical score: absolute deviation from entropy
        typ    = np.abs(-log_p - H)                # (V,)
        # Sort by ascending typical score (most typical first)
        order  = np.argsort(typ)
        cumsum = np.cumsum(probs[order])
        cutoff = int(np.searchsorted(cumsum, self.p) + 1)
        cutoff = max(cutoff, self.min_keep)

        out          = np.full_like(scores, _NEG_INF, dtype=np.float32)
        keep         = order[:cutoff]
        out[keep]    = scores[keep]
        return out


# ─────────────────────────────────────────────────────────────────────
# 7. Nucleus Filter  (Holtzman et al. 2020 — The Curious Case of NN)
# ─────────────────────────────────────────────────────────────────────

class NucleusFilter(IECNNScoreProcessor):
    """Top-p (nucleus) sampling filter.

    Restricts sampling to the smallest set of tokens whose cumulative
    probability sums to at least ``p``.

    Parameters
    ----------
    p        : nucleus mass cutoff (default 0.90)
    min_keep : always keep at least this many tokens
    """

    def __init__(self, p: float = 0.90, min_keep: int = 1):
        self.p        = float(p)
        self.min_keep = int(min_keep)

    def __call__(self, scores: np.ndarray, p: float = None, **kw) -> np.ndarray:
        """Filter scores to nucleus top-p.

        Parameters
        ----------
        scores : (V,) score array
        p      : override the instance ``p`` for this call (enables adaptive
                 top-p scheduling from the generation loop without rebuilding
                 the processor object)
        """
        _p          = float(p) if p is not None else self.p
        sorted_idx  = np.argsort(scores)[::-1]         # descending
        s           = scores[sorted_idx]
        s_shifted   = s - s.max()
        probs       = np.exp(s_shifted); probs /= probs.sum()
        cum_probs   = np.cumsum(probs)
        # Find first index where cumulative prob >= _p
        cutoff = int(np.searchsorted(cum_probs, _p) + 1)
        cutoff = max(cutoff, self.min_keep)

        out          = np.full_like(scores, _NEG_INF, dtype=np.float32)
        keep         = sorted_idx[:cutoff]
        out[keep]    = scores[keep]
        return out


# ─────────────────────────────────────────────────────────────────────
# 8. Min-P Filter  (Menhguin & Kalomaze, 2023)
# ─────────────────────────────────────────────────────────────────────

class MinPFilter(IECNNScoreProcessor):
    """Adaptive min-p sampling filter.

    Removes tokens below  min_p × p_max.  The threshold scales with the
    strongest token — so when the model is confident (high p_max) the
    filter becomes more aggressive, and when uncertain it stays open.

    Often performs better than top-p because the threshold adapts to
    the sharpness of the distribution rather than a fixed quantile.

    Parameters
    ----------
    min_p    : base probability fraction (default 0.05)
    min_keep : always keep at least this many tokens
    """

    def __init__(self, min_p: float = 0.05, min_keep: int = 1):
        self.min_p    = float(min_p)
        self.min_keep = int(min_keep)

    def __call__(self, scores: np.ndarray,
                 min_p: Optional[float] = None,
                 **kw) -> np.ndarray:
        """Apply min-p filter.

        Parameters
        ----------
        scores : (V,) score array
        min_p  : optional override for adaptive scheduling (default: use self.min_p)
        """
        _mp      = float(min_p) if min_p is not None else self.min_p
        s        = scores - scores.max()
        probs    = np.exp(s); probs /= probs.sum()
        p_max    = float(probs.max())
        thresh   = _mp * p_max
        mask     = probs < thresh

        out = scores.copy()
        out[mask] = _NEG_INF

        # Ensure at least min_keep tokens remain finite
        finite = (out > _NEG_INF / 2).sum()
        if finite < self.min_keep:
            keep_idx       = np.argsort(scores)[::-1][: self.min_keep]
            out[keep_idx]  = scores[keep_idx]
        return out


# ─────────────────────────────────────────────────────────────────────
# 9. Dynamic Temperature
# ─────────────────────────────────────────────────────────────────────

class DynamicTemperature:
    """Adaptive temperature that reacts to the recent confidence trend.

    • Confidence trending UP   → lower temperature (sharpen output)
    • Confidence trending DOWN → higher temperature (more exploration)
    • Base temperature sets the minimum floor

    Returns a float temperature value; not a score processor because
    temperature is applied inside softmax_sample(), not as an additive
    score delta.

    Parameters
    ----------
    base     : minimum temperature floor (default 0.15)
    max_temp : hard ceiling (default 0.70)
    window   : how many recent steps to look at for the trend
    """

    def __init__(self, base: float = 0.15, max_temp: float = 0.70,
                 window: int = 4):
        self.base     = float(base)
        self.max_temp = float(max_temp)
        self.window   = int(window)

    def get(self, confidence: float,
            conf_history: Sequence[float]) -> float:
        """Compute temperature for current step."""
        if len(conf_history) < 2:
            return float(np.clip(self.base + 0.60 * (1.0 - confidence),
                                 self.base, self.max_temp))

        recent = np.array(list(conf_history)[-self.window:], dtype=np.float32)
        trend  = float(recent[-1] - recent[0]) / max(len(recent) - 1, 1)
        # Negative trend → raise temp; positive trend → lower temp
        adj    = self.base + 0.40 * (1.0 - confidence) - 0.30 * trend
        return float(np.clip(adj, self.base, self.max_temp))


# ─────────────────────────────────────────────────────────────────────
# 10. Eta (Truncation) Sampling  (Hewitt et al. 2022)
# ─────────────────────────────────────────────────────────────────────

class EtaFilter(IECNNScoreProcessor):
    """Entropy-adaptive truncation sampling filter.

    Computes a dynamic cutoff value:
        eta = min(epsilon, sqrt(epsilon × exp(−H))
    where H is the entropy of the distribution, then keeps only tokens
    with probability ≥ eta.

    At high entropy (uncertain model): eta is small → wide net kept.
    At low entropy (confident model):  eta is large → tight selection.

    This is strictly more principled than a fixed threshold because it
    adapts to how "spread" the current distribution is.

    Reference: Hewitt et al. 2022, "Truncation Sampling as Language
    Model Desmoothing" — EMNLP Findings.

    Parameters
    ----------
    epsilon  : base threshold (default 3e-4; paper suggests 3e-4 to 4e-3)
    min_keep : always keep at least this many tokens
    """

    def __init__(self, epsilon: float = 3e-4, min_keep: int = 1):
        self.epsilon  = float(epsilon)
        self.min_keep = int(min_keep)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        log_p = _log_softmax(scores)          # (V,)
        probs  = np.exp(log_p)
        H      = float(-(probs * log_p).sum())    # entropy of distribution

        # Dynamic threshold
        eta   = min(self.epsilon, float(np.sqrt(self.epsilon * np.exp(-H))))
        mask  = probs < eta

        out = scores.copy()
        out[mask] = _NEG_INF

        # Ensure at least min_keep tokens remain
        finite = (out > _NEG_INF / 2).sum()
        if finite < self.min_keep:
            keep_idx      = np.argsort(scores)[::-1][: self.min_keep]
            out[keep_idx] = scores[keep_idx]
        return out


# ─────────────────────────────────────────────────────────────────────
# 11. Exponential Decay Length Penalty
# ─────────────────────────────────────────────────────────────────────

class ExponentialDecayLength(IECNNScoreProcessor):
    """Exponentially boost stop-token scores after a start index.

    After ``start_idx`` tokens have been generated, the score of every
    stop token is boosted by ``(factor − 1) × factor^(step − start_idx)``.
    This creates a smooth, increasing pressure to terminate naturally
    rather than hitting a hard max-length cutoff.

    Transformer equivalent: ``ExponentialDecayLengthPenalty``.

    Parameters
    ----------
    start_idx : generation step after which boosting begins (default 8)
    factor    : exponential base (default 1.08; range 1.01–1.30)
    stop_ids  : set of vocab indices for stop tokens
    """

    def __init__(self, start_idx: int = 8, factor: float = 1.08,
                 stop_ids: Optional[set] = None):
        self.start_idx = int(start_idx)
        self.factor    = float(factor)
        self.stop_ids  = stop_ids or set()

    def __call__(self, scores: np.ndarray, step: int = 0,
                 stop_ids: Optional[set] = None, **kw) -> np.ndarray:
        sids = stop_ids or self.stop_ids
        if step < self.start_idx or not sids:
            return scores

        boost = (self.factor - 1.0) * (self.factor ** (step - self.start_idx))
        out   = scores.copy()
        for idx in sids:
            if 0 <= idx < len(out) and out[idx] > _NEG_INF / 2:
                out[idx] += float(boost)
        return out


# ─────────────────────────────────────────────────────────────────────
# 12. Vocabulary Frequency Prior
# ─────────────────────────────────────────────────────────────────────

class VocabFrequencyPrior(IECNNScoreProcessor):
    """Log-frequency prior that penalises ultra-rare tokens.

    Adds a small bonus proportional to log(training_count + 1) × strength.
    The bias is zero-meaned so it only shifts the relative ordering of
    words — it doesn't inflate or deflate the overall score scale.

    Motivation: in a well-trained IECNN, common words are more often the
    correct next token than hapax legomena.  A frequency prior makes the
    model's choice more robust against cosine noise that can elevate
    obscure vocabulary to spurious top positions.

    Parameters
    ----------
    words     : the generation vocabulary list (must match scores dim)
    word_freq : Counter / dict of {word: count} from the BaseMapper
    strength  : log-frequency scale factor (default 0.04)
    """

    def __init__(self, words: List[str], word_freq: dict,
                 strength: float = 0.04):
        bias = np.zeros(len(words), dtype=np.float32)
        for i, w in enumerate(words):
            cnt    = word_freq.get(w, 0)
            bias[i] = float(np.log1p(cnt)) * strength
        bias  -= bias.mean()     # zero-centre
        self._bias = bias.astype(np.float32)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        return scores + self._bias


# ─────────────────────────────────────────────────────────────────────
# 13. Mirostat v2 Scheduler  (Basu et al. 2020, ICLR 2021)
# ─────────────────────────────────────────────────────────────────────

class MirostatScheduler:
    """Feedback-based adaptive temperature targeting a confidence level.

    Inspired by Mirostat (Basu et al. 2020), which targets a fixed
    perplexity level via a feedback loop over the token surprisal.

    IECNN adaptation:
      • Works in confidence space (cosine scores) instead of log-prob space
      • Maintains an internal estimate ``mu`` of the model's current
        confidence level
      • After each token, updates: mu += lr × (observed_conf − target)
      • Temperature = base / (mu + ε)  — inversely proportional to mu

    Warmup schedule:
      • During the first ``warmup_steps`` tokens the temperature ceiling is
        linearly decayed from ``warmup_max_temp`` down to ``max_temp``.
      • This gives richer exploration in the early tokens (when the model
        has less context to constrain sampling) and tighter focus once the
        sentence is established.

    Effect:
      • When confidence drops below target → mu decreases → temperature
        rises → more exploratory sampling
      • When confidence exceeds target   → mu increases → temperature
        falls → sharper, more focused sampling

    This gives consistent quality throughout a sequence regardless of
    how confident the model is at each individual step.

    Parameters
    ----------
    target          : desired average confidence level (default 0.38)
    lr              : feedback learning rate (default 0.08)
    base            : temperature floor (default 0.18)
    max_temp        : temperature ceiling at steady state (default 0.65)
    warmup_max_temp : temperature ceiling at step 0 (default 0.85)
    warmup_steps    : steps over which ceiling decays to max_temp (default 6)
    """

    def __init__(self, target: float = 0.38, lr: float = 0.08,
                 base: float = 0.18, max_temp: float = 0.65,
                 warmup_max_temp: float = 0.85, warmup_steps: int = 6):
        self.target          = float(target)
        self.lr              = float(lr)
        self.base            = float(base)
        self.max_temp        = float(max_temp)
        self.warmup_max_temp = float(warmup_max_temp)
        self.warmup_steps    = int(warmup_steps)
        self._mu             = float(target)   # running confidence estimate
        self._step           = 0               # tokens emitted so far

    def update(self, observed_conf: float) -> None:
        """Update mu after observing a token's confidence."""
        self._mu += self.lr * (observed_conf - self.target)
        self._mu  = float(np.clip(self._mu, 0.05, 1.0))
        self._step += 1

    def get(self) -> float:
        """Current temperature derived from the Mirostat feedback state.

        Temperature ceiling is linearly interpolated from ``warmup_max_temp``
        to ``max_temp`` over the first ``warmup_steps`` tokens so that early
        sampling is exploratory and later sampling is focused.
        """
        frac    = min(1.0, self._step / max(self.warmup_steps, 1))
        cur_max = self.warmup_max_temp * (1.0 - frac) + self.max_temp * frac
        temp    = self.base / (self._mu + 0.01)
        return float(np.clip(temp, self.base, cur_max))

    def reset(self) -> None:
        self._mu   = float(self.target)
        self._step = 0


# ─────────────────────────────────────────────────────────────────────
# 14. Bigram Continuation Bonus
# ─────────────────────────────────────────────────────────────────────

class BigramContinuationBonus(IECNNScoreProcessor):
    """Data-driven bigram continuation scoring.

    If the phrase ``last_token + " " + candidate`` exists in the training
    vocabulary, the candidate receives a bonus proportional to the phrase
    embedding norm (proxy for observed co-occurrence frequency).

    This is IECNN's equivalent of n-gram language model interpolation:

        score(w | ctx) ≈ MHC_score(w | ctx) + λ × LM_bigram(w | last_token)

    Effect: common collocations ("the house", "his father", "she said",
    "never mind") are gently boosted, improving syntactic fluency without
    any additional training.

    The bonus is lazily cached per unique ``last_token`` — O(V) build cost
    once per unique predecessor, then O(1) lookups thereafter.  Cache is
    capped at 2000 entries.

    Parameters
    ----------
    words        : ordered list of generation vocab words
    phrase_vocab : dict[phrase_str → embedding_array] (base_mapper._base_vocab)
    strength     : maximum additive bonus (default 0.07)
    """

    def __init__(self,
                 words:        List[str],
                 phrase_vocab: Dict[str, np.ndarray],
                 strength:     float = 0.07):
        self._words       = words
        self._phrase_vocab = phrase_vocab
        self._strength    = float(strength)
        self._cache: Dict[str, np.ndarray] = {}

    def __call__(self, scores: np.ndarray,
                 last_token: str = "", **kw) -> np.ndarray:
        if not last_token or not self._phrase_vocab:
            return scores

        if last_token not in self._cache:
            bonus = np.zeros(len(scores), dtype=np.float32)
            for i, w in enumerate(self._words):
                phrase = last_token + " " + w
                emb = self._phrase_vocab.get(phrase)
                if emb is not None:
                    bonus[i] = float(np.linalg.norm(emb))

            # Normalise so the maximum bonus equals self._strength
            peak = float(bonus.max())
            if peak > 1e-9:
                bonus = bonus * (self._strength / peak)

            if len(self._cache) < 2000:
                self._cache[last_token] = bonus
            else:
                return scores + bonus   # cache full — apply without storing

        return scores + self._cache[last_token]


# ─────────────────────────────────────────────────────────────────────
# 15. Semantic Proximity Penalty
# ─────────────────────────────────────────────────────────────────────

class SemanticProximityPenalty(IECNNScoreProcessor):
    """Penalises candidates that are semantically near-duplicate of recent tokens.

    String-based repetition penalties (``RepetitionPenalty``, ``NoRepeatNGram``)
    block exact token re-use but cannot catch synonymous repetition such as
    "big … large … huge" or "said … told … replied".  This processor fills
    that gap by working in embedding space.

    Algorithm (fully vectorised):
        1. Collect unit-norm embeddings of the last ``window`` emitted tokens
           → R  (K, E)
        2. Compute cosine-similarity matrix: S = word_vecs_n @ R.T  → (V, K)
        3. Take max across recent tokens:  max_sim = S.max(axis=1)  → (V,)
        4. Penalty = proximity_scale × max(0, max_sim − sim_threshold)²

    The quadratic form concentrates punishment on near-duplicates (similarity
    ≥ 0.90) while leaving loosely-related words (~0.50) completely untouched.

    Cost: O(V × K × E) ≈ 5 000 × 4 × 224 = 4.5 M FP32 ops/step — fast.

    Parameters
    ----------
    window          : number of recent tokens to check against (default 4)
    sim_threshold   : cosine similarity above which penalty activates (default 0.82)
    proximity_scale : penalty coefficient (default 3.5)
    """

    def __init__(self, window: int = 4, sim_threshold: float = 0.82,
                 proximity_scale: float = 3.5):
        self.window          = int(window)
        self.sim_threshold   = float(sim_threshold)
        self.proximity_scale = float(proximity_scale)

    def __call__(self,
                 scores:        np.ndarray,
                 word_vecs_n:   Optional[np.ndarray] = None,
                 recent_vecs_n: Optional[np.ndarray] = None,
                 **kw) -> np.ndarray:
        """
        Parameters
        ----------
        scores        : (V,) score array
        word_vecs_n   : (V, E) unit-norm word embeddings — same E as recent_vecs_n
        recent_vecs_n : (K, E) unit-norm embeddings of recent tokens (K ≤ window)
        """
        if word_vecs_n is None or recent_vecs_n is None:
            return scores
        K = recent_vecs_n.shape[0]
        if K == 0:
            return scores

        R = recent_vecs_n[-self.window:]                   # (K, E)

        # Vectorised cosine-similarity matrix and per-word maximum
        sims    = word_vecs_n @ R.T                        # (V, K)
        max_sim = sims.max(axis=1)                         # (V,)

        excess  = np.maximum(0.0, max_sim - self.sim_threshold).astype(np.float32)
        penalty = (self.proximity_scale * excess * excess).astype(np.float32)

        # Preserve already-blocked tokens (do not un-block or double-penalise)
        blocked = scores <= _NEG_INF / 2
        out     = scores - penalty
        out[blocked] = scores[blocked]
        return out


# ─────────────────────────────────────────────────────────────────────
# 16. Tail-Free Sampling Filter
# ─────────────────────────────────────────────────────────────────────

class TopKFilter(IECNNScoreProcessor):
    """Adaptive top-K sampling filter.

    Restricts sampling to the top-K scoring tokens.  Acts as a hard cap
    before the softer distribution-shaping filters (TypicalFilter, TFS,
    NucleusFilter) — prevents any low-probability tail token from appearing
    regardless of distribution shape.

    Adaptive k (k_adaptive=True, default):
        k_eff = round(k × (1 + entropy_scale × H_norm))
        where H_norm = H / log(valid_count) ∈ [0, 1].
        When the model is uncertain (high entropy) → wider beam.
        When the model is confident (low entropy) → narrower focus.

    This avoids the failure mode of fixed top-K where confident distributions
    are still padded to K candidates and uncertain ones are harshly truncated.

    Parameters
    ----------
    k            : base maximum tokens to keep (default 40)
    min_keep     : hard minimum survivors (default 1)
    k_adaptive   : whether to scale k by distribution entropy (default True)
    entropy_scale: entropy scaling coefficient (default 0.50 → k range: k..1.5k)
    """

    def __init__(self, k: int = 40, min_keep: int = 1,
                 k_adaptive: bool = True, entropy_scale: float = 0.50):
        self.k             = int(k)
        self.min_keep      = int(min_keep)
        self.k_adaptive    = bool(k_adaptive)
        self.entropy_scale = float(entropy_scale)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        V   = len(scores)
        k_e = self.k

        if self.k_adaptive:
            _valid = scores[scores > _NEG_INF / 2]
            if len(_valid) > 1:
                _v       = _valid - _valid.max()
                _p       = np.exp(_v); _p /= _p.sum()
                _H       = float(-np.dot(_p, np.log(_p + 1e-30)))
                _H_norm  = float(np.clip(_H / np.log(len(_p)), 0.0, 1.0))
                k_e      = int(round(self.k * (1.0 + self.entropy_scale * _H_norm)))

        k_e = max(self.min_keep, min(k_e, V))

        if k_e >= V:
            return scores

        top_idx     = np.argpartition(scores, -k_e)[-k_e:]   # O(V) partial sort
        out         = np.full_like(scores, _NEG_INF, dtype=np.float32)
        out[top_idx] = scores[top_idx]
        return out


class PromptDriftPenalty(IECNNScoreProcessor):
    """Penalises tokens that would pull context away from the prompt direction.

    Complementary to ``SemanticFieldBias`` (which *adds* a bonus to
    prompt-similar tokens): ``PromptDriftPenalty`` *subtracts* from
    prompt-distant tokens, creating a symmetric push-pull constraint
    around the prompt's semantic direction.

    Effect on the combined pipeline
    --------------------------------
    SemanticFieldBias adds up to ``sfb_strength`` to tokens with
    sim > sfb_threshold.  PromptDriftPenalty subtracts up to ``strength``
    from tokens with sim < ``threshold``.  Together they form a soft
    "semantic attractor" centred on the prompt direction.

    Unlike ``ContextAnchor`` (which corrects the context vector each step),
    this processor acts directly on the score array — giving the filter
    chain a prompt-proximity signal independent of the running context.

    Parameters
    ----------
    strength  : maximum penalty magnitude (default 0.06)
    threshold : cosine-sim below which penalty starts (default 0.08)
                Tokens with sim ≥ threshold receive zero penalty.
    """

    def __init__(self, word_vecs_n: np.ndarray, prompt_embed: np.ndarray,
                 strength: float = 0.06, threshold: float = 0.08):
        pv  = prompt_embed[:word_vecs_n.shape[1]].astype(np.float32)
        pv /= float(np.linalg.norm(pv)) + 1e-9
        sims   = word_vecs_n @ pv             # (V,) cosine sim with prompt
        # penalty = strength × relu(threshold − sim) / threshold
        # → 0 at sim=threshold, ramps up to `strength` at sim=0
        below  = sims < threshold
        pen    = np.where(below,
                          strength * np.clip(threshold - sims, 0.0, None)
                          / (threshold + 1e-9),
                          0.0).astype(np.float32)
        self._penalty = pen   # (V,) precomputed — zero-cost per step

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        return scores - self._penalty


class LocalSemanticFilter(IECNNScoreProcessor):
    """Local semantic vocabulary restriction.

    Restricts sampling to the top-K vocabulary words most cosine-similar
    to the current enriched context direction.  Acts as a content-aware
    pre-filter complementary to the speculative two-pass filter:

      Speculative filter  → fast draft using a subset of dots
      LocalSemanticFilter → semantic locality: keeps only words *near*
                            the current context in embedding space

    Together they cut the live vocab from ~5 000 to a tightly focused pool
    before the heavier penalty processors run, improving both quality and speed.

    Parameters
    ----------
    top_k : maximum words to keep (default 200)
            Applied only when the vocab exceeds top_k.
    """

    def __init__(self, top_k: int = 200):
        self.top_k = int(top_k)

    def __call__(self, scores: np.ndarray,
                 word_vecs_n: Optional[np.ndarray] = None,
                 ctx_eff: Optional[np.ndarray] = None,
                 **kw) -> np.ndarray:
        if word_vecs_n is None or ctx_eff is None:
            return scores

        V = len(scores)
        k = min(self.top_k, V)
        if k >= V:
            return scores

        ed     = word_vecs_n.shape[1]             # EMBED_DIM
        ctx_e  = ctx_eff[:ed]
        ctx_n  = ctx_e / (float(np.linalg.norm(ctx_e)) + 1e-9)  # unit (ed,)
        sims   = word_vecs_n @ ctx_n              # (V,) cosine sims

        top_idx         = np.argpartition(sims, -k)[-k:]   # O(V)
        out             = np.full_like(scores, _NEG_INF, dtype=np.float32)
        out[top_idx]    = scores[top_idx]
        return out


class DotVariancePenalty(IECNNScoreProcessor):
    """Cross-dot prediction variance penalty.

    For each vocabulary word, measures the variance of cosine similarity
    scores across all neural dots.  High variance indicates strong dot
    disagreement about this word — an unreliable prediction.  Low variance
    signals consensus, suggesting a more trustworthy token choice.

    Unlike the cross-head agreement bonus (which counts top-K hits), this
    penalty operates on the full per-token score distribution across every
    dot and therefore catches cases where a token scores highly in some
    heads but near-zero in others — a sign of spurious activation.

    Algorithm (vectorised):
        dot_means_n  = unit(raw_preds[:, :embed_dim])         (n_dots, E)
        dot_scores   = dot_means_n @ word_vecs_n.T             (n_dots, V)
        var_scores   = variance over dots, per word             (V,)
        penalty      = strength × (var_scores / max_var)
        output       = scores − penalty   (hard-blocked slots untouched)

    Parameters
    ----------
    strength : maximum penalty subtraction (default 0.08)
    """

    def __init__(self, strength: float = 0.08):
        self.strength = float(strength)

    def __call__(self, scores: np.ndarray,
                 raw_preds: Optional[np.ndarray] = None,
                 word_vecs_n: Optional[np.ndarray] = None,
                 **kw) -> np.ndarray:
        if raw_preds is None or word_vecs_n is None:
            return scores

        ed      = word_vecs_n.shape[1]         # EMBED_DIM (224)
        dm      = raw_preds[:, :ed]            # (n_dots, embed_dim)
        nrms    = np.linalg.norm(dm, axis=1, keepdims=True).clip(1e-9)
        dm_n    = dm / nrms                    # (n_dots, embed_dim) unit
        dot_sc  = dm_n @ word_vecs_n.T         # (n_dots, V)

        var_sc  = dot_sc.var(axis=0)           # (V,) variance across dots
        max_var = float(var_sc.max())
        if max_var < 1e-9:
            return scores

        var_norm = var_sc / max_var            # (V,) in [0, 1]
        blocked  = scores <= _NEG_INF / 2
        out      = scores - self.strength * var_norm.astype(np.float32)
        out[blocked] = scores[blocked]         # restore hard-blocked sentinels
        return out


class SurpriseBonus(IECNNScoreProcessor):
    """Context-relevant surprise bonus.

    Rewards tokens that are semantically near the current context
    (cosine sim ≥ ``ctx_threshold``) but *not* among the top-N by raw
    score — these are "surprise" tokens: contextually plausible but
    underweighted by the dot voting.  Giving them a small bonus opens
    the effective beam slightly beyond what the dot consensus picks,
    producing more varied yet still on-topic continuations.

    This complements ``LocalSemanticFilter`` (which restricts to a
    context-close pool) by giving secondary-ranked tokens within that
    pool a second chance to surface.

    Parameters
    ----------
    strength      : bonus magnitude (default 0.04)
    top_skip      : number of already-top-ranked tokens that do NOT
                    receive the bonus (they don't need it) (default 50)
    ctx_threshold : minimum cosine sim to current context (default 0.15)
    """

    def __init__(self, strength: float = 0.04,
                 top_skip: int = 50, ctx_threshold: float = 0.15):
        self.strength      = float(strength)
        self.top_skip      = int(top_skip)
        self.ctx_threshold = float(ctx_threshold)

    def __call__(self, scores: np.ndarray,
                 word_vecs_n: Optional[np.ndarray] = None,
                 ctx_eff: Optional[np.ndarray] = None,
                 **kw) -> np.ndarray:
        if word_vecs_n is None or ctx_eff is None:
            return scores

        V = len(scores)
        # Context direction in embed space
        ed    = word_vecs_n.shape[1]
        ctx_e = ctx_eff[:ed]
        ctx_n = ctx_e / (float(np.linalg.norm(ctx_e)) + 1e-9)
        sims  = word_vecs_n @ ctx_n           # (V,) context cosine sims

        # Mask: contextually plausible (sim ≥ threshold)
        ctx_close = sims >= self.ctx_threshold

        # Mask: NOT in top-N by current scores (these already have signal)
        n_skip     = min(self.top_skip, V)
        top_idx    = np.argpartition(scores, -n_skip)[-n_skip:]
        top_mask   = np.zeros(V, dtype=bool)
        top_mask[top_idx] = True

        # Bonus: context-close AND not already top-ranked AND not hard-blocked
        finite_mask = scores > _NEG_INF / 2
        bonus_mask  = ctx_close & ~top_mask & finite_mask

        out = scores.copy()
        out[bonus_mask] += self.strength
        return out


class TailFreeFilter(IECNNScoreProcessor):
    """Tail-free sampling (TFS) filter.

    Removes tokens in the statistical "tail" of the sorted probability
    distribution by examining the second derivative of the sorted
    probability sequence.  The tail is defined as the region where
    the absolute second derivative crosses the cumulative threshold z.

    Reference: Tails of the Unexpected — Phénix & Egan (2019, unpublished
    preprint; widely adopted in LLM samplers including LLaMA.cpp).

    Algorithm:
        1. Sort tokens by descending score → compute softmax probs p
        2. d1 = |diff(p)|        (1st derivative of sorted probs)
        3. d2 = |diff(d1)|       (2nd derivative)
        4. d2 = d2 / sum(d2)     (normalise)
        5. cum_d2 = cumsum(d2)
        6. Cut at first index where cum_d2 >= z; keep at least min_keep tokens
        7. Block all tokens beyond the cut

    Rationale:
        Near the top of the distribution, consecutive probabilities differ
        by large, smooth amounts (genuine signal).  In the tail, they are
        roughly flat but noisy.  The second derivative detects the boundary
        between the two regimes without needing a fixed probability mass
        like nucleus sampling.

    z ≈ 1.00 = keep nearly everything (disabled); z ≈ 0.90 = moderate tail cut.
    Combines well with NucleusFilter — use TFS first, then nucleus as backstop.

    Parameters
    ----------
    z        : cumulative second-derivative threshold (default 0.95)
    min_keep : always keep at least this many tokens (default 1)
    """

    def __init__(self, z: float = 0.95, min_keep: int = 1):
        self.z        = float(z)
        self.min_keep = int(min_keep)

    def __call__(self, scores: np.ndarray, **kw) -> np.ndarray:
        sorted_idx = np.argsort(scores)[::-1]           # descending
        s          = scores[sorted_idx]
        s_shifted  = s - s.max()
        probs      = np.exp(s_shifted); probs /= probs.sum()

        # First and second absolute derivatives of sorted probs
        d1 = np.abs(np.diff(probs))
        d2 = np.abs(np.diff(d1))

        if len(d2) == 0:
            return scores

        d2_sum = float(d2.sum())
        if d2_sum < 1e-30:
            return scores

        d2_norm = d2 / d2_sum
        cum_d2  = np.cumsum(d2_norm)

        # First index where cumulative second derivative >= z
        # d2 has length len(probs) - 2, so cutoff index is offset by +2
        cutoff = int(np.searchsorted(cum_d2, self.z) + 2)
        cutoff = max(cutoff, self.min_keep)

        out       = np.full_like(scores, _NEG_INF, dtype=np.float32)
        keep      = sorted_idx[:cutoff]
        out[keep] = scores[keep]
        return out
