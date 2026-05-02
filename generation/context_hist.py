"""
IECNN Context History Manager
================================
Replaces the simple EMA context update with an attention-weighted
rolling buffer of recent token embeddings.

IECNN-native equivalent of Transformer KV-cache + multi-head self-attention:
  • No Q/K/V weight matrices
  • Attention weights computed as dot-product cosine between current
    context and stored history embeddings
  • Temporal position decay ensures recent tokens have stronger influence
  • Blends with current context via configurable alpha

This gives the generation loop genuine "soft memory" of earlier tokens
rather than an exponential moving average that forgets within 5-6 steps.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional


class ContextHistory:
    """Rolling attention-weighted context window.

    At each generation step:
      1. The current context vector is used as a "query"
      2. Cosine attention scores are computed against stored history
      3. Position decay down-weights older entries
      4. Softmax normalisation → attention weights
      5. Context = alpha × ctx_current + (1−alpha) × Σ attn_i × hist_i

    Parameters
    ----------
    window      : ring-buffer capacity in tokens (default 24)
    dim         : vector dimension (must match IECNN FEATURE_DIM = 256)
    decay       : per-step recency decay (0 < decay < 1, default 0.88)
                  decay=1.0 → uniform attention over window
    ctx_alpha   : weight of current context vs history blend (default 0.70)
    """

    def __init__(self, window: int = 24, dim: int = 256,
                 decay: float = 0.88, ctx_alpha: float = 0.70):
        self.window    = int(window)
        self.dim       = int(dim)
        self.decay     = float(decay)
        self.ctx_alpha = float(ctx_alpha)

        self._buf   = np.zeros((window, dim), dtype=np.float32)
        self._ptr   = 0    # write pointer
        self._count = 0    # total pushes (capped at window for indices)

    # ── Public API ──────────────────────────────────────────────────

    def push(self, vec: np.ndarray) -> None:
        """Store a (normalised) token embedding in the ring buffer."""
        v = vec.astype(np.float32)
        nrm = np.linalg.norm(v)
        self._buf[self._ptr] = v / nrm if nrm > 1e-9 else v
        self._ptr   = (self._ptr + 1) % self.window
        self._count += 1

    def attend(self, ctx: np.ndarray) -> np.ndarray:
        """Return attention-enriched context.

        Uses cosine dot-product attention over stored token history:
          - Older entries are down-weighted by position decay
          - Softmax normalises the attention distribution
          - Result is blended with the current context via ctx_alpha

        Returns the same-shape (dim,) unit-norm vector as the input.
        """
        n = min(self._count, self.window)
        if n == 0:
            return _normalise(ctx)

        # Gather entries in chronological order (oldest → newest)
        indices = [(self._ptr - n + i) % self.window for i in range(n)]
        hist    = self._buf[indices]          # (n, dim)

        # Temporal decay: most recent entry has weight 1.0
        pos_w = self.decay ** np.arange(n - 1, -1, -1, dtype=np.float32)

        # Cosine attention: score_i = cos(ctx, hist_i) × pos_w_i
        ctx_n      = _normalise(ctx)
        attn_raw   = (hist @ ctx_n) * pos_w  # (n,)

        # Softmax
        attn_raw  -= attn_raw.max()
        attn_w     = np.exp(attn_raw); attn_w /= (attn_w.sum() + 1e-9)

        # History summary: attention-weighted sum of stored embeddings
        hist_ctx = (attn_w[:, None] * hist).sum(axis=0)   # (dim,)

        # Blend current context with history summary
        blended  = self.ctx_alpha * ctx + (1.0 - self.ctx_alpha) * hist_ctx
        return _normalise(blended)

    def top_similar(self, ctx: np.ndarray, k: int = 4) -> np.ndarray:
        """Return the k history embeddings most similar to ctx (for debug)."""
        n = min(self._count, self.window)
        if n == 0:
            return np.zeros((0, self.dim), dtype=np.float32)
        indices = [(self._ptr - n + i) % self.window for i in range(n)]
        hist    = self._buf[indices]
        sims    = hist @ _normalise(ctx)
        top_k   = np.argsort(sims)[::-1][:k]
        return hist[top_k]

    def reset(self) -> None:
        """Clear the buffer (call at generation start)."""
        self._buf[:] = 0.0
        self._ptr    = 0
        self._count  = 0


class ContextAnchor:
    """Semantic drift detector and corrector.

    Maintains the prompt embedding as a fixed "topic anchor".
    At each generation step:
      1. Measures how far the current context has drifted from the anchor
      2. If drift exceeds the threshold, applies a correction that pulls
         the context back toward the anchor topic

    This is IECNN's inference-time prompt-following mechanism — no
    fine-tuning or instruction-tuning data required.

    Parameters
    ----------
    anchor              : prompt embedding (will be unit-normalised)
    drift_threshold     : minimum cosine similarity before correction
                          fires (default 0.20, range 0.10–0.40)
    correction_strength : how hard to pull back (default 0.15)
    """

    def __init__(self, anchor: np.ndarray,
                 drift_threshold: float = 0.20,
                 correction_strength: float = 0.15):
        self.anchor     = _normalise(anchor.astype(np.float32))
        self.threshold  = float(drift_threshold)
        self.strength   = float(correction_strength)

    def correct(self, ctx: np.ndarray) -> np.ndarray:
        """Apply correction if the context has drifted from the anchor.

        Returns the (possibly corrected) unit-norm context vector.
        """
        ctx_n   = _normalise(ctx.astype(np.float32))
        cosine  = float(ctx_n @ self.anchor[:len(ctx_n)])

        if cosine < self.threshold:
            # Drift detected: interpolate toward anchor
            pull   = self.strength * (1.0 - cosine / max(self.threshold, 1e-6))
            pull   = float(np.clip(pull, 0.0, 0.40))
            anchor = self.anchor[:len(ctx)]   # match length
            fixed  = ctx + pull * anchor
            return _normalise(fixed)

        return ctx_n

    def drift(self, ctx: np.ndarray) -> float:
        """Return 1 − cosine(ctx, anchor); higher = more drift."""
        ctx_n  = _normalise(ctx.astype(np.float32))
        cosine = float(ctx_n @ self.anchor[:len(ctx_n)])
        return float(1.0 - cosine)


# ── Internal helpers ──────────────────────────────────────────────────

def _normalise(v: np.ndarray) -> np.ndarray:
    n = np.linalg.norm(v)
    return v / n if n > 1e-9 else v
