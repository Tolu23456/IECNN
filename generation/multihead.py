"""
IECNN Multi-Head Convergence Voting
=====================================
Divides the Neural Dot pool into H parallel "heads" (analogous to
Transformer multi-head attention heads).  Each head computes an
independent score distribution over the vocabulary; the final
distribution is a confidence-weighted average across all heads.

Design rationale
----------------
In a standard Transformer, multi-head attention learns to attend to
different positional and semantic aspects of the input simultaneously.
In IECNN, dots are analogously specialised (via Peep + dot evolution),
so grouping them ensures that:
  • No single dot monopolises the vote
  • Heads that specialise in different semantic regions contribute
    proportionally to their local confidence
  • Low-confidence heads are down-weighted automatically via softmax

This gives more coherent outputs than top-3 single-head selection because
8 independent predictions are combined rather than 3.

Head assignment
---------------
Round-robin: dot i → head (i % n_heads).
This distributes specialisations evenly because Peep calibration also
uses round-robin when dots are sorted by hit count.
"""

from __future__ import annotations
import numpy as np
from typing import List, Optional, Tuple


class MultiHeadConvergence:
    """8-head convergence voting over the full Neural Dot pool.

    Parameters
    ----------
    n_heads   : number of parallel heads (default 8)
    embed_dim : vocabulary embedding dimension, EMBED_DIM=224 (default 224)
    """

    def __init__(self, n_heads: int = 8, embed_dim: int = 224):
        self.n_heads   = int(n_heads)
        self.embed_dim = int(embed_dim)
        self._assign: Optional[List[List[int]]] = None

    # ── Setup ───────────────────────────────────────────────────────

    def build(self, n_dots: int) -> None:
        """Assign dots to heads via round-robin."""
        H = self.n_heads
        self._assign = [[] for _ in range(H)]
        for i in range(n_dots):
            self._assign[i % H].append(i)

    # ── Core forward pass ───────────────────────────────────────────

    def forward(self,
                raw_preds:    np.ndarray,
                word_vecs_n:  np.ndarray,
                gram_w:       np.ndarray,
                block_mask:   Optional[np.ndarray] = None,
                ) -> Tuple[np.ndarray, float]:
        """Compute multi-head score distribution.

        Parameters
        ----------
        raw_preds  : (n_dots, feature_dim) — W @ ctx for each dot
        word_vecs_n: (V, embed_dim)        — unit-norm word embeddings
        gram_w     : (V,)                  — grammar weight bias
        block_mask : (V,) bool             — tokens to hard-block

        Returns
        -------
        combined_scores : (V,) confidence-weighted sum across heads
        confidence      : scalar — weighted average of head max-scores
        """
        if self._assign is None:
            self.build(raw_preds.shape[0])

        H   = self.n_heads
        V   = word_vecs_n.shape[0]
        ed  = self.embed_dim

        head_scores = np.zeros((H, V), dtype=np.float32)
        head_maxes  = np.zeros(H,      dtype=np.float32)

        for h, idxs in enumerate(self._assign):
            if not idxs:
                continue
            hp  = raw_preds[idxs]           # (nh, feature_dim)
            hm  = hp.mean(axis=0)           # (feature_dim,) head mean
            res = hp - hm                   # (nh, feature_dim) residuals

            res_e = res[:, :ed]             # (nh, embed_dim) embedding part
            rn    = np.linalg.norm(res_e, axis=1, keepdims=True).clip(1e-9)
            res_n = res_e / rn              # unit residuals

            # Each dot in this head votes for vocab; average across head
            hs  = (res_n @ word_vecs_n.T).mean(axis=0)  # (V,)
            hs  = hs + gram_w

            # Per-head z-score normalisation: prevents dominant heads whose raw
            # cosines happen to be larger in magnitude from drowning out quieter
            # but semantically focused heads.  Normalise before block-masking so
            # blocked tokens (-2.0) don't skew the mean/std estimate.
            _hs_std = float(hs.std())
            if _hs_std > 1e-6:
                hs = (hs - float(hs.mean())) / _hs_std

            if block_mask is not None:
                hs[block_mask] = -2.0

            head_scores[h] = hs
            head_maxes[h]  = float(hs.max())

        # Confidence-weighted combination via softmax over head max-scores
        shifted = head_maxes - head_maxes.max()
        wt      = np.exp(shifted); wt /= wt.sum()
        combined = (wt[:, None] * head_scores).sum(axis=0)   # (V,)

        # Overall confidence: softmax-weighted mean of head max-scores
        confidence = float(np.clip((wt * head_maxes).sum(), 0.0, 1.0))

        return combined, confidence

    # ── Peep-augmented forward ──────────────────────────────────────

    def peep_forward(self,
                     raw_preds:    np.ndarray,
                     word_vecs_n:  np.ndarray,
                     gram_w:       np.ndarray,
                     peep_top_k:   np.ndarray,
                     peep_weights: np.ndarray,
                     mean_pred:    np.ndarray,
                     block_mask:   Optional[np.ndarray] = None,
                     ) -> Tuple[np.ndarray, float]:
        """Peep-guided + multi-head fusion.

        The Peep mechanism selects the top-K most specialised dots.
        Their weighted residual is computed (as in the standard Peep path)
        to form a "primary" score distribution.

        The multi-head vote forms a "secondary" distribution.
        The two are linearly blended (70/30) so the Peep signal leads
        but is stabilised by the broader head consensus.

        Parameters
        ----------
        peep_top_k   : (K,) indices of Peep-selected dots
        peep_weights : (K,) cosine-based weights for those dots
        mean_pred    : (feature_dim,) global mean of raw_preds
        """
        ed = self.embed_dim

        # ── Peep signal ────────────────────────────────────────────
        residuals_top = (raw_preds[peep_top_k] - mean_pred[None, :])  # (K, fd)
        blended       = (peep_weights[:, None] * residuals_top).sum(axis=0)  # (fd,)
        res_e         = blended[:ed]
        res_n_val     = float(np.linalg.norm(res_e))
        if res_n_val < 1e-9:
            res_e = np.zeros(ed, dtype=np.float32)
        else:
            res_e = res_e / res_n_val
        peep_scores = (word_vecs_n @ res_e) + gram_w  # (V,)

        # Z-score normalise Peep scores so their magnitude is compatible with
        # the z-normalised MHC head scores (added in forward()/_forward_sub()).
        # Without this, the 70/30 blend is numerically dominated by whichever
        # path happens to have larger raw cosines.
        _ps_std = float(peep_scores.std())
        if _ps_std > 1e-6:
            peep_scores = (peep_scores - float(peep_scores.mean())) / _ps_std

        if block_mask is not None:
            peep_scores[block_mask] = -2.0

        # ── Multi-head signal ──────────────────────────────────────
        mhc_scores, mhc_conf = self.forward(
            raw_preds, word_vecs_n, gram_w, block_mask
        )

        # ── Blend (70% Peep, 30% heads) ───────────────────────────
        combined   = 0.70 * peep_scores + 0.30 * mhc_scores
        peep_conf  = float(np.clip(peep_scores.max(), 0.0, 1.0))
        confidence = 0.70 * peep_conf + 0.30 * mhc_conf

        return combined, confidence

    # ── Internal helper ─────────────────────────────────────────────

    def _forward_sub(self,
                     raw_preds_sub: np.ndarray,
                     word_vecs_n:   np.ndarray,
                     gram_w:        np.ndarray,
                     block_mask:    Optional[np.ndarray] = None,
                     ) -> Tuple[np.ndarray, float]:
        """Like forward() but builds head-assignments freshly for the given
        sub-array so indices never go out of bounds.

        Used by contrastive_forward() which passes proper sub-arrays to avoid
        the index-collision bug that arises when self._assign (built for 128
        dots) is used to index into a 64-dot sub-array.
        """
        n  = raw_preds_sub.shape[0]
        H  = self.n_heads
        ed = self.embed_dim
        V  = word_vecs_n.shape[0]

        # Build local (per-call) head assignment — O(n), small
        local_assign: List[List[int]] = [[] for _ in range(H)]
        for i in range(n):
            local_assign[i % H].append(i)

        head_scores = np.zeros((H, V), dtype=np.float32)
        head_maxes  = np.zeros(H,      dtype=np.float32)

        for h, idxs in enumerate(local_assign):
            if not idxs:
                continue
            hp    = raw_preds_sub[idxs]
            hm    = hp.mean(axis=0)
            res   = hp - hm
            res_e = res[:, :ed]
            rn    = np.linalg.norm(res_e, axis=1, keepdims=True).clip(1e-9)
            res_n = res_e / rn
            hs    = (res_n @ word_vecs_n.T).mean(axis=0) + gram_w
            # Z-score normalisation (mirrors forward() to ensure consistent
            # score magnitudes between expert and amateur sub-arrays in
            # contrastive_forward; prevents alpha-weighted subtraction from
            # being dominated by scale differences between the two groups).
            _hs_std = float(hs.std())
            if _hs_std > 1e-6:
                hs = (hs - float(hs.mean())) / _hs_std
            if block_mask is not None:
                hs[block_mask] = -2.0
            head_scores[h] = hs
            head_maxes[h]  = float(hs.max())

        shifted  = head_maxes - head_maxes.max()
        wt       = np.exp(shifted); wt /= wt.sum()
        combined = (wt[:, None] * head_scores).sum(axis=0)
        conf     = float(np.clip((wt * head_maxes).sum(), 0.0, 1.0))
        return combined, conf

    # ── Cross-head agreement bonus ──────────────────────────────────

    def agreement_bonus(self,
                        raw_preds:   np.ndarray,
                        word_vecs_n: np.ndarray,
                        top_k:       int   = 10,
                        strength:    float = 0.10,
                        ) -> np.ndarray:
        """Per-word cross-head agreement bonus.

        Counts how many of the H independent heads place each word in their
        local top-K and returns a proportional bonus.  Words that appear in
        the top-K of many heads receive the full bonus; words ranked highly
        by only one head (possibly noise) receive little or none.

        This is IECNN's native ensemble-agreement signal — analogous to a
        Transformer's multi-head attention where all heads ``attend'' to the
        same position, amplifying reliable signal and suppressing noise.

        Algorithm (vectorised per head):
            for each head h:
                mean_pred_h = raw_preds[head_dots].mean(axis=0)[:E]
                sims_h      = word_vecs_n @ unit(mean_pred_h)   → (V,)
                top-K words in sims_h → counts[those words] += 1
            bonus = counts / n_heads × strength

        Cost: O(H × V × E) ≈ 8 × 5000 × 224 = 9 M FP32 ops — fast.

        Parameters
        ----------
        raw_preds   : (n_dots, feature_dim) — W @ ctx_eff for each dot
        word_vecs_n : (V, embed_dim)        — unit-norm word embeddings
        top_k       : words per head counted as "votes" (default 10)
        strength    : maximum additive bonus (default 0.10)

        Returns
        -------
        bonus : (V,) float32 — cross-head agreement bonus
        """
        if self._assign is None:
            self.build(raw_preds.shape[0])

        V      = word_vecs_n.shape[0]
        ed     = self.embed_dim
        counts = np.zeros(V, dtype=np.float32)
        k      = min(top_k, V)

        for h_idxs in self._assign:
            if not h_idxs:
                continue
            h_mean = raw_preds[h_idxs].mean(axis=0)[:ed]   # (ed,)
            h_norm = float(np.linalg.norm(h_mean))
            if h_norm < 1e-9:
                continue
            h_sims   = word_vecs_n @ (h_mean / h_norm)     # (V,)
            top_idx  = np.argpartition(h_sims, -k)[-k:]    # (k,) fast top-K
            counts[top_idx] += 1.0

        # Normalise: full agreement (count == n_heads) → bonus = strength
        bonus = counts * (strength / max(self.n_heads, 1))
        return bonus.astype(np.float32)

    # ── Head-spread penalty ─────────────────────────────────────────

    def head_spread_penalty(self,
                            raw_preds:   np.ndarray,
                            word_vecs_n: np.ndarray,
                            strength:    float = 0.05,
                            ) -> np.ndarray:
        """Per-word penalty proportional to cross-head score spread.

        For each word, computes the range (max − min) of per-head cosine
        scores across all H heads and normalises it to [0, 1].  Words
        where the heads strongly *disagree* (high spread) receive a
        larger penalty, discouraging selection of unreliably-scored tokens.

        Complements the cross-head *agreement bonus* (which rewards tokens
        ranked in the top-K by many heads) by also penalising tokens where
        the numerical spread of head scores is large even if no head puts
        the token at the very top.

        Cost: O(H × V × E) — same order as agreement_bonus.

        Parameters
        ----------
        raw_preds   : (n_dots, feature_dim)
        word_vecs_n : (V, embed_dim)
        strength    : maximum penalty (default 0.05)

        Returns
        -------
        penalty : (V,) float32 — penalty to subtract from scores
        """
        if self._assign is None:
            self.build(raw_preds.shape[0])

        V    = word_vecs_n.shape[0]
        ed   = self.embed_dim
        sims = []   # list of (V,) per-head sim arrays

        for h_idxs in self._assign:
            if not h_idxs:
                continue
            hm   = raw_preds[h_idxs].mean(axis=0)[:ed]
            hn   = float(np.linalg.norm(hm))
            if hn < 1e-9:
                continue
            sims.append(word_vecs_n @ (hm / hn))   # (V,)

        if len(sims) < 2:
            return np.zeros(V, dtype=np.float32)

        stacked = np.stack(sims, axis=0)            # (n_active_heads, V)
        spread  = stacked.max(axis=0) - stacked.min(axis=0)  # (V,) range
        mx      = float(spread.max())
        if mx < 1e-9:
            return np.zeros(V, dtype=np.float32)

        pen = (spread / mx) * strength
        return pen.astype(np.float32)

    # ── Contrastive voting ──────────────────────────────────────────

    def contrastive_forward(self,
                            raw_preds:   np.ndarray,
                            ctx_eff:     np.ndarray,
                            word_vecs_n: np.ndarray,
                            gram_w:      np.ndarray,
                            block_mask:  Optional[np.ndarray] = None,
                            alpha:       float = 0.12,
                            ) -> Tuple[np.ndarray, float]:
        """IECNN Contrastive Voting  (cf. Li et al. 2022, Contrastive Decoding).

        Split the dot pool into two groups by alignment with the current
        context vector ``ctx_eff``:

          Expert  — top-½ dots (most context-aligned):  context-specific signal
          Amateur — bot-½ dots (least context-aligned): generic language signal

        Contrastive score = (1+α) × expert_score − α × amateur_score

        Subtracting the generic signal amplifies the part of the distribution
        that is *specific* to the current context, producing more topically
        coherent token choices.

        Uses ``_forward_sub()`` internally so sub-array head-assignments are
        always built fresh — avoids the index-out-of-bounds that would occur
        if ``self._assign`` (built for the full pool) was used on a half-sized
        sub-array.

        Parameters
        ----------
        raw_preds  : (n_dots, feature_dim) — W @ ctx_eff for each dot
        ctx_eff    : (feature_dim,)        — enriched context vector
        word_vecs_n: (V, embed_dim)        — unit-norm word embeddings
        gram_w     : (V,)                  — grammar weight bias
        block_mask : (V,) bool             — tokens to hard-block
        alpha      : contrastive mixing coefficient (default 0.12)
        """
        n_dots = raw_preds.shape[0]
        if n_dots < 4:
            return self.forward(raw_preds, word_vecs_n, gram_w, block_mask)

        # Rank dots by cosine alignment with ctx_eff
        ctx_n     = ctx_eff / (float(np.linalg.norm(ctx_eff)) + 1e-9)
        ed        = self.embed_dim
        preds_e   = raw_preds[:, :ed]
        norms     = np.linalg.norm(preds_e, axis=1, keepdims=True).clip(1e-9)
        preds_n   = preds_e / norms
        alignment = preds_n @ ctx_n[:ed]

        order       = np.argsort(alignment)[::-1]    # most aligned first
        half        = max(n_dots // 2, 1)
        expert_idx  = order[:half]
        amateur_idx = order[half:]

        expert_scores,  expert_conf = self._forward_sub(
            raw_preds[expert_idx],  word_vecs_n, gram_w, block_mask)
        amateur_scores, _           = self._forward_sub(
            raw_preds[amateur_idx], word_vecs_n, gram_w, block_mask)

        combined = (1.0 + alpha) * expert_scores - alpha * amateur_scores

        # Final z-score normalisation of the combined contrastive score:
        # The subtraction can shift the distribution unpredictably; centering
        # and scaling to unit variance ensures downstream filters (top-K,
        # typical, nucleus) receive a stable, comparable score range each step.
        _c_valid = combined[combined > -1.0]   # exclude hard-blocked slots
        if len(_c_valid) > 1:
            _c_std = float(_c_valid.std())
            if _c_std > 1e-6:
                combined = (combined - float(_c_valid.mean())) / _c_std

        return combined, expert_conf
