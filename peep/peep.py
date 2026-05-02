"""
Peep Mechanism — execution/selector layer of IECNN.

The Peep Mechanism sits above the 128 Neural Dots and acts as the
execution layer: it reads what every dot is thinking and selects the
one whose learned specialisation best matches the current context.

Role separation
---------------
  Dots        — understanding.  Each dot learned a directional speciality
                from the training corpus via winner-take-all causal training.
  Peep        — execution.  It does not understand language; it knows which
                dot to listen to.  It routes the current context to the most
                qualified dot and reports that dot's prediction as output.

Calibration (training time)
---------------------------
After each causal training batch the winner-take-all result already tells
us which dot was closest to the target token at each position.  PeepMechanism
receives those winners via observe_batch() and updates the winning dot's
specialisation vector via a slow EMA toward the mean context direction that
triggered those wins.

Selection (generation time)
---------------------------
Given the current context vector and all 128 dot predictions, Peep computes
  score[d] = cos(ctx, specialisation[d]) × log1p(hit_count[d])
and returns the argmax dot.  The log-count factor gives well-calibrated dots
an edge over freshly-seeded ones in tie situations.

Persistence
-----------
Saved/loaded at <persistence_path>.peep.pkl alongside the main brain.
"""

import numpy as np
import pickle
from typing import Optional


class PeepMechanism:
    """
    Selector / arbitration layer over the Neural Dot ensemble.

    Parameters
    ----------
    n_dots : int
        Number of Neural Dots (usually 128).
    dim : int
        Feature/working dimension (usually 256).
    specialisation_lr : float
        EMA learning rate when updating specialisation vectors (default 0.04).
        Kept small for stability: fast-moving specialisations collapse.
    """

    def __init__(self, n_dots: int, dim: int,
                 specialisation_lr: float = 0.04):
        self.n_dots            = n_dots
        self.dim               = dim
        self.specialisation_lr = specialisation_lr

        self.specialisations   = np.zeros((n_dots, dim), dtype=np.float32)
        self.hit_counts        = np.zeros(n_dots, dtype=np.int64)
        self.calibrated        = False

    def observe_batch(self,
                      ctxs:      np.ndarray,
                      best_dots: np.ndarray) -> None:
        """
        Update dot specialisations from one training batch.

        Called once per batch after the winner-take-all step in
        causal_train_pass.  For each dot, averages the contexts where it
        was the winner and EMA-updates its specialisation vector.

        Parameters
        ----------
        ctxs : (n, dim) float32
            Normalised context vectors for each position in the batch.
        best_dots : (n,) int
            Index of the winning dot per position (argmax causal cosine).
        """
        bd  = best_dots.astype(np.int32)
        alpha = self.specialisation_lr

        for d in range(self.n_dots):
            mask   = (bd == d)
            n_wins = int(mask.sum())
            if n_wins == 0:
                continue

            mean_ctx = ctxs[mask].mean(axis=0).astype(np.float32)
            n = float(np.linalg.norm(mean_ctx))
            if n < 1e-9:
                continue
            mean_ctx /= n

            if self.hit_counts[d] == 0:
                self.specialisations[d] = mean_ctx
            else:
                self.specialisations[d] = (
                    (1.0 - alpha) * self.specialisations[d]
                    + alpha        * mean_ctx
                )
                sn = float(np.linalg.norm(self.specialisations[d]))
                if sn > 1e-9:
                    self.specialisations[d] /= sn

            self.hit_counts[d] += n_wins

        self.calibrated = True

        # ── Diversity push: subtract global centroid × 0.45, renorm ──────────
        # Prevents all specialisations from converging toward the corpus-mean
        # direction after many batches.  Each dot's unique direction is amplified
        # by removing the shared "average Wikipedia direction".
        centroid  = self.specialisations.mean(axis=0)               # (dim,)
        cn        = float(np.linalg.norm(centroid))
        if cn > 1e-9:
            self.specialisations -= 0.45 * centroid[None, :]
            sn = np.linalg.norm(self.specialisations, axis=1, keepdims=True)
            self.specialisations /= sn.clip(1e-9)

        # ── Pairwise repulsion: push highly similar specialisations apart ─────
        # Vectorised: for each dot, subtract the weighted sum of directions
        # whose cosine similarity exceeds the repulsion threshold.
        # Runs in O(n_dots²) ≈ 16k ops — negligible for n_dots=128.
        REP_THRESH = 0.80
        REP_LR     = 0.025
        sims = self.specialisations @ self.specialisations.T         # (n, n)
        np.fill_diagonal(sims, 0.0)
        too_similar  = np.clip(sims - REP_THRESH, 0.0, None)        # (n, n) ≥ 0
        if too_similar.sum() > 0:
            repulsion    = too_similar @ self.specialisations         # (n, dim)
            pair_counts  = (too_similar > 0).sum(axis=1, keepdims=True).clip(1)
            self.specialisations -= REP_LR * repulsion / pair_counts
            sn = np.linalg.norm(self.specialisations, axis=1, keepdims=True)
            self.specialisations /= sn.clip(1e-9)

    def diversity_score(self) -> float:
        """Mean pairwise (1 − cosine) across all dot specialisations.

        Range: 0 (all identical) → 1 (all orthogonal).
        A score > 0.35 indicates healthy specialisation diversity.
        """
        sp   = self.specialisations                                  # (n, dim)
        sims = sp @ sp.T                                             # (n, n)
        mask = ~np.eye(self.n_dots, dtype=bool)
        return float(1.0 - sims[mask].mean())

    def select(self,
               ctx_eff:   np.ndarray,
               raw_preds: np.ndarray) -> int:
        """
        Return the index of the best dot for the current context.

        Score = cos(ctx_eff, specialisation[d]) × log1p(hit_count[d])

        Falls back to highest prediction magnitude when not yet calibrated.
        """
        if not self.calibrated or self.hit_counts.sum() == 0:
            return int(np.argmax(np.linalg.norm(raw_preds, axis=1)))

        # Pure cosine similarity — no hit-count weighting.
        # High-hit-count dots averaged over too many diverse contexts and
        # converge toward the corpus mean, making them generic.
        # Weighting by log(hits) amplifies the wrong (most-general) dots.
        ctx_n    = ctx_eff / (float(np.linalg.norm(ctx_eff)) + 1e-9)
        scores   = self.specialisations @ ctx_n          # (n_dots,) cosine
        return int(np.argmax(scores))

    def top_k(self,
              ctx_eff:   np.ndarray,
              raw_preds: np.ndarray,
              k: int = 3) -> np.ndarray:
        """Return top-k dot indices sorted by Peep score (best first)."""
        if not self.calibrated or self.hit_counts.sum() == 0:
            mags = np.linalg.norm(raw_preds, axis=1)
            return np.argsort(mags)[::-1][:k]

        ctx_n  = ctx_eff / (float(np.linalg.norm(ctx_eff)) + 1e-9)
        scores = self.specialisations @ ctx_n            # pure cosine
        return np.argsort(scores)[::-1][:k]

    def stats(self) -> dict:
        active = int((self.hit_counts > 0).sum())
        total  = int(self.hit_counts.sum())
        div    = self.diversity_score() if self.calibrated else 0.0
        return {
            "calibrated":   self.calibrated,
            "active_dots":  active,
            "total_hits":   total,
            "max_hits":     int(self.hit_counts.max()) if active else 0,
            "top5_dots":    list(np.argsort(self.hit_counts)[::-1][:5].tolist()),
            "hit_counts":   self.hit_counts.tolist(),
            "diversity":    round(div, 4),
        }

    def save(self, path: str) -> None:
        data = {
            "n_dots":            self.n_dots,
            "dim":               self.dim,
            "specialisation_lr": self.specialisation_lr,
            "specialisations":   self.specialisations,
            "hit_counts":        self.hit_counts,
            "calibrated":        self.calibrated,
        }
        with open(path, "wb") as fh:
            pickle.dump(data, fh, protocol=4)

    @classmethod
    def load(cls, path: str) -> "PeepMechanism":
        with open(path, "rb") as fh:
            data = pickle.load(fh)
        p = cls(data["n_dots"], data["dim"],
                specialisation_lr=data.get("specialisation_lr", 0.04))
        p.specialisations = data["specialisations"].astype(np.float32)
        p.hit_counts      = data["hit_counts"].astype(np.int64)
        p.calibrated      = data["calibrated"]
        return p
