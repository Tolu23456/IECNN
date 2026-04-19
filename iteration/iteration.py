"""
Iteration Controller — manages the IECNN feedback loop.

Stopping conditions:
  1. Iteration Budget       — hard cap on rounds (T_max)
  2. Convergence Dominance  — one cluster holds ≥ delta of total weight (F9)
  3. No Utility Gain (EUG)  — F16: expected next-round gain ≤ eug_threshold
  4. Temporal Stability     — centroid barely moved (F12 > 0.99)
  5. Score Decline          — top score drops for `decline_patience` rounds

Additional safeguards:
  - Rollback: if a round produces worse clusters than the previous, revert

Adaptive learning rate (Formula 14):
  eta(t) = base_lr * (1 - 0.8 * dominance²)
  The system slows its bias updates when convergence is near.

EUG (Formula 16):
  U(t) = E[C_{t+1}(p)] - C_t(p)
  Measures whether the next iteration is expected to improve structure.
  Replaces the old cluster-ID-based novelty_gain check which was always
  zero because cluster IDs reset to 0,1,2… every round.
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import (
    dominance_score, adaptive_learning_rate, temporal_stability,
    emergent_utility_gradient,
)
from convergence.convergence import Cluster


class StopReason:
    BUDGET    = "iteration_budget"
    DOMINANCE = "convergence_dominance"
    EUG       = "no_utility_gain"
    STABILITY = "temporal_stability"
    DECLINE   = "score_decline"
    NONE      = "continuing"


class IterationController:
    """
    Manages the IECNN iteration loop.

    Usage:
      ctrl = IterationController(...)
      ctrl.reset()
      while True:
          ...run clusters...
          stop, reason = ctrl.should_stop(clusters)
          if stop: break
          ctrl.record_round(clusters, stats)
    """

    def __init__(self,
                 max_iterations:       int   = 12,
                 dominance_threshold:  float = 0.70,
                 novelty_threshold:    float = 0.05,
                 eug_threshold:        float = 0.001,
                 stability_threshold:  float = 0.99,
                 decline_patience:     int   = 3,
                 base_lr:              float = 0.10,
                 alpha:                float = 0.70):
        self.max_iterations    = max_iterations
        self.dom_thresh        = dominance_threshold
        self.novelty_thresh    = novelty_threshold   # kept for API compat
        self.eug_thresh        = eug_threshold
        self.stability_thresh  = stability_threshold
        self.decline_patience  = decline_patience
        self.base_lr           = base_lr
        self.alpha             = alpha

        self._round:            int = 0
        self._centroid_history: List[Optional[np.ndarray]] = []
        self._score_history:    List[float] = []
        self._stop_reason:      Optional[str] = None
        self._history:          List[Dict] = []

        # Best state for rollback
        self._best_round:       int = 0
        self._best_clusters:    Optional[List[Cluster]] = None
        self._best_centroid:    Optional[np.ndarray] = None

    def reset(self):
        self._round = 0
        self._centroid_history.clear()
        self._score_history.clear()
        self._stop_reason = None
        self._history.clear()
        self._best_round = 0
        self._best_clusters = None
        self._best_centroid = None

    @property
    def current_round(self) -> int: return self._round

    @property
    def stop_reason(self) -> Optional[str]: return self._stop_reason

    def current_dominance(self) -> float:
        """Dominance score from the most recently recorded round."""
        if not self._history: return 0.0
        return float(self._history[-1].get("dominance", 0.0))

    def current_lr(self) -> float:
        """Formula 14: adaptive learning rate based on current dominance."""
        return adaptive_learning_rate(self.base_lr, self.current_dominance())

    def current_eug(self) -> float:
        """Formula 16: Emergent Utility Gradient from current score history."""
        return emergent_utility_gradient(self._score_history)

    # ── Stopping conditions ──────────────────────────────────────────

    def should_stop(self, clusters: List[Cluster]) -> Tuple[bool, str]:
        """Check all stopping conditions; return (should_stop, reason)."""

        # 1. Budget
        if self._round >= self.max_iterations:
            self._stop_reason = StopReason.BUDGET
            return True, StopReason.BUDGET

        if not clusters:
            self._stop_reason = StopReason.BUDGET
            return True, StopReason.BUDGET

        # 2. Dominance
        scores = [c.score for c in clusters]
        dom = dominance_score(scores[0], scores)
        if dom >= self.dom_thresh:
            self._stop_reason = StopReason.DOMINANCE
            return True, StopReason.DOMINANCE

        # 3. EUG — Emergent Utility Gradient (F16) (requires ≥ 3 rounds)
        #    Replaces the old cluster-ID novelty_gain check, which always
        #    returned 0 because cluster IDs reset to 0,1,2… each round.
        if self._round >= 3:
            eug = emergent_utility_gradient(self._score_history)
            if eug <= self.eug_thresh:
                self._stop_reason = StopReason.EUG
                return True, StopReason.EUG

        # 4. Temporal stability (centroid barely moved)
        if self._round >= 2 and len(self._centroid_history) >= 1:
            curr_cent = clusters[0].centroid
            prev_cent = self._centroid_history[-1]
            if curr_cent is not None and prev_cent is not None:
                ts = temporal_stability(curr_cent, prev_cent, self.alpha)
                if ts >= self.stability_thresh:
                    self._stop_reason = StopReason.STABILITY
                    return True, StopReason.STABILITY

        # 5. Score decline for `decline_patience` consecutive rounds
        if self._round >= self.decline_patience + 1:
            recent = self._score_history[-self.decline_patience:]
            if all(recent[i] > recent[i+1] for i in range(len(recent)-1)):
                self._stop_reason = StopReason.DECLINE
                return True, StopReason.DECLINE

        return False, StopReason.NONE

    # ── Round recording ──────────────────────────────────────────────

    def record_round(self, clusters: List[Cluster], stats: Dict):
        if clusters and clusters[0].centroid is not None:
            self._centroid_history.append(clusters[0].centroid.copy())
            self._score_history.append(float(clusters[0].score))

            # Rollback tracking: keep the best round's clusters
            if (self._best_clusters is None or
                    float(clusters[0].score) >= float(self._best_clusters[0].score)):
                self._best_round    = self._round
                self._best_clusters = clusters[:]
                self._best_centroid = clusters[0].centroid.copy()
        else:
            self._centroid_history.append(None)
            self._score_history.append(0.0)

        scores = [c.score for c in clusters]
        total  = sum(scores)
        dom    = scores[0] / total if total > 1e-10 and scores else 0.0
        eug    = emergent_utility_gradient(self._score_history)
        lr     = self.current_lr()

        self._history.append({
            "round":     self._round,
            "clusters":  len(clusters),
            "top_score": float(clusters[0].score) if clusters else 0.0,
            "dominance": float(dom),
            "eug":       float(eug),
            "lr":        float(lr),
        })
        self._round += 1

    # ── State access ─────────────────────────────────────────────────

    def advance(self, clusters: List[Cluster], fallback: np.ndarray) -> np.ndarray:
        """Return the best available centroid for seeding next round."""
        if clusters and clusters[0].centroid is not None:
            return clusters[0].centroid.copy()
        if self._best_centroid is not None:
            return self._best_centroid.copy()
        return fallback

    def best_clusters(self) -> Optional[List[Cluster]]:
        """Return the clusters from the best-scoring round (for rollback)."""
        return self._best_clusters

    def summary(self) -> Dict:
        return {
            "rounds":       self._round,
            "stop_reason":  self._stop_reason,
            "best_round":   self._best_round,
            "history":      self._history,
            "final_lr":     self.current_lr(),
            "final_eug":    self.current_eug(),
        }
