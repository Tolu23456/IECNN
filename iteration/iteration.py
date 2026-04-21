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
    emergent_utility_gradient, global_energy, system_objective,
    stability_energy, exploration_pressure,
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
    Manages the IECNN iteration loop with an Energy-Based Objective.

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
        self._eug_history:      List[float] = []
        self._entropy_history:  List[float] = []
        self._stability_history: List[float] = []
        self._energy_history:   List[float] = []
        self._objective_history: List[float] = []
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
        self._eug_history.clear()
        self._entropy_history.clear()
        self._stability_history.clear()
        self._energy_history.clear()
        self._objective_history.clear()
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
        """Formula 14: adaptive learning rate based on current dominance and stability."""
        dom = self.current_dominance()
        stab = self._stability_history[-1] if self._stability_history else 0.5
        # FIXED F14: η(t) = η0 * (1 - 0.8*D(t)^2) * (1 + 0.2*S(t))
        return adaptive_learning_rate(self.base_lr, dom) * (1.0 + 0.2 * stab)

    def current_eug(self) -> float:
        """Formula 16: Emergent Utility Gradient (improvement + entropy + stability)."""
        return emergent_utility_gradient(
            self._score_history, self._entropy_history, self._stability_history
        )

    def current_objective(self) -> float:
        """F22: Master System Objective J(t)."""
        if not self._objective_history: return 0.0
        return self._objective_history[-1]

    def current_energy(self) -> float:
        """F21: Global Energy E(t)."""
        if not self._energy_history: return 1.0
        return self._energy_history[-1]

    def current_stability(self) -> float:
        """F25: Stability Energy S(t)."""
        if not self._stability_history: return 0.0
        return self._stability_history[-1]

    def exploration_pressure(self) -> float:
        """F26: Exploration Pressure X(t)."""
        return exploration_pressure(self.current_stability(), self.current_dominance())

    def utility_acceleration(self) -> float:
        """ΔU = U(t) - U(t-1): rate of change of EUG across the last two rounds."""
        if len(self._eug_history) < 2:
            return 0.0
        return float(self._eug_history[-1] - self._eug_history[-2])

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
        if self._round >= 3:
            eug = self.current_eug()
            if eug <= self.eug_thresh:
                self._stop_reason = StopReason.EUG
                return True, StopReason.EUG

        # 4. Temporal stability (centroid barely moved)
        if self._round >= 2 and len(self._centroid_history) >= 1:
            ts = self.current_stability()
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
        ent = stats.get("entropy", 0.5)
        self._entropy_history.append(float(ent))

        if clusters and clusters[0].centroid is not None:
            curr_cent = clusters[0].centroid
            self._centroid_history.append(curr_cent.copy())
            self._score_history.append(float(clusters[0].score))

            # Temporal instability
            if len(self._centroid_history) >= 2:
                prev_cent = self._centroid_history[-2]
                instab = 1.0 - temporal_stability(curr_cent, prev_cent, self.alpha)
            else:
                instab = 0.5

            # Stability Energy (F25)
            stab = stability_energy(ent, instab)
            self._stability_history.append(stab)

            # Global Energy (F21)
            scores = [c.score for c in clusters]
            dom = dominance_score(scores[0], scores)
            energy = global_energy(ent, dom, instab)
            self._energy_history.append(energy)

            # EUG (F16)
            eug = self.current_eug()
            self._eug_history.append(eug)

            # Master Objective J(t) (F22)
            obj = system_objective(float(clusters[0].score), eug, energy)
            self._objective_history.append(obj)

            # Rollback tracking: keep the best round's objective
            if (self._best_clusters is None or
                    obj >= self._objective_history[self._best_round]):
                self._best_round    = self._round
                self._best_clusters = clusters[:]
                self._best_centroid = curr_cent.copy()
        else:
            self._centroid_history.append(None)
            self._score_history.append(0.0)
            self._stability_history.append(0.0)
            self._energy_history.append(1.0)
            self._eug_history.append(0.0)
            self._objective_history.append(-1.0)

        self._history.append({
            "round":     self._round,
            "clusters":  len(clusters),
            "top_score": float(clusters[0].score) if clusters else 0.0,
            "dominance": float(dom) if clusters else 0.0,
            "energy":    float(self.current_energy()),
            "objective": float(self.current_objective()),
            "stability": float(self.current_stability()),
            "eug":       float(self.current_eug()),
            "delta_u":   float(self.utility_acceleration()),
            "lr":        float(self.current_lr()),
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
