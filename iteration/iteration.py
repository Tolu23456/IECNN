"""
Iteration Controller — manages the IECNN loop.

Three stopping conditions:
  1. Iteration Budget  — hard cap on rounds (prevents infinite loops)
  2. Convergence Dominance — one cluster holds >= delta of total weight
     (captures genuine agreement, not just lack of change)
  3. Low Novelty Gain — new iterations stop producing different candidates
     (signals exploration is exhausted; Formula 6)
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import novelty_gain, dominance_score
from convergence.convergence import Cluster


class StopReason:
    BUDGET    = "iteration_budget"
    DOMINANCE = "convergence_dominance"
    NOVELTY   = "low_novelty_gain"
    NONE      = "continuing"


class IterationController:
    def __init__(self, max_iterations: int = 10,
                 dominance_threshold: float = 0.70,
                 novelty_threshold: float = 0.05):
        self.max_iterations   = max_iterations
        self.dom_thresh       = dominance_threshold
        self.novelty_thresh   = novelty_threshold
        self._round           = 0
        self._cluster_history: List[List[int]] = []
        self._stop_reason: Optional[str] = None
        self._history: List[Dict] = []

    def reset(self):
        self._round = 0; self._cluster_history = []
        self._stop_reason = None; self._history = []

    @property
    def current_round(self) -> int: return self._round

    @property
    def stop_reason(self) -> Optional[str]: return self._stop_reason

    def should_stop(self, clusters: List[Cluster]) -> Tuple[bool, str]:
        if self._round >= self.max_iterations:
            self._stop_reason = StopReason.BUDGET
            return True, StopReason.BUDGET

        if clusters:
            scores = [c.score for c in clusters]
            dom = dominance_score(scores[0], scores)
            if dom >= self.dom_thresh:
                self._stop_reason = StopReason.DOMINANCE
                return True, StopReason.DOMINANCE

        if self._round >= 2 and clusters:
            cur  = set(c.cluster_id for c in clusters)
            prev = set(self._cluster_history[-1]) if self._cluster_history else set()
            ng   = novelty_gain(len(cur - prev), len(cur))
            if ng < self.novelty_thresh:
                self._stop_reason = StopReason.NOVELTY
                return True, StopReason.NOVELTY

        return False, StopReason.NONE

    def record_round(self, clusters: List[Cluster], stats: Dict):
        self._cluster_history.append([c.cluster_id for c in clusters])
        scores = [c.score for c in clusters]
        total  = sum(scores)
        dom    = scores[0] / total if total > 1e-10 and scores else 0.0
        cur    = set(c.cluster_id for c in clusters)
        prev   = set(self._cluster_history[-2]) if len(self._cluster_history) >= 2 else set()
        ng     = novelty_gain(len(cur - prev), len(cur))
        self._history.append({
            "round": self._round, "clusters": len(clusters),
            "top_score": float(clusters[0].score) if clusters else 0.0,
            "dominance": float(dom), "novelty_gain": float(ng),
        })
        self._round += 1

    def advance(self, clusters: List[Cluster], fallback: np.ndarray) -> np.ndarray:
        if clusters and clusters[0].centroid is not None:
            return clusters[0].centroid.copy()
        return fallback

    def summary(self) -> Dict:
        return {"rounds": self._round, "stop_reason": self._stop_reason,
                "history": self._history}
