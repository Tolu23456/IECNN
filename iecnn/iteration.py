import numpy as np
from typing import List, Optional, Dict, Tuple
from .formulas import novelty_gain
from .convergence import Cluster


class StopReason:
    BUDGET = "iteration_budget"
    DOMINANCE = "convergence_dominance"
    NOVELTY = "low_novelty_gain"
    NONE = "continuing"


class IterationController:
    """
    Controls the IECNN iterative loop.

    Stops when any of three conditions are met:

    Condition 1 — Iteration Budget
      Hard cap on total rounds. Prevents infinite loops.

    Condition 2 — Convergence Dominance
      One cluster holds >= delta fraction of total weight.
      Captures genuine agreement, not just lack of change.

    Condition 3 — Low Novelty Gain
      New iterations stop producing meaningfully different candidates.
      Everything new maps into existing clusters.
      Signals that exploration is exhausted.

    The system stops when: Budget OR Dominance OR LowNovelty.
    """

    def __init__(
        self,
        max_iterations: int = 10,
        dominance_threshold: float = 0.75,
        novelty_threshold: float = 0.05,
    ):
        self.max_iterations = max_iterations
        self.dominance_threshold = dominance_threshold
        self.novelty_threshold = novelty_threshold

        self._round = 0
        self._cluster_history: List[List[int]] = []
        self._stop_reason: Optional[str] = None
        self._history: List[Dict] = []

    def reset(self):
        self._round = 0
        self._cluster_history = []
        self._stop_reason = None
        self._history = []

    @property
    def current_round(self) -> int:
        return self._round

    @property
    def stop_reason(self) -> Optional[str]:
        return self._stop_reason

    def _check_budget(self) -> bool:
        """Condition 1: have we exceeded the iteration budget?"""
        return self._round >= self.max_iterations

    def _check_dominance(self, clusters: List[Cluster]) -> bool:
        """
        Condition 2: does one cluster hold >= delta of total weight?
        Formula 9: Dominance(k*) = C(k*) / sum_k C(k)
        """
        if not clusters:
            return False
        scores = [c.score for c in clusters]
        total = sum(scores)
        if total < 1e-10:
            return False
        top_ratio = scores[0] / total
        return top_ratio >= self.dominance_threshold

    def _check_novelty(self, clusters: List[Cluster]) -> bool:
        """
        Condition 3: is the novelty gain too low?
        Formula 6: NG(t) = |NewClusters(t)| / |TotalClusters(t)|

        A cluster is "new" if its id does not appear in previous rounds.
        """
        if self._round < 2:
            return False

        current_ids = set(c.cluster_id for c in clusters)
        prev_ids = set(self._cluster_history[-1]) if self._cluster_history else set()
        new_count = len(current_ids - prev_ids)
        total_count = len(current_ids)

        ng = novelty_gain(new_count, total_count)
        return ng < self.novelty_threshold

    def should_stop(self, clusters: List[Cluster]) -> Tuple[bool, str]:
        """
        Check all three stopping conditions.
        Returns (should_stop, reason).
        """
        if self._check_budget():
            self._stop_reason = StopReason.BUDGET
            return True, StopReason.BUDGET

        if self._check_dominance(clusters):
            self._stop_reason = StopReason.DOMINANCE
            return True, StopReason.DOMINANCE

        if self._check_novelty(clusters):
            self._stop_reason = StopReason.NOVELTY
            return True, StopReason.NOVELTY

        return False, StopReason.NONE

    def record_round(self, clusters: List[Cluster], stats: Dict):
        """Record the state of this round for novelty tracking."""
        cluster_ids = [c.cluster_id for c in clusters]
        self._cluster_history.append(cluster_ids)

        scores = [c.score for c in clusters]
        total_score = sum(scores)
        dom = scores[0] / total_score if total_score > 1e-10 and scores else 0.0

        current_ids = set(cluster_ids)
        prev_ids = set(self._cluster_history[-2]) if len(self._cluster_history) >= 2 else set()
        new_count = len(current_ids - prev_ids)
        ng = novelty_gain(new_count, len(current_ids))

        self._history.append({
            "round": self._round,
            "num_clusters": len(clusters),
            "top_score": float(clusters[0].score) if clusters else 0.0,
            "dominance": float(dom),
            "novelty_gain": float(ng),
            "pruning_stats": stats,
        })

        self._round += 1

    def advance(self, clusters: List[Cluster], surviving_centroid: np.ndarray) -> np.ndarray:
        """
        Prepare the merged output to feed into the next iteration.
        Returns the centroid of the top surviving cluster,
        which becomes the "refined input" for the next round.
        """
        if clusters and clusters[0].centroid is not None:
            return clusters[0].centroid.copy()
        return surviving_centroid

    @property
    def history(self) -> List[Dict]:
        return self._history

    def summary(self) -> Dict:
        return {
            "rounds_completed": self._round,
            "stop_reason": self._stop_reason,
            "max_iterations": self.max_iterations,
            "history": self._history,
        }
