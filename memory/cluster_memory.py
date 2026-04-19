"""
Cluster Memory — tracks cluster evolution across rounds and across calls.

Maintains a timeline of:
  - Cluster centroids at each round
  - Score trajectories of top clusters
  - Temporal stability between consecutive rounds
  - Cross-call context (persistent pattern library)

Used by:
  - IterationController: to compute temporal stability (Formula 12)
  - Pipeline: to seed the next call with known stable patterns
  - Evaluation: for historical analysis
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
from collections import deque

import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score


class RoundSnapshot:
    """Snapshot of the cluster state at one round."""
    def __init__(self, round_idx: int, centroids: List[np.ndarray],
                 scores: List[float], stop_reason: str = ""):
        self.round_idx   = round_idx
        self.centroids   = [c.copy() for c in centroids]
        self.scores      = list(scores)
        self.stop_reason = stop_reason

    @property
    def top_centroid(self) -> Optional[np.ndarray]:
        if self.centroids:
            return self.centroids[0]
        return None

    @property
    def top_score(self) -> float:
        return self.scores[0] if self.scores else 0.0

    @property
    def num_clusters(self) -> int:
        return len(self.centroids)


class ClusterMemory:
    """
    Records the cluster timeline across rounds (within one call) and across
    multiple calls (persistent pattern library for long-running sessions).
    """

    def __init__(self, feature_dim: int = 128, max_patterns: int = 64,
                 window_rounds: int = 10):
        self.feature_dim   = feature_dim
        self.max_patterns  = max_patterns
        self.window_rounds = window_rounds

        # Per-call timeline
        self._round_snapshots: List[RoundSnapshot] = []

        # Cross-call pattern library: stable patterns that recur
        self._pattern_library: List[Tuple[np.ndarray, float]] = []  # (centroid, weight)
        self._pattern_counts: List[int] = []

    # ── Within-call recording ────────────────────────────────────────

    def record_round(self, round_idx: int, centroids: List[np.ndarray],
                     scores: List[float], stop_reason: str = ""):
        snap = RoundSnapshot(round_idx, centroids, scores, stop_reason)
        self._round_snapshots.append(snap)
        # Trim to window
        if len(self._round_snapshots) > self.window_rounds:
            self._round_snapshots.pop(0)

    def temporal_stability(self, alpha: float = 0.7) -> float:
        """
        Formula 12: S(centroid_t, centroid_{t-1}).
        Returns 0.0 if fewer than 2 rounds have been recorded.
        """
        snaps = self._round_snapshots
        if len(snaps) < 2:
            return 0.0
        c_curr = snaps[-1].top_centroid
        c_prev = snaps[-2].top_centroid
        if c_curr is None or c_prev is None:
            return 0.0
        return similarity_score(c_curr, c_prev, alpha)

    def score_trajectory(self) -> List[float]:
        """Top cluster scores across recorded rounds."""
        return [s.top_score for s in self._round_snapshots]

    def is_score_declining(self) -> bool:
        """True if top score has consistently declined over last 3 rounds."""
        traj = self.score_trajectory()
        if len(traj) < 3:
            return False
        return traj[-1] < traj[-2] < traj[-3]

    def current_centroid(self) -> Optional[np.ndarray]:
        """Centroid of the top cluster from the most recent round."""
        if not self._round_snapshots:
            return None
        return self._round_snapshots[-1].top_centroid

    def previous_centroid(self) -> Optional[np.ndarray]:
        """Centroid from the round before the most recent."""
        if len(self._round_snapshots) < 2:
            return None
        return self._round_snapshots[-2].top_centroid

    def reset_call(self):
        """Clear the per-call timeline (call after each `run()`)."""
        self._round_snapshots.clear()

    # ── Cross-call pattern library ───────────────────────────────────

    def commit_pattern(self, centroid: np.ndarray, score: float, alpha: float = 0.7):
        """
        After a successful convergence, store the winning centroid as a pattern.
        If a similar pattern already exists, update its weight.
        """
        n = np.linalg.norm(centroid)
        if n < 1e-10:
            return
        c_norm = centroid / n

        # Check if similar pattern exists
        for i, (pat, w) in enumerate(self._pattern_library):
            if similarity_score(c_norm, pat, alpha) > 0.85:
                # Update existing pattern (exponential moving average)
                self._pattern_library[i] = (
                    (0.8 * pat + 0.2 * c_norm),
                    w * 0.9 + score * 0.1,
                )
                self._pattern_counts[i] += 1
                return

        # Add new pattern
        self._pattern_library.append((c_norm, score))
        self._pattern_counts.append(1)

        # Trim to max_patterns by keeping highest-weighted
        if len(self._pattern_library) > self.max_patterns:
            order = np.argsort([w for _, w in self._pattern_library])[::-1]
            self._pattern_library = [self._pattern_library[i] for i in order[:self.max_patterns]]
            self._pattern_counts  = [self._pattern_counts[i] for i in order[:self.max_patterns]]

    def closest_pattern(self, query: np.ndarray, alpha: float = 0.7) -> Optional[np.ndarray]:
        """Return the closest known pattern to `query`, or None."""
        if not self._pattern_library:
            return None
        q = query / (np.linalg.norm(query) + 1e-10)
        best_sim, best_pat = -1.0, None
        for pat, _ in self._pattern_library:
            s = similarity_score(q, pat, alpha)
            if s > best_sim:
                best_sim, best_pat = s, pat
        return best_pat if best_sim > 0.3 else None

    def pattern_library_size(self) -> int:
        return len(self._pattern_library)

    def summary(self) -> dict:
        return {
            "rounds_recorded":  len(self._round_snapshots),
            "patterns_stored":  len(self._pattern_library),
            "temporal_stability": round(self.temporal_stability(), 4),
            "score_trajectory": self.score_trajectory(),
            "declining":        self.is_score_declining(),
        }

    def __repr__(self):
        s = self.summary()
        return (f"ClusterMemory(rounds={s['rounds_recorded']}, "
                f"patterns={s['patterns_stored']}, "
                f"stability={s['temporal_stability']:.3f})")
