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
from formulas.formulas import similarity_score, phase_aware_similarity


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

    def __init__(self, feature_dim: int = 256, max_patterns: int = 128,
                 window_rounds: int = 10, phase_coding: bool = False):
        self.feature_dim   = feature_dim
        self.max_patterns  = max_patterns
        self.window_rounds = window_rounds
        self.phase_coding  = phase_coding

        # Per-call timeline
        self._round_snapshots: List[RoundSnapshot] = []

        # Cross-call pattern library: stable patterns that recur
        self._pattern_library: List[Tuple[np.ndarray, float]] = []  # (centroid, weight)
        self._pattern_counts: List[int] = []
        # Parallel to _pattern_library: each entry is either None (no phase
        # data, e.g. legacy patterns or phase coding disabled) or a tuple
        # (re_sum, im_sum, count) accumulating circular statistics.
        self._pattern_phase_acc: List[Optional[Tuple[float, float, int]]] = []

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

    def _pattern_phase(self, idx: int) -> Tuple[Optional[float], float]:
        """Return (mean_phase, concentration) for pattern idx, or (None, 0.0)."""
        acc = self._pattern_phase_acc[idx] if idx < len(self._pattern_phase_acc) else None
        if acc is None:
            return (None, 0.0)
        re_s, im_s, cnt = acc
        if cnt <= 0:
            return (None, 0.0)
        mean = float(np.arctan2(im_s, re_s))
        conc = float(np.sqrt(re_s * re_s + im_s * im_s) / cnt)
        return (mean, conc)

    def _accum_phase(self, idx: int, phase: float):
        """Add a sample to pattern idx's circular accumulator."""
        re = float(np.cos(phase)); im = float(np.sin(phase))
        cur = self._pattern_phase_acc[idx] if idx < len(self._pattern_phase_acc) else None
        if cur is None:
            self._pattern_phase_acc[idx] = (re, im, 1)
        else:
            r0, i0, c0 = cur
            self._pattern_phase_acc[idx] = (r0 + re, i0 + im, c0 + 1)

    def commit_pattern(self, centroid: np.ndarray, score: float,
                       alpha: float = 0.7,
                       phase: Optional[float] = None):
        """
        After a successful convergence, store the winning centroid as a pattern.
        If a similar pattern already exists, update its weight.

        In Phase-Coded Mode, 'centroid' is a complex vector.
        """
        n = np.linalg.norm(centroid)
        if n < 1e-10:
            return
        c_norm = (centroid / n).astype(np.complex64) if np.iscomplexobj(centroid) else (centroid / n).astype(np.complex64)

        # Make sure parallel arrays stay in lock-step length, even on legacy
        # state that predates phase tracking.
        while len(self._pattern_phase_acc) < len(self._pattern_library):
            self._pattern_phase_acc.append(None)

        # Check if similar pattern exists
        for i, (pat, w) in enumerate(self._pattern_library):
            # Complex similarity handles phase automatically
            sim = similarity_score(c_norm, pat, alpha)

            if sim > 0.85:
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
        # We don't need a separate phase_acc if the vector itself is complex
        self._pattern_phase_acc.append(None)

        # Trim to max_patterns by keeping highest-weighted
        if len(self._pattern_library) > self.max_patterns:
            order = np.argsort([w for _, w in self._pattern_library])[::-1]
            keep = list(order[:self.max_patterns])
            self._pattern_library   = [self._pattern_library[i]   for i in keep]
            self._pattern_counts    = [self._pattern_counts[i]    for i in keep]
            self._pattern_phase_acc = [self._pattern_phase_acc[i] for i in keep]

    def closest_pattern(self, query: np.ndarray, alpha: float = 0.7,
                        query_phase: Optional[float] = None) -> Optional[np.ndarray]:
        """Return the closest known pattern to `query`, or None."""
        if not self._pattern_library:
            return None
        q = query / (np.linalg.norm(query) + 1e-10)
        best_sim, best_pat = -1.0, None
        for i, (pat, _) in enumerate(self._pattern_library):
            if self.phase_coding and query_phase is not None:
                pat_phase, pat_conc = self._pattern_phase(i)
                s = phase_aware_similarity(q, pat,
                                           phase_a=query_phase, phase_b=pat_phase,
                                           concentration=pat_conc, alpha=alpha)
            else:
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

    # ── Persistence ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "feature_dim":      self.feature_dim,
            "max_patterns":     self.max_patterns,
            "window_rounds":    self.window_rounds,
            "phase_coding":     self.phase_coding,
            "pattern_library":  [(c.copy(), float(w)) for c, w in self._pattern_library],
            "pattern_counts":   list(self._pattern_counts),
            "pattern_phase_acc": [
                None if acc is None else (float(acc[0]), float(acc[1]), int(acc[2]))
                for acc in self._pattern_phase_acc
            ],
        }

    def load_state(self, state: dict):
        self.feature_dim   = state.get("feature_dim", self.feature_dim)
        self.max_patterns  = state.get("max_patterns", self.max_patterns)
        self.window_rounds = state.get("window_rounds", self.window_rounds)
        # Don't override phase_coding from state — the IECNN constructor sets
        # it intentionally; legacy state simply has no phase data.
        self._pattern_library = [
            (np.asarray(c, dtype=np.float32), float(w))
            for c, w in state.get("pattern_library", [])
        ]
        self._pattern_counts  = list(state.get("pattern_counts", []))
        loaded_phases = state.get("pattern_phase_acc", [])
        self._pattern_phase_acc = [
            None if acc is None else (float(acc[0]), float(acc[1]), int(acc[2]))
            for acc in loaded_phases
        ]
        # Pad missing entries (legacy pickles) so parallel arrays stay aligned.
        while len(self._pattern_phase_acc) < len(self._pattern_library):
            self._pattern_phase_acc.append(None)
        self._round_snapshots = []
