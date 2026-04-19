"""
Dot Memory — tracks per-dot prediction history and effectiveness.

Each neural dot accumulates a record of:
  - How many times its predictions entered the winning cluster
  - How many total predictions it has made
  - A rolling window of its recent prediction vectors

This allows the system to:
  1. Rank dots by effectiveness (for evolution selection)
  2. Bias dot attention toward historically successful patterns
  3. Detect dots that are specializing vs generalizing
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple


class DotMemory:
    """
    Records each dot's prediction history and computes effectiveness scores.

    Effectiveness(d) = success_count(d) / total_count(d)
    where a "success" means the dot's prediction landed in the winning cluster.
    """

    def __init__(self, num_dots: int, window_size: int = 20):
        self.num_dots    = num_dots
        self.window_size = window_size

        self.success_count = np.zeros(num_dots, dtype=np.float32)
        self.total_count   = np.zeros(num_dots, dtype=np.float32)

        # Rolling window of recent predictions per dot
        self._windows: Dict[int, deque] = {
            i: deque(maxlen=window_size) for i in range(num_dots)
        }

        # Per-dot prediction variance (measures specialization)
        self._var_sums: Dict[int, np.ndarray] = {}
        self._var_counts: Dict[int, int] = {i: 0 for i in range(num_dots)}

    def record(self, dot_id: int, prediction: np.ndarray, in_winner: bool):
        """Record whether a dot's prediction ended in the winning cluster."""
        if dot_id < 0 or dot_id >= self.num_dots:
            return
        self.total_count[dot_id] += 1
        if in_winner:
            self.success_count[dot_id] += 1
        self._windows[dot_id].append(prediction.copy())
        # Update rolling variance for specialization score
        d = len(self._windows[dot_id])
        if d >= 2:
            stack = np.stack(list(self._windows[dot_id]))
            self._var_sums[dot_id] = np.var(stack, axis=0)
        self._var_counts[dot_id] += 1

    def effectiveness(self, dot_id: int) -> float:
        """Fraction of predictions that entered the winning cluster (0.5 prior)."""
        total = self.total_count[dot_id]
        if total < 1:
            return 0.5  # uninformed prior
        return float(self.success_count[dot_id] / total)

    def all_effectivenesses(self) -> np.ndarray:
        """Return effectiveness scores for all dots."""
        totals = self.total_count.copy()
        mask = totals >= 1
        eff = np.full(self.num_dots, 0.5, dtype=np.float32)
        eff[mask] = self.success_count[mask] / totals[mask]
        return eff

    def specialization_score(self, dot_id: int) -> float:
        """
        How consistent (specialized) is the dot's output?
        Low variance = high specialization (dot focuses on a niche).
        High variance = generalist (dot explores broadly).
        Returns [0, 1]: 1 = fully specialized, 0 = fully general.
        """
        if dot_id not in self._var_sums:
            return 0.5
        mean_var = float(np.mean(self._var_sums[dot_id]))
        return float(1.0 / (1.0 + mean_var))

    def recent_centroid(self, dot_id: int) -> Optional[np.ndarray]:
        """Return the mean of the dot's recent predictions as a guidance signal."""
        w = self._windows[dot_id]
        if len(w) == 0:
            return None
        return np.mean(np.stack(list(w)), axis=0).astype(np.float32)

    def rankings(self) -> List[Tuple[int, float]]:
        """Return list of (dot_id, effectiveness) sorted highest first."""
        eff = self.all_effectivenesses()
        order = np.argsort(eff)[::-1]
        return [(int(i), float(eff[i])) for i in order]

    def reset_round(self):
        """Call between iterations — does NOT erase long-term history."""
        pass  # Long-term counts accumulate; only per-round state is tracked in pipeline

    def reset_all(self):
        """Full reset — wipe all history."""
        self.success_count[:] = 0.0
        self.total_count[:] = 0.0
        for i in range(self.num_dots):
            self._windows[i].clear()
        self._var_sums.clear()
        self._var_counts = {i: 0 for i in range(self.num_dots)}

    def summary(self) -> dict:
        eff = self.all_effectivenesses()
        active = int(np.sum(self.total_count >= 1))
        return {
            "num_dots":        self.num_dots,
            "active_dots":     active,
            "mean_eff":        float(np.mean(eff)),
            "max_eff":         float(np.max(eff)),
            "min_eff":         float(np.min(eff)),
            "top5":            self.rankings()[:5],
        }

    def __repr__(self):
        s = self.summary()
        return (f"DotMemory(dots={self.num_dots}, active={s['active_dots']}, "
                f"mean_eff={s['mean_eff']:.3f}, max_eff={s['max_eff']:.3f})")
