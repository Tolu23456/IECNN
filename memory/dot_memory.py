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
  4. Apply Dot Reinforcement Pressure (F17) within-call to decay weak dots
"""

import numpy as np
from collections import deque
from typing import Dict, List, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


class DotMemory:
    """
    Records each dot's prediction history and computes effectiveness scores.

    Effectiveness(d) = success_count(d) / total_count(d)
    where a "success" means the dot's prediction landed in the winning cluster.
    """

    def __init__(self, num_dots: int, window_size: int = 20):
        self.num_dots    = num_dots
        self.window_size = window_size

        # F17 Reinforcement Weights (Meta-learnable)
        self.lambda1 = 0.40  # convergence contribution
        self.lambda2 = 0.20  # specialization
        self.lambda3 = 0.30  # utility (EUG)
        self.lambda4 = 0.10  # failure penalty
        self.beta    = 0.50  # utility acceleration

        self._success_counts: Dict[int, float] = {}
        self._total_counts:   Dict[int, float] = {}

        # Rolling window of recent predictions per dot
        self._windows: Dict[int, deque] = {}

        # Per-dot prediction variance (measures specialization)
        self._var_sums: Dict[int, np.ndarray] = {}
        self._var_counts: Dict[int, int] = {}

        # Per-dot circular phase accumulator: (re_sum, im_sum, count)
        # Populated only when phase coding is active and predictions carry phase.
        self._phase_acc: Dict[int, Tuple[float, float, int]] = {}
        # Multiplier on fitness for narrowly phase-concentrated dots; 0 disables.
        self.phase_bonus_weight: float = 0.0

    def _ensure_id(self, dot_id: int):
        if dot_id not in self._windows:
            self._windows[dot_id] = deque(maxlen=self.window_size)
            self._var_counts[dot_id] = 0
            self._success_counts[dot_id] = 0.0
            self._total_counts[dot_id] = 0.0
            self._phase_acc[dot_id] = (0.0, 0.0, 0)

    def record(self, dot_id: int, prediction: np.ndarray, in_winner: bool,
               phase: Optional[float] = None):
        """Record whether a dot's prediction ended in the winning cluster.

        If `phase` (radians) is provided, also accumulate it into the dot's
        circular phase distribution. A narrow distribution (high concentration)
        means the dot fires consistently from the same positional slot.
        """
        self._ensure_id(dot_id)

        self._total_counts[dot_id] += 1.0
        if in_winner:
            self._success_counts[dot_id] += 1.0

        self._windows[dot_id].append(prediction.copy())
        # Update rolling variance for specialization score
        d = len(self._windows[dot_id])
        if d >= 2:
            stack = np.stack(list(self._windows[dot_id]))
            self._var_sums[dot_id] = np.var(stack, axis=0)
        self._var_counts[dot_id] += 1

        if phase is not None:
            re_s, im_s, cnt = self._phase_acc.get(dot_id, (0.0, 0.0, 0))
            self._phase_acc[dot_id] = (
                re_s + float(np.cos(phase)),
                im_s + float(np.sin(phase)),
                cnt + 1,
            )

    def phase_concentration(self, dot_id: int) -> float:
        """Resultant length R/n of the dot's circular phase distribution.

        Returns 0.0 when no phase samples have been recorded — this disables
        any phase-related bonus for dots that have never seen phase data.
        Returns ~1.0 when the dot fires from a single positional slice.
        """
        re_s, im_s, cnt = self._phase_acc.get(dot_id, (0.0, 0.0, 0))
        if cnt < 1:
            return 0.0
        return float(np.sqrt(re_s * re_s + im_s * im_s) / cnt)

    def all_phase_concentrations(self, dot_ids: List[int]) -> np.ndarray:
        out = np.zeros(len(dot_ids), dtype=np.float32)
        for i, did in enumerate(dot_ids):
            out[i] = self.phase_concentration(did)
        return out

    def effectiveness(self, dot_id: int) -> float:
        """Fraction of predictions that entered the winning cluster (0.5 prior)."""
        total = self._total_counts.get(dot_id, 0.0)
        if total < 1:
            return 0.5  # uninformed prior
        return float(self._success_counts.get(dot_id, 0.0) / total)

    def all_effectivenesses(self, dot_ids: Optional[List[int]] = None) -> np.ndarray:
        """Return effectiveness scores for given dots or all known dots."""
        if dot_ids is None:
            dot_ids = list(self._total_counts.keys())

        eff = np.zeros(len(dot_ids), dtype=np.float32)
        for i, did in enumerate(dot_ids):
            eff[i] = self.effectiveness(did)
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
        if dot_id not in self._windows:
            return None
        w = self._windows[dot_id]
        if len(w) == 0:
            return None
        return np.mean(np.stack(list(w)), axis=0).astype(np.float32)

    def rankings(self, dot_ids: Optional[List[int]] = None) -> List[Tuple[int, float]]:
        """Return list of (dot_id, fitness) sorted highest first."""
        if dot_ids is None:
            dot_ids = list(self._total_counts.keys())
        fit = self.all_fitness_scores(dot_ids)
        order = np.argsort(fit)[::-1]
        return [(int(dot_ids[i]), float(fit[i])) for i in order]

    def all_fitness_scores(self, dot_ids: List[int], j_norm: float = 0.0) -> np.ndarray:
        """
        F24: Dot Fitness Function
        F_d = R_d + alpha*C_d + beta*S_d + gamma*U_d - delta*N_d
        """
        from formulas.formulas import dot_fitness
        drp = self.drp_scores(dot_ids, j_norm)
        eff = self.all_effectivenesses(dot_ids)
        fit = np.zeros(len(dot_ids), dtype=np.float32)
        for i, did in enumerate(dot_ids):
            spec = self.specialization_score(did)
            # Use effectiveness as proxy for U_d (utility impact) for now
            fit[i] = dot_fitness(
                rd=float(drp[i]), cd=float(eff[i]), sd=spec, ud=float(eff[i]), nd=1.0 - float(eff[i])
            )
        # Phase-narrowness bonus: dots with a tightly concentrated phase
        # distribution get a multiplicative boost. Inactive when bonus weight
        # is 0 or when the dot has no phase samples (concentration == 0.0).
        if self.phase_bonus_weight > 0.0:
            conc = self.all_phase_concentrations(dot_ids)
            fit = fit * (1.0 + self.phase_bonus_weight * conc)
        return fit

    # ── Dot Reinforcement Pressure (F17) ─────────────────────────────

    def drp_scores(self, dot_ids: List[int], system_health: float) -> np.ndarray:
        """
        Compute FIXED F17 Dot Reinforcement Pressure scores for every dot.

        R_d = λ1·effectiveness + λ2·specialization + λ3·J_norm − λ4·failure_rate

        Returns an array of shape (len(dot_ids),) with a pressure score per dot.
        """
        from formulas.formulas import dot_reinforcement_pressure
        eff    = self.all_effectivenesses(dot_ids)
        scores = np.zeros(len(dot_ids), dtype=np.float32)
        for i, did in enumerate(dot_ids):
            spec         = self.specialization_score(did)
            failure_rate = 1.0 - float(eff[i])
            scores[i]    = dot_reinforcement_pressure(
                convergence_contrib = float(eff[i]),
                specialization      = spec,
                system_health       = system_health,
                failure_rate        = failure_rate,
                lambda1             = self.lambda1,
                lambda2             = self.lambda2,
                lambda3             = self.lambda3,
                lambda4             = self.lambda4,
            )
        return scores

    def apply_memory_decay(self, dot_ids: List[int], rho: float, x: float = 0.0):
        """
        F23: Memory Decay Function
        M_{t+1} = (1 - rho) * M_t + rho * X

        Applies decay to success_counts based on plasticity rate rho.
        """
        for did in dot_ids:
            if did in self._success_counts:
                self._success_counts[did] = (1.0 - rho) * self._success_counts[did] + rho * x

    def apply_floor_pressure(self, dot_ids: List[int], drp: np.ndarray, floor: float = 0.05,
                              decay: float = 0.90):
        """
        Apply pressure floor: dots whose DRP score falls below `floor`
        have their success_count decayed by `decay`, reducing future effectiveness.
        """
        for i, did in enumerate(dot_ids):
            if float(drp[i]) < floor:
                self._success_counts[did] = max(0.0, self._success_counts.get(did, 0.0) * decay)

    def hard_selection(self, dot_ids: List[int], drp: np.ndarray, keep_frac: float = 0.60,
                        penalty: float = 0.50):
        """
        Hard elimination pressure: the bottom (1 - keep_frac) of dots by DRP
        score receive a 50% effectiveness penalty.
        """
        n     = len(dot_ids)
        top_k = max(1, int(n * keep_frac))
        order = np.argsort(drp)[::-1]
        losers = order[top_k:]
        for idx in losers:
            did = dot_ids[idx]
            self._success_counts[did] = max(0.0, self._success_counts.get(did, 0.0) * penalty)

    def competition_decay(self, dot_ids: List[int], drp: np.ndarray, top_k_frac: float = 0.70,
                           decay: float = 0.90):
        """
        Top-k competition: rank dots by their DRP score, keep the top
        `top_k_frac` fraction intact, and decay the rest.
        """
        n     = len(dot_ids)
        top_k = max(1, int(n * top_k_frac))
        order = np.argsort(drp)[::-1]   # highest DRP first
        losers = order[top_k:]           # bottom 30%
        for idx in losers:
            did = dot_ids[idx]
            self._success_counts[did] = max(0.0, self._success_counts.get(did, 0.0) * decay)

    def prune(self, keep_ids):
        """
        Drop all per-dot records whose dot_id is not in `keep_ids`.

        Used by the joint dot_pool / dot_memory pruner to evict orphaned
        history left behind by dead dots. This is the change that actually
        shrinks .dotmem.pkl on disk.
        """
        keep = set(int(i) for i in keep_ids)
        for store in (self._success_counts, self._total_counts,
                      self._windows, self._var_sums, self._var_counts,
                      self._phase_acc):
            for did in list(store.keys()):
                if int(did) not in keep:
                    del store[did]

    def reset_round(self):
        """Call between iterations — does NOT erase long-term history."""
        pass  # Long-term counts accumulate; only per-round state is tracked in pipeline

    def reset_all(self):
        """Full reset — wipe all history."""
        self._success_counts.clear()
        self._total_counts.clear()
        self._windows.clear()
        self._var_sums.clear()
        self._var_counts.clear()
        self._phase_acc.clear()

    def summary(self, current_dot_ids: Optional[List[int]] = None) -> dict:
        eff = self.all_effectivenesses(current_dot_ids)
        active = len(self._total_counts)
        return {
            "num_dots":        self.num_dots,
            "active_dots":     active,
            "mean_eff":        float(np.mean(eff)) if len(eff) > 0 else 0.5,
            "max_eff":         float(np.max(eff)) if len(eff) > 0 else 0.5,
            "min_eff":         float(np.min(eff)) if len(eff) > 0 else 0.5,
            "top5":            self.rankings(current_dot_ids)[:5],
        }

    def __repr__(self):
        s = self.summary()
        return (f"DotMemory(dots={self.num_dots}, active={s['active_dots']}, "
                f"mean_eff={s['mean_eff']:.3f}, max_eff={s['max_eff']:.3f})")

    # ── Persistence ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "num_dots":           self.num_dots,
            "window_size":        self.window_size,
            "lambda1":            self.lambda1,
            "lambda2":            self.lambda2,
            "lambda3":            self.lambda3,
            "lambda4":            self.lambda4,
            "beta":               self.beta,
            "phase_bonus_weight": float(self.phase_bonus_weight),
            "success_counts":     dict(self._success_counts),
            "total_counts":       dict(self._total_counts),
            "windows":            {k: list(v) for k, v in self._windows.items()},
            "var_sums":           dict(self._var_sums),
            "var_counts":         dict(self._var_counts),
            "phase_acc":          {int(k): tuple(v) for k, v in self._phase_acc.items()},
        }

    def load_state(self, state: dict):
        from collections import deque
        self.num_dots    = state.get("num_dots", self.num_dots)
        self.window_size = state.get("window_size", self.window_size)
        self.lambda1 = state.get("lambda1", self.lambda1)
        self.lambda2 = state.get("lambda2", self.lambda2)
        self.lambda3 = state.get("lambda3", self.lambda3)
        self.lambda4 = state.get("lambda4", self.lambda4)
        self.beta    = state.get("beta", self.beta)
        self.phase_bonus_weight = float(state.get("phase_bonus_weight", self.phase_bonus_weight))
        self._success_counts = dict(state.get("success_counts", {}))
        self._total_counts   = dict(state.get("total_counts", {}))
        self._windows = {
            int(k): deque(v, maxlen=self.window_size)
            for k, v in state.get("windows", {}).items()
        }
        self._var_sums   = dict(state.get("var_sums", {}))
        self._var_counts = dict(state.get("var_counts", {}))
        self._phase_acc  = {
            int(k): (float(v[0]), float(v[1]), int(v[2]))
            for k, v in state.get("phase_acc", {}).items()
        }
