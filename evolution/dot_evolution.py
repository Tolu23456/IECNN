"""
Dot Evolution — genetic-style selection and mutation of the neural dot pool.

After each complete `run()`, dots are ranked by effectiveness (how often their
predictions ended in the winning cluster). The evolution engine:

  1. Selects the top `keep_frac` dots (elites)
  2. Clones elites with small perturbations (mutation)
  3. Produces crossover offspring from pairs of elites
  4. Replaces the least effective dots with new offspring

This is NOT gradient-based. It is a discrete, population-level update.
The bias vector and weight matrix of each dot are both subject to evolution.
"""

import numpy as np
from dataclasses import dataclass
from typing import List, Tuple, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from memory.dot_memory import DotMemory


@dataclass
class EvolutionConfig:
    keep_fraction:    float = 0.40   # fraction of elite dots to keep unchanged
    clone_fraction:   float = 0.30   # fraction of population from clone+mutate
    crossover_fraction: float = 0.20 # fraction from crossover of two elites
    random_fraction:  float = 0.10   # fraction of fresh random dots
    bias_noise_std:   float = 0.05   # std of Gaussian noise added to bias during mutation
    weight_noise_std: float = 0.01   # std of Gaussian noise added to weights during mutation
    tournament_size:  int   = 4      # k for tournament selection
    min_generations:  int   = 3      # minimum calls before evolution kicks in
    max_dot_pool_size: int  = 1000   # Hard cap to prevent OOM


class DotEvolution:
    """
    Manages the evolution of the neural dot pool across calls.

    Call `evolve(dots, memory)` after each `run()` to produce an evolved pool
    that favours dots that historically produced winning predictions.
    """

    def __init__(self, config: Optional[EvolutionConfig] = None, seed: int = 42):
        self.config     = config or EvolutionConfig()
        self._rng       = np.random.RandomState(seed)
        self._generation = 0

    @property
    def generation(self) -> int:
        return self._generation

    def evolve(self, dots: list, memory: DotMemory) -> list:
        """
        Produce an evolved dot pool from the current pool + effectiveness memory.

        Returns a new list of NeuralDot objects (same length as `dots`).
        """
        from neural_dot.neural_dot import NeuralDot, BiasVector, DotType

        # 0. Pool Culling: if memory is tracking too many dead dots, prune it.
        # This keeps the save files compact.
        if len(memory._total_counts) > self.config.max_dot_pool_size * 2:
            current_ids = [d.dot_id for d in dots]
            memory.prune(current_ids)

        n = len(dots)
        if n == 0:
            return dots

        dot_ids = [d.dot_id for d in dots]
        # Use F24 Dot Fitness for selection
        fitness = memory.all_fitness_scores(dot_ids)
        cfg = self.config

        # Minimum runs before evolution kicks in
        # We check the mean total count of the current dots
        avg_total_count = np.mean([memory._total_counts.get(did, 0.0) for did in dot_ids])
        if avg_total_count < cfg.min_generations:
            self._generation += 1
            return dots  # too early — not enough data yet

        # Compute pool sizes
        n_keep     = max(1, int(n * cfg.keep_fraction))
        n_clone    = max(1, int(n * cfg.clone_fraction))
        n_cross    = max(1, int(n * cfg.crossover_fraction))
        n_random   = n - n_keep - n_clone - n_cross

        # Sort indices by fitness
        order = np.argsort(fitness)[::-1]
        elite_indices   = list(order[:n_keep])
        replace_indices = list(order[n_keep:])

        new_pool = [None] * n

        # 1. Elites — keep unchanged (same dot_id)
        for idx in elite_indices:
            new_pool[idx] = dots[idx]

        # Slot the offspring into replaced positions
        slots = iter(replace_indices)

        # 2. Clones with mutation
        for _ in range(n_clone):
            parent_idx = self._tournament_select(elite_indices, fitness, cfg.tournament_size)
            parent    = dots[parent_idx]
            child     = self._mutate(parent, cfg)
            try:
                slot = next(slots)
                new_pool[slot] = child
            except StopIteration:
                break

        # 3. Crossover offspring
        for _ in range(n_cross):
            p1_idx = self._tournament_select(elite_indices, fitness, cfg.tournament_size)
            p2_idx = self._tournament_select(elite_indices, fitness, cfg.tournament_size)
            child = self._crossover(dots[p1_idx], dots[p2_idx])
            try:
                slot = next(slots)
                new_pool[slot] = child
            except StopIteration:
                break

        # 4. Random new dots (for diversity)
        for _ in range(max(0, n_random)):
            try:
                slot = next(slots)
                new_dot = NeuralDot(
                    dot_id=None, # New unique ID
                    feature_dim=dots[0].feature_dim,
                    bias=BiasVector.random(self._rng),
                    dot_type=DotType(self._rng.randint(0, len(DotType.__members__))),
                    seed=int(self._rng.randint(0, 2**31)),
                    birth_generation=self._generation + 1,
                )
                new_pool[slot] = new_dot
            except StopIteration:
                break

        # Fill any remaining None slots
        for i in range(n):
            if new_pool[i] is None:
                new_pool[i] = dots[i]

        self._generation += 1
        return new_pool

    def _tournament_select(self, candidates_indices: List[int], eff: np.ndarray, k: int) -> int:
        """k-tournament selection among candidate dot indices."""
        if len(candidates_indices) <= 1:
            return candidates_indices[0] if candidates_indices else 0
        k = min(k, len(candidates_indices))
        subset_indices = self._rng.choice(len(candidates_indices), size=k, replace=False)
        contestant_indices = [candidates_indices[i] for i in subset_indices]

        best_idx = contestant_indices[0]
        best_eff = eff[best_idx]
        for idx in contestant_indices[1:]:
            if eff[idx] > best_eff:
                best_eff = eff[idx]
                best_idx = idx
        return best_idx

    def _mutate(self, dot, cfg: EvolutionConfig):
        """Clone a dot with Gaussian noise added to its bias and weights.

        Experimental Stable Mutation: The child gets a new ID but we record
        its parentage in the metadata for lineage tracking.
        """
        from neural_dot.neural_dot import NeuralDot, BiasVector

        # Mutate bias
        b_arr = dot.bias.to_array()
        noise = self._rng.randn(len(b_arr)).astype(np.float32) * cfg.bias_noise_std
        new_b = np.clip(b_arr + noise, 0.0, 2.0)
        new_b[-1] = max(new_b[-1], 0.05)  # temperature must be positive

        child = NeuralDot(
            dot_id=None, # New unique ID for the mutant child
            feature_dim=dot.feature_dim,
            bias=BiasVector.from_array(new_b),
            dot_type=dot.dot_type,
            seed=int(self._rng.randint(0, 2**31)),
            birth_generation=self._generation + 1,
        )
        # Mutate weights slightly
        child.W = dot.W + self._rng.randn(*dot.W.shape).astype(np.float32) * cfg.weight_noise_std
        child.b_offset = dot.b_offset + self._rng.randn(*dot.b_offset.shape).astype(np.float32) * cfg.weight_noise_std * 0.1

        # Record lineage
        child.metadata = getattr(dot, "metadata", {}).copy()
        child.metadata["parent_id"] = dot.dot_id
        child.metadata["lineage_depth"] = child.metadata.get("lineage_depth", 0) + 1
        return child

    def _crossover(self, p1, p2):
        """Blend two parent dots: average bias, uniform-crossover on weight rows. Gets a new unique ID."""
        from neural_dot.neural_dot import NeuralDot, BiasVector

        # Average the bias vectors
        b_arr = (p1.bias.to_array() + p2.bias.to_array()) / 2.0

        child = NeuralDot(
            dot_id=None, # New unique ID for the crossover child
            feature_dim=p1.feature_dim,
            bias=BiasVector.from_array(b_arr),
            dot_type=p1.dot_type,
            seed=int(self._rng.randint(0, 2**31)),
            birth_generation=self._generation + 1,
        )
        # Uniform row-wise crossover on W matrix
        mask = self._rng.rand(p1.W.shape[0]) > 0.5
        child.W = np.where(mask[:, None], p1.W, p2.W)
        child.b_offset = (p1.b_offset + p2.b_offset) / 2.0
        return child

    def mutate_weak_dots(self, dots: list, effectivenesses: np.ndarray,
                          threshold: float = 0.10,
                          mutation_std: float = 0.05) -> list:
        """Inline (within-call) mutation. Keeps the same dot_id as it is a transformation."""
        from neural_dot.neural_dot import NeuralDot, BiasVector, DotType

        n_types = len(DotType.__members__)
        # effectivenesses matches the dots list order here
        for i, dot in enumerate(dots):
            eff = float(effectivenesses[i])
            if eff < threshold:
                # Weight mutation
                dot.W = dot.W + self._rng.randn(*dot.W.shape).astype(np.float32) * mutation_std
                dot.b_offset = (dot.b_offset
                                + self._rng.randn(*dot.b_offset.shape).astype(np.float32)
                                * mutation_std * 0.1)
                # Bias mutation
                b_arr = dot.bias.to_array()
                noise = self._rng.randn(len(b_arr)).astype(np.float32) * mutation_std
                b_arr = np.clip(b_arr + noise, 0.0, 2.0)
                b_arr[-1] = max(float(b_arr[-1]), 0.05)
                dot.bias = BiasVector.from_array(b_arr)
                # Optional type switch (20% chance)
                if self._rng.rand() < 0.20:
                    new_type = DotType(self._rng.randint(0, n_types))
                    dot.dot_type = new_type
        return dots

    # ── Persistence ──────────────────────────────────────────────────

    def state_dict(self) -> dict:
        return {
            "generation": int(self._generation),
            "rng_state":  self._rng.get_state(),
            "config":     {
                "keep_fraction":      self.config.keep_fraction,
                "clone_fraction":     self.config.clone_fraction,
                "crossover_fraction": self.config.crossover_fraction,
                "random_fraction":    self.config.random_fraction,
                "bias_noise_std":     self.config.bias_noise_std,
                "weight_noise_std":   self.config.weight_noise_std,
                "tournament_size":    self.config.tournament_size,
                "min_generations":    self.config.min_generations,
            },
        }

    def load_state(self, state: dict):
        self._generation = int(state.get("generation", 0))
        rng_state = state.get("rng_state")
        if rng_state is not None:
            try:
                self._rng.set_state(rng_state)
            except Exception:
                pass
        cfg = state.get("config")
        if cfg:
            for k, v in cfg.items():
                if hasattr(self.config, k):
                    setattr(self.config, k, v)

    def stats(self, memory: DotMemory, dot_ids: Optional[List[int]] = None) -> dict:
        eff = memory.all_effectivenesses(dot_ids)
        return {
            "generation":   self._generation,
            "top_dot_eff":  float(np.max(eff)) if len(eff) > 0 else 0.5,
            "mean_eff":     float(np.mean(eff)) if len(eff) > 0 else 0.5,
            "bottom_eff":   float(np.min(eff)) if len(eff) > 0 else 0.5,
        }
