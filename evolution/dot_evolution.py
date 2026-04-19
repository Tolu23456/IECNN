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

        n = len(dots)
        if n == 0:
            return dots

        eff = memory.all_effectivenesses()
        cfg = self.config

        # Minimum runs before evolution kicks in
        total_runs = float(np.sum(memory.total_count))
        if total_runs < cfg.min_generations * n:
            self._generation += 1
            return dots  # too early — not enough data yet

        # Compute pool sizes
        n_keep     = max(1, int(n * cfg.keep_fraction))
        n_clone    = max(1, int(n * cfg.clone_fraction))
        n_cross    = max(1, int(n * cfg.crossover_fraction))
        n_random   = n - n_keep - n_clone - n_cross

        # Sort dots by effectiveness
        order = np.argsort(eff)[::-1]
        elite_ids   = list(order[:n_keep])
        replace_ids = list(order[n_keep:])

        new_pool = [None] * n

        # 1. Elites — keep unchanged
        for i, did in enumerate(elite_ids):
            new_pool[did] = dots[did]

        # Slot the offspring into replaced positions
        slots = iter(replace_ids)

        # 2. Clones with mutation
        for _ in range(n_clone):
            parent_id = self._tournament_select(elite_ids, eff, cfg.tournament_size)
            parent    = dots[parent_id]
            child     = self._mutate(parent, cfg)
            try:
                slot = next(slots)
                new_pool[slot] = child
            except StopIteration:
                break

        # 3. Crossover offspring
        for _ in range(n_cross):
            p1_id = self._tournament_select(elite_ids, eff, cfg.tournament_size)
            p2_id = self._tournament_select(elite_ids, eff, cfg.tournament_size)
            child = self._crossover(dots[p1_id], dots[p2_id])
            try:
                slot = next(slots)
                new_pool[slot] = child
            except StopIteration:
                break

        # 4. Random new dots (for diversity)
        for _ in range(max(0, n_random)):
            try:
                slot = next(slots)
                dot_id = slot
                new_dot = NeuralDot(
                    dot_id=dot_id,
                    feature_dim=dots[0].feature_dim,
                    bias=BiasVector.random(self._rng),
                    dot_type=DotType(self._rng.randint(0, len(DotType.__members__))),
                    seed=int(self._rng.randint(0, 2**31)),
                )
                new_pool[slot] = new_dot
            except StopIteration:
                break

        # Fill any remaining None slots (shouldn't happen, but be safe)
        for i in range(n):
            if new_pool[i] is None:
                new_pool[i] = dots[i]

        self._generation += 1
        return new_pool

    def _tournament_select(self, candidates: List[int], eff: np.ndarray, k: int) -> int:
        """k-tournament selection among candidate dot IDs."""
        if len(candidates) <= 1:
            return candidates[0] if candidates else 0
        k = min(k, len(candidates))
        contestants = self._rng.choice(candidates, size=k, replace=False)
        return int(contestants[np.argmax(eff[contestants])])

    def _mutate(self, dot, cfg: EvolutionConfig):
        """Clone a dot with Gaussian noise added to its bias and weights."""
        from neural_dot.neural_dot import NeuralDot, BiasVector

        # Mutate bias
        b_arr = dot.bias.to_array()
        noise = self._rng.randn(len(b_arr)).astype(np.float32) * cfg.bias_noise_std
        new_b = np.clip(b_arr + noise, 0.0, 2.0)
        new_b[-1] = max(new_b[-1], 0.05)  # temperature must be positive

        child = NeuralDot(
            dot_id=dot.dot_id,
            feature_dim=dot.feature_dim,
            bias=BiasVector.from_array(new_b),
            dot_type=dot.dot_type,
            seed=int(self._rng.randint(0, 2**31)),
        )
        # Mutate weights slightly
        child.W = dot.W + self._rng.randn(*dot.W.shape).astype(np.float32) * cfg.weight_noise_std
        child.b_offset = dot.b_offset + self._rng.randn(*dot.b_offset.shape).astype(np.float32) * cfg.weight_noise_std * 0.1
        return child

    def _crossover(self, p1, p2):
        """Blend two parent dots: average bias, uniform-crossover on weight rows."""
        from neural_dot.neural_dot import NeuralDot, BiasVector

        # Average the bias vectors
        b_arr = (p1.bias.to_array() + p2.bias.to_array()) / 2.0

        child = NeuralDot(
            dot_id=p1.dot_id,
            feature_dim=p1.feature_dim,
            bias=BiasVector.from_array(b_arr),
            dot_type=p1.dot_type,
            seed=int(self._rng.randint(0, 2**31)),
        )
        # Uniform row-wise crossover on W matrix
        mask = self._rng.rand(p1.W.shape[0]) > 0.5
        child.W = np.where(mask[:, None], p1.W, p2.W)
        child.b_offset = (p1.b_offset + p2.b_offset) / 2.0
        return child

    def mutate_weak_dots(self, dots: list, effectivenesses: np.ndarray,
                          threshold: float = 0.10,
                          mutation_std: float = 0.05) -> list:
        """
        Inline (within-call) mutation of dots whose effectiveness is below `threshold`.

        Unlike the between-call `evolve()` which replaces dots entirely, this
        method transforms them in-place: adds Gaussian noise to weights and bias,
        and with 20% probability switches the dot to a randomly chosen type.

        `mutation_std` is adaptive — the pipeline passes a higher value when EUG
        is stagnant (|U| < 0.01) to increase exploration breadth.

        Returns the same dot list (mutated in-place where applicable).
        """
        from neural_dot.neural_dot import NeuralDot, BiasVector, DotType

        n_types = len(DotType.__members__)
        for dot in dots:
            eff = float(effectivenesses[dot.dot_id])
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
                # Optional type switch (20% chance — creates new specialisation)
                if self._rng.rand() < 0.20:
                    new_type = DotType(self._rng.randint(0, n_types))
                    dot.dot_type = new_type
        return dots

    def stats(self, memory: DotMemory) -> dict:
        eff = memory.all_effectivenesses()
        return {
            "generation":   self._generation,
            "top_dot_eff":  float(np.max(eff)),
            "mean_eff":     float(np.mean(eff)),
            "bottom_eff":   float(np.min(eff)),
        }
