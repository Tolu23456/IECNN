"""
AGI Control Layer — manages system-level cognition and internal actions.

The AGI Control Layer implements the 'Self-Model' (SM), which allows the
system to monitor its own performance via the Cognitive State Vector (CSV)
and perform internal actions to modulate its behavior.
"""

import numpy as np
from typing import Dict, Any, Optional, List
from dataclasses import dataclass

@dataclass
class CognitiveStateVector:
    """Represents the internal mental state of the system."""
    entropy: float = 0.0      # Current cluster uncertainty (F11)
    dominance: float = 0.0    # Leader strength (F9)
    stability: float = 0.0    # Temporal centroid consistency (F12)
    energy: float = 0.0       # Global system energy (F21)
    eug: float = 0.0          # Emergent Utility Gradient (F16)
    call_count: int = 0       # Cumulative call experience
    reasoning_depth: int = 2  # Current depth of internal simulations

    def to_array(self) -> np.ndarray:
        return np.array([
            self.entropy, self.dominance, self.stability,
            self.energy, self.eug, self.call_count / 1000.0,
            self.reasoning_depth / 10.0
        ], dtype=np.float32)

class InternalCognitiveActions:
    """Modulations the Self-Model can apply to the IECNN pipeline."""
    def __init__(self):
        self.reasoning_depth_delta: float = 0.0 # Delta for dot reasoning
        self.iteration_budget_delta: int = 0    # Dynamic computation budget
        self.threshold_shift: float = 0.0
        self.mutation_pressure: float = 1.0
        self.exploration_noise: float = 0.0
        self.attention_allocation: float = 0.5 # AAF weight

class SelfModel:
    """
    The 'Ego' of the system. Maps CSV to InternalCognitiveActions.

    The Self-Model learns which parameter modulations lead to higher
    utility and lower energy over time.
    """
    def __init__(self, state_dim: int = 7, seed: int = 42):
        self.state_dim = state_dim
        self._rng = np.random.RandomState(seed)

        # Policy weight matrix (State -> Action deltas)
        # 0: reasoning_depth, 1: threshold, 2: mutation, 3: noise, 4: AAF
        self.policy = self._rng.randn(5, state_dim).astype(np.float32) * 0.1
        self.bias = np.zeros(5, dtype=np.float32)

        # Performance history to refine the policy
        self.history: List[Dict[str, Any]] = []

    def decide(self, csv: CognitiveStateVector) -> InternalCognitiveActions:
        """Analyze current CSV and return parameter modulations."""
        state = csv.to_array()
        raw_actions = np.tanh(self.policy @ state + self.bias)

        actions = InternalCognitiveActions()

        # 1. Reasoning Depth: high entropy/energy increases depth
        actions.reasoning_depth_delta = float(raw_actions[0] * 0.2)

        # 2. Threshold Shift: high stability allows tighter thresholds
        actions.threshold_shift = float(raw_actions[1] * 0.1)

        # 3. Mutation Pressure: low EUG (stagnation) increases pressure
        actions.mutation_pressure = float(1.0 + raw_actions[2] * 0.5)

        # 4. Exploration Noise
        actions.exploration_noise = float(max(0.0, raw_actions[3] * 0.2))

        # 5. Attention Allocation Field (AAF)
        actions.attention_allocation = float(np.clip(0.5 + raw_actions[4] * 0.4, 0.1, 0.9))

        # 6. Iteration Budget (Dynamic Budget): high complexity needs more rounds
        # Complexity estimated by entropy and lack of dominance
        complexity = csv.entropy * (1.0 - csv.dominance)
        actions.iteration_budget_delta = int(np.round(complexity * 5 + raw_actions[0] * 2))

        # 7. Thinking Policy (Fast vs Deep):
        # If surprise (EUG delta) is high, switch to deep reasoning mode.
        if abs(csv.eug) > 0.15:
            actions.reasoning_depth_delta += 0.3
            actions.iteration_budget_delta += 4

        return actions

    def learn(self, last_csv: CognitiveStateVector, actions: InternalCognitiveActions,
              utility_delta: float, energy_delta: float):
        """
        Policy Gradient-style update for the Self-Model.
        Rewards modulations that increase EUG and decrease System Energy.
        """
        reward = utility_delta - energy_delta
        # In IECNN, we don't use backprop, so we use a simple Dot Reinforcement
        # style update for the policy weights.
        state = last_csv.to_array()

        # Action vector that was taken (simplified)
        act_vec = np.array([
            actions.reasoning_depth_delta / 3.0,
            actions.threshold_shift / 0.1,
            (actions.mutation_pressure - 1.0) / 0.5,
            actions.exploration_noise / 0.2,
            (actions.attention_allocation - 0.5) / 0.4
        ], dtype=np.float32)

        # Nudge policy: state-action pairs that led to reward are reinforced
        lr = 0.01
        self.policy += lr * reward * np.outer(act_vec, state)
        self.bias += lr * reward * act_vec

    def state_dict(self) -> Dict:
        return {"policy": self.policy, "bias": self.bias}

    def load_state(self, state: Dict):
        self.policy = state.get("policy", self.policy)
        self.bias = state.get("bias", self.bias)
