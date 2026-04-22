"""
IECNN Cognition Layer — AGI Control Layer (F27–F35).

Manages the system's "mind state" and cognitive control policy.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formulas.formulas import (
    reasoning_depth, abstraction_gradient, planning_horizon,
    goal_stability, self_model_update, memory_plasticity
)
from .world import WorldState
from .planning import PlanningSystem
from .memory import LongTermMemory
from .reasoning import DeepReasoningLayer

class CognitionLayer:
    """
    AGI Control Layer for IECNN.

    Operates on top of the convergence process to modulate behavior.
    """

    TASKS = ["reasoning", "planning", "abstraction", "memory", "exploration"]

    def __init__(self, state_dim: int = 5, feature_dim: int = 256, seed: int = 42):
        self.state_dim = state_dim
        self.feature_dim = feature_dim
        self._rng = np.random.RandomState(seed)

        # Core Cognitive Pillars
        self.world = WorldState(feature_dim=feature_dim)
        self.planning = PlanningSystem(seed=seed)
        self.memory = LongTermMemory(feature_dim=feature_dim)
        self.reasoning = DeepReasoningLayer(feature_dim=feature_dim, seed=seed)

        # F35: Self-Model (Persistent across runs)
        self.self_model = np.zeros(state_dim, dtype=np.float32)

        # F28: Attention Allocation Field (AAF) - Hybrid Matrix (Task Embeddings)
        # 5 rows (tasks), state_dim columns
        # Initialize with fixed symbolic anchors + random modulation
        self.aaf = self._rng.randn(len(self.TASKS), state_dim).astype(np.float32) * 0.1

        # Learnable modulation for task embeddings
        self.task_momentum = np.zeros_like(self.aaf)

        self.last_csv = np.zeros(state_dim, dtype=np.float32)
        self.last_policy = np.zeros(len(self.TASKS), dtype=np.float32)

    def observe(self, convergence: float, utility: float, entropy: float,
                dominance: float, stability: float) -> np.ndarray:
        """F27: Cognitive State Vector (CSV)"""
        csv = np.array([convergence, utility, entropy, dominance, stability], dtype=np.float32)
        self.last_csv = csv
        return csv

    def process(self, csv: np.ndarray) -> Dict:
        """Run the full cognitive process (F28–F35)."""

        # F28: Attention & Policy (MCPF)
        logits = self.aaf @ csv
        policy = self._softmax(logits)
        self.last_policy = policy

        # F29: Reasoning Depth
        cs_norm = float(np.linalg.norm(csv))
        rdf = reasoning_depth(cs_norm, csv[4]) # csv[4] = stability

        # F30: Abstraction Gradient
        ag = abstraction_gradient(csv[2], csv[0]) # entropy - convergence

        # F32: Planning Horizon
        phf = planning_horizon(csv[4], csv[2], rdf)

        # F33: Goal Stability
        gsf = goal_stability(csv[0], csv[3]) # convergence, dominance

        # F35: Self-Model Update
        # Use stability from F25 to drive plasticity rho
        rho = memory_plasticity(csv[4])
        self_model_update(self.self_model, csv, rho)

        cognition_report = {
            "csv": csv,
            "policy": {task: float(policy[i]) for i, task in enumerate(self.TASKS)},
            "reasoning_depth": float(rdf),
            "abstraction_gradient": float(ag),
            "planning_horizon": float(phf),
            "goal_stability": float(gsf),
            "self_model": self.self_model.copy()
        }

        return cognition_report

    def modulate_parameters(self, report: Dict, base_params: Dict) -> Dict:
        """F34: Apply Meta-Controller Policy to modulate system parameters."""
        policy = report["policy"]

        # Modulate max_iterations based on reasoning depth
        # Scale: RDF typically 0.1-2.0. Base iterations 12.
        # Deep reasoning -> more iterations.
        rdf = report["reasoning_depth"]
        new_params = base_params.copy()

        if "max_iterations" in new_params:
            new_params["max_iterations"] = int(np.clip(
                new_params["max_iterations"] * (0.5 + rdf), 5, 30
            ))

        # Modulate exploration noise based on policy["exploration"]
        if "exploration_noise" in new_params:
            new_params["exploration_noise"] *= (0.5 + 2.0 * policy["exploration"])

        # Abstraction Gradient influences dot granularity (hypothetical dim focus)
        # If AG > 0, system should abstract more (higher-level focus)

        return new_params

    def update_aaf(self, delta_j: float, lr: float = 0.01):
        """Update task embeddings based on success (change in master objective J)."""
        # If J improved, reinforce the tasks that were prioritized in the last policy
        # This is a simple policy-gradient-like reinforcement
        update = np.outer(self.last_policy, self.last_csv)
        if delta_j > 0:
            self.aaf += lr * delta_j * update
        else:
            # Weaken the policy that led to a decline
            self.aaf += lr * delta_j * update # delta_j is negative

    def _softmax(self, x: np.ndarray) -> np.ndarray:
        e_x = np.exp(x - np.max(x))
        return e_x / e_x.sum()

    def compress_concepts(self, clusters: List, ag: float) -> List:
        """
        F31: Concept Formation Operator (CFO) - Stage B Abstraction.
        Merges clusters into higher-level concepts if AG is high.
        """
        if ag < 0.2:
            return clusters

        # High abstraction gradient -> merge top-N clusters into a single 'concept'
        # if they are reasonably similar.
        if len(clusters) < 2:
            return clusters

        # Just a placeholder for now: if AG is very high, we could return
        # a single cluster representing the weighted mean of top 3.
        return clusters

    def run_agi_loop(self, current_latent: np.ndarray, j_score: float,
                     concepts: List = None) -> Dict:
        """
        Executes the full AGI loop (F36–F45) after a pipeline run.
        """
        # 1. Update World Model (F37)
        self.world.update(current_latent)

        # 2. Causal Discovery (F38)
        # Use simple surprise signal for now (norm of change)
        surprise = float(np.linalg.norm(current_latent - self.world.global_vector))
        delta_j = j_score - 0.5 # relative to a prior
        self.world.construct_causal_graph(surprise, delta_j)

        # 3. Simulate and Plan (F39–F42)
        # Simulate k-steps ahead using current policy
        futures = self.planning.simulate(self.world.global_vector, self.last_policy)

        # Evaluate plan (crude score based on future stability)
        plan_score = self.planning.evaluate_plan([j_score] * len(futures))

        # 4. Long-Term Memory (F43–F45)
        self.memory.encode(self.world.global_vector, concepts or [], j_score)

        # Memory retrieval based on cognitive state
        associative_recall = self.memory.retrieve(self.last_csv)

        # 5. Deep Reasoning (triggered by policy)
        if self.last_policy[0] > 0.25: # task 0 is 'reasoning'
            current_latent = self.reasoning.analyze(
                current_latent, self.world, self.planning, self.last_policy
            )

        # Consolidation if surprise was high
        if surprise > 0.5:
            # Predict next state (dummy prediction = last global vector)
            w_pred = self.world.global_vector
            self.memory.consolidate(current_latent, w_pred)

        return {
            "surprise": surprise,
            "plan_score": plan_score,
            "recall_norm": float(np.linalg.norm(associative_recall)),
            "reasoned_latent": current_latent
        }

    def save(self, path: str):
        np.savez(path,
                 self_model=self.self_model,
                 aaf=self.aaf,
                 world_vector=self.world.global_vector,
                 memories=self.memory.memories,
                 mem_count=self.memory.count)

    def load(self, path: str):
        if os.path.exists(path):
            data = np.load(path)
            self.self_model = data["self_model"]
            self.aaf = data["aaf"]
            if "world_vector" in data:
                self.world.global_vector = data["world_vector"]
            if "memories" in data:
                self.memory.memories = data["memories"]
                self.memory.count = int(data["mem_count"])
