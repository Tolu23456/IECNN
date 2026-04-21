"""
IECNN World Model — internal representation of reality (F36–F38).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formulas.formulas import world_update

class WorldState:
    """
    F36: World State Representation (WSR).
    Maintains a hybrid Graph + Tensor representation of reality.
    """
    def __init__(self, feature_dim: int = 256, max_entities: int = 100):
        self.feature_dim = feature_dim
        self.max_entities = max_entities

        # Entities: ID -> {attributes, latent_vector}
        self.entities: Dict[int, Dict] = {}

        # Relationships: (ID1, ID2) -> {type, weight, causal_score}
        self.relationships: Dict[Tuple[int, int], Dict] = {}

        # Tensor Shadow: Adj matrix for fast computation (N x N x F)
        # For now, a simple N x N weight matrix
        self.adj_matrix = np.zeros((max_entities, max_entities), dtype=np.float32)

        # Global World Vector (summary for F37)
        self.global_vector = np.zeros(feature_dim, dtype=np.float32)

    def update(self, observations: np.ndarray, lambda_rate: float = 0.9):
        """F37: World Update Function (WUF)"""
        # observation should be a pooled latent vector from BaseMap
        world_update(self.global_vector, observations, lambda_rate)

    def add_entity(self, entity_id: int, latent: np.ndarray, attributes: Dict = None):
        self.entities[entity_id] = {
            "latent": latent.copy(),
            "attributes": attributes or {}
        }

    def add_relationship(self, id1: int, id2: int, rel_type: str, weight: float = 1.0):
        self.relationships[(id1, id2)] = {
            "type": rel_type,
            "weight": weight,
            "causal_score": 0.0
        }
        if id1 < self.max_entities and id2 < self.max_entities:
            self.adj_matrix[id1, id2] = weight

    def construct_causal_graph(self, surprise: float, delta_j: float):
        """
        F38: Causal Graph Constructor (CGC).
        Estimates P(R|E,P) using counterfactual reward sensitivity.
        """
        # If objective J improved and surprise was high, reinforce existing
        # relationships that were active during the intervention.
        if abs(delta_j) < 1e-5:
            return

        for (id1, id2), rel in self.relationships.items():
            # Heuristic: causality ∝ Surprise * ΔJ
            # If ΔJ is positive and surprise is high, this relationship is likely causal
            rel["causal_score"] += surprise * delta_j
            # Normalize/Clip
            rel["causal_score"] = np.clip(rel["causal_score"], -1.0, 1.0)

    def get_tensor_view(self) -> np.ndarray:
        return self.adj_matrix
