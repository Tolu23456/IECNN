"""
IECNN Long-Term Memory — structured experience storage (F43–F45).
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Any
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from formulas.formulas import memory_retrieval_attention, experience_consolidation

class LongTermMemory:
    """
    F43–F45: Long-Term Memory encoding, retrieval, and consolidation.
    """
    def __init__(self, feature_dim: int = 256, max_memories: int = 1000):
        self.feature_dim = feature_dim
        self.max_memories = max_memories

        # M_long: Structured knowledge events
        self.memories = np.zeros((max_memories, feature_dim), dtype=np.float32)
        self.count = 0

        # Metadata for each memory (importance, concepts, etc.)
        self.metadata: List[Dict] = []

    def encode(self, world_vec: np.ndarray, concepts: List, j_score: float):
        """
        F43: Long-Term Memory Encoding (LTME).
        Only important experiences are stored.
        """
        # Importance threshold
        if j_score < 0.3:
            return

        if self.count < self.max_memories:
            idx = self.count
            self.count += 1
        else:
            # Simple FIFO or replace least important (not implemented)
            idx = self.count % self.max_memories
            self.count += 1

        self.memories[idx] = world_vec.copy()
        self.metadata.append({
            "j_score": j_score,
            "concepts": concepts
        })

    def retrieve(self, cs_vector: np.ndarray) -> np.ndarray:
        """
        F44: Memory Retrieval Attention (MRA).
        Associative recall based on current cognitive state.
        """
        if self.count == 0:
            return np.zeros(self.feature_dim, dtype=np.float32)

        active_mems = self.memories[:min(self.count, self.max_memories)]
        # Map CS vector to FEATURE_DIM if needed (crude expansion)
        query = np.zeros(self.feature_dim, dtype=np.float32)
        query[:len(cs_vector)] = cs_vector

        weights = memory_retrieval_attention(query, active_mems)
        return (active_mems * weights[:, None]).sum(axis=0)

    def consolidate(self, world_vec: np.ndarray, predicted_vec: np.ndarray, eta: float = 0.05):
        """
        F45: Experience Consolidation Function (ECF).
        Learns from surprise signal (unexpected experiences).
        """
        # Surprise = W - W_pred
        # Consolidate into the most recent or relevant memory
        if self.count > 0:
            idx = (self.count - 1) % self.max_memories
            experience_consolidation(self.memories[idx], world_vec, predicted_vec, eta)

            # Re-normalize after update
            n = np.linalg.norm(self.memories[idx])
            if n > 1e-10: self.memories[idx] /= n
