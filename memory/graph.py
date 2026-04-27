"""
World Knowledge Graph — persistent long-term memory for IECNN.

Consolidates frequently recurring stable patterns from ClusterMemory
into a permanent graph structure. Nodes represent stable concepts (centroids),
and edges represent co-occurrence or transition probabilities.
"""

import numpy as np
from typing import List, Dict, Optional, Tuple, Set
import pickle
import os
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score

class Node:
    """A stable conceptual anchor in the world graph."""
    def __init__(self, node_id: int, centroid: np.ndarray):
        self.node_id = node_id
        self.centroid = centroid
        self.weight = 1.0
        self.occurrences = 1
        self.last_updated = 0
        self.metadata: Dict = {}

class Edge:
    """Relationship between two conceptual nodes."""
    def __init__(self, source_id: int, target_id: int):
        self.source_id = source_id
        self.target_id = target_id
        self.strength = 1.0
        self.co_occurrences = 1

class WorldGraph:
    """
    Graph-based Long-Term Memory (M_long).

    Consolidates volatile patterns into permanent conceptual nodes.
    """
    def __init__(self, feature_dim: int = 256, threshold: float = 0.85):
        self.feature_dim = feature_dim
        self.threshold = threshold
        self.nodes: Dict[int, Node] = {}
        self.edges: Dict[Tuple[int, int], Edge] = {}
        self._next_node_id = 10000

    def consolidate(self, cluster_memory_patterns: List[Tuple[np.ndarray, float]], alpha: float = 0.7):
        """
        Merge new patterns from ClusterMemory into the permanent graph.
        Patterns with high weight that recur across many calls become graph nodes.
        Automatically links concepts that appear in the same consolidation batch.
        """
        new_node_ids = []

        for centroid, weight in cluster_memory_patterns:
            if weight < 0.2: continue # Ignore weak patterns

            best_node = self.find_closest(centroid, alpha)
            if best_node and similarity_score(centroid, best_node.centroid, alpha) > self.threshold:
                # Update existing node (consolidation)
                best_node.centroid = 0.95 * best_node.centroid + 0.05 * centroid
                best_node.weight += weight
                best_node.occurrences += 1
                new_node_ids.append(best_node.node_id)
            else:
                # Add as new candidate node
                new_id = self._next_node_id
                self._next_node_id += 1
                self.nodes[new_id] = Node(new_id, centroid.copy())
                self.nodes[new_id].weight = weight
                new_node_ids.append(new_id)

        # Link concepts that frequently appear together (Co-occurrence edges)
        for i in range(len(new_node_ids)):
            for j in range(i + 1, len(new_node_ids)):
                self.add_edge(new_node_ids[i], new_node_ids[j])

    def find_closest(self, query: np.ndarray, alpha: float = 0.7) -> Optional[Node]:
        """Find the most similar conceptual anchor in the graph."""
        if not self.nodes: return None

        best_sim = -1.0
        best_node = None

        # In a real SOTA implementation, this would use a spatial index (HNSW/KD-tree)
        # For now, we use a linear sweep over the graph nodes.
        for node in self.nodes.values():
            sim = similarity_score(query, node.centroid, alpha)
            if sim > best_sim:
                best_sim = sim
                best_node = node

        return best_node if best_sim > 0.3 else None

    def add_edge(self, source_id: int, target_id: int):
        """Record a relationship between two concepts."""
        key = tuple(sorted((source_id, target_id)))
        if key in self.edges:
            self.edges[key].strength += 1.0
            self.edges[key].co_occurrences += 1
        else:
            self.edges[key] = Edge(key[0], key[1])

    def query(self, query_vec: np.ndarray, alpha: float = 0.7, top_k: int = 5) -> List[Tuple[Node, float]]:
        """Retrieve related conceptual anchors."""
        results = []
        # Flatten query if it's a matrix
        q_v = np.mean(query_vec, axis=0) if query_vec.ndim > 1 else query_vec

        for node in self.nodes.values():
            sim = similarity_score(q_v, node.centroid, alpha)
            if sim > 0.1:
                results.append((node, sim))

        results.sort(key=lambda x: -x[1])
        return results[:top_k]

    def retrieve_facts(self, context_vec: np.ndarray, alpha: float = 0.7) -> Optional[np.ndarray]:
        """
        Factual Retrieval: Find the most relevant fact-centroid in the graph.
        Returns a blended centroid of related nodes to serve as a memory hint.
        """
        matches = self.query(context_vec, alpha, top_k=3)
        if not matches:
            return None

        total_sim = sum(s for _, s in matches)
        if total_sim < 1e-10: return matches[0][0].centroid

        blended = np.zeros(self.feature_dim, dtype=np.complex64)
        for node, sim in matches:
            blended += (sim / total_sim) * node.centroid.astype(np.complex64)

        return blended

    def state_dict(self) -> Dict:
        return {
            "nodes": {nid: (n.centroid, n.weight, n.occurrences) for nid, n in self.nodes.items()},
            "edges": {(e.source_id, e.target_id): (e.strength, e.co_occurrences) for e in self.edges.values()},
            "next_id": self._next_node_id
        }

    def load_state(self, state: Dict):
        self._next_node_id = state.get("next_id", 10000)
        for nid, (centroid, weight, occ) in state.get("nodes", {}).items():
            n = Node(nid, centroid)
            n.weight = weight
            n.occurrences = occ
            self.nodes[nid] = n

        for (s_id, t_id), (strength, co_occ) in state.get("edges", {}).items():
            e = Edge(s_id, t_id)
            e.strength = strength
            e.co_occurrences = co_occ
            self.edges[(s_id, t_id)] = e
