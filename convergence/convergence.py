"""
Convergence Layer

Finds agreement among predictions using the unified BaseMapping
representation space. A single similarity metric over structured maps
is consistent across all data types (text, image, video).
"""

import numpy as np
import ctypes
import os
from typing import List, Tuple, Dict, Optional

import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import (
    similarity_score, convergence_score, dominance_score,
    pairwise_similarity_matrix,
)


class Cluster:
    def __init__(self, cluster_id: int):
        self.cluster_id  = cluster_id
        self.predictions: List[np.ndarray] = []
        self.confidences: List[float]       = []
        self.infos: List[Dict]              = []
        self.centroid: Optional[np.ndarray] = None
        self.score: float                   = 0.0

    def add(self, p: np.ndarray, c: float, info: Dict):
        self.predictions.append(p); self.confidences.append(c); self.infos.append(info)
        self.centroid = np.mean(np.stack(self.predictions), axis=0)

    def compute_score(self, alpha: float = 0.7):
        self.score = convergence_score(self.predictions, self.confidences, alpha)

    @property
    def size(self) -> int: return len(self.predictions)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    def sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in self.infos:
            s = info.get("source", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def __repr__(self):
        return f"Cluster(id={self.cluster_id}, size={self.size}, score={self.score:.4f})"


class ConvergenceLayer:
    """
    Clusters predictions by similarity + agreement strength (Formula 1).
    Uses C-accelerated pairwise similarity matrix when available.

    The unified BaseMapping space means one metric works for all modalities.
    """

    def __init__(self, similarity_threshold: float = 0.45,
                 alpha: float = 0.7, dominance_threshold: float = 0.70):
        self.threshold  = similarity_threshold
        self.alpha      = alpha
        self.dom_thresh = dominance_threshold

    def run(self, candidates: List[Tuple]) -> Tuple[List[Cluster], List[int]]:
        if not candidates: return [], []
        preds  = [c[0] for c in candidates]
        confs  = [c[1] for c in candidates]
        infos  = [c[2] for c in candidates]

        sim    = pairwise_similarity_matrix(preds, self.alpha)
        assign = self._cluster(preds, sim)

        cmap: Dict[int, Cluster] = {}
        for i, (p, c, info) in enumerate(zip(preds, confs, infos)):
            cid = assign[i]
            if cid not in cmap: cmap[cid] = Cluster(cid)
            cmap[cid].add(p, c, info)
        for cl in cmap.values(): cl.compute_score(self.alpha)

        clusters = sorted(cmap.values(), key=lambda c: c.score, reverse=True)
        return clusters, assign

    def _cluster(self, preds: List[np.ndarray], sim: np.ndarray) -> List[int]:
        n = len(preds)
        assign: List[int] = [-1] * n
        centroids: Dict[int, np.ndarray] = {}
        members: Dict[int, List[int]]    = {}
        nxt = 0
        for i in range(n):
            best_cid, best_sim = -1, self.threshold
            for cid, cent in centroids.items():
                s = similarity_score(preds[i], cent, self.alpha)
                if s > best_sim: best_sim, best_cid = s, cid
            if best_cid == -1:
                cid = nxt; nxt += 1
                assign[i] = cid; members[cid] = [i]; centroids[cid] = preds[i].copy()
            else:
                assign[i] = best_cid; members[best_cid].append(i)
                centroids[best_cid] = np.mean([preds[j] for j in members[best_cid]], axis=0)
        return assign

    def dominance(self, clusters: List[Cluster]) -> Tuple[float, bool]:
        if not clusters: return 0.0, False
        scores = [c.score for c in clusters]
        d = dominance_score(scores[0], scores)
        return d, d >= self.dom_thresh

    def summarize(self, clusters: List[Cluster]) -> Dict:
        if not clusters: return {"num_clusters": 0}
        dom, is_dom = self.dominance(clusters)
        return {
            "num_clusters": len(clusters),
            "top_score":    float(clusters[0].score),
            "top_size":     clusters[0].size,
            "dominance":    float(dom),
            "is_dominant":  is_dom,
        }
