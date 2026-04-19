"""
Convergence Layer — hierarchical two-level clustering with consensus.

Level 1 (Micro-clustering):
  Groups very similar predictions (threshold=0.60) into fine-grained clusters.
  Produces micro-cluster centroids.

Level 2 (Macro-clustering):
  Groups micro-clusters whose centroids are similar (threshold=0.40) into
  macro-clusters. Scores each macro-cluster using Formula 15 (Hierarchical
  Convergence Score).

Consensus centroid:
  Instead of the plain mean, the final centroid of a macro-cluster is
  computed as an attention-weighted combination of its micro-centroids,
  biased toward the most confident micro-cluster.

Cross-type agreement bonus:
  If predictions from multiple dot types agree on a macro-cluster, a
  bonus is applied to its Formula 2 convergence score (Formula 13).
"""

import numpy as np
import ctypes
import os
from typing import List, Tuple, Dict, Optional, Set
import sys
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import (
    similarity_score, convergence_score, dominance_score,
    pairwise_similarity_matrix, hierarchical_convergence_score,
    cross_type_agreement, cluster_entropy, temporal_stability,
)


class MicroCluster:
    """Fine-grained cluster of very similar predictions."""
    def __init__(self, micro_id: int):
        self.micro_id    = micro_id
        self.predictions: List[np.ndarray] = []
        self.confidences: List[float] = []
        self.infos: List[Dict] = []
        self.centroid: Optional[np.ndarray] = None
        self.score: float = 0.0

    def add(self, p, c, info):
        self.predictions.append(p); self.confidences.append(c); self.infos.append(info)
        self.centroid = np.mean(np.stack(self.predictions), axis=0)

    def compute_score(self, alpha):
        self.score = convergence_score(self.predictions, self.confidences, alpha)

    @property
    def size(self): return len(self.predictions)


class Cluster:
    """
    Macro-cluster of micro-clusters. The primary convergence unit.
    """
    def __init__(self, cluster_id: int):
        self.cluster_id   = cluster_id
        self.predictions: List[np.ndarray] = []
        self.confidences: List[float] = []
        self.infos: List[Dict] = []
        self.centroid: Optional[np.ndarray] = None
        self.score: float = 0.0
        self._micro_clusters: List[MicroCluster] = []

    def add(self, p: np.ndarray, c: float, info: Dict):
        self.predictions.append(p); self.confidences.append(c); self.infos.append(info)
        self._recompute_centroid()

    def _recompute_centroid(self):
        if not self.predictions: return
        stack = np.stack(self.predictions)
        confs = np.array(self.confidences, np.float32)
        # Confidence-weighted centroid (more weight to confident predictions)
        w = confs / (confs.sum() + 1e-10)
        self.centroid = (w[:, None] * stack).sum(axis=0)

    def add_micro(self, mc: MicroCluster):
        """Absorb a micro-cluster into this macro-cluster."""
        self._micro_clusters.append(mc)
        self.predictions.extend(mc.predictions)
        self.confidences.extend(mc.confidences)
        self.infos.extend(mc.infos)
        self._recompute_centroid()

    def compute_score(self, alpha: float = 0.7, gamma: float = 0.3):
        """Score using Formula 15 (hierarchical) if micro-clusters available."""
        if len(self._micro_clusters) >= 2:
            centroids = [mc.centroid for mc in self._micro_clusters if mc.centroid is not None]
            scores    = [mc.score for mc in self._micro_clusters]
            if centroids and scores:
                self.score = hierarchical_convergence_score(centroids, scores, alpha, gamma)
                return
        self.score = convergence_score(self.predictions, self.confidences, alpha)

    def apply_cross_type_bonus(self, alpha: float = 0.7, bonus_weight: float = 0.15):
        """Formula 13: boost score if multiple dot types agree."""
        type_preds: Dict[str, List[np.ndarray]] = {}
        for p, info in zip(self.predictions, self.infos):
            t = info.get("dot_type", "unknown")
            type_preds.setdefault(t, []).append(p)
        if len(type_preds) < 2: return
        type_centroids = {t: np.mean(np.stack(ps), axis=0) for t, ps in type_preds.items()}
        cta = cross_type_agreement(type_centroids, alpha)
        self.score = self.score * (1.0 + bonus_weight * cta)

    @property
    def size(self) -> int: return len(self.predictions)

    @property
    def mean_confidence(self) -> float:
        return float(np.mean(self.confidences)) if self.confidences else 0.0

    @property
    def num_micro(self) -> int: return len(self._micro_clusters)

    def sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in self.infos:
            s = info.get("source", "unknown")
            counts[s] = counts.get(s, 0) + 1
        return counts

    def dot_types(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in self.infos:
            t = info.get("dot_type", "unknown")
            counts[t] = counts.get(t, 0) + 1
        return counts

    def __repr__(self):
        return (f"Cluster(id={self.cluster_id}, size={self.size}, "
                f"score={self.score:.4f}, micro={self.num_micro})")


class ConvergenceLayer:
    """
    Two-level hierarchical clustering for IECNN predictions.

    Stage 1 — Micro-clustering (threshold = micro_threshold):
      Groups very similar predictions into fine-grained micro-clusters.

    Stage 2 — Macro-clustering (threshold = macro_threshold):
      Groups micro-cluster centroids into macro-clusters.
      Scores each with Formula 15 (Hierarchical Convergence Score).
      Applies cross-type agreement bonus (Formula 13).
    """

    def __init__(self, micro_threshold: float = 0.60,
                 macro_threshold: float = 0.40,
                 alpha: float = 0.7,
                 gamma: float = 0.3,
                 dominance_threshold: float = 0.70,
                 cross_type_bonus: float = 0.15):
        self.micro_thresh   = micro_threshold
        self.macro_thresh   = macro_threshold
        self.alpha          = alpha
        self.gamma          = gamma
        self.dom_thresh     = dominance_threshold
        self.ct_bonus       = cross_type_bonus

    # ── Main entry point ─────────────────────────────────────────────

    def run(self, candidates: List[Tuple]) -> Tuple[List[Cluster], List[int]]:
        if not candidates: return [], []
        preds = [c[0] for c in candidates]
        confs = [c[1] for c in candidates]
        infos = [c[2] for c in candidates]

        # Stage 1: micro-clustering
        micro_clusters = self._micro_cluster(preds, confs, infos)
        for mc in micro_clusters:
            mc.compute_score(self.alpha)

        # Stage 2: macro-clustering over micro-cluster centroids
        macro_clusters = self._macro_cluster(micro_clusters)
        for cl in macro_clusters:
            cl.compute_score(self.alpha, self.gamma)
            cl.apply_cross_type_bonus(self.alpha, self.ct_bonus)

        macro_clusters.sort(key=lambda c: c.score, reverse=True)

        # Build assignment array (prediction index → macro cluster id)
        pred_to_micro: Dict[int, int] = {}
        for mc in micro_clusters:
            for i, info in enumerate(infos):
                if info in mc.infos and i not in pred_to_micro:
                    pred_to_micro[i] = mc.micro_id

        assign = [-1] * len(candidates)
        for cl in macro_clusters:
            for mc in cl._micro_clusters:
                for i, info in enumerate(infos):
                    if info in mc.infos:
                        assign[i] = cl.cluster_id

        return macro_clusters, assign

    # ── Stage 1: Micro-clustering ────────────────────────────────────

    def _micro_cluster(self, preds: List[np.ndarray], confs: List[float],
                       infos: List[Dict]) -> List[MicroCluster]:
        """
        Sequential greedy micro-clustering.
        A prediction joins an existing micro-cluster if its similarity to
        the cluster centroid exceeds micro_threshold.
        """
        clusters: List[MicroCluster] = []
        centroids: List[np.ndarray]  = []

        for i, (p, c, info) in enumerate(zip(preds, confs, infos)):
            best_cid, best_sim = -1, self.micro_thresh
            for cid, cent in enumerate(centroids):
                s = similarity_score(p, cent, self.alpha)
                if s > best_sim:
                    best_sim, best_cid = s, cid

            if best_cid == -1:
                mc = MicroCluster(len(clusters))
                mc.add(p, c, info)
                clusters.append(mc)
                centroids.append(p.copy())
            else:
                clusters[best_cid].add(p, c, info)
                centroids[best_cid] = clusters[best_cid].centroid

        return clusters

    # ── Stage 2: Macro-clustering ────────────────────────────────────

    def _macro_cluster(self, micros: List[MicroCluster]) -> List[Cluster]:
        """
        Cluster the micro-cluster centroids into macro-clusters.
        """
        if not micros: return []
        macro_list: List[Cluster] = []
        macro_centroids: List[np.ndarray] = []

        for mc in micros:
            if mc.centroid is None: continue
            best_cid, best_sim = -1, self.macro_thresh
            for cid, cent in enumerate(macro_centroids):
                s = similarity_score(mc.centroid, cent, self.alpha)
                if s > best_sim:
                    best_sim, best_cid = s, cid

            if best_cid == -1:
                cl = Cluster(len(macro_list))
                cl.add_micro(mc)
                macro_list.append(cl)
                macro_centroids.append(mc.centroid.copy())
            else:
                macro_list[best_cid].add_micro(mc)
                macro_centroids[best_cid] = macro_list[best_cid].centroid

        return macro_list

    # ── Metrics ──────────────────────────────────────────────────────

    def dominance(self, clusters: List[Cluster]) -> Tuple[float, bool]:
        if not clusters: return 0.0, False
        scores = [c.score for c in clusters]
        d = dominance_score(scores[0], scores)
        return d, d >= self.dom_thresh

    def entropy(self, clusters: List[Cluster]) -> float:
        """Formula 11: entropy of the cluster score distribution."""
        if not clusters: return 0.0
        return cluster_entropy([c.score for c in clusters])

    def stability(self, clusters: List[Cluster], prev_centroid: Optional[np.ndarray]) -> float:
        """Formula 12: similarity of top centroid to previous round."""
        if not clusters or prev_centroid is None: return 0.0
        curr = clusters[0].centroid
        if curr is None: return 0.0
        return temporal_stability(curr, prev_centroid, self.alpha)

    def summarize(self, clusters: List[Cluster],
                   prev_centroid: Optional[np.ndarray] = None) -> Dict:
        if not clusters:
            return {"num_clusters": 0, "num_micro": 0}
        dom, is_dom = self.dominance(clusters)
        return {
            "num_clusters":  len(clusters),
            "num_micro":     sum(c.num_micro for c in clusters),
            "top_score":     float(clusters[0].score),
            "top_size":      clusters[0].size,
            "dominance":     float(dom),
            "is_dominant":   is_dom,
            "entropy":       self.entropy(clusters),
            "stability":     self.stability(clusters, prev_centroid),
            "centroid":      clusters[0].centroid,
        }
