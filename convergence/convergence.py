"""
Convergence Layer — hierarchical two-level clustering with consensus.
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
    cross_type_agreement, cluster_entropy, temporal_stability, _fp,
    convergence_score_ultra
)

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "convergence_c.so")
    if os.path.exists(so_path):
        try:
            import ctypes
            _lib = ctypes.CDLL(so_path)
            _lib.greedy_cluster.restype = ctypes.c_int
            _lib.score_cluster_ultra.restype = ctypes.c_float
        except Exception:
            _lib = None
    return _lib

class MicroCluster:
    def __init__(self, micro_id: int):
        self.micro_id = micro_id
        self.predictions = []
        self.confidences = []
        self.infos = []
        self.centroid = None
        self.score = 0.0

    def add(self, p, c, info):
        self.predictions.append(p); self.confidences.append(c); self.infos.append(info)

        # Phase-Coded Aggregation (Complex Centroid)
        # Predictions 'p' are already complex-valued activations from NeuralDot.predict
        stack = np.stack(self.predictions).astype(np.complex64)

        # Constructive/Destructive Interference happens here via mean
        self.centroid = np.mean(stack, axis=0)

    def compute_score(self, alpha):
        self.score = convergence_score(self.predictions, self.confidences, alpha)

class Cluster:
    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.predictions = []
        self.confidences = []
        self.infos = []
        self.centroid = None
        self.score = 0.0
        self._micro_clusters = []

    def add(self, p, c, info):
        self.predictions.append(p); self.confidences.append(c); self.infos.append(info)
        self._recompute_centroid()

    def _recompute_centroid(self):
        if not self.predictions: return
        # Predictions are already complex-valued activations from NeuralDot.predict
        stack = np.stack(self.predictions).astype(np.complex64)
        confs = np.array(self.confidences, np.float32)

        # Weights normalized by confidence
        w = confs / (confs.sum() + 1e-10)

        # Constructive/Destructive Interference
        self.centroid = (w[:, None].astype(np.complex64) * stack).sum(axis=0)

    def add_micro(self, mc):
        self._micro_clusters.append(mc)
        self.predictions.extend(mc.predictions)
        self.confidences.extend(mc.confidences)
        self.infos.extend(mc.infos)
        self._recompute_centroid()

    def compute_score(self, alpha=0.7, gamma=0.3):
        if len(self._micro_clusters) >= 2:
            centroids = [mc.centroid for mc in self._micro_clusters if mc.centroid is not None]
            scores = [mc.score for mc in self._micro_clusters]
            if centroids and scores:
                self.score = hierarchical_convergence_score(centroids, scores, alpha, gamma)
                return
        self.score = convergence_score(self.predictions, self.confidences, alpha)

    def apply_cross_type_bonus(self, alpha=0.7, bonus_weight=0.15):
        type_preds = {}
        for p, info in zip(self.predictions, self.infos):
            t = info.get("dot_type", "unknown")
            type_preds.setdefault(t, []).append(p)
        if len(type_preds) < 2: return
        type_centroids = {t: np.mean(np.stack(ps), axis=0) for t, ps in type_preds.items()}
        cta = cross_type_agreement(type_centroids, alpha)
        self.score = self.score * (1.0 + bonus_weight * cta)

    @property
    def size(self): return len(self.predictions)
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

    def modalities(self) -> Dict[str, int]:
        """Distribution of modalities within this cluster."""
        counts: Dict[str, int] = {}
        for info in self.infos:
            m = info.get("modality", "unknown")
            counts[m] = counts.get(m, 0) + 1
        return counts

    def mean_phase(self) -> Tuple[float, float]:
        """
        Circular mean and concentration of member phases.

        Returns (mean_phase_radians, concentration in [0, 1]).
        Concentration of 1.0 means all members agree on a single phase;
        0.0 means uniformly spread. Returns (0.0, 0.0) if no members carry
        a 'phase' key (i.e. phase coding is disabled or this is legacy data).
        """
        re_sum = 0.0
        im_sum = 0.0
        n = 0
        for info in self.infos:
            ph = info.get("phase")
            if ph is None:
                continue
            re_sum += float(np.cos(ph))
            im_sum += float(np.sin(ph))
            n += 1
        if n == 0:
            return (0.0, 0.0)
        mean = float(np.arctan2(im_sum, re_sum))
        conc = float(np.sqrt(re_sum * re_sum + im_sum * im_sum) / n)
        return (mean, conc)

    def __repr__(self):
        return (f"Cluster(id={self.cluster_id}, size={self.size}, "
                f"score={self.score:.4f}, micro={self.num_micro})")


class ConvergenceLayer:
    def __init__(self, micro_threshold=0.60, macro_threshold=0.40, alpha=0.7, gamma=0.3, dominance_threshold=0.70, cross_type_bonus=0.15):
        self.micro_thresh = micro_threshold
        self.macro_thresh = macro_threshold
        self.alpha = alpha
        self.gamma = gamma
        self.dom_thresh = dominance_threshold
        self.ct_bonus = cross_type_bonus

    def run(self, candidates):
        if not candidates: return [], []
        preds = [c[0] for c in candidates]
        confs = [c[1] for c in candidates]
        infos = [c[2] for c in candidates]

        # Multi-modal boost: slightly lower threshold for cross-modal agreement
        # to encourage the emergence of unified concepts.
        is_mixed = len(set(info.get("modality", "unknown") for info in infos)) > 1
        current_micro_thresh = self.micro_thresh * 0.9 if is_mixed else self.micro_thresh

        # Stage 1: micro-clustering
        micro_clusters = self._micro_cluster(preds, confs, infos, current_micro_thresh)
        for mc in micro_clusters:
            mc.compute_score(self.alpha)

        # Stage 2: macro-clustering over micro-cluster centroids
        macro_clusters = self._macro_cluster(micro_clusters)
        for cl in macro_clusters:
            cl.compute_score(self.alpha, self.gamma)
            cl.apply_cross_type_bonus(self.alpha, self.ct_bonus)
        macro_clusters.sort(key=lambda c: c.score, reverse=True)
        assign = [-1] * len(candidates)
        for cl in macro_clusters:
            for mc in cl._micro_clusters:
                for i, info in enumerate(infos):
                    if info in mc.infos: assign[i] = cl.cluster_id
        return macro_clusters, assign

    # ── Stage 1: Micro-clustering ────────────────────────────────────

    def _micro_cluster(self, preds: List[np.ndarray], confs: List[float],
                       infos: List[Dict], threshold: Optional[float] = None) -> List[MicroCluster]:
        """
        Sequential greedy micro-clustering.
        C-accelerated for performance.
        """
        if threshold is None: threshold = self.micro_thresh
        n = len(preds)
        if n == 0: return []

        lib = _load_lib()
        if lib and hasattr(lib, "greedy_cluster"):
            stk = np.ascontiguousarray(np.real(np.stack(preds)), np.float32)
            dim = stk.shape[1]
            assign = np.zeros(n, dtype=np.int32)

            num_c = lib.greedy_cluster(
                stk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n), ctypes.c_int(dim),
                ctypes.c_float(self.alpha), ctypes.c_float(threshold),
                assign.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )

            clusters = [MicroCluster(i) for i in range(num_c)]
            for i in range(n):
                clusters[assign[i]].add(preds[i], confs[i], infos[i])
            return clusters

        # Fallback
        clusters: List[MicroCluster] = []
        centroids: List[np.ndarray]  = []

        for i, (p, c, info) in enumerate(zip(preds, confs, infos)):
            best_cid, best_sim = -1, threshold
            for cid, cent in enumerate(centroids):
                s = similarity_score(p, cent, self.alpha)
                if s > best_sim: best_sim, best_cid = s, cid
            if best_cid == -1:
                mc = MicroCluster(len(clusters))
                mc.add(p, c, info)
                clusters.append(mc); centroids.append(p.copy())
            else:
                clusters[best_cid].add(p, c, info)
                centroids[best_cid] = clusters[best_cid].centroid
        return clusters

    def _macro_cluster(self, micros):
        if not micros: return []
        n = len(micros)
        centroids_list = [mc.centroid for mc in micros if mc.centroid is not None]
        if not centroids_list: return []

        lib = _load_lib()
        if lib and hasattr(lib, "greedy_cluster"):
            stk = np.ascontiguousarray(np.real(np.stack(centroids_list)), np.float32)
            n_valid = len(centroids_list)
            dim = stk.shape[1]
            assign = np.zeros(n_valid, dtype=np.int32)

            num_c = lib.greedy_cluster(
                stk.ctypes.data_as(ctypes.POINTER(ctypes.c_float)),
                ctypes.c_int(n_valid), ctypes.c_int(dim),
                ctypes.c_float(self.alpha), ctypes.c_float(self.macro_thresh),
                assign.ctypes.data_as(ctypes.POINTER(ctypes.c_int))
            )

            macro_list = [Cluster(i) for i in range(num_c)]
            for i, mc in enumerate(micros):
                if mc.centroid is not None:
                    macro_list[assign[i]].add_micro(mc)
            return macro_list

        macro_list = []
        macro_centroids = []
        for mc in micros:
            if mc.centroid is None: continue
            best_cid, best_sim = -1, self.macro_thresh
            for cid, cent in enumerate(macro_centroids):
                s = similarity_score(mc.centroid, cent, self.alpha)
                if s > best_sim: best_sim, best_cid = s, cid
            if best_cid == -1:
                cl = Cluster(len(macro_list))
                cl.add_micro(mc)
                macro_list.append(cl); macro_centroids.append(mc.centroid.copy())
            else:
                macro_list[best_cid].add_micro(mc)
                macro_centroids[best_cid] = macro_list[best_cid].centroid
        return macro_list

    def dominance(self, clusters):
        if not clusters: return 0.0, False
        scores = [c.score for c in clusters]
        d = dominance_score(scores[0], scores)
        return d, d >= self.dom_thresh

    def entropy(self, clusters):
        return cluster_entropy([c.score for c in clusters]) if clusters else 0.0

    def stability(self, clusters, prev_centroid):
        if not clusters or prev_centroid is None: return 0.0
        return temporal_stability(clusters[0].centroid, prev_centroid, self.alpha)

    def summarize(self, clusters, prev_centroid=None):
        if not clusters: return {"num_clusters": 0, "num_micro": 0}
        dom, is_dom = self.dominance(clusters)
        return {
            "num_clusters": len(clusters), "num_micro": sum(c.num_micro for c in clusters),
            "top_score": float(clusters[0].score), "dominance": float(dom),
            "is_dominant": is_dom, "entropy": self.entropy(clusters),
            "stability": self.stability(clusters, prev_centroid),
            "centroid": clusters[0].centroid
        }

    def run_ultra(self, candidates, repellent=None):
        clusters, assign = self.run(candidates)
        for cl in clusters:
            cl.score = convergence_score_ultra(cl.predictions, cl.confidences, self.alpha, repellent=repellent)
        clusters.sort(key=lambda c: c.score, reverse=True)
        return clusters, assign
