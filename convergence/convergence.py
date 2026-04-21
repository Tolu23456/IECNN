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
        self.centroid = np.mean(np.stack(self.predictions), axis=0)

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
        stack = np.stack(self.predictions)
        confs = np.array(self.confidences, np.float32)
        w = confs / (confs.sum() + 1e-10)
        self.centroid = (w[:, None] * stack).sum(axis=0)

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
    def num_micro(self): return len(self._micro_clusters)

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
        micro_clusters = self._micro_cluster(preds, confs, infos)
        for mc in micro_clusters: mc.compute_score(self.alpha)
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

    def _micro_cluster(self, preds, confs, infos, threshold=None):
        if threshold is None: threshold = self.micro_thresh
        lib = _load_lib()
        if lib and preds:
            n = len(preds)
            dim = len(preds[0])
            stk = np.ascontiguousarray(np.stack(preds), np.float32)
            assigns = np.zeros(n, dtype=np.int32)
            cents_buf = np.zeros((n, dim), dtype=np.float32)
            num_c = lib.greedy_cluster(_fp(stk)[0], ctypes.c_int(n), ctypes.c_int(dim),
                                      ctypes.c_float(threshold), ctypes.c_float(self.alpha),
                                      assigns.ctypes.data_as(ctypes.POINTER(ctypes.c_int)),
                                      _fp(cents_buf)[0])
            clusters = [MicroCluster(i) for i in range(num_c)]
            for i, cid in enumerate(assigns): clusters[cid].add(preds[i], confs[i], infos[i])
            return clusters

        # Fallback
        clusters = []
        centroids = []
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
