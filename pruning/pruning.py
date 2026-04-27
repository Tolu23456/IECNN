"""
Pruning Layer — three-stage candidate and cluster pruning.
"""

import numpy as np
import ctypes
from typing import List, Tuple, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score, _fp
from convergence.convergence import Cluster

# ── Load C shared library ────────────────────────────────────────────
_lib = None

def _load_lib():
    global _lib
    if _lib is not None:
        return _lib
    here = os.path.dirname(os.path.abspath(__file__))
    so_path = os.path.join(here, "pruning_c.so")
    if os.path.exists(so_path):
        try:
            _lib = ctypes.CDLL(so_path)
            _lib.deduplicate_fast.restype = ctypes.c_int
        except Exception:
            _lib = None
    return _lib

class PruningLayer:
    def __init__(self, soft_conf=0.06, near_dup=0.92, max_aim_per_dot=4, merge_thresh=0.80, hard_thresh=0.04, floor_fraction=0.08, min_keep=1, min_survivors=30, alpha=0.7):
        self.soft_conf = soft_conf
        self.near_dup = near_dup
        self.max_aim = max_aim_per_dot
        self.merge_thresh = merge_thresh
        self.hard_thresh = hard_thresh
        self.floor_frac = floor_fraction
        self.min_keep = min_keep
        self.min_survivors = min_survivors
        self.alpha = alpha

    def stage1(self, candidates, dominance=0.0):
        if not candidates: return [], {}
        lib = _load_lib()
        conf_thresh = self.soft_conf * (1.0 + 0.5 * dominance)
        kept = [c for c in candidates if c[1] >= conf_thresh]

        # Near-duplicate removal
        deduped = []
        if lib and kept:
            n_cap = len(kept)
            dim = len(kept[0][0])
            # Explicitly take real part for C-accelerated deduplication
            stk = np.ascontiguousarray(np.real(np.stack([x[0] for x in kept])), np.float32)
            kept_indices = np.zeros(n_cap, dtype=np.int32)
            num_kept = lib.deduplicate_fast(_fp(stk)[0], ctypes.c_int(n_cap), ctypes.c_int(dim),
                                           ctypes.c_float(self.near_dup), ctypes.c_float(self.alpha),
                                           kept_indices.ctypes.data_as(ctypes.POINTER(ctypes.c_int)))
            deduped = [kept[idx] for idx in kept_indices[:num_kept]]
        else:
            for c in kept:
                if not any(similarity_score(c[0], d[0], self.alpha) > self.near_dup for d in deduped):
                    deduped.append(c)

        if len(deduped) < self.min_survivors:
            # deduplicated candidates are already in 'deduped', we just need to pick the top remaining
            # By using index-based tracking, we avoid the array truthiness issue.
            seen_indices = set()
            # If we used the C library, we know which indices we kept.
            # But the 'c not in deduped' is the problem because 'c' contains a numpy array.
            # Let's use a simpler and safer approach:

            # 1. Identify all candidate indices that were NOT picked for deduped
            picked_candidates = set(id(c) for c in deduped)
            remaining = [c for c in candidates if id(c) not in picked_candidates]
            remaining.sort(key=lambda x: x[1], reverse=True)
            deduped.extend(remaining[:self.min_survivors - len(deduped)])

        return deduped, {"in": len(candidates), "out": len(deduped)}

    def stage2(self, clusters):
        return clusters, {"merged": 0, "out": len(clusters)}

    def stage3(self, clusters, dominance=0.0):
        if not clusters: return [], {}
        max_s = max(c.score for c in clusters)
        thr = max(self.hard_thresh * (1.0 + 2.0 * dominance), max_s * self.floor_frac)
        kept = [c for c in clusters if c.score >= thr]
        if len(kept) < self.min_keep: kept = clusters[:self.min_keep]
        return kept, {"threshold": float(thr), "kept": len(kept)}

    def run(self, candidates, clusters, dominance=0.0):
        filt, s1 = self.stage1(candidates, dominance)
        surv, s3 = self.stage3(clusters, dominance)
        return filt, surv, {"s1": s1, "s3": s3}
