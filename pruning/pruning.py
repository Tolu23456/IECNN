"""
Pruning Layer — Three-stage candidate and cluster pruning.

Stage 1 — Early / Soft Filtering  (before full convergence)
  Goal: prevent the candidate pool from exploding.
  - Drop low-confidence predictions
  - Remove near-duplicates (sim > tight threshold)
  - Cap AIM variants per dot at max_variants

Stage 2 — Mid-stage / Cluster Compression  (as grouping begins)
  Goal: reduce redundancy before final scoring.
  - Merge clusters whose centroids are very similar
  - Represent each merged group with a combined centroid

Stage 3 — Final / Hard Selection  (after convergence scoring)
  Goal: enforce clear convergence.
  - Keep only clusters above the dynamic score threshold
  - Always keep at least min_keep clusters
"""

import numpy as np
from typing import List, Tuple, Dict
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score
from convergence.convergence import Cluster


class PruningLayer:
    def __init__(self, soft_conf: float = 0.08, near_dup: float = 0.95,
                 max_aim_per_dot: int = 3, merge_thresh: float = 0.80,
                 hard_thresh: float = 0.05, min_keep: int = 1, alpha: float = 0.7):
        self.soft_conf    = soft_conf
        self.near_dup     = near_dup
        self.max_aim      = max_aim_per_dot
        self.merge_thresh = merge_thresh
        self.hard_thresh  = hard_thresh
        self.min_keep     = min_keep
        self.alpha        = alpha

    # ── Stage 1 ──────────────────────────────────────────────────────

    def stage1(self, candidates: List[Tuple]) -> Tuple[List[Tuple], Dict]:
        if not candidates: return [], {}
        # Confidence filter
        kept = [(p,c,i) for p,c,i in candidates if c >= self.soft_conf]
        dropped_conf = len(candidates) - len(kept)

        # AIM cap per dot
        aim_counts: Dict[int, int] = {}
        capped = []
        dropped_aim = 0
        for p, c, info in kept:
            dot_id = info.get("original_dot_id", info.get("dot_id", -1))
            if info.get("source", "original") == "original":
                aim_counts[dot_id] = 0
                capped.append((p, c, info))
            else:
                cnt = aim_counts.get(dot_id, 0)
                if cnt < self.max_aim:
                    aim_counts[dot_id] = cnt + 1
                    capped.append((p, c, info))
                else:
                    dropped_aim += 1

        # Near-duplicate removal
        deduped = []
        dropped_dup = 0
        for p, c, info in capped:
            if not any(similarity_score(p, q, self.alpha) > self.near_dup for q, _, _ in deduped):
                deduped.append((p, c, info))
            else:
                dropped_dup += 1

        return deduped, {"dropped_conf": dropped_conf, "dropped_aim": dropped_aim,
                          "dropped_dup": dropped_dup, "out": len(deduped)}

    # ── Stage 2 ──────────────────────────────────────────────────────

    def stage2(self, clusters: List[Cluster]) -> Tuple[List[Cluster], Dict]:
        if len(clusters) <= 1: return clusters, {"merged": 0}
        merged_flags = [False] * len(clusters)
        result = []
        merge_count = 0
        for i, ci in enumerate(clusters):
            if merged_flags[i]: continue
            group = [i]
            for j in range(i+1, len(clusters)):
                if merged_flags[j]: continue
                if ci.centroid is not None and clusters[j].centroid is not None:
                    s = similarity_score(ci.centroid, clusters[j].centroid, self.alpha)
                    if s > self.merge_thresh:
                        group.append(j); merged_flags[j] = True
            if len(group) == 1:
                result.append(ci)
            else:
                nc = Cluster(ci.cluster_id)
                for idx in group:
                    for p,c,info in zip(clusters[idx].predictions, clusters[idx].confidences, clusters[idx].infos):
                        nc.add(p, c, info)
                nc.compute_score(self.alpha)
                result.append(nc)
                merge_count += len(group) - 1
        result.sort(key=lambda c: c.score, reverse=True)
        return result, {"merged": merge_count, "out": len(result)}

    # ── Stage 3 ──────────────────────────────────────────────────────

    def stage3(self, clusters: List[Cluster]) -> Tuple[List[Cluster], Dict]:
        if not clusters: return [], {}
        max_s = max(c.score for c in clusters)
        thr   = max(self.hard_thresh, max_s * 0.1)
        kept  = [c for c in clusters if c.score >= thr]
        if len(kept) < self.min_keep: kept = clusters[:self.min_keep]
        return kept, {"threshold": float(thr), "kept": len(kept),
                      "discarded": len(clusters) - len(kept)}

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, candidates: List[Tuple], clusters: List[Cluster]) -> Tuple[List[Tuple], List[Cluster], Dict]:
        filt, s1 = self.stage1(candidates)
        comp, s2 = self.stage2(clusters)
        surv, s3 = self.stage3(comp)
        return filt, surv, {"s1": s1, "s2": s2, "s3": s3}
