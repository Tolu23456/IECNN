"""
Pruning Layer — three-stage candidate and cluster pruning.

Stage 1 — Soft Filter (before full convergence):
  - Drop low-confidence predictions (below soft_conf threshold)
  - Remove near-duplicates (similarity > near_dup threshold)
  - Cap AIM variants per original dot (max_aim)
  - Keep at least min_survivors regardless of thresholds

Stage 2 — Cluster Compression (after micro-clustering):
  - Merge macro-clusters whose centroids are very similar (merge_thresh)
  - Represent merged group with a combined consensus centroid

Stage 3 — Hard Selection (after scoring):
  - Dynamic threshold: max(hard_thresh, top_score * floor_fraction)
  - Always keep at least min_keep clusters
  - Track which pruned clusters were close (for feedback)

Dynamic threshold adaptation:
  The hard threshold rises as convergence progresses (when dominance is
  high, keep fewer clusters; when exploratory, keep more diversity).
"""

import numpy as np
from typing import List, Tuple, Dict, Optional
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score
from convergence.convergence import Cluster


class PruningLayer:
    def __init__(self,
                 soft_conf: float        = 0.06,
                 near_dup: float         = 0.92,
                 max_aim_per_dot: int    = 4,
                 merge_thresh: float     = 0.80,
                 hard_thresh: float      = 0.04,
                 floor_fraction: float   = 0.08,
                 min_keep: int           = 1,
                 min_survivors: int      = 30,
                 alpha: float            = 0.7):
        self.soft_conf    = soft_conf
        self.near_dup     = near_dup
        self.max_aim      = max_aim_per_dot
        self.merge_thresh = merge_thresh
        self.hard_thresh  = hard_thresh
        self.floor_frac   = floor_fraction
        self.min_keep     = min_keep
        self.min_survivors = min_survivors
        self.alpha        = alpha

    # ── Stage 1: Soft Filtering ──────────────────────────────────────

    def stage1(self, candidates: List[Tuple],
               dominance: float = 0.0) -> Tuple[List[Tuple], Dict]:
        if not candidates: return [], {}

        # Adaptive confidence threshold: relax when convergence is weak
        conf_thresh = self.soft_conf * (1.0 + 0.5 * dominance)

        # Confidence filter
        kept = [(p, c, i) for p, c, i in candidates if c >= conf_thresh]
        dropped_conf = len(candidates) - len(kept)

        # AIM cap per original dot
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

        # Near-duplicate removal (sequential, O(n²) but n is bounded)
        deduped = []
        dropped_dup = 0
        for p, c, info in capped:
            is_dup = any(
                similarity_score(p, q, self.alpha) > self.near_dup
                for q, _, _ in deduped
            )
            if is_dup:
                dropped_dup += 1
            else:
                deduped.append((p, c, info))

        # Safety net: never drop below min_survivors (relax conf requirement)
        if len(deduped) < self.min_survivors:
            extras = [(p, c, i) for p, c, i in candidates if (p, c, i) not in deduped]
            extras.sort(key=lambda x: x[1], reverse=True)
            need = self.min_survivors - len(deduped)
            deduped.extend(extras[:need])

        return deduped, {
            "in":           len(candidates),
            "dropped_conf": dropped_conf,
            "dropped_aim":  dropped_aim,
            "dropped_dup":  dropped_dup,
            "out":          len(deduped),
        }

    # ── Stage 2: Cluster Compression ────────────────────────────────

    def stage2(self, clusters: List[Cluster]) -> Tuple[List[Cluster], Dict]:
        if len(clusters) <= 1: return clusters, {"merged": 0, "out": len(clusters)}
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
                # Create merged cluster
                from convergence.convergence import Cluster as CL
                nc = CL(ci.cluster_id)
                for idx in group:
                    for mc in clusters[idx]._micro_clusters:
                        nc.add_micro(mc)
                nc.compute_score(self.alpha)
                result.append(nc)
                merge_count += len(group) - 1
        result.sort(key=lambda c: c.score, reverse=True)
        return result, {"merged": merge_count, "out": len(result)}

    # ── Stage 3: Hard Selection ──────────────────────────────────────

    def stage3(self, clusters: List[Cluster],
               dominance: float = 0.0) -> Tuple[List[Cluster], Dict]:
        if not clusters: return [], {}
        max_s = max(c.score for c in clusters)
        # Dynamic threshold: rises with dominance (keep fewer when converging)
        dyn_hard = self.hard_thresh * (1.0 + 2.0 * dominance)
        thr = max(dyn_hard, max_s * self.floor_frac)
        kept = [c for c in clusters if c.score >= thr]
        if len(kept) < self.min_keep:
            kept = clusters[:self.min_keep]
        discarded_scores = [c.score for c in clusters if c not in kept]
        return kept, {
            "threshold":  float(thr),
            "kept":       len(kept),
            "discarded":  len(clusters) - len(kept),
            "near_miss":  sum(1 for s in discarded_scores if s >= thr * 0.8),
        }

    # ── Full pipeline ─────────────────────────────────────────────────

    def run(self, candidates: List[Tuple], clusters: List[Cluster],
            dominance: float = 0.0) -> Tuple[List[Tuple], List[Cluster], Dict]:
        filt, s1 = self.stage1(candidates, dominance)
        comp, s2 = self.stage2(clusters)
        surv, s3 = self.stage3(comp, dominance)
        return filt, surv, {"s1": s1, "s2": s2, "s3": s3}
