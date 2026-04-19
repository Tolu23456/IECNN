import numpy as np
from typing import List, Tuple, Dict
from .formulas import similarity_score
from .convergence import Cluster


class PruningLayer:
    """
    Three-stage pruning pipeline.

    Stage 1 — Early / Soft Filtering (before full convergence)
      Goal: prevent the candidate pool from exploding.
      - Drop low-confidence predictions
      - Remove near-duplicates within a tight similarity threshold
      - Cap AIM variants per dot

    Stage 2 — Mid-stage / Cluster Compression (as grouping begins)
      Goal: reduce redundancy before final scoring.
      - Merge clusters with high inter-cluster similarity
      - Represent each merged group with its centroid

    Stage 3 — Final / Hard Selection (after convergence scoring)
      Goal: enforce clear convergence.
      - Keep only clusters above the dominance cutoff
      - Discard weak or inconsistent clusters
    """

    def __init__(
        self,
        soft_confidence_threshold: float = 0.1,
        near_duplicate_threshold: float = 0.95,
        max_aim_variants_per_dot: int = 3,
        cluster_merge_threshold: float = 0.8,
        hard_score_threshold: float = 0.05,
        min_clusters_to_keep: int = 1,
        alpha: float = 0.7,
    ):
        self.soft_confidence_threshold = soft_confidence_threshold
        self.near_duplicate_threshold = near_duplicate_threshold
        self.max_aim_variants_per_dot = max_aim_variants_per_dot
        self.cluster_merge_threshold = cluster_merge_threshold
        self.hard_score_threshold = hard_score_threshold
        self.min_clusters_to_keep = min_clusters_to_keep
        self.alpha = alpha

    def stage1_soft_filter(
        self,
        candidates: List[Tuple[np.ndarray, float, Dict]],
    ) -> Tuple[List[Tuple[np.ndarray, float, Dict]], Dict]:
        """
        Stage 1 — Early Soft Filtering.

        Removes:
          - Predictions with confidence below the soft threshold
          - Near-duplicate predictions (sim > near_duplicate_threshold)
          - Excess AIM variants beyond max_aim_variants_per_dot per dot
        """
        if not candidates:
            return [], {"removed_low_conf": 0, "removed_duplicates": 0, "removed_excess_aim": 0}

        conf_filtered = []
        removed_low_conf = 0
        for pred, conf, info in candidates:
            if conf >= self.soft_confidence_threshold:
                conf_filtered.append((pred, conf, info))
            else:
                removed_low_conf += 1

        aim_cap_filtered = []
        removed_excess_aim = 0
        aim_count_per_dot: Dict[int, int] = {}
        for pred, conf, info in conf_filtered:
            dot_id = info.get("original_dot_id", info.get("dot_id", -1))
            source = info.get("source", "original")
            if source == "original":
                aim_count_per_dot[dot_id] = 0
                aim_cap_filtered.append((pred, conf, info))
            else:
                count = aim_count_per_dot.get(dot_id, 0)
                if count < self.max_aim_variants_per_dot:
                    aim_count_per_dot[dot_id] = count + 1
                    aim_cap_filtered.append((pred, conf, info))
                else:
                    removed_excess_aim += 1

        deduped = []
        removed_duplicates = 0
        for i, (pred_i, conf_i, info_i) in enumerate(aim_cap_filtered):
            is_duplicate = False
            for pred_j, _, _ in deduped:
                s = similarity_score(pred_i, pred_j, self.alpha)
                if s > self.near_duplicate_threshold:
                    is_duplicate = True
                    break
            if not is_duplicate:
                deduped.append((pred_i, conf_i, info_i))
            else:
                removed_duplicates += 1

        stats = {
            "input_count": len(candidates),
            "output_count": len(deduped),
            "removed_low_conf": removed_low_conf,
            "removed_excess_aim": removed_excess_aim,
            "removed_duplicates": removed_duplicates,
        }
        return deduped, stats

    def stage2_cluster_compression(
        self, clusters: List[Cluster]
    ) -> Tuple[List[Cluster], Dict]:
        """
        Stage 2 — Mid-stage Cluster Compression.

        Merges clusters with high inter-cluster centroid similarity.
        Represents the merged group with a combined centroid.
        """
        if not clusters:
            return [], {"merged": 0, "output_clusters": 0}

        if len(clusters) == 1:
            return clusters, {"merged": 0, "output_clusters": 1}

        merged_flags = [False] * len(clusters)
        merged_clusters: List[Cluster] = []
        merge_count = 0

        for i in range(len(clusters)):
            if merged_flags[i]:
                continue

            base_cluster = clusters[i]
            to_merge = [i]

            for j in range(i + 1, len(clusters)):
                if merged_flags[j]:
                    continue
                if base_cluster.centroid is None or clusters[j].centroid is None:
                    continue
                s = similarity_score(base_cluster.centroid, clusters[j].centroid, self.alpha)
                if s > self.cluster_merge_threshold:
                    to_merge.append(j)
                    merged_flags[j] = True

            if len(to_merge) == 1:
                merged_clusters.append(base_cluster)
            else:
                new_cluster = Cluster(cluster_id=base_cluster.cluster_id)
                for idx in to_merge:
                    for pred, conf, info in zip(
                        clusters[idx].predictions,
                        clusters[idx].confidences,
                        clusters[idx].infos,
                    ):
                        new_cluster.add(pred, conf, info)
                new_cluster.compute_score(self.alpha)
                merged_clusters.append(new_cluster)
                merge_count += len(to_merge) - 1

        merged_clusters.sort(key=lambda c: c.score, reverse=True)

        stats = {
            "input_clusters": len(clusters),
            "output_clusters": len(merged_clusters),
            "merged": merge_count,
        }
        return merged_clusters, stats

    def stage3_hard_selection(
        self, clusters: List[Cluster]
    ) -> Tuple[List[Cluster], Dict]:
        """
        Stage 3 — Final Hard Selection.

        Keeps only clusters above the score threshold.
        Always retains at least min_clusters_to_keep clusters.
        """
        if not clusters:
            return [], {"kept": 0, "discarded": 0}

        all_scores = [c.score for c in clusters]
        max_score = max(all_scores)

        dynamic_threshold = max(
            self.hard_score_threshold,
            max_score * 0.1,
        )

        kept = [c for c in clusters if c.score >= dynamic_threshold]

        if len(kept) < self.min_clusters_to_keep:
            kept = clusters[:self.min_clusters_to_keep]

        discarded = len(clusters) - len(kept)
        stats = {
            "input_clusters": len(clusters),
            "kept": len(kept),
            "discarded": discarded,
            "threshold_used": float(dynamic_threshold),
        }
        return kept, stats

    def run(
        self,
        candidates: List[Tuple[np.ndarray, float, Dict]],
        clusters: List[Cluster],
    ) -> Tuple[
        List[Tuple[np.ndarray, float, Dict]],
        List[Cluster],
        Dict,
    ]:
        """
        Run the full three-stage pruning pipeline.

        Returns:
          filtered_candidates: candidates surviving Stage 1
          surviving_clusters: clusters surviving Stages 2 and 3
          stats: dict summarizing each stage's pruning activity
        """
        filtered_candidates, s1_stats = self.stage1_soft_filter(candidates)
        compressed_clusters, s2_stats = self.stage2_cluster_compression(clusters)
        surviving_clusters, s3_stats = self.stage3_hard_selection(compressed_clusters)

        stats = {
            "stage1": s1_stats,
            "stage2": s2_stats,
            "stage3": s3_stats,
            "total_candidates_in": len(candidates),
            "total_candidates_out": len(filtered_candidates),
            "total_clusters_in": len(clusters),
            "total_clusters_out": len(surviving_clusters),
        }
        return filtered_candidates, surviving_clusters, stats
