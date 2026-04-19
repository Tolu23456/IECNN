"""
IECNN Evaluation Metrics

Measures the quality, diversity, and stability of IECNN runs.
All metrics are derived from the architecture's own representations;
no external labels or ground truth are required.

Metrics defined here:
  - cluster_entropy         (Formula 11) — certainty in the score distribution
  - temporal_stability      (Formula 12) — how much the top cluster moves between rounds
  - prediction_diversity    — average pairwise dissimilarity among all candidates
  - cross_type_agreement    (Formula 13) — agreement between dots of different types
  - agreement_rate          — fraction of predictions in the top cluster
  - dot_specialization      (Formula 10) — consistency of individual dot outputs
  - convergence_quality     — composite quality score for a run
"""

import numpy as np
from typing import List, Optional, Dict, Tuple
from dataclasses import dataclass, field
import sys, os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from formulas.formulas import similarity_score, pairwise_similarity_matrix, dominance_score


@dataclass
class RunMetrics:
    """All metrics for a single IECNN run."""
    cluster_entropy:       float = 0.0
    temporal_stability:    float = 0.0
    prediction_diversity:  float = 0.0
    cross_type_agreement:  float = 0.0
    agreement_rate:        float = 0.0
    dot_specialization:    float = 0.0
    convergence_quality:   float = 0.0
    dominance:             float = 0.0
    num_rounds:            int   = 0
    num_clusters:          int   = 0
    stop_reason:           str   = ""
    per_round:             list  = field(default_factory=list)

    def __repr__(self):
        return (
            f"RunMetrics(quality={self.convergence_quality:.3f}, "
            f"entropy={self.cluster_entropy:.3f}, "
            f"stability={self.temporal_stability:.3f}, "
            f"diversity={self.prediction_diversity:.3f}, "
            f"dominance={self.dominance:.3f}, "
            f"rounds={self.num_rounds})"
        )


class IECNNMetrics:
    """Computes quality metrics for IECNN runs."""

    def __init__(self, alpha: float = 0.7):
        self.alpha = alpha

    # ── Formula 11: Cluster Entropy ──────────────────────────────────
    def cluster_entropy(self, scores: List[float]) -> float:
        """
        H_C = -Σ_k [C(k)/Z] * log[C(k)/Z]
        Low entropy → clear winner. High entropy → confused/exploratory.
        Returns entropy normalized to [0, 1] by dividing by log(n).
        """
        if not scores: return 0.0
        s = np.array(scores, np.float32); s = np.clip(s, 0.0, None)
        Z = s.sum()
        if Z < 1e-10: return 1.0  # maximum uncertainty
        p = s / Z
        p = p[p > 1e-10]
        H = float(-np.sum(p * np.log(p)))
        H_max = np.log(len(scores))
        return float(H / H_max) if H_max > 1e-10 else 0.0

    # ── Formula 12: Temporal Stability ───────────────────────────────
    def temporal_stability(self, centroid_prev: Optional[np.ndarray],
                           centroid_curr: Optional[np.ndarray]) -> float:
        """
        TS(t) = S(centroid_t, centroid_{t-1})
        Returns 0 if either centroid is missing, 1 if identical.
        """
        if centroid_prev is None or centroid_curr is None: return 0.0
        return similarity_score(centroid_curr, centroid_prev, self.alpha)

    # ── Prediction Diversity ──────────────────────────────────────────
    def prediction_diversity(self, predictions: List[np.ndarray]) -> float:
        """
        Average pairwise *dissimilarity* among candidate predictions.
        Diversity = 1 - mean(pairwise_similarity).
        High diversity = exploratory pool. Low diversity = focused pool.
        """
        if len(predictions) < 2: return 0.0
        # Sample up to 80 predictions for speed
        sample = predictions[:80]
        sim_mat = pairwise_similarity_matrix(sample, self.alpha)
        n = len(sample)
        # Exclude diagonal
        mask = ~np.eye(n, dtype=bool)
        return float(1.0 - np.mean(sim_mat[mask]))

    # ── Formula 13: Cross-Type Agreement ─────────────────────────────
    def cross_type_agreement(self, type_centroids: Dict[str, np.ndarray]) -> float:
        """
        CDA = Σ_{a≠b} S(centroid_a, centroid_b) / num_pairs
        High CDA: different dot types converge on the same answer.
        """
        keys = list(type_centroids.keys())
        if len(keys) < 2: return 0.0
        total, count = 0.0, 0
        for i in range(len(keys)):
            for j in range(i+1, len(keys)):
                total += similarity_score(type_centroids[keys[i]], type_centroids[keys[j]], self.alpha)
                count += 1
        return total / count if count > 0 else 0.0

    # ── Agreement Rate ────────────────────────────────────────────────
    def agreement_rate(self, top_cluster_size: int, total_candidates: int) -> float:
        """Fraction of all candidates that agreed (landed in top cluster)."""
        if total_candidates == 0: return 0.0
        return min(1.0, top_cluster_size / total_candidates)

    # ── Formula 10: Dot Specialization ───────────────────────────────
    def dot_specialization(self, predictions_per_dot: Dict[int, List[np.ndarray]]) -> float:
        """
        S_spec(d) = mean pairwise similarity of a dot's own predictions.
        Averaged over all dots.
        High score: each dot consistently predicts the same thing (specialized).
        """
        scores = []
        for dot_id, preds in predictions_per_dot.items():
            if len(preds) < 2:
                scores.append(1.0); continue
            sim = pairwise_similarity_matrix(preds[:10], self.alpha)
            n = len(preds[:10])
            mask = ~np.eye(n, dtype=bool)
            scores.append(float(np.mean(sim[mask])))
        return float(np.mean(scores)) if scores else 0.0

    # ── Convergence Quality ───────────────────────────────────────────
    def convergence_quality(self, cluster_scores: List[float], rounds: int,
                            temporal_stability: float, entropy: float) -> float:
        """
        Composite quality metric: combines dominance, stability, and low entropy.
        Q = dom * (0.5 * stability + 0.5 * (1 - entropy)) * round_penalty
        """
        if not cluster_scores: return 0.0
        dom = dominance_score(cluster_scores[0], cluster_scores)
        clarity = 0.5 * temporal_stability + 0.5 * (1.0 - entropy)
        round_penalty = 1.0 / (1.0 + 0.05 * max(0, rounds - 2))  # slight penalty for many rounds
        return float(dom * clarity * round_penalty)

    # ── Full run evaluation ───────────────────────────────────────────
    def evaluate(self, result) -> RunMetrics:
        """
        Evaluate a completed IECNNResult and return a RunMetrics object.
        """
        m = RunMetrics()
        m.num_rounds  = result.summary.get("rounds", 0)
        m.stop_reason = result.stop_reason or ""

        if result.top_cluster is None:
            return m

        # Cluster scores from all_rounds history
        scores_list = [r.get("top_score", 0.0) for r in result.rounds]
        m.cluster_entropy = self.cluster_entropy(scores_list) if scores_list else 0.0

        # Temporal stability: compare centroids in last two rounds
        centroids = [r.get("centroid") for r in result.rounds if r.get("centroid") is not None]
        if len(centroids) >= 2:
            m.temporal_stability = self.temporal_stability(centroids[-2], centroids[-1])

        m.num_clusters = result.top_cluster.size
        m.dominance    = float(result.rounds[-1].get("dominance", 0.0)) if result.rounds else 0.0

        m.convergence_quality = self.convergence_quality(
            scores_list, m.num_rounds, m.temporal_stability, m.cluster_entropy
        )
        m.per_round = list(result.rounds)
        return m

    def compare(self, results: list) -> dict:
        """Compare multiple IECNNResult objects on the same input."""
        metrics = [self.evaluate(r) for r in results]
        best_idx = max(range(len(metrics)), key=lambda i: metrics[i].convergence_quality)
        return {
            "count":    len(results),
            "best_idx": best_idx,
            "metrics":  metrics,
            "best_quality": metrics[best_idx].convergence_quality,
        }
