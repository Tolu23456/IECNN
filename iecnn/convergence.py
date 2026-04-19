import numpy as np
from typing import List, Tuple, Dict, Optional
from .formulas import similarity_score, convergence_score, dominance_score


class Cluster:
    """
    A group of predictions that have converged on a similar answer.
    Represented by its member predictions, a centroid, and a score.
    """

    def __init__(self, cluster_id: int):
        self.cluster_id = cluster_id
        self.predictions: List[np.ndarray] = []
        self.confidences: List[float] = []
        self.infos: List[Dict] = []
        self.centroid: Optional[np.ndarray] = None
        self.score: float = 0.0

    def add(self, prediction: np.ndarray, confidence: float, info: Dict):
        self.predictions.append(prediction)
        self.confidences.append(confidence)
        self.infos.append(info)
        self._update_centroid()

    def _update_centroid(self):
        if self.predictions:
            self.centroid = np.mean(np.stack(self.predictions), axis=0)

    def compute_score(self, alpha: float = 0.7):
        self.score = convergence_score(self.predictions, self.confidences, alpha)
        return self.score

    @property
    def size(self) -> int:
        return len(self.predictions)

    @property
    def mean_confidence(self) -> float:
        if not self.confidences:
            return 0.0
        return float(np.mean(self.confidences))

    def sources(self) -> Dict[str, int]:
        counts: Dict[str, int] = {}
        for info in self.infos:
            src = info.get("source", "unknown")
            counts[src] = counts.get(src, 0) + 1
        return counts

    def __repr__(self):
        return (
            f"Cluster(id={self.cluster_id}, size={self.size}, "
            f"score={self.score:.4f}, conf={self.mean_confidence:.3f})"
        )


class ConvergenceLayer:
    """
    Identifies agreement among predictions using similarity + agreement strength.

    Process:
      1. Build pairwise similarity matrix over all predictions
         (in unified BaseMapping representation space)
      2. Assign predictions to clusters via threshold-based grouping
      3. Score each cluster using the Convergence Score formula
      4. Report dominance of the leading cluster

    Design principle: A single similarity metric over BaseMapping
    representations, consistent across all data types.
    """

    def __init__(
        self,
        similarity_threshold: float = 0.5,
        alpha: float = 0.7,
        dominance_threshold: float = 0.75,
    ):
        self.similarity_threshold = similarity_threshold
        self.alpha = alpha
        self.dominance_threshold = dominance_threshold

    def _build_similarity_matrix(
        self, predictions: List[np.ndarray]
    ) -> np.ndarray:
        """Compute pairwise similarity matrix using Formula 1."""
        n = len(predictions)
        sim_matrix = np.zeros((n, n), dtype=np.float32)
        for i in range(n):
            for j in range(i, n):
                s = similarity_score(predictions[i], predictions[j], self.alpha)
                sim_matrix[i, j] = s
                sim_matrix[j, i] = s
        return sim_matrix

    def _cluster_predictions(
        self,
        predictions: List[np.ndarray],
        sim_matrix: np.ndarray,
    ) -> List[int]:
        """
        Greedy threshold-based clustering.
        Each prediction joins the first cluster whose centroid similarity
        exceeds the threshold. If none found, starts a new cluster.
        """
        n = len(predictions)
        assignments = [-1] * n
        cluster_members: Dict[int, List[int]] = {}
        cluster_centroids: Dict[int, np.ndarray] = {}
        next_cluster_id = 0

        for i in range(n):
            best_cluster = -1
            best_sim = self.similarity_threshold

            for cid, centroid in cluster_centroids.items():
                s = similarity_score(predictions[i], centroid, self.alpha)
                if s > best_sim:
                    best_sim = s
                    best_cluster = cid

            if best_cluster == -1:
                cid = next_cluster_id
                next_cluster_id += 1
                assignments[i] = cid
                cluster_members[cid] = [i]
                cluster_centroids[cid] = predictions[i].copy()
            else:
                assignments[i] = best_cluster
                cluster_members[best_cluster].append(i)
                members = [predictions[j] for j in cluster_members[best_cluster]]
                cluster_centroids[best_cluster] = np.mean(np.stack(members), axis=0)

        return assignments

    def run(
        self,
        candidates: List[Tuple[np.ndarray, float, Dict]],
    ) -> Tuple[List[Cluster], List[int], np.ndarray]:
        """
        Run convergence on all candidate predictions.

        Args:
          candidates: list of (prediction, confidence, info)

        Returns:
          clusters: list of Cluster objects sorted by score descending
          assignments: cluster id for each candidate
          sim_matrix: pairwise similarity matrix
        """
        if not candidates:
            return [], [], np.array([])

        predictions = [c[0] for c in candidates]
        confidences = [c[1] for c in candidates]
        infos = [c[2] for c in candidates]

        sim_matrix = self._build_similarity_matrix(predictions)
        assignments = self._cluster_predictions(predictions, sim_matrix)

        cluster_map: Dict[int, Cluster] = {}
        for i, (pred, conf, info) in enumerate(zip(predictions, confidences, infos)):
            cid = assignments[i]
            if cid not in cluster_map:
                cluster_map[cid] = Cluster(cluster_id=cid)
            cluster_map[cid].add(pred, conf, info)

        for cluster in cluster_map.values():
            cluster.compute_score(self.alpha)

        clusters = sorted(cluster_map.values(), key=lambda c: c.score, reverse=True)
        return clusters, assignments, sim_matrix

    def compute_dominance(self, clusters: List[Cluster]) -> Tuple[float, bool]:
        """
        Formula 9 — compute dominance of the leading cluster.
        Returns (dominance_ratio, is_dominant).
        """
        if not clusters:
            return 0.0, False
        scores = [c.score for c in clusters]
        dom = dominance_score(scores[0], scores)
        return dom, dom > self.dominance_threshold

    def summarize(self, clusters: List[Cluster]) -> Dict:
        """Return a summary of the convergence results."""
        if not clusters:
            return {"num_clusters": 0, "top_score": 0.0, "dominance": 0.0}

        scores = [c.score for c in clusters]
        dom, is_dom = self.compute_dominance(clusters)

        return {
            "num_clusters": len(clusters),
            "top_score": float(clusters[0].score),
            "top_cluster_size": clusters[0].size,
            "dominance": float(dom),
            "is_dominant": is_dom,
            "all_scores": [float(s) for s in scores],
            "top_sources": clusters[0].sources() if clusters else {},
        }
