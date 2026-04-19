import numpy as np
from typing import List, Dict, Optional, Any
from .basemapping import BaseMapper, BaseMap
from .neural_dot import BiasVector, DotGenerator
from .aim import AIMLayer
from .convergence import ConvergenceLayer, Cluster
from .pruning import PruningLayer
from .iteration import IterationController, StopReason


class IECNNResult:
    """
    The output of a full IECNN pipeline run.

    Contains:
      - output: the final stable prediction vector
      - basemap: the structured BaseMap of the input
      - top_cluster: the winning convergence cluster
      - iteration_summary: full record of all rounds
      - stop_reason: why iteration halted
    """

    def __init__(
        self,
        output: np.ndarray,
        basemap: BaseMap,
        top_cluster: Optional[Cluster],
        iteration_summary: Dict,
        stop_reason: str,
        all_rounds: List[Dict],
    ):
        self.output = output
        self.basemap = basemap
        self.top_cluster = top_cluster
        self.iteration_summary = iteration_summary
        self.stop_reason = stop_reason
        self.all_rounds = all_rounds

    def __repr__(self):
        return (
            f"IECNNResult("
            f"rounds={self.iteration_summary.get('rounds_completed', 0)}, "
            f"stop='{self.stop_reason}', "
            f"output_norm={np.linalg.norm(self.output):.3f})"
        )


class IECNN:
    """
    Iterative Emergent Convergent Neural Network.

    A novel AI architecture where many independent neural dots
    each produce a candidate prediction. The system identifies
    convergent (most agreed-upon) results, merges them, discards
    the rest, and iterates until a stable output emerges.

    Unlike transformers or ANNs, IECNN:
      - Uses neural dots (complete mini-predictors) not neurons (signal passers)
      - Learns through convergence selection, not gradient descent
      - Builds memory through iteration, not internal dot state
      - Uses a unified BaseMapping representation for all modalities

    Layers (in order):
      1. Input          — receives raw data
      2. BaseMapping    — converts to structured maps
      3. Dot Generation — creates many neural dots
      4. Prediction     — each dot independently predicts
      5. AIM            — attention inverse mechanism (parallel with originals)
      6. Convergence    — finds agreement clusters
      7. Merge          — centroid of winning cluster
      8. Pruning        — removes weak candidates and clusters
      9. Iteration Ctrl — checks stopping conditions, loops back
     10. Output         — returns final stable result
    """

    def __init__(
        self,
        feature_dim: int = 128,
        num_dots: int = 64,
        max_iterations: int = 10,
        similarity_threshold: float = 0.45,
        dominance_threshold: float = 0.70,
        novelty_threshold: float = 0.05,
        alpha: float = 0.7,
        max_aim_variants: int = 3,
        soft_conf_threshold: float = 0.08,
        seed: int = 42,
    ):
        self.feature_dim = feature_dim
        self.num_dots = num_dots
        self.seed = seed

        self.base_mapper = BaseMapper(feature_dim=feature_dim)

        self.base_bias = BiasVector(
            attention_bias=0.5,
            granularity_bias=0.5,
            abstraction_bias=0.5,
            inversion_bias=0.3,
            sampling_temperature=1.0,
        )

        self.dot_generator = DotGenerator(
            num_dots=num_dots,
            feature_dim=feature_dim,
            base_bias=self.base_bias,
            seed=seed,
        )

        self.aim = AIMLayer(
            max_variants_per_dot=max_aim_variants,
            seed=seed,
        )

        self.convergence = ConvergenceLayer(
            similarity_threshold=similarity_threshold,
            alpha=alpha,
            dominance_threshold=dominance_threshold,
        )

        self.pruning = PruningLayer(
            soft_confidence_threshold=soft_conf_threshold,
            max_aim_variants_per_dot=max_aim_variants,
            alpha=alpha,
        )

        self.iteration_ctrl = IterationController(
            max_iterations=max_iterations,
            dominance_threshold=dominance_threshold,
            novelty_threshold=novelty_threshold,
        )

    def fit(self, texts: List[str]) -> "IECNN":
        """
        Train BaseMapping vocabulary on a corpus of texts.
        Optional — IECNN operates in zero-shot mode without fitting.
        """
        self.base_mapper.fit(texts)
        return self

    def run(self, text: str, verbose: bool = False) -> IECNNResult:
        """
        Run the full IECNN pipeline on input text.

        Process:
          1. BaseMap the input
          2. Generate all dots
          3. Iterative loop:
             a. All dots predict on current basemap
             b. AIM generates inverted variants (parallel)
             c. Pruning Stage 1: soft filter candidates
             d. Convergence: cluster all candidates
             e. Pruning Stage 2 & 3: compress and hard-select clusters
             f. Check stopping conditions
             g. If not stopped, merge top cluster → new input
          4. Return final stable output
        """
        self.iteration_ctrl.reset()
        basemap = self.base_mapper.transform(text)
        dots = self.dot_generator.generate()
        current_basemap = basemap
        all_rounds = []
        top_cluster = None
        final_output = basemap.pool("mean")

        if verbose:
            print(f"\n{'='*60}")
            print(f"IECNN Processing: '{text[:60]}{'...' if len(text)>60 else ''}'")
            print(f"BaseMap: {basemap}")
            print(f"Dots: {len(dots)}")
            print(f"{'='*60}")

        while True:
            round_num = self.iteration_ctrl.current_round

            dot_predictions = self.dot_generator.run_all(current_basemap, dots)

            all_candidates = self.aim.transform(dot_predictions, current_basemap)

            filtered_candidates, s1_stats = self.pruning.stage1_soft_filter(all_candidates)

            if not filtered_candidates:
                if verbose:
                    print(f"Round {round_num}: No candidates survived soft filter. Stopping.")
                break

            clusters, assignments, sim_matrix = self.convergence.run(filtered_candidates)

            _, surviving_clusters, pruning_stats = self.pruning.run(
                filtered_candidates, clusters
            )

            if not surviving_clusters:
                surviving_clusters = clusters[:1] if clusters else []

            self.iteration_ctrl.record_round(surviving_clusters, pruning_stats)

            conv_summary = self.convergence.summarize(surviving_clusters)
            dom, is_dominant = self.convergence.compute_dominance(surviving_clusters)

            round_info = {
                "round": round_num,
                "num_candidates": len(all_candidates),
                "num_filtered": len(filtered_candidates),
                "num_clusters": len(surviving_clusters),
                "convergence": conv_summary,
                "pruning": pruning_stats,
                "dominance": float(dom),
            }
            all_rounds.append(round_info)

            if verbose:
                print(
                    f"Round {round_num}: "
                    f"{len(all_candidates)} candidates → "
                    f"{len(filtered_candidates)} filtered → "
                    f"{len(surviving_clusters)} clusters | "
                    f"dom={dom:.3f} top_score={conv_summary['top_score']:.4f}"
                )

            if surviving_clusters:
                top_cluster = surviving_clusters[0]
                final_output = top_cluster.centroid.copy()

            should_stop, reason = self.iteration_ctrl.should_stop(surviving_clusters)
            if should_stop:
                if verbose:
                    print(f"\nStopping: {reason}")
                break

            refined = self.iteration_ctrl.advance(surviving_clusters, final_output)
            current_basemap = self._blend_basemap(basemap, refined)

            self._update_bias(surviving_clusters, all_candidates, assignments)

        norm = np.linalg.norm(final_output)
        if norm > 1e-10:
            final_output = final_output / norm * np.sqrt(self.feature_dim)

        if verbose:
            summary = self.iteration_ctrl.summary()
            print(f"\nCompleted in {summary['rounds_completed']} rounds.")
            print(f"Stop reason: {summary['stop_reason']}")
            if top_cluster:
                print(f"Top cluster: {top_cluster}")

        return IECNNResult(
            output=final_output,
            basemap=basemap,
            top_cluster=top_cluster,
            iteration_summary=self.iteration_ctrl.summary(),
            stop_reason=self.iteration_ctrl.stop_reason or StopReason.BUDGET,
            all_rounds=all_rounds,
        )

    def _blend_basemap(self, original: BaseMap, refined_vec: np.ndarray) -> BaseMap:
        """
        Blend the original BaseMap matrix with the refined convergence output.
        This is how the Iteration Controller feeds merged output back into the system.
        The refined vector acts as a "soft prompt" that biases the next round.
        """
        blended_matrix = original.matrix.copy()
        norm = np.linalg.norm(refined_vec)
        if norm > 1e-10:
            refined_normed = refined_vec / norm
            for i in range(len(blended_matrix)):
                blended_matrix[i] = 0.8 * blended_matrix[i] + 0.2 * refined_normed

        from .basemapping import BaseMap as BM
        return BM(
            matrix=blended_matrix,
            bases=original.bases,
            modifiers=original.modifiers,
            metadata={**original.metadata, "blended": True},
        )

    def _update_bias(
        self,
        surviving_clusters: List[Cluster],
        candidates: List,
        assignments: List[int],
    ):
        """
        AIM-assisted learning: if inversion variants consistently win,
        shift the base bias toward those strategies.
        """
        if not surviving_clusters:
            return

        winning_ids = {c.cluster_id for c in surviving_clusters[:1]}
        winning_aim_count = 0
        total_winning = 0

        for i, (_, _, info) in enumerate(candidates):
            if i < len(assignments) and assignments[i] in winning_ids:
                total_winning += 1
                if info.get("inversion_type") is not None:
                    winning_aim_count += 1

        if total_winning > 0:
            aim_win_ratio = winning_aim_count / total_winning
            current_arr = self.base_bias.to_array()
            target_inv_bias = min(0.9, current_arr[3] + aim_win_ratio * 0.1)
            target_arr = current_arr.copy()
            target_arr[3] = target_inv_bias

            winning_bias = BiasVector.from_array(target_arr)
            self.base_bias = self.base_bias.update(winning_bias, learning_rate=0.05)
            self.dot_generator.base_bias = self.base_bias

    def similarity(self, text_a: str, text_b: str) -> float:
        """
        Compute semantic similarity between two texts by running IECNN
        on both and measuring the similarity of their stable outputs.
        """
        from .formulas import similarity_score
        result_a = self.run(text_a)
        result_b = self.run(text_b)
        return similarity_score(result_a.output, result_b.output, self.convergence.alpha)

    def encode(self, text: str) -> np.ndarray:
        """
        Encode text to a stable IECNN representation vector.
        """
        return self.run(text).output
